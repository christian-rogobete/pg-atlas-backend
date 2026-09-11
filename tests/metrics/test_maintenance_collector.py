"""
Unit tests for the GitHub maintenance-signal collector.

Exercises the fail-closed gate, actor/association classification (including
the event-actor authorization proxy and unknown attribution), cohort
assembly against canned GraphQL payloads, rate-limit halts, and the
per-signal persistence merge — all without network or database.

SPDX-FileCopyrightText: 2026 PG Atlas contributors
SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from github import GithubException

import pg_atlas.procrastinate.github_maintenance as gm
from pg_atlas.config import settings
from pg_atlas.metrics.maintenance import (
    STATE_INCOMPLETE,
    STATE_NOT_APPLICABLE,
    STATE_OK,
    BacklogSignal,
    MaintenanceSignals,
    ResponsivenessSignal,
)
from pg_atlas.procrastinate.github_maintenance import (
    MaintenanceCollectionFailed,
    classify_pr_events,
    collect_maintenance_for_repo,
    external_tracker_declared,
    maintenance_metric_allowed,
    merge_collected_signals,
    resolve_response_item,
)

NOW = dt.datetime(2026, 9, 9, 12, 0, tzinfo=dt.UTC)


def _iso(days_ago: float) -> str:
    return (NOW - dt.timedelta(days=days_ago)).isoformat()


def _actor(login: str, typename: str = "User") -> dict[str, str]:
    return {"__typename": typename, "login": login}


_QueryHandler = Callable[[str, dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]


class _FakeRequester:
    """Dispatch GraphQL queries to a canned handler, recording every call."""

    def __init__(self, handler: _QueryHandler) -> None:
        self._handler = handler
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def graphql_query(self, query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append((query, variables))

        return self._handler(query, variables)


def _patch_client(monkeypatch: pytest.MonkeyPatch, requester: _FakeRequester, **extra: Any) -> None:
    client = SimpleNamespace(**{"_Github__requester": requester}, **extra)
    monkeypatch.setattr(gm, "get_github_client", lambda: client)


@pytest.fixture
def default_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the contested parameters so canned payloads stay deterministic."""

    monkeypatch.setattr(settings, "MAINTENANCE_WINDOW_DAYS", 180)
    monkeypatch.setattr(settings, "MAINTENANCE_RESPONSE_INTERVAL_DAYS", 7)
    monkeypatch.setattr(settings, "MAINTENANCE_ITEM_PAGE_CAP", 10)
    monkeypatch.setattr(settings, "MAINTENANCE_REQUEST_CAP", 120)
    monkeypatch.setattr(settings, "MAINTENANCE_TIME_CAP_SECONDS", 300.0)
    monkeypatch.setattr(settings, "MAINTENANCE_RATE_LIMIT_MAX_WAIT_SECONDS", 120.0)
    monkeypatch.setattr(settings, "MAINTENANCE_INCLUDE_DRAFT_PRS", True)
    monkeypatch.setattr(settings, "MAINTENANCE_EXTERNAL_TRACKER_REPOS", "")


# ---------------------------------------------------------------------------
# Fail-closed gate
# ---------------------------------------------------------------------------


class TestGate:
    def test_disabled_flag_allows_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "MAINTENANCE_METRIC_ENABLED", False)
        monkeypatch.setattr(settings, "MAINTENANCE_METRIC_ALLOWLIST", "*")
        assert not maintenance_metric_allowed("Soneso", "stellar-php-sdk")

    def test_empty_allowlist_allows_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "MAINTENANCE_METRIC_ENABLED", True)
        monkeypatch.setattr(settings, "MAINTENANCE_METRIC_ALLOWLIST", "")
        assert not maintenance_metric_allowed("Soneso", "stellar-php-sdk")

    def test_allowlist_match_is_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "MAINTENANCE_METRIC_ENABLED", True)
        monkeypatch.setattr(settings, "MAINTENANCE_METRIC_ALLOWLIST", "soneso/STELLAR-php-sdk, Other/repo")
        assert maintenance_metric_allowed("Soneso", "stellar-php-sdk")
        assert maintenance_metric_allowed("other", "REPO")
        assert not maintenance_metric_allowed("Soneso", "stellar_flutter_sdk")

    def test_star_allows_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "MAINTENANCE_METRIC_ENABLED", True)
        monkeypatch.setattr(settings, "MAINTENANCE_METRIC_ALLOWLIST", "*")
        assert maintenance_metric_allowed("any", "repo")

    def test_malformed_entries_are_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "MAINTENANCE_METRIC_ENABLED", True)
        monkeypatch.setattr(settings, "MAINTENANCE_METRIC_ALLOWLIST", "just-a-name, owner/, /repo, a/b/c")
        assert not maintenance_metric_allowed("just-a-name", "")
        assert not maintenance_metric_allowed("owner", "")
        assert not maintenance_metric_allowed("a", "b")

    def test_external_tracker_declaration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "MAINTENANCE_EXTERNAL_TRACKER_REPOS", "Soneso/stellar-php-sdk")
        assert external_tracker_declared("soneso", "STELLAR-PHP-SDK")
        assert not external_tracker_declared("soneso", "other")


# ---------------------------------------------------------------------------
# Actor and event classification
# ---------------------------------------------------------------------------


class TestActorClassification:
    def test_bot_typename_is_bot(self) -> None:
        assert gm._is_bot_actor(gm._ActorRef(typename="Bot", login="github-actions"))

    def test_bot_login_pattern_is_bot(self) -> None:
        assert gm._is_bot_actor(gm._ActorRef(typename="User", login="dependabot"))

    def test_regular_user_is_not_bot(self) -> None:
        assert not gm._is_bot_actor(gm._ActorRef(typename="User", login="alice"))

    def test_missing_actor_is_not_classified_as_bot(self) -> None:
        assert not gm._is_bot_actor(None)


class TestScanCommentPage:
    def test_first_maintainer_comment_wins(self) -> None:
        comments = [
            gm._CommentNode(createdAt=_iso(29.5), authorAssociation="NONE", author=gm._ActorRef(login="rando")),
            gm._CommentNode(createdAt=_iso(29), authorAssociation="MEMBER", author=gm._ActorRef(login="bob")),
            gm._CommentNode(createdAt=_iso(28), authorAssociation="OWNER", author=gm._ActorRef(login="carol")),
        ]
        scan = gm._scan_comment_page(comments, "alice")
        assert scan.qualifying_time == NOW - dt.timedelta(days=29)
        assert scan.unknown_times == []

    def test_bot_and_author_comments_do_not_qualify(self) -> None:
        comments = [
            gm._CommentNode(createdAt=_iso(29), authorAssociation="MEMBER", author=gm._ActorRef(typename="Bot", login="x")),
            gm._CommentNode(createdAt=_iso(28), authorAssociation="OWNER", author=gm._ActorRef(login="alice")),
        ]
        scan = gm._scan_comment_page(comments, "alice")
        assert scan.qualifying_time is None
        assert scan.unknown_times == []

    def test_unresolvable_author_with_qualifying_association_is_unknown(self) -> None:
        """Attribution is reported, never guessed — mirroring the event rule."""
        comments = [gm._CommentNode(createdAt=_iso(29), authorAssociation="MEMBER", author=None)]
        scan = gm._scan_comment_page(comments, "alice")
        assert scan.qualifying_time is None
        assert scan.unknown_times == [NOW - dt.timedelta(days=29)]

    def test_unresolvable_author_without_qualifying_association_is_nothing(self) -> None:
        """Deleted accounts normally report NONE and simply fail the filter."""
        comments = [gm._CommentNode(createdAt=_iso(29), authorAssociation="NONE", author=None)]
        scan = gm._scan_comment_page(comments, "alice")
        assert scan.qualifying_time is None
        assert scan.unknown_times == []


class TestScanReviewPage:
    def test_earliest_qualifying_review_wins(self) -> None:
        reviews = [
            gm._ReviewNode(submittedAt=_iso(28), authorAssociation="MEMBER", author=gm._ActorRef(login="bob")),
            gm._ReviewNode(submittedAt=_iso(29), authorAssociation="OWNER", author=gm._ActorRef(login="carol")),
        ]
        scan = gm._scan_review_page(reviews, "alice")
        assert scan.qualifying_time == NOW - dt.timedelta(days=29)

    def test_unsubmitted_bot_and_unknown_reviews(self) -> None:
        reviews = [
            gm._ReviewNode(submittedAt=None, authorAssociation="MEMBER", author=gm._ActorRef(login="bob")),
            gm._ReviewNode(submittedAt=_iso(29), authorAssociation="MEMBER", author=gm._ActorRef(typename="Bot", login="b")),
            gm._ReviewNode(submittedAt=_iso(28), authorAssociation="COLLABORATOR", author=None),
        ]
        scan = gm._scan_review_page(reviews, "alice")
        assert scan.qualifying_time is None
        assert scan.unknown_times == [NOW - dt.timedelta(days=28)]


class TestClassifyPrEvents:
    def test_merge_by_other_human_qualifies(self) -> None:
        events = [gm._TimelineNode(typename="MergedEvent", createdAt=_iso(29), actor=gm._ActorRef(login="bob"))]
        result = classify_pr_events(events, "alice")
        assert result.qualifying_times == [NOW - dt.timedelta(days=29)]
        assert result.unknown_times == []

    def test_close_by_other_human_qualifies_as_rejection(self) -> None:
        """A prompt human rejection is a response."""
        events = [gm._TimelineNode(typename="ClosedEvent", createdAt=_iso(29), actor=gm._ActorRef(login="bob"))]
        assert classify_pr_events(events, "alice").qualifying_times

    def test_author_withdrawal_does_not_qualify(self) -> None:
        events = [gm._TimelineNode(typename="ClosedEvent", createdAt=_iso(29), actor=gm._ActorRef(login="alice"))]
        result = classify_pr_events(events, "alice")
        assert result.qualifying_times == []
        assert result.unknown_times == []

    def test_bot_merge_does_not_qualify(self) -> None:
        events = [gm._TimelineNode(typename="MergedEvent", createdAt=_iso(29), actor=gm._ActorRef(typename="Bot", login="b"))]
        assert classify_pr_events(events, "alice").qualifying_times == []

    def test_missing_actor_is_unknown_attribution(self) -> None:
        events = [gm._TimelineNode(typename="ClosedEvent", createdAt=_iso(29), actor=None)]
        result = classify_pr_events(events, "alice")
        assert result.qualifying_times == []
        assert result.unknown_times == [NOW - dt.timedelta(days=29)]


class TestResolveResponseItem:
    def test_qualifying_within_interval_settles_item(self) -> None:
        item = resolve_response_item(
            NOW - dt.timedelta(days=30),
            [NOW - dt.timedelta(days=29)],
            [NOW - dt.timedelta(days=29.5)],
            interval_days=7,
        )
        assert item.first_response_at == NOW - dt.timedelta(days=29)
        assert not item.indeterminate

    def test_unknown_within_interval_without_qualifying_is_indeterminate(self) -> None:
        item = resolve_response_item(
            NOW - dt.timedelta(days=30),
            [],
            [NOW - dt.timedelta(days=29)],
            interval_days=7,
        )
        assert item.indeterminate

    def test_late_qualifying_with_late_unknown_stays_determinate(self) -> None:
        item = resolve_response_item(
            NOW - dt.timedelta(days=30),
            [NOW - dt.timedelta(days=10)],
            [NOW - dt.timedelta(days=12)],
            interval_days=7,
        )
        assert not item.indeterminate
        assert item.first_response_at == NOW - dt.timedelta(days=10)

    def test_truncated_detection_is_indeterminate(self) -> None:
        item = resolve_response_item(NOW - dt.timedelta(days=30), [], [], interval_days=7, detection_truncated=True)
        assert item.indeterminate

    def test_nothing_found_is_unanswered(self) -> None:
        item = resolve_response_item(NOW - dt.timedelta(days=30), [], [], interval_days=7)
        assert item.first_response_at is None
        assert not item.indeterminate


# ---------------------------------------------------------------------------
# Full collection against canned payloads
# ---------------------------------------------------------------------------


def _happy_path_handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if query == gm._OVERVIEW_QUERY:
        return {}, {"data": {"repository": {"pushedAt": _iso(2), "hasIssuesEnabled": True, "isArchived": False}}}

    if query == gm._OPEN_ISSUES_QUERY:
        return {}, {
            "data": {
                "repository": {
                    "issues": {
                        "totalCount": 2,
                        "nodes": [{"createdAt": _iso(100)}, {"createdAt": _iso(10)}],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        }

    if query == gm._OPEN_PULLS_QUERY:
        return {}, {
            "data": {
                "repository": {
                    "pullRequests": {
                        "totalCount": 1,
                        "nodes": [{"createdAt": _iso(20)}],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        }

    if query == gm._ISSUE_COHORT_QUERY:
        assert variables["after"] is None, "listing must stop at the window boundary"
        return {}, {
            "data": {
                "repository": {
                    "issues": {
                        "nodes": [
                            {
                                "number": 1,
                                "createdAt": _iso(30),
                                "author": _actor("alice"),
                                "comments": {
                                    "nodes": [{"createdAt": _iso(29), "authorAssociation": "MEMBER", "author": _actor("bob")}],
                                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                                },
                            },
                            {
                                "number": 2,
                                "createdAt": _iso(40),
                                "author": _actor("carol"),
                                "comments": {
                                    "nodes": [
                                        {"createdAt": _iso(39), "authorAssociation": "OWNER", "author": _actor("carol")},
                                        {
                                            "createdAt": _iso(38),
                                            "authorAssociation": "MEMBER",
                                            "author": _actor("helper[bot]", "Bot"),
                                        },
                                        {"createdAt": _iso(37), "authorAssociation": "NONE", "author": _actor("rando")},
                                    ],
                                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                                },
                            },
                            {
                                "number": 4,
                                "createdAt": _iso(35),
                                "authorAssociation": "MEMBER",
                                "author": _actor("bob"),
                                "comments": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}},
                            },
                            {
                                "number": 3,
                                "createdAt": _iso(200),
                                "author": _actor("old"),
                                "comments": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}},
                            },
                        ],
                        "pageInfo": {"hasNextPage": True, "endCursor": "cursor-must-not-be-followed"},
                    }
                }
            }
        }

    if query == gm._PULL_COHORT_QUERY:
        return {}, {
            "data": {
                "repository": {
                    "pullRequests": {
                        "nodes": [
                            {
                                "number": 13,
                                "createdAt": _iso(25),
                                "isDraft": False,
                                "authorAssociation": "OWNER",
                                "author": _actor("grace"),
                                "reviews": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}},
                                "timelineItems": {"nodes": [], "pageInfo": {"hasNextPage": False}},
                            },
                            {
                                "number": 10,
                                "createdAt": _iso(30),
                                "isDraft": False,
                                "author": _actor("dave"),
                                "reviews": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}},
                                "timelineItems": {
                                    "nodes": [{"__typename": "MergedEvent", "createdAt": _iso(29), "actor": _actor("bob")}],
                                    "pageInfo": {"hasNextPage": False},
                                },
                            },
                            {
                                "number": 11,
                                "createdAt": _iso(50),
                                "isDraft": False,
                                "author": _actor("erin"),
                                "reviews": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}},
                                "timelineItems": {
                                    "nodes": [{"__typename": "ClosedEvent", "createdAt": _iso(49), "actor": _actor("erin")}],
                                    "pageInfo": {"hasNextPage": False},
                                },
                            },
                            {
                                "number": 12,
                                "createdAt": _iso(20),
                                "isDraft": False,
                                "author": _actor("frank"),
                                "reviews": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}},
                                "timelineItems": {
                                    "nodes": [{"__typename": "ClosedEvent", "createdAt": _iso(19), "actor": None}],
                                    "pageInfo": {"hasNextPage": False},
                                },
                            },
                        ],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        }

    raise AssertionError(f"Unexpected query: {query[:60]}")


class TestCollectMaintenanceForRepo:
    async def test_happy_path_signals(self, monkeypatch: pytest.MonkeyPatch, default_parameters: None) -> None:
        requester = _FakeRequester(_happy_path_handler)
        _patch_client(monkeypatch, requester)

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=False, now=NOW)
        signals = collected.signals

        assert collected.pushed_at == NOW - dt.timedelta(days=2)
        assert signals.issues_enabled is True
        assert signals.archived is False
        assert signals.requests_used == len(requester.calls)

        assert signals.issue_backlog is not None
        assert signals.issue_backlog.state == STATE_OK
        assert signals.issue_backlog.open_count == 2
        assert signals.issue_backlog.median_open_age_days == 55.0

        assert signals.pr_backlog is not None
        assert signals.pr_backlog.open_count == 1
        assert signals.pr_backlog.median_open_age_days == 20.0

        # Issue #1 answered by a MEMBER within the interval; #2 got only the
        # author's own, a bot's, and an unassociated comment; #4 is the
        # maintainer's own work and leaves the cohort; #3 is outside the
        # window and stops the listing.
        assert signals.issue_responsiveness is not None
        assert signals.issue_responsiveness.state == STATE_OK
        assert signals.issue_responsiveness.cohort_size == 2
        assert signals.issue_responsiveness.maintainer_authored_count == 1
        assert signals.issue_responsiveness.eligible_size == 2
        assert signals.issue_responsiveness.responded_within_interval == 1
        assert signals.issue_responsiveness.response_fraction == 0.5

        # PR #10 merged by another human, #11 withdrawn by its author
        # (unanswered), #12 closed by an unresolvable actor (indeterminate),
        # #13 authored by a maintainer and excluded from the cohort.
        assert signals.pr_responsiveness is not None
        assert signals.pr_responsiveness.cohort_size == 2
        assert signals.pr_responsiveness.maintainer_authored_count == 1
        assert signals.pr_responsiveness.responded_within_interval == 1
        assert signals.pr_responsiveness.response_fraction == 0.5
        assert signals.pr_responsiveness.unknown_attribution_count == 1

        assert signals.release_fallback is None

    async def test_issues_disabled_is_not_applicable(self, monkeypatch: pytest.MonkeyPatch, default_parameters: None) -> None:
        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            if query == gm._OVERVIEW_QUERY:
                return {}, {"data": {"repository": {"pushedAt": _iso(2), "hasIssuesEnabled": False, "isArchived": False}}}

            return _happy_path_handler(query, variables)

        requester = _FakeRequester(handler)
        _patch_client(monkeypatch, requester)

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=False, now=NOW)
        signals = collected.signals

        assert signals.issue_responsiveness is not None
        assert signals.issue_responsiveness.state == STATE_NOT_APPLICABLE
        assert signals.issue_backlog is not None
        assert signals.issue_backlog.state == STATE_NOT_APPLICABLE
        assert signals.pr_responsiveness is not None
        assert signals.pr_responsiveness.state == STATE_OK
        assert all(query != gm._OPEN_ISSUES_QUERY and query != gm._ISSUE_COHORT_QUERY for query, _ in requester.calls)

    async def test_declared_external_tracker_is_not_applicable(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        monkeypatch.setattr(settings, "MAINTENANCE_EXTERNAL_TRACKER_REPOS", "owner/repo")
        requester = _FakeRequester(_happy_path_handler)
        _patch_client(monkeypatch, requester)

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=False, now=NOW)

        assert collected.signals.external_tracker_declared is True
        assert collected.signals.issue_responsiveness is not None
        assert collected.signals.issue_responsiveness.state == STATE_NOT_APPLICABLE

    async def test_comment_continuation_is_followed(self, monkeypatch: pytest.MonkeyPatch, default_parameters: None) -> None:
        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            if query == gm._ISSUE_COHORT_QUERY:
                return {}, {
                    "data": {
                        "repository": {
                            "issues": {
                                "nodes": [
                                    {
                                        "number": 5,
                                        "createdAt": _iso(30),
                                        "author": _actor("alice"),
                                        "comments": {
                                            "nodes": [
                                                {
                                                    "createdAt": _iso(29.9),
                                                    "authorAssociation": "NONE",
                                                    "author": _actor("rando"),
                                                }
                                            ],
                                            "pageInfo": {"hasNextPage": True, "endCursor": "c1"},
                                        },
                                    }
                                ],
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                            }
                        }
                    }
                }

            if query == gm._ISSUE_COMMENTS_QUERY:
                assert variables["number"] == 5
                assert variables["after"] == "c1"
                return {}, {
                    "data": {
                        "repository": {
                            "issue": {
                                "comments": {
                                    "nodes": [
                                        {"createdAt": _iso(29), "authorAssociation": "COLLABORATOR", "author": _actor("bob")}
                                    ],
                                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                                }
                            }
                        }
                    }
                }

            return _happy_path_handler(query, variables)

        requester = _FakeRequester(handler)
        _patch_client(monkeypatch, requester)

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=False, now=NOW)

        assert collected.signals.issue_responsiveness is not None
        assert collected.signals.issue_responsiveness.responded_within_interval == 1

    async def test_draft_prs_excluded_when_configured(self, monkeypatch: pytest.MonkeyPatch, default_parameters: None) -> None:
        monkeypatch.setattr(settings, "MAINTENANCE_INCLUDE_DRAFT_PRS", False)

        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            if query == gm._PULL_COHORT_QUERY:
                return {}, {
                    "data": {
                        "repository": {
                            "pullRequests": {
                                "nodes": [
                                    {
                                        "number": 20,
                                        "createdAt": _iso(30),
                                        "isDraft": True,
                                        "author": _actor("dave"),
                                        "reviews": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}},
                                        "timelineItems": {"nodes": [], "pageInfo": {"hasNextPage": False}},
                                    }
                                ],
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                            }
                        }
                    }
                }

            return _happy_path_handler(query, variables)

        _patch_client(monkeypatch, _FakeRequester(handler))

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=False, now=NOW)

        assert collected.signals.pr_responsiveness is not None
        assert collected.signals.pr_responsiveness.cohort_size == 0

    async def test_rate_limit_halt_marks_remaining_signals_incomplete(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """An over-cap reset wait halts without sleeping; nothing is guessed."""

        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            if query == gm._OVERVIEW_QUERY:
                return {}, {"data": {"repository": {"pushedAt": _iso(2), "hasIssuesEnabled": True, "isArchived": False}}}

            raise GithubException(
                400,
                {"errors": [{"type": "RATE_LIMITED", "message": "API rate limit exceeded"}]},
                {"x-ratelimit-reset": str(time.time() + 3600)},
            )

        _patch_client(monkeypatch, _FakeRequester(handler))

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=True, now=NOW)
        signals = collected.signals

        for signal in (signals.issue_responsiveness, signals.issue_backlog, signals.pr_responsiveness, signals.pr_backlog):
            assert signal is not None
            assert signal.state == STATE_INCOMPLETE
            assert signal.incomplete_reason == gm.REASON_RATE_LIMITED

        assert signals.release_fallback is not None
        assert signals.release_fallback.state == STATE_INCOMPLETE
        assert collected.pushed_at is not None

    async def test_overview_failure_raises_instead_of_writing_garbage(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            raise GithubException(500, {"message": "boom"}, {})

        _patch_client(monkeypatch, _FakeRequester(handler))

        with pytest.raises(MaintenanceCollectionFailed):
            await collect_maintenance_for_repo("owner", "repo", need_release_fallback=False, now=NOW)

    async def test_release_fallback_collects_dated_non_draft_releases(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        monkeypatch.setattr(settings, "MAINTENANCE_INCLUDE_PRERELEASES", False)

        fake_releases = [
            SimpleNamespace(draft=False, prerelease=False, published_at=NOW - dt.timedelta(days=10)),
            SimpleNamespace(draft=True, prerelease=False, published_at=NOW - dt.timedelta(days=20)),
            SimpleNamespace(draft=False, prerelease=True, published_at=NOW - dt.timedelta(days=30)),
            SimpleNamespace(draft=False, prerelease=False, published_at=None),
            SimpleNamespace(draft=False, prerelease=False, published_at=NOW - dt.timedelta(days=40)),
        ]
        fake_repo = SimpleNamespace(get_releases=lambda: fake_releases)

        def _get_repo(path: str) -> SimpleNamespace:
            return fake_repo

        requester = _FakeRequester(_happy_path_handler)
        _patch_client(monkeypatch, requester, get_repo=_get_repo)

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=True, now=NOW)

        fallback = collected.signals.release_fallback
        assert fallback is not None
        assert fallback.state == STATE_OK
        assert fallback.publication_dates == [
            (NOW - dt.timedelta(days=40)).date().isoformat(),
            (NOW - dt.timedelta(days=10)).date().isoformat(),
        ]
        assert fallback.include_prereleases is False


# ---------------------------------------------------------------------------
# Per-signal persistence merge
# ---------------------------------------------------------------------------


def _signals_with(
    issue_responsiveness: ResponsivenessSignal | None = None,
    issue_backlog: BacklogSignal | None = None,
    collected_at: str = "",
) -> MaintenanceSignals:
    return MaintenanceSignals(
        schema_version=1,
        collected_at=collected_at or NOW.isoformat(),
        window_days=180,
        response_interval_days=7,
        issue_responsiveness=issue_responsiveness,
        issue_backlog=issue_backlog,
    )


class TestMergeCollectedSignals:
    def test_incomplete_never_overwrites_complete(self) -> None:
        old_ok = ResponsivenessSignal(state=STATE_OK, as_of=_iso(3), response_fraction=0.8)
        new_incomplete = ResponsivenessSignal(state=STATE_INCOMPLETE, as_of=_iso(0), incomplete_reason="rate-limited")

        merged = merge_collected_signals(_signals_with(old_ok), _signals_with(new_incomplete))

        assert merged.issue_responsiveness == old_ok
        assert merged.collected_at == NOW.isoformat()

    def test_fresh_ok_replaces_old_ok(self) -> None:
        old_ok = BacklogSignal(state=STATE_OK, as_of=_iso(3), open_count=5)
        new_ok = BacklogSignal(state=STATE_OK, as_of=_iso(0), open_count=6)

        merged = merge_collected_signals(_signals_with(None, old_ok), _signals_with(None, new_ok))

        assert merged.issue_backlog == new_ok

    def test_not_applicable_replaces_old_ok(self) -> None:
        """A repo that disabled its tracker must not keep stale real values."""
        old_ok = BacklogSignal(state=STATE_OK, as_of=_iso(3), open_count=5)
        new_na = BacklogSignal(state=STATE_NOT_APPLICABLE, as_of=_iso(0))

        merged = merge_collected_signals(_signals_with(None, old_ok), _signals_with(None, new_na))

        assert merged.issue_backlog == new_na

    def test_no_previous_payload_keeps_new(self) -> None:
        new = _signals_with(ResponsivenessSignal(state=STATE_INCOMPLETE, as_of=_iso(0), incomplete_reason="page-cap"))
        assert merge_collected_signals(None, new) == new


# ---------------------------------------------------------------------------
# Cap containment
# ---------------------------------------------------------------------------


class TestCapContainment:
    async def test_cohort_page_cap_leaves_other_signals_collectable(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """A capped issue cohort is incomplete by itself; PR collection continues."""

        monkeypatch.setattr(settings, "MAINTENANCE_ITEM_PAGE_CAP", 1)

        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            if query == gm._ISSUE_COHORT_QUERY:
                return {}, {
                    "data": {
                        "repository": {
                            "issues": {
                                "nodes": [
                                    {
                                        "number": 1,
                                        "createdAt": _iso(30),
                                        "author": _actor("alice"),
                                        "comments": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}},
                                    }
                                ],
                                "pageInfo": {"hasNextPage": True, "endCursor": "more"},
                            }
                        }
                    }
                }

            return _happy_path_handler(query, variables)

        _patch_client(monkeypatch, _FakeRequester(handler))

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=False, now=NOW)
        signals = collected.signals

        assert signals.issue_responsiveness is not None
        assert signals.issue_responsiveness.state == STATE_INCOMPLETE
        assert signals.issue_responsiveness.incomplete_reason == gm.REASON_PAGE_CAP
        assert signals.issue_responsiveness.response_fraction is None
        assert signals.pr_responsiveness is not None
        assert signals.pr_responsiveness.state == STATE_OK
        assert signals.pr_responsiveness.response_fraction == 0.5

    async def test_release_fallback_item_cap_applies_to_filtered_releases(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """A long run of drafts must hit the walk cap, not drain the budget."""

        monkeypatch.setattr(gm, "_RELEASE_ITEM_CAP", 5)

        fake_releases = [
            SimpleNamespace(draft=True, prerelease=False, published_at=NOW - dt.timedelta(days=i)) for i in range(50)
        ]
        fake_repo = SimpleNamespace(get_releases=lambda: fake_releases)

        def _get_repo(path: str) -> SimpleNamespace:
            return fake_repo

        _patch_client(monkeypatch, _FakeRequester(_happy_path_handler), get_repo=_get_repo)

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=True, now=NOW)

        fallback = collected.signals.release_fallback
        assert fallback is not None
        assert fallback.state == STATE_INCOMPLETE
        assert fallback.complete is False
        assert fallback.incomplete_reason == gm.REASON_PAGE_CAP
        assert fallback.publication_dates == []


# ---------------------------------------------------------------------------
# Rate-limit machinery
# ---------------------------------------------------------------------------


def _rate_limited_exception(headers: dict[str, str] | None) -> GithubException:
    return GithubException(400, {"errors": [{"type": "RATE_LIMITED", "message": "API rate limit exceeded"}]}, headers)


class TestRateLimitWaitSeconds:
    def test_graphql_envelope_with_reset_header(self) -> None:
        exc = _rate_limited_exception({"X-RateLimit-Reset": str(time.time() + 30)})
        wait = gm._rate_limit_wait_seconds(exc)
        assert wait is not None
        assert 25 < wait <= 31

    def test_retry_after_takes_precedence(self) -> None:
        exc = _rate_limited_exception({"Retry-After": "17", "X-RateLimit-Reset": str(time.time() + 3000)})
        assert gm._rate_limit_wait_seconds(exc) == 17.0

    def test_malformed_headers_fall_back_to_default_wait(self) -> None:
        exc = _rate_limited_exception({"Retry-After": "soon", "X-RateLimit-Reset": "later"})
        assert gm._rate_limit_wait_seconds(exc) == 60.0

    def test_secondary_limit_429_with_retry_after(self) -> None:
        exc = GithubException(429, {"message": "slow down"}, {"Retry-After": "9"})
        assert gm._rate_limit_wait_seconds(exc) == 9.0

    def test_403_without_rate_headers_is_not_a_rate_limit(self) -> None:
        exc = GithubException(403, {"message": "forbidden"}, {})
        assert gm._rate_limit_wait_seconds(exc) is None

    def test_ordinary_error_is_not_a_rate_limit(self) -> None:
        exc = GithubException(500, {"message": "boom"}, {"Retry-After": "9"})
        assert gm._rate_limit_wait_seconds(exc) is None


class TestRateLimitRetry:
    async def test_bounded_wait_then_retry_succeeds(self, monkeypatch: pytest.MonkeyPatch, default_parameters: None) -> None:
        """A short advertised reset is waited out once and collection completes."""

        open_issue_attempts = {"count": 0}

        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            if query == gm._OPEN_ISSUES_QUERY:
                open_issue_attempts["count"] += 1
                if open_issue_attempts["count"] == 1:
                    raise _rate_limited_exception({"X-RateLimit-Reset": str(time.time() - 5)})

            return _happy_path_handler(query, variables)

        _patch_client(monkeypatch, _FakeRequester(handler))

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=False, now=NOW)

        assert open_issue_attempts["count"] == 2
        assert collected.signals.issue_backlog is not None
        assert collected.signals.issue_backlog.state == STATE_OK
        assert collected.signals.pr_responsiveness is not None
        assert collected.signals.pr_responsiveness.state == STATE_OK


class TestReleaseFallbackBudget:
    def test_budget_charged_per_rest_page(self, monkeypatch: pytest.MonkeyPatch, default_parameters: None) -> None:
        """Walking filtered releases charges the shared budget once per page."""

        fake_releases = [
            SimpleNamespace(draft=True, prerelease=False, published_at=NOW - dt.timedelta(days=i)) for i in range(65)
        ]
        fake_repo = SimpleNamespace(get_releases=lambda: fake_releases)

        def _get_repo(path: str) -> SimpleNamespace:
            return fake_repo

        _patch_client(monkeypatch, _FakeRequester(_happy_path_handler), get_repo=_get_repo)
        collector = gm._MaintenanceCollector("owner", "repo", now=NOW)

        fallback = collector.collect_release_fallback()

        # One charge up front plus one per full page of 30 items walked.
        assert collector.requests_used == 3
        assert fallback.state == STATE_OK
        assert fallback.complete is True
        assert fallback.publication_dates == []
