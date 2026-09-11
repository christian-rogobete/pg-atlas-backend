"""
Unit tests for the GitHub maintenance-signal collector.

Exercises the fail-closed gate, actor/association classification (including
the event-actor authorization proxy and unknown attribution), code-driven
issue closes, cohort assembly against canned GraphQL payloads, rate-limit
halts, and the per-signal persistence merge — all without network or
database.

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
    monkeypatch.setattr(gm, "_get_maintenance_client", lambda: client)


class _FakePaginatedReleases:
    """Serve canned releases in 30-item pages, counting page fetches."""

    def __init__(self, items: list[SimpleNamespace]) -> None:
        self._items = items
        self.page_calls = 0

    def get_page(self, index: int) -> list[SimpleNamespace]:
        self.page_calls += 1

        return self._items[index * 30 : (index + 1) * 30]


def _releases_repo(items: list[SimpleNamespace]) -> tuple[SimpleNamespace, _FakePaginatedReleases]:
    paginated = _FakePaginatedReleases(items)

    return SimpleNamespace(get_releases=lambda: paginated), paginated


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
        result = classify_pr_events(events, "alice")
        assert result.qualifying_times == [NOW - dt.timedelta(days=29)]
        assert result.unknown_times == []

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
                                "number": 6,
                                "createdAt": _iso(33),
                                "author": _actor("erin"),
                                "comments": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}},
                                "timelineItems": {
                                    "nodes": [
                                        {
                                            "__typename": "ClosedEvent",
                                            "createdAt": _iso(31),
                                            "closer": {"__typename": "PullRequest"},
                                        }
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
                                "number": 7,
                                "createdAt": _iso(45),
                                "author": _actor("frank"),
                                "comments": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}},
                                "timelineItems": {
                                    "nodes": [{"__typename": "ClosedEvent", "createdAt": _iso(44), "closer": None}],
                                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                                },
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

        # Issue #1 answered by a MEMBER comment within the interval; #6 closed
        # by a merged PR within the interval (a code close is a response); #2
        # got only the author's own, a bot's, and an unassociated comment; #7
        # was closed manually (never a response); #4 is the maintainer's own
        # work and leaves the cohort; #3 is outside the window and stops the
        # listing.
        assert signals.issue_responsiveness is not None
        assert signals.issue_responsiveness.state == STATE_OK
        assert signals.issue_responsiveness.cohort_size == 4
        assert signals.issue_responsiveness.maintainer_authored_count == 1
        assert signals.issue_responsiveness.eligible_size == 4
        assert signals.issue_responsiveness.responded_within_interval == 2
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

        fake_repo, _ = _releases_repo(
            [
                SimpleNamespace(draft=False, prerelease=False, published_at=NOW - dt.timedelta(days=10)),
                SimpleNamespace(draft=True, prerelease=False, published_at=NOW - dt.timedelta(days=20)),
                SimpleNamespace(draft=False, prerelease=True, published_at=NOW - dt.timedelta(days=30)),
                SimpleNamespace(draft=False, prerelease=False, published_at=None),
                SimpleNamespace(draft=False, prerelease=False, published_at=NOW - dt.timedelta(days=40)),
            ]
        )

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

    def test_newer_complete_signal_survives_a_late_older_run(self) -> None:
        """Observation order wins between two complete measurements."""
        newer = BacklogSignal(state=STATE_OK, as_of=_iso(1), open_count=6)
        older = BacklogSignal(state=STATE_OK, as_of=_iso(3), open_count=5)

        merged = merge_collected_signals(_signals_with(None, newer), _signals_with(None, older))

        assert merged.issue_backlog == newer

    def test_preserved_signal_keeps_its_own_parameters(self) -> None:
        """A kept signal travels with the window and interval it measured."""
        old_ok = ResponsivenessSignal(state=STATE_OK, as_of=_iso(3), window_days=180, interval_days=7, response_fraction=0.8)
        new_incomplete = ResponsivenessSignal(state=STATE_INCOMPLETE, as_of=_iso(0), incomplete_reason="rate-limited")

        merged = merge_collected_signals(_signals_with(old_ok), _signals_with(new_incomplete))

        assert merged.issue_responsiveness is not None
        assert merged.issue_responsiveness.window_days == 180
        assert merged.issue_responsiveness.interval_days == 7

    def test_late_older_ok_cannot_undo_a_newer_not_applicable(self) -> None:
        """Applicability transitions are observations; ordering applies to them too."""
        stored_na = BacklogSignal(state=STATE_NOT_APPLICABLE, as_of=_iso(0))
        late_ok = BacklogSignal(state=STATE_OK, as_of=_iso(2), open_count=25)

        merged = merge_collected_signals(
            _signals_with(None, stored_na, collected_at=_iso(0)),
            _signals_with(None, late_ok, collected_at=_iso(2)),
        )

        assert merged.issue_backlog == stored_na

    def test_newer_ok_survives_a_late_older_not_applicable(self) -> None:
        stored_ok = BacklogSignal(state=STATE_OK, as_of=_iso(0), open_count=3)
        late_na = BacklogSignal(state=STATE_NOT_APPLICABLE, as_of=_iso(2))

        merged = merge_collected_signals(
            _signals_with(None, stored_ok, collected_at=_iso(0)),
            _signals_with(None, late_na, collected_at=_iso(2)),
        )

        assert merged.issue_backlog == stored_ok

    def test_run_level_fields_come_from_the_newer_run(self) -> None:
        newer = _signals_with(None, None, collected_at=_iso(0))
        newer.issues_enabled = False
        older = _signals_with(None, None, collected_at=_iso(2))
        older.issues_enabled = True

        merged = merge_collected_signals(newer, older)

        assert merged.collected_at == _iso(0)
        assert merged.issues_enabled is False

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

    async def test_release_fallback_page_cap_marks_incomplete(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """A listing longer than the page cap is incomplete; no page beyond the cap is fetched."""

        monkeypatch.setattr(gm, "_RELEASE_PAGE_CAP", 1)

        fake_repo, paginated = _releases_repo(
            [SimpleNamespace(draft=True, prerelease=False, published_at=NOW - dt.timedelta(days=i)) for i in range(60)]
        )

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
        assert paginated.page_calls == 1

    async def test_release_fallback_finds_latest_publication_regardless_of_listing_order(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """The listing carries no publication-order contract; the walk must not stop at the first N distinct dates."""

        monkeypatch.setattr(settings, "MAINTENANCE_CADENCE_LAST_N_EVENTS", 3)

        # Page one already holds three distinct (older) publication dates; the
        # newest publication sits on page two. An implementation that stops
        # after N distinct dates never sees it.
        distinct_old = [NOW - dt.timedelta(days=age) for age in (40, 30, 20)]
        items = [SimpleNamespace(draft=False, prerelease=False, published_at=date) for date in distinct_old]
        items += [SimpleNamespace(draft=False, prerelease=False, published_at=distinct_old[0]) for _ in range(27)]
        items.append(SimpleNamespace(draft=False, prerelease=False, published_at=NOW - dt.timedelta(days=2)))
        fake_repo, paginated = _releases_repo(items)

        def _get_repo(path: str) -> SimpleNamespace:
            return fake_repo

        _patch_client(monkeypatch, _FakeRequester(_happy_path_handler), get_repo=_get_repo)

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=True, now=NOW)

        fallback = collected.signals.release_fallback
        assert fallback is not None
        assert fallback.state == STATE_OK
        assert paginated.page_calls == 2
        assert fallback.publication_dates == sorted(
            (NOW - dt.timedelta(days=age)).date().isoformat() for age in (40, 30, 20, 2)
        )


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
    async def test_budget_charged_at_every_request_boundary(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """The repository lookup and each fetched page cost one charge each."""

        fake_repo, paginated = _releases_repo(
            [SimpleNamespace(draft=True, prerelease=False, published_at=NOW - dt.timedelta(days=i)) for i in range(65)]
        )

        def _get_repo(path: str) -> SimpleNamespace:
            return fake_repo

        _patch_client(monkeypatch, _FakeRequester(_happy_path_handler), get_repo=_get_repo)
        collector = gm._MaintenanceCollector("owner", "repo", now=NOW)

        fallback = await collector.collect_release_fallback()

        assert paginated.page_calls == 3
        assert collector.requests_used == 1 + paginated.page_calls
        assert fallback.state == STATE_OK
        assert fallback.complete is True
        assert fallback.publication_dates == []

    async def test_request_cap_stops_before_the_next_page_fetch(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """Once the cap is exhausted, no further page request starts."""

        fake_repo, paginated = _releases_repo(
            [SimpleNamespace(draft=True, prerelease=False, published_at=NOW - dt.timedelta(days=i)) for i in range(65)]
        )

        def _get_repo(path: str) -> SimpleNamespace:
            return fake_repo

        _patch_client(monkeypatch, _FakeRequester(_happy_path_handler), get_repo=_get_repo)
        collector = gm._MaintenanceCollector("owner", "repo", now=NOW)
        collector._budget.request_cap = 2

        with pytest.raises(gm._CollectionHalt) as halt:
            await collector.collect_release_fallback()

        assert halt.value.reason == gm.REASON_REQUEST_CAP
        assert paginated.page_calls == 1
        assert collector.requests_used == 2


# ---------------------------------------------------------------------------
# Code-driven issue closes
# ---------------------------------------------------------------------------


class TestClassifyIssueCloseEvents:
    def test_close_by_merged_pr_qualifies(self) -> None:
        events = [gm._TimelineNode(typename="ClosedEvent", createdAt=_iso(29), closer=gm._CloserRef(typename="PullRequest"))]
        assert gm.classify_issue_close_events(events) == [NOW - dt.timedelta(days=29)]

    def test_close_by_commit_qualifies(self) -> None:
        events = [gm._TimelineNode(typename="ClosedEvent", createdAt=_iso(29), closer=gm._CloserRef(typename="Commit"))]
        assert gm.classify_issue_close_events(events) == [NOW - dt.timedelta(days=29)]

    def test_manual_close_never_qualifies(self) -> None:
        """A click-close is ambiguous and mass-produceable; only code counts."""
        events = [gm._TimelineNode(typename="ClosedEvent", createdAt=_iso(29), closer=None)]
        assert gm.classify_issue_close_events(events) == []

    def test_non_close_events_are_skipped(self) -> None:
        events = [gm._TimelineNode(typename="ReopenedEvent", createdAt=_iso(29), closer=gm._CloserRef(typename="Commit"))]
        assert gm.classify_issue_close_events(events) == []

    def test_times_are_sorted(self) -> None:
        events = [
            gm._TimelineNode(typename="ClosedEvent", createdAt=_iso(10), closer=gm._CloserRef(typename="Commit")),
            gm._TimelineNode(typename="ClosedEvent", createdAt=_iso(20), closer=gm._CloserRef(typename="PullRequest")),
        ]
        assert gm.classify_issue_close_events(events) == [
            NOW - dt.timedelta(days=20),
            NOW - dt.timedelta(days=10),
        ]

    async def test_truncated_issue_timeline_is_indeterminate(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """Unfetched close pages that could hold the response are never guessed."""

        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            if query == gm._ISSUE_COHORT_QUERY:
                return {}, {
                    "data": {
                        "repository": {
                            "issues": {
                                "nodes": [
                                    {
                                        "number": 9,
                                        "createdAt": _iso(30),
                                        "author": _actor("alice"),
                                        "comments": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}},
                                        "timelineItems": {
                                            "nodes": [],
                                            "pageInfo": {"hasNextPage": True, "endCursor": "more"},
                                        },
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

        assert collected.signals.issue_responsiveness is not None
        assert collected.signals.issue_responsiveness.cohort_size == 0
        assert collected.signals.issue_responsiveness.unknown_attribution_count == 1


class TestCodeCloseCommentInteraction:
    async def test_out_of_interval_close_does_not_stop_comment_search(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """A late close must not hide a within-interval comment on later pages."""

        continuation_calls = {"count": 0}

        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            if query == gm._ISSUE_COHORT_QUERY:
                return {}, {
                    "data": {
                        "repository": {
                            "issues": {
                                "nodes": [
                                    {
                                        "number": 11,
                                        "createdAt": _iso(33),
                                        "author": _actor("alice"),
                                        "comments": {
                                            "nodes": [
                                                {
                                                    "createdAt": _iso(32),
                                                    "authorAssociation": "NONE",
                                                    "author": _actor("rando"),
                                                }
                                            ],
                                            "pageInfo": {"hasNextPage": True, "endCursor": "c1"},
                                        },
                                        "timelineItems": {
                                            "nodes": [
                                                {
                                                    "__typename": "ClosedEvent",
                                                    "createdAt": _iso(20),  # 13 days after creation
                                                    "closer": {"__typename": "PullRequest"},
                                                }
                                            ],
                                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                                        },
                                    }
                                ],
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                            }
                        }
                    }
                }

            if query == gm._ISSUE_COMMENTS_QUERY:
                continuation_calls["count"] += 1
                return {}, {
                    "data": {
                        "repository": {
                            "issue": {
                                "comments": {
                                    "nodes": [
                                        {
                                            "createdAt": _iso(29),  # 4 days after creation
                                            "authorAssociation": "MEMBER",
                                            "author": _actor("bob"),
                                        }
                                    ],
                                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                                }
                            }
                        }
                    }
                }

            return _happy_path_handler(query, variables)

        _patch_client(monkeypatch, _FakeRequester(handler))

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=False, now=NOW)

        assert continuation_calls["count"] == 1
        signals = collected.signals
        assert signals.issue_responsiveness is not None
        # Issue #11 was answered by the MEMBER comment 4 days in, even though
        # the code close came 13 days in — after the interval.
        assert signals.issue_responsiveness.cohort_size == 1
        assert signals.issue_responsiveness.responded_within_interval == 1
        assert signals.issue_responsiveness.response_fraction == 1.0

    async def test_within_interval_close_skips_comment_pagination(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """A close that already settles the item spends no extra requests."""

        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            if query == gm._ISSUE_COHORT_QUERY:
                return {}, {
                    "data": {
                        "repository": {
                            "issues": {
                                "nodes": [
                                    {
                                        "number": 12,
                                        "createdAt": _iso(30),
                                        "author": _actor("alice"),
                                        "comments": {
                                            "nodes": [],
                                            "pageInfo": {"hasNextPage": True, "endCursor": "c1"},
                                        },
                                        "timelineItems": {
                                            "nodes": [
                                                {
                                                    "__typename": "ClosedEvent",
                                                    "createdAt": _iso(28),  # 2 days after creation
                                                    "closer": {"__typename": "Commit"},
                                                }
                                            ],
                                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                                        },
                                    }
                                ],
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                            }
                        }
                    }
                }

            assert query != gm._ISSUE_COMMENTS_QUERY, "settled item must not paginate comments"

            return _happy_path_handler(query, variables)

        _patch_client(monkeypatch, _FakeRequester(handler))

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=False, now=NOW)

        assert collected.signals.issue_responsiveness is not None
        assert collected.signals.issue_responsiveness.cohort_size == 1
        assert collected.signals.issue_responsiveness.responded_within_interval == 1


class TestTransportBoundaries:
    async def test_transport_failure_becomes_typed_halt(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """A connection failure surfaces as a query-error halt, never raw."""

        import requests

        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            raise requests.exceptions.ConnectionError("connection reset")

        _patch_client(monkeypatch, _FakeRequester(handler))
        collector = gm._MaintenanceCollector("owner", "repo", now=NOW)

        with pytest.raises(gm._CollectionHalt) as halt:
            await collector.fetch_overview()

        assert halt.value.reason == gm.REASON_QUERY_ERROR
        assert "ConnectionError" in halt.value.detail

    async def test_rate_limit_wait_exceeding_remaining_deadline_halts(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """A reset wait longer than the remaining time budget halts immediately."""

        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            raise _rate_limited_exception({"X-RateLimit-Reset": str(time.time() + 30)})

        _patch_client(monkeypatch, _FakeRequester(handler))
        collector = gm._MaintenanceCollector("owner", "repo", now=NOW)
        collector._budget.deadline_monotonic = time.monotonic() + 5.0

        started = time.monotonic()
        with pytest.raises(gm._CollectionHalt) as halt:
            await collector.fetch_overview()

        assert halt.value.reason == gm.REASON_RATE_LIMITED
        assert time.monotonic() - started < 2.0


# ---------------------------------------------------------------------------
# Backlog salvage and median short-circuit
# ---------------------------------------------------------------------------


def _open_issues_page(total: int, ages: list[float], has_next: bool) -> dict[str, Any]:
    return {
        "data": {
            "repository": {
                "issues": {
                    "totalCount": total,
                    "nodes": [{"createdAt": _iso(age)} for age in ages],
                    "pageInfo": {"hasNextPage": has_next, "endCursor": "next" if has_next else None},
                }
            }
        }
    }


class TestBacklogResilience:
    async def test_median_short_circuit_stops_paging(self, monkeypatch: pytest.MonkeyPatch, default_parameters: None) -> None:
        """Once the oldest-first prefix holds the middle order statistic, no more pages are fetched."""

        issue_page_calls = {"count": 0}
        ages = [float(170 - i) for i in range(100)]

        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            if query == gm._OPEN_ISSUES_QUERY:
                issue_page_calls["count"] += 1
                return {}, _open_issues_page(150, ages, has_next=True)

            return _happy_path_handler(query, variables)

        _patch_client(monkeypatch, _FakeRequester(handler))

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=False, now=NOW)

        assert issue_page_calls["count"] == 1
        backlog = collected.signals.issue_backlog
        assert backlog is not None
        assert backlog.state == STATE_OK
        assert backlog.open_count == 150
        assert backlog.median_age_exact is True
        # Median of 150 ages = mean of the 75th and 76th oldest: (96 + 95) / 2.
        assert backlog.median_open_age_days == 95.5

    async def test_budget_halt_on_continuation_keeps_exact_count(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """A shared-cap halt mid-listing keeps the server-side count and stops later phases."""

        monkeypatch.setattr(settings, "MAINTENANCE_REQUEST_CAP", 2)
        ages = [float(170 - i) for i in range(100)]

        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            if query == gm._OPEN_ISSUES_QUERY:
                return {}, _open_issues_page(300, ages, has_next=True)

            return _happy_path_handler(query, variables)

        _patch_client(monkeypatch, _FakeRequester(handler))

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=False, now=NOW)
        signals = collected.signals

        assert signals.issue_backlog is not None
        assert signals.issue_backlog.state == STATE_INCOMPLETE
        assert signals.issue_backlog.open_count == 300
        assert signals.issue_backlog.incomplete_reason == gm.REASON_REQUEST_CAP
        assert signals.issue_backlog.median_open_age_days is None

        for later in (signals.pr_backlog, signals.issue_responsiveness, signals.pr_responsiveness):
            assert later is not None
            assert later.state == STATE_INCOMPLETE
            assert later.incomplete_reason == gm.REASON_REQUEST_CAP


# ---------------------------------------------------------------------------
# Review pagination across pages
# ---------------------------------------------------------------------------


class TestReviewPagination:
    async def test_late_first_page_review_does_not_stop_the_search(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """Reviews carry no submission-order contract; an earlier submission can sit on a later page."""

        continuation_calls = {"count": 0}

        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            if query == gm._PULL_COHORT_QUERY:
                return {}, {
                    "data": {
                        "repository": {
                            "pullRequests": {
                                "nodes": [
                                    {
                                        "number": 30,
                                        "createdAt": _iso(30),
                                        "isDraft": False,
                                        "author": _actor("alice"),
                                        "reviews": {
                                            "nodes": [
                                                {
                                                    "submittedAt": _iso(20),  # 10 days after creation
                                                    "authorAssociation": "MEMBER",
                                                    "author": _actor("bob"),
                                                }
                                            ],
                                            "pageInfo": {"hasNextPage": True, "endCursor": "r1"},
                                        },
                                        "timelineItems": {"nodes": [], "pageInfo": {"hasNextPage": False}},
                                    }
                                ],
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                            }
                        }
                    }
                }

            if query == gm._PULL_REVIEWS_QUERY:
                continuation_calls["count"] += 1
                return {}, {
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "reviews": {
                                    "nodes": [
                                        {
                                            "submittedAt": _iso(28),  # 2 days after creation
                                            "authorAssociation": "OWNER",
                                            "author": _actor("carol"),
                                        }
                                    ],
                                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                                }
                            }
                        }
                    }
                }

            return _happy_path_handler(query, variables)

        _patch_client(monkeypatch, _FakeRequester(handler))

        collected = await collect_maintenance_for_repo("owner", "repo", need_release_fallback=False, now=NOW)

        assert continuation_calls["count"] == 1
        pr = collected.signals.pr_responsiveness
        assert pr is not None
        assert pr.cohort_size == 1
        assert pr.responded_within_interval == 1
        assert pr.response_fraction == 1.0

    async def test_truncated_pr_timeline_is_indeterminate(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            if query == gm._PULL_COHORT_QUERY:
                return {}, {
                    "data": {
                        "repository": {
                            "pullRequests": {
                                "nodes": [
                                    {
                                        "number": 31,
                                        "createdAt": _iso(30),
                                        "isDraft": False,
                                        "author": _actor("alice"),
                                        "reviews": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}},
                                        "timelineItems": {
                                            "nodes": [],
                                            "pageInfo": {"hasNextPage": True, "endCursor": "t1"},
                                        },
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

        pr = collected.signals.pr_responsiveness
        assert pr is not None
        assert pr.cohort_size == 0
        assert pr.unknown_attribution_count == 1


class TestEventLoopAndWaitBounds:
    async def test_event_loop_stays_live_during_a_slow_request(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """The synchronous transport call runs in a worker thread."""

        import asyncio

        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            time.sleep(0.05)
            return {}, {"data": {"repository": {"pushedAt": _iso(2), "hasIssuesEnabled": True, "isArchived": False}}}

        _patch_client(monkeypatch, _FakeRequester(handler))
        collector = gm._MaintenanceCollector("owner", "repo", now=NOW)

        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.005)

        task = asyncio.create_task(ticker())
        await asyncio.sleep(0)
        await collector.fetch_overview()
        task.cancel()

        assert ticks >= 2

    async def test_wait_margin_counts_against_the_cap(self, monkeypatch: pytest.MonkeyPatch, default_parameters: None) -> None:
        """An advertised reset just under the cap still halts once the retry margin is added."""

        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            raise _rate_limited_exception({"X-RateLimit-Reset": str(time.time() + 119.5)})

        _patch_client(monkeypatch, _FakeRequester(handler))
        collector = gm._MaintenanceCollector("owner", "repo", now=NOW)

        started = time.monotonic()
        with pytest.raises(gm._CollectionHalt) as halt:
            await collector.fetch_overview()

        assert halt.value.reason == gm.REASON_RATE_LIMITED
        assert time.monotonic() - started < 2.0


class TestDeadlineGovernsInFlightRequests:
    async def test_graphql_request_is_cancelled_at_the_deadline(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """A slow GraphQL request becomes a time-cap halt, not a late success."""

        def handler(query: str, variables: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            time.sleep(0.2)
            return {}, {"data": {"repository": {"pushedAt": _iso(2), "hasIssuesEnabled": True, "isArchived": False}}}

        _patch_client(monkeypatch, _FakeRequester(handler))
        collector = gm._MaintenanceCollector("owner", "repo", now=NOW)
        collector._budget.deadline_monotonic = time.monotonic() + 0.02

        started = time.monotonic()
        with pytest.raises(gm._CollectionHalt) as halt:
            await collector.fetch_overview()

        assert halt.value.reason == gm.REASON_TIME_CAP
        assert time.monotonic() - started < 0.15

    async def test_rest_lookup_is_cancelled_at_the_deadline(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        def _slow_get_repo(path: str) -> SimpleNamespace:
            time.sleep(0.2)
            raise AssertionError("unreachable in this test")

        _patch_client(monkeypatch, _FakeRequester(_happy_path_handler), get_repo=_slow_get_repo)
        collector = gm._MaintenanceCollector("owner", "repo", now=NOW)
        collector._budget.deadline_monotonic = time.monotonic() + 0.02

        with pytest.raises(gm._CollectionHalt) as halt:
            await collector.collect_release_fallback()

        assert halt.value.reason == gm.REASON_TIME_CAP

    async def test_final_release_page_cannot_complete_past_the_deadline(
        self, monkeypatch: pytest.MonkeyPatch, default_parameters: None
    ) -> None:
        """The last page finishing after the cap halts instead of reporting complete."""

        fake_repo, paginated = _releases_repo(
            [SimpleNamespace(draft=False, prerelease=False, published_at=NOW - dt.timedelta(days=10))]
        )
        original_get_page = paginated.get_page

        def _slow_get_page(index: int) -> list[SimpleNamespace]:
            time.sleep(0.2)
            return original_get_page(index)

        paginated.get_page = _slow_get_page  # type: ignore[method-assign]

        def _get_repo(path: str) -> SimpleNamespace:
            return fake_repo

        _patch_client(monkeypatch, _FakeRequester(_happy_path_handler), get_repo=_get_repo)
        collector = gm._MaintenanceCollector("owner", "repo", now=NOW)
        collector._budget.deadline_monotonic = time.monotonic() + 0.05

        with pytest.raises(gm._CollectionHalt) as halt:
            await collector.collect_release_fallback()

        assert halt.value.reason == gm.REASON_TIME_CAP
