"""
GitHub maintenance-signal collector.

Collects the API-backed maintenance signals for one repository — issue and PR
responsiveness cohorts, open-item backlog snapshots, ``pushed_at``, issue
tracker applicability, and (when the stored deps.dev releases lack usable
dated coverage) a GitHub Releases publication-date fallback — and persists
them under ``Repo.repo_metadata["maintenance_signals"]`` plus the
``Repo.pushed_at`` column. Release cadence and commit activity are computed
later by ``pg_atlas.metrics.materialize_maintenance`` from stored data.

Cohort selection filters by creation date client-side (GitHub's REST ``since``
filters by update time, so listings walk GraphQL creation-ordered pages).
Responsiveness cohorts cover community-authored work: items authored by
maintainers themselves (author association OWNER/MEMBER/COLLABORATOR) are
excluded and counted separately — a maintainer's own self-merged PR is not an
unanswered request. Maintainer qualification is event-specific: comments and
submitted reviews require an ``OWNER``/``MEMBER``/``COLLABORATOR`` author
association; merge and close timeline events expose only an actor, so a
non-bot actor other than the PR author counts — an authorization-based proxy
(merging or closing another author's PR requires write or triage rights). An
event, comment, or review that would qualify but whose actor cannot be
resolved is unknown attribution: the item is excluded from the response
fraction and counted separately, never guessed.

Collection is bounded by per-repo page, request, and time caps. GraphQL
rate-limit errors arrive inside HTTP-200 envelopes (PyGithub surfaces them as
``GithubException``); the collector waits for the advertised reset once,
bounded, then records the affected signals as ``incomplete`` — capped or
failed collection is recorded, never silently mixed into results. Persistence
merges per signal: a newly incomplete signal never overwrites a previously
complete one, whose own ``as_of`` keeps it honest at materialization time.

Scheduling fails closed behind ``MAINTENANCE_METRIC_ENABLED`` plus
``MAINTENANCE_METRIC_ALLOWLIST``; the task layer re-checks the same gate at
execution time. Explicit CLI runs bypass the gate::

    uv run python -m pg_atlas.procrastinate.github_maintenance Soneso/stellar-php-sdk

SPDX-FileCopyrightText: 2026 PG Atlas contributors
SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, cast

import msgspec
from github import GithubException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pg_atlas.config import settings
from pg_atlas.db_models.release import Release
from pg_atlas.db_models.repo_vertex import Repo
from pg_atlas.gitlog.filters import DEFAULT_BOT_NAME_PATTERNS
from pg_atlas.metrics.maintenance import (
    MAINTENANCE_SIGNALS_KEY,
    MAINTENANCE_SIGNALS_SCHEMA_VERSION,
    REASON_PAGE_CAP,
    REASON_QUERY_ERROR,
    REASON_RATE_LIMITED,
    REASON_REQUEST_CAP,
    REASON_TIME_CAP,
    STATE_INCOMPLETE,
    STATE_NOT_APPLICABLE,
    STATE_OK,
    BacklogSignal,
    MaintenanceSignals,
    ReleaseFallback,
    ResponseItem,
    ResponsivenessSignal,
    compute_backlog,
    compute_responsiveness,
    distinct_release_dates,
    repo_metadata_merge_expression,
    signals_from_metadata,
    signals_to_payload,
)
from pg_atlas.procrastinate.github import get_github_client

logger = logging.getLogger(__name__)

#: Comment/review author associations accepted as maintainer responses. A
#: documented heuristic (MEMBER means org membership), not proof of
#: responsibility — accepted as the best rule the API offers.
MAINTAINER_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

#: GraphQL page sizes, interpolated into the query documents below so the
#: cap arithmetic (pages x page size) has one source of truth.
_COHORT_PAGE_SIZE = 25
_OPEN_ITEMS_PAGE_SIZE = 100
_COMMENTS_FIRST_PAGE_SIZE = 50
_CONTINUATION_PAGE_SIZE = 100
_REVIEWS_FIRST_PAGE_SIZE = 20
_TIMELINE_PAGE_SIZE = 10
_PAGE_SIZES = {
    "cohort_page": _COHORT_PAGE_SIZE,
    "open_page": _OPEN_ITEMS_PAGE_SIZE,
    "comments_first": _COMMENTS_FIRST_PAGE_SIZE,
    "continuation": _CONTINUATION_PAGE_SIZE,
    "reviews_first": _REVIEWS_FIRST_PAGE_SIZE,
    "timeline": _TIMELINE_PAGE_SIZE,
}

#: GitHub Releases fallback: REST page size and total items walked.
_RELEASES_PER_PAGE = 30
_RELEASE_ITEM_CAP = 300


# ---------------------------------------------------------------------------
# Gate and declared overrides
# ---------------------------------------------------------------------------


def _parse_owner_repo_set(raw: str, *, setting_name: str) -> set[str]:
    """Parse a comma-separated ``owner/repo`` list, rejecting malformed entries."""

    entries: set[str] = set()
    for item in raw.split(","):
        entry = item.strip().lower()
        if not entry:
            continue

        parts = entry.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            logger.warning(f"{setting_name} entry rejected: {item.strip()!r}")
            continue

        entries.add(entry)

    return entries


def maintenance_metric_allowed(owner: str, repo: str) -> bool:
    """
    Fail-closed gate for maintenance-signal collection.

    Requires the enable flag AND an allowlist match: a comma-separated
    ``owner/repo`` list (case-insensitive), or the explicit value ``"*"`` for
    all repositories. An empty allowlist allows nothing. Checked at scheduling
    time AND re-checked by the task at execution time, so disabling stops
    already-queued work. Explicit CLI runs bypass this gate. Lives in this
    module (not the task layer) so it stays importable without a database.
    """

    if not settings.MAINTENANCE_METRIC_ENABLED:
        return False

    raw = settings.MAINTENANCE_METRIC_ALLOWLIST.strip()
    if raw == "*":
        return True

    allowed = _parse_owner_repo_set(raw, setting_name="maintenance-metric allowlist")

    return f"{owner}/{repo}".lower() in allowed


def external_tracker_declared(owner: str, repo: str) -> bool:
    """
    Return whether this repo's issue tracking is declared to live off GitHub.

    An empty GitHub tracker cannot by itself distinguish an external tracker
    from no incoming work, so applicability is a declared per-repo override;
    declared repos get not-applicable issue signals instead of real zeros.
    """

    declared = _parse_owner_repo_set(
        settings.MAINTENANCE_EXTERNAL_TRACKER_REPOS,
        setting_name="maintenance-metric external-tracker list",
    )

    return f"{owner}/{repo}".lower() in declared


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MaintenanceSourceRepoNotFound(Exception):
    """Raised when no tracked repo matches the requested source identity."""


class AmbiguousMaintenanceSourceRepo(Exception):
    """Raised when more than one tracked repo matches the source identity."""


class MaintenanceCollectionFailed(Exception):
    """Raised when collection produced nothing usable for the repository."""


class _CollectionHalt(Exception):
    """Internal: stop collecting; remaining signals become incomplete."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------
# GraphQL payload shapes (typed subsets)
# ---------------------------------------------------------------------------


class _ActorRef(msgspec.Struct, omit_defaults=True):
    """Author/actor subset used for bot and identity classification."""

    typename: str | None = msgspec.field(name="__typename", default=None)
    login: str | None = None


class _PageInfo(msgspec.Struct, omit_defaults=True):
    """Page info shape for GraphQL pagination."""

    hasNextPage: bool
    endCursor: str | None = None


class _OverviewRepository(msgspec.Struct, omit_defaults=True):
    """Repo-level facts fetched once per collection run."""

    pushedAt: str | None = None
    hasIssuesEnabled: bool = True
    isArchived: bool = False


class _OverviewData(msgspec.Struct, omit_defaults=True):
    repository: _OverviewRepository | None = None


class _CreatedAtNode(msgspec.Struct, omit_defaults=True):
    createdAt: str


class _OpenItemsConnection(msgspec.Struct, omit_defaults=True):
    totalCount: int = 0
    nodes: list[_CreatedAtNode] | None = None
    pageInfo: _PageInfo | None = None


class _OpenIssuesRepository(msgspec.Struct, omit_defaults=True):
    issues: _OpenItemsConnection | None = None


class _OpenIssuesData(msgspec.Struct, omit_defaults=True):
    repository: _OpenIssuesRepository | None = None


class _OpenPullsRepository(msgspec.Struct, omit_defaults=True):
    pullRequests: _OpenItemsConnection | None = None


class _OpenPullsData(msgspec.Struct, omit_defaults=True):
    repository: _OpenPullsRepository | None = None


class _CommentNode(msgspec.Struct, omit_defaults=True):
    createdAt: str
    authorAssociation: str | None = None
    author: _ActorRef | None = None


class _CommentsConnection(msgspec.Struct, omit_defaults=True):
    nodes: list[_CommentNode] | None = None
    pageInfo: _PageInfo | None = None


class _IssueCohortNode(msgspec.Struct, omit_defaults=True):
    number: int
    createdAt: str
    authorAssociation: str | None = None
    author: _ActorRef | None = None
    comments: _CommentsConnection | None = None


class _IssueCohortConnection(msgspec.Struct, omit_defaults=True):
    nodes: list[_IssueCohortNode] | None = None
    pageInfo: _PageInfo | None = None


class _IssueCohortRepository(msgspec.Struct, omit_defaults=True):
    issues: _IssueCohortConnection | None = None


class _IssueCohortData(msgspec.Struct, omit_defaults=True):
    repository: _IssueCohortRepository | None = None


class _IssueCommentsOnly(msgspec.Struct, omit_defaults=True):
    comments: _CommentsConnection | None = None


class _IssueByNumberRepository(msgspec.Struct, omit_defaults=True):
    issue: _IssueCommentsOnly | None = None


class _IssueByNumberData(msgspec.Struct, omit_defaults=True):
    repository: _IssueByNumberRepository | None = None


class _ReviewNode(msgspec.Struct, omit_defaults=True):
    submittedAt: str | None = None
    authorAssociation: str | None = None
    author: _ActorRef | None = None


class _ReviewsConnection(msgspec.Struct, omit_defaults=True):
    nodes: list[_ReviewNode] | None = None
    pageInfo: _PageInfo | None = None


class _TimelineNode(msgspec.Struct, omit_defaults=True):
    """One CLOSED_EVENT / MERGED_EVENT timeline entry."""

    typename: str | None = msgspec.field(name="__typename", default=None)
    createdAt: str | None = None
    actor: _ActorRef | None = None


class _TimelineConnection(msgspec.Struct, omit_defaults=True):
    nodes: list[_TimelineNode] | None = None
    pageInfo: _PageInfo | None = None


class _PullCohortNode(msgspec.Struct, omit_defaults=True):
    number: int
    createdAt: str
    isDraft: bool = False
    authorAssociation: str | None = None
    author: _ActorRef | None = None
    reviews: _ReviewsConnection | None = None
    timelineItems: _TimelineConnection | None = None


class _PullCohortConnection(msgspec.Struct, omit_defaults=True):
    nodes: list[_PullCohortNode] | None = None
    pageInfo: _PageInfo | None = None


class _PullCohortRepository(msgspec.Struct, omit_defaults=True):
    pullRequests: _PullCohortConnection | None = None


class _PullCohortData(msgspec.Struct, omit_defaults=True):
    repository: _PullCohortRepository | None = None


class _PullReviewsOnly(msgspec.Struct, omit_defaults=True):
    reviews: _ReviewsConnection | None = None


class _PullByNumberRepository(msgspec.Struct, omit_defaults=True):
    pullRequest: _PullReviewsOnly | None = None


class _PullByNumberData(msgspec.Struct, omit_defaults=True):
    repository: _PullByNumberRepository | None = None


# ---------------------------------------------------------------------------
# GraphQL queries
# ---------------------------------------------------------------------------

_OVERVIEW_QUERY = (
    """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    pushedAt
    hasIssuesEnabled
    isArchived
  }
}
""".strip()
    % _PAGE_SIZES
)

_OPEN_ISSUES_QUERY = (
    """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    issues(states: OPEN, first: %(open_page)d, after: $after, orderBy: {field: CREATED_AT, direction: ASC}) {
      totalCount
      nodes { createdAt }
      pageInfo { hasNextPage endCursor }
    }
  }
}
""".strip()
    % _PAGE_SIZES
)

_OPEN_PULLS_QUERY = (
    """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: %(open_page)d, after: $after, orderBy: {field: CREATED_AT, direction: ASC}) {
      totalCount
      nodes { createdAt }
      pageInfo { hasNextPage endCursor }
    }
  }
}
""".strip()
    % _PAGE_SIZES
)

_ISSUE_COHORT_QUERY = (
    """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    issues(first: %(cohort_page)d, after: $after, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        number
        createdAt
        authorAssociation
        author { __typename login }
        comments(first: %(comments_first)d) {
          nodes { createdAt authorAssociation author { __typename login } }
          pageInfo { hasNextPage endCursor }
        }
      }
      pageInfo { hasNextPage endCursor }
    }
  }
}
""".strip()
    % _PAGE_SIZES
)

_ISSUE_COMMENTS_QUERY = (
    """
query($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      comments(first: %(continuation)d, after: $after) {
        nodes { createdAt authorAssociation author { __typename login } }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
""".strip()
    % _PAGE_SIZES
)

_PULL_COHORT_QUERY = (
    """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: %(cohort_page)d, after: $after, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        number
        createdAt
        isDraft
        authorAssociation
        author { __typename login }
        reviews(first: %(reviews_first)d) {
          nodes { submittedAt authorAssociation author { __typename login } }
          pageInfo { hasNextPage endCursor }
        }
        timelineItems(itemTypes: [CLOSED_EVENT, MERGED_EVENT], first: %(timeline)d) {
          nodes {
            __typename
            ... on ClosedEvent { createdAt actor { __typename login } }
            ... on MergedEvent { createdAt actor { __typename login } }
          }
          pageInfo { hasNextPage endCursor }
        }
      }
      pageInfo { hasNextPage endCursor }
    }
  }
}
""".strip()
    % _PAGE_SIZES
)

_PULL_REVIEWS_QUERY = (
    """
query($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviews(first: %(continuation)d, after: $after) {
        nodes { submittedAt authorAssociation author { __typename login } }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
""".strip()
    % _PAGE_SIZES
)


# ---------------------------------------------------------------------------
# Classification helpers (pure)
# ---------------------------------------------------------------------------


def _parse_iso(value: str) -> dt.datetime:
    """Parse one GraphQL ISO-8601 timestamp into an aware UTC datetime."""

    return dt.datetime.fromisoformat(value).astimezone(dt.UTC)


def _is_bot_actor(actor: _ActorRef | None) -> bool:
    """
    Return whether an author/actor reference is an automated account.

    Primary signal is the GraphQL ``Bot`` actor type; the pony metric's name
    patterns supplement it for machine accounts that present as ``User``
    (e.g. ``dependabot``). An absent reference is not classified here.
    """

    if actor is None:
        return False

    if actor.typename == "Bot":
        return True

    if actor.login:
        return any(pattern.search(actor.login) for pattern in DEFAULT_BOT_NAME_PATTERNS)

    return False


@dataclass(frozen=True)
class _AttributionScan:
    """One page's earliest qualifying response and unknown-attribution times."""

    qualifying_time: dt.datetime | None
    unknown_times: list[dt.datetime]


def _scan_comment_page(
    comments: list[_CommentNode],
    author_login: str | None,
) -> _AttributionScan:
    """
    Scan one chronological comment page for maintainer responses.

    Qualifying: author association OWNER/MEMBER/COLLABORATOR, author not a bot
    and not the item's own author. A comment whose association qualifies but
    whose author cannot be resolved is unknown attribution — reported, never
    guessed either way — mirroring the merge/close event rule. (In practice
    GitHub reports association NONE for deleted accounts, so such comments
    normally fail the association filter instead.)
    """

    qualifying_time: dt.datetime | None = None
    unknown_times: list[dt.datetime] = []
    for comment in comments:
        if comment.authorAssociation not in MAINTAINER_ASSOCIATIONS:
            continue

        if comment.author is None:
            unknown_times.append(_parse_iso(comment.createdAt))
            continue

        if _is_bot_actor(comment.author):
            continue

        if author_login is not None and comment.author.login == author_login:
            continue

        qualifying_time = _parse_iso(comment.createdAt)
        break

    return _AttributionScan(qualifying_time=qualifying_time, unknown_times=unknown_times)


def _scan_review_page(
    reviews: list[_ReviewNode],
    author_login: str | None,
) -> _AttributionScan:
    """Scan one review page under the same attribution rules as comments."""

    qualifying: list[dt.datetime] = []
    unknown_times: list[dt.datetime] = []
    for review in reviews:
        if review.submittedAt is None or review.authorAssociation not in MAINTAINER_ASSOCIATIONS:
            continue

        if review.author is None:
            unknown_times.append(_parse_iso(review.submittedAt))
            continue

        if _is_bot_actor(review.author):
            continue

        if author_login is not None and review.author.login == author_login:
            continue

        qualifying.append(_parse_iso(review.submittedAt))

    return _AttributionScan(qualifying_time=min(qualifying) if qualifying else None, unknown_times=unknown_times)


@dataclass(frozen=True)
class _EventClassification:
    """Merge/close timeline events split into qualifying and unknown times."""

    qualifying_times: list[dt.datetime]
    unknown_times: list[dt.datetime]


def classify_pr_events(
    events: list[_TimelineNode],
    author_login: str | None,
) -> _EventClassification:
    """
    Classify merge/close timeline events by their actor.

    These events expose only an actor, so a non-bot actor other than the PR
    author qualifies (authorization proxy: acting on another author's PR
    requires write or triage rights). The author's own close is a withdrawal,
    not a response. An event without a resolvable actor is unknown
    attribution and is reported as such, never guessed either way.
    """

    qualifying: list[dt.datetime] = []
    unknown: list[dt.datetime] = []
    for event in events:
        if event.typename not in ("ClosedEvent", "MergedEvent") or event.createdAt is None:
            continue

        occurred_at = _parse_iso(event.createdAt)

        if event.actor is None or not event.actor.login:
            unknown.append(occurred_at)
            continue

        if _is_bot_actor(event.actor):
            continue

        if author_login is not None and event.actor.login == author_login:
            continue

        qualifying.append(occurred_at)

    return _EventClassification(qualifying_times=sorted(qualifying), unknown_times=sorted(unknown))


def resolve_response_item(
    created_at: dt.datetime,
    qualifying_times: list[dt.datetime],
    unknown_times: list[dt.datetime],
    *,
    interval_days: int,
    detection_truncated: bool = False,
) -> ResponseItem:
    """
    Combine qualifying and unknown-attribution response times into one item.

    An unknown-attribution event matters only when it could change the
    within-interval outcome: a qualifying response inside the interval settles
    the item regardless, while an unknown event inside the interval with no
    qualifying response inside it makes the item indeterminate. Truncated
    detection (unfetched pages that could still hold the first response) also
    yields indeterminate unless the item is already settled.
    """

    interval = dt.timedelta(days=interval_days)
    first_qualifying = min(qualifying_times) if qualifying_times else None

    if first_qualifying is not None and first_qualifying - created_at <= interval:
        return ResponseItem(created_at=created_at, first_response_at=first_qualifying)

    if any(unknown - created_at <= interval for unknown in unknown_times):
        return ResponseItem(created_at=created_at, indeterminate=True)

    if detection_truncated:
        return ResponseItem(created_at=created_at, indeterminate=True)

    return ResponseItem(created_at=created_at, first_response_at=first_qualifying)


# ---------------------------------------------------------------------------
# Collection budget
# ---------------------------------------------------------------------------


@dataclass
class _CollectionBudget:
    """Per-repo request and wall-clock caps shared across all phases."""

    request_cap: int
    deadline_monotonic: float
    requests_used: int = 0

    def charge(self) -> None:
        """Account for one API request; halt collection when a cap is hit."""

        if self.requests_used >= self.request_cap:
            raise _CollectionHalt(REASON_REQUEST_CAP, f"request cap {self.request_cap} reached")

        if time.monotonic() > self.deadline_monotonic:
            raise _CollectionHalt(REASON_TIME_CAP, "collection time cap reached")

        self.requests_used += 1


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollectedMaintenance:
    """Everything one collection run produced for one repository."""

    signals: MaintenanceSignals
    pushed_at: dt.datetime | None


def _has_rate_limited_error(data: object) -> bool:
    """Return whether an exception payload carries a GraphQL RATE_LIMITED error."""

    if not isinstance(data, dict):
        return False

    errors = cast("dict[str, object]", data).get("errors")
    if not isinstance(errors, list):
        return False

    for entry in cast("list[object]", errors):
        if isinstance(entry, dict) and cast("dict[str, object]", entry).get("type") == "RATE_LIMITED":
            return True

    return False


def _rate_limit_wait_seconds(exc: GithubException) -> float | None:
    """
    Return a bounded wait if the exception is a rate-limit condition.

    GraphQL rate limits arrive inside HTTP-200 envelopes that PyGithub
    re-raises with the error list in ``exc.data``; secondary limits surface as
    HTTP 403/429. Returns seconds until the advertised reset (or the
    ``retry-after`` value), or ``None`` when the exception is not a rate
    limit.
    """

    is_rate_limited = _has_rate_limited_error(exc.data)

    if not is_rate_limited and exc.status not in (403, 429):
        return None

    headers = {key.lower(): value for key, value in (exc.headers or {}).items()}

    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            pass

    reset_epoch = headers.get("x-ratelimit-reset")
    if reset_epoch is not None:
        try:
            return max(float(reset_epoch) - time.time(), 0.0)
        except ValueError:
            pass

    if is_rate_limited:
        return 60.0

    return None


class _MaintenanceCollector:
    """One bounded collection pass over a single repository."""

    def __init__(self, owner: str, repo: str, *, now: dt.datetime) -> None:
        self._owner = owner
        self._repo = repo
        self._now = now
        self._window_start = now - dt.timedelta(days=settings.MAINTENANCE_WINDOW_DAYS)
        self._budget = _CollectionBudget(
            request_cap=settings.MAINTENANCE_REQUEST_CAP,
            deadline_monotonic=time.monotonic() + settings.MAINTENANCE_TIME_CAP_SECONDS,
        )

    @property
    def requests_used(self) -> int:
        return self._budget.requests_used

    async def _run_graphql(self, query: str, variables: dict[str, Any], data_type: type[Any]) -> Any:
        """
        Run one GraphQL query with budget accounting and bounded retry.

        A rate-limit response waits for the advertised reset (capped by
        ``MAINTENANCE_RATE_LIMIT_MAX_WAIT_SECONDS``) and retries once; a
        second limit, an over-cap wait, or any other API error halts
        collection with a typed reason.
        """

        requester = getattr(get_github_client(), "_Github__requester", None)
        if requester is None:
            raise _CollectionHalt(REASON_QUERY_ERROR, "PyGithub requester is unavailable")

        for attempt in range(2):
            self._budget.charge()
            try:
                _, payload = requester.graphql_query(query, variables)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            except GithubException as exc:
                wait_seconds = _rate_limit_wait_seconds(exc)
                if wait_seconds is None:
                    raise _CollectionHalt(REASON_QUERY_ERROR, f"{type(exc).__name__}: {exc}") from exc

                if attempt == 1 or wait_seconds > settings.MAINTENANCE_RATE_LIMIT_MAX_WAIT_SECONDS:
                    raise _CollectionHalt(REASON_RATE_LIMITED, f"rate limited, reset in {wait_seconds:.0f}s") from exc

                logger.warning(
                    f"maintenance-collect rate limited: repo={self._owner}/{self._repo} waiting {wait_seconds:.0f}s"
                )
                await asyncio.sleep(wait_seconds + 1.0)
                continue

            try:
                return msgspec.convert(payload.get("data"), type=data_type)  # pyright: ignore[reportUnknownMemberType]
            except msgspec.ValidationError as exc:
                raise _CollectionHalt(REASON_QUERY_ERROR, f"invalid GraphQL payload: {exc}") from exc

        raise _CollectionHalt(REASON_RATE_LIMITED, "rate limited after retry")

    # --- phases ---

    async def fetch_overview(self) -> _OverviewRepository:
        """Fetch repo-level facts; a missing repository halts the run."""

        data: _OverviewData = await self._run_graphql(
            _OVERVIEW_QUERY,
            {"owner": self._owner, "name": self._repo},
            _OverviewData,
        )
        if data.repository is None:
            raise _CollectionHalt(REASON_QUERY_ERROR, "repository not returned")

        return data.repository

    async def collect_open_ages(self, *, pulls: bool) -> BacklogSignal:
        """Walk the open listing oldest-first and build the backlog snapshot."""

        query = _OPEN_PULLS_QUERY if pulls else _OPEN_ISSUES_QUERY
        created_ats: list[dt.datetime] = []
        total_count = 0
        listing_complete = False
        after: str | None = None

        for _ in range(settings.MAINTENANCE_ITEM_PAGE_CAP):
            variables = {"owner": self._owner, "name": self._repo, "after": after}
            connection: _OpenItemsConnection | None
            if pulls:
                pulls_data: _OpenPullsData = await self._run_graphql(query, variables, _OpenPullsData)
                connection = pulls_data.repository.pullRequests if pulls_data.repository is not None else None
            else:
                issues_data: _OpenIssuesData = await self._run_graphql(query, variables, _OpenIssuesData)
                connection = issues_data.repository.issues if issues_data.repository is not None else None

            if connection is None:
                raise _CollectionHalt(REASON_QUERY_ERROR, "open listing not returned")

            total_count = connection.totalCount
            created_ats.extend(_parse_iso(node.createdAt) for node in connection.nodes or [])

            page_info = connection.pageInfo
            if page_info is None or not page_info.hasNextPage or not page_info.endCursor:
                listing_complete = True
                break

            after = page_info.endCursor

        return compute_backlog(
            created_ats,
            total_count,
            now=self._now,
            listing_complete=listing_complete,
            as_of=self._now.isoformat(),
            incomplete_reason=None if listing_complete else REASON_PAGE_CAP,
        )

    async def collect_issue_cohort(self) -> ResponsivenessSignal:
        """Walk issues newest-first, classify responses, aggregate the cohort."""

        items: list[ResponseItem] = []
        maintainer_authored = 0
        cohort_complete = False
        after: str | None = None

        for _ in range(settings.MAINTENANCE_ITEM_PAGE_CAP):
            data: _IssueCohortData = await self._run_graphql(
                _ISSUE_COHORT_QUERY,
                {"owner": self._owner, "name": self._repo, "after": after},
                _IssueCohortData,
            )
            connection = data.repository.issues if data.repository is not None else None
            if connection is None:
                raise _CollectionHalt(REASON_QUERY_ERROR, "issue cohort listing not returned")

            reached_window_start = False
            for node in connection.nodes or []:
                created_at = _parse_iso(node.createdAt)
                if created_at < self._window_start:
                    reached_window_start = True
                    break

                if _is_bot_actor(node.author):
                    continue

                if node.authorAssociation in MAINTAINER_ASSOCIATIONS:
                    maintainer_authored += 1
                    continue

                items.append(await self._classify_issue(node, created_at))

            page_info = connection.pageInfo
            if reached_window_start or page_info is None or not page_info.hasNextPage or not page_info.endCursor:
                cohort_complete = True
                break

            after = page_info.endCursor

        if not cohort_complete:
            # A truncated cohort would bias the fraction (newest-first walk
            # misses the oldest, most response-eligible items), so no values
            # are reported. Only this signal is affected: the shared budget
            # survives, so collection continues with the remaining phases.
            logger.warning(f"maintenance-collect issue cohort capped before window start: repo={self._owner}/{self._repo}")

            return ResponsivenessSignal(
                state=STATE_INCOMPLETE,
                as_of=self._now.isoformat(),
                incomplete_reason=REASON_PAGE_CAP,
            )

        return compute_responsiveness(
            items,
            now=self._now,
            window_days=settings.MAINTENANCE_WINDOW_DAYS,
            interval_days=settings.MAINTENANCE_RESPONSE_INTERVAL_DAYS,
            as_of=self._now.isoformat(),
            maintainer_authored_count=maintainer_authored,
        )

    async def _classify_issue(self, node: _IssueCohortNode, created_at: dt.datetime) -> ResponseItem:
        """Find the first maintainer comment, paginating comments as needed."""

        author_login = node.author.login if node.author is not None else None
        comments = node.comments.nodes if node.comments is not None else None
        scan = _scan_comment_page(comments or [], author_login)
        response_at = scan.qualifying_time
        unknown_times = list(scan.unknown_times)

        page_info = node.comments.pageInfo if node.comments is not None else None
        after = page_info.endCursor if page_info is not None else None
        has_more = page_info is not None and page_info.hasNextPage and after is not None

        while response_at is None and has_more:
            data: _IssueByNumberData = await self._run_graphql(
                _ISSUE_COMMENTS_QUERY,
                {"owner": self._owner, "name": self._repo, "number": node.number, "after": after},
                _IssueByNumberData,
            )
            issue = data.repository.issue if data.repository is not None else None
            connection = issue.comments if issue is not None else None
            if connection is None:
                raise _CollectionHalt(REASON_QUERY_ERROR, f"comment continuation not returned for issue {node.number}")

            scan = _scan_comment_page(connection.nodes or [], author_login)
            response_at = scan.qualifying_time
            unknown_times.extend(scan.unknown_times)
            page_info = connection.pageInfo
            after = page_info.endCursor if page_info is not None else None
            has_more = page_info is not None and page_info.hasNextPage and after is not None

        return resolve_response_item(
            created_at,
            [response_at] if response_at is not None else [],
            unknown_times,
            interval_days=settings.MAINTENANCE_RESPONSE_INTERVAL_DAYS,
        )

    async def collect_pull_cohort(self) -> ResponsivenessSignal:
        """Walk PRs newest-first, classify maintainer actions, aggregate."""

        items: list[ResponseItem] = []
        maintainer_authored = 0
        cohort_complete = False
        after: str | None = None

        for _ in range(settings.MAINTENANCE_ITEM_PAGE_CAP):
            data: _PullCohortData = await self._run_graphql(
                _PULL_COHORT_QUERY,
                {"owner": self._owner, "name": self._repo, "after": after},
                _PullCohortData,
            )
            connection = data.repository.pullRequests if data.repository is not None else None
            if connection is None:
                raise _CollectionHalt(REASON_QUERY_ERROR, "pull request cohort listing not returned")

            reached_window_start = False
            for node in connection.nodes or []:
                created_at = _parse_iso(node.createdAt)
                if created_at < self._window_start:
                    reached_window_start = True
                    break

                if _is_bot_actor(node.author):
                    continue

                if node.authorAssociation in MAINTAINER_ASSOCIATIONS:
                    maintainer_authored += 1
                    continue

                if node.isDraft and not settings.MAINTENANCE_INCLUDE_DRAFT_PRS:
                    continue

                items.append(await self._classify_pull(node, created_at))

            page_info = connection.pageInfo
            if reached_window_start or page_info is None or not page_info.hasNextPage or not page_info.endCursor:
                cohort_complete = True
                break

            after = page_info.endCursor

        if not cohort_complete:
            # Same containment as the issue cohort: report this signal
            # incomplete without biased values and keep collecting.
            logger.warning(f"maintenance-collect PR cohort capped before window start: repo={self._owner}/{self._repo}")

            return ResponsivenessSignal(
                state=STATE_INCOMPLETE,
                as_of=self._now.isoformat(),
                incomplete_reason=REASON_PAGE_CAP,
            )

        return compute_responsiveness(
            items,
            now=self._now,
            window_days=settings.MAINTENANCE_WINDOW_DAYS,
            interval_days=settings.MAINTENANCE_RESPONSE_INTERVAL_DAYS,
            as_of=self._now.isoformat(),
            maintainer_authored_count=maintainer_authored,
        )

    async def _classify_pull(self, node: _PullCohortNode, created_at: dt.datetime) -> ResponseItem:
        """Resolve one PR's first maintainer action from reviews and events."""

        author_login = node.author.login if node.author is not None else None

        review_scan = _scan_review_page(node.reviews.nodes or [] if node.reviews is not None else [], author_login)
        review_time = review_scan.qualifying_time
        unknown_times = list(review_scan.unknown_times)
        review_page_info = node.reviews.pageInfo if node.reviews is not None else None
        after = review_page_info.endCursor if review_page_info is not None else None
        has_more_reviews = review_page_info is not None and review_page_info.hasNextPage and after is not None

        while review_time is None and has_more_reviews:
            data: _PullByNumberData = await self._run_graphql(
                _PULL_REVIEWS_QUERY,
                {"owner": self._owner, "name": self._repo, "number": node.number, "after": after},
                _PullByNumberData,
            )
            pull = data.repository.pullRequest if data.repository is not None else None
            connection = pull.reviews if pull is not None else None
            if connection is None:
                raise _CollectionHalt(REASON_QUERY_ERROR, f"review continuation not returned for PR {node.number}")

            review_scan = _scan_review_page(connection.nodes or [], author_login)
            review_time = review_scan.qualifying_time
            unknown_times.extend(review_scan.unknown_times)
            review_page_info = connection.pageInfo
            after = review_page_info.endCursor if review_page_info is not None else None
            has_more_reviews = review_page_info is not None and review_page_info.hasNextPage and after is not None

        events = classify_pr_events(
            node.timelineItems.nodes or [] if node.timelineItems is not None else [],
            author_login,
        )
        timeline_page_info = node.timelineItems.pageInfo if node.timelineItems is not None else None
        timeline_truncated = timeline_page_info is not None and timeline_page_info.hasNextPage and not events.qualifying_times

        qualifying_times = list(events.qualifying_times)
        if review_time is not None:
            qualifying_times.append(review_time)

        return resolve_response_item(
            created_at,
            qualifying_times,
            unknown_times + events.unknown_times,
            interval_days=settings.MAINTENANCE_RESPONSE_INTERVAL_DAYS,
            detection_truncated=timeline_truncated,
        )

    def collect_release_fallback(self) -> ReleaseFallback:
        """
        Fetch dated GitHub Releases publication dates via REST.

        Newest-first, so truncation never affects the most recent shipping
        events; drafts are always excluded, prereleases follow
        ``MAINTENANCE_INCLUDE_PRERELEASES``. Stops once
        ``MAINTENANCE_CADENCE_LAST_N_EVENTS`` distinct dates are known.
        """

        gh = get_github_client()
        dates: set[dt.date] = set()
        complete = True
        items_walked = 0

        self._budget.charge()
        try:
            releases = gh.get_repo(f"{self._owner}/{self._repo}").get_releases()
            for release in releases:
                if items_walked >= _RELEASE_ITEM_CAP:
                    complete = False
                    break

                items_walked += 1
                if items_walked % _RELEASES_PER_PAGE == 0:
                    self._budget.charge()

                # The stubs declare published_at non-optional; the API returns
                # null for unpublished entries.
                if release.draft or release.published_at is None:  # pyright: ignore[reportUnnecessaryComparison]
                    continue

                if release.prerelease and not settings.MAINTENANCE_INCLUDE_PRERELEASES:
                    continue

                dates.add(release.published_at.astimezone(dt.UTC).date())
                if len(dates) >= settings.MAINTENANCE_CADENCE_LAST_N_EVENTS:
                    break

        except GithubException as exc:
            raise _CollectionHalt(REASON_QUERY_ERROR, f"GitHub releases fetch failed: {exc}") from exc

        return ReleaseFallback(
            state=STATE_OK if complete else STATE_INCOMPLETE,
            as_of=self._now.isoformat(),
            publication_dates=[date.isoformat() for date in sorted(dates)],
            include_prereleases=settings.MAINTENANCE_INCLUDE_PRERELEASES,
            complete=complete,
            incomplete_reason=None if complete else REASON_PAGE_CAP,
        )


async def collect_maintenance_for_repo(
    owner: str,
    repo: str,
    *,
    need_release_fallback: bool,
    now: dt.datetime | None = None,
) -> CollectedMaintenance:
    """
    Run one bounded collection pass and assemble the signals payload.

    Phases run in fixed order (overview, backlogs, cohorts, release
    fallback); a halt marks the interrupted and unattempted signals
    ``incomplete`` with the halt's typed reason while completed signals keep
    their values. A halt before anything was collected raises
    ``MaintenanceCollectionFailed`` so a previously stored payload is never
    replaced by an empty one.
    """

    observed_now = now if now is not None else dt.datetime.now(dt.UTC)
    collector = _MaintenanceCollector(owner, repo, now=observed_now)
    as_of = observed_now.isoformat()

    try:
        overview = await collector.fetch_overview()
    except _CollectionHalt as halt:
        raise MaintenanceCollectionFailed(f"{owner}/{repo}: {halt.reason}: {halt.detail}") from halt

    pushed_at = _parse_iso(overview.pushedAt) if overview.pushedAt else None
    tracker_declared_external = external_tracker_declared(owner, repo)
    issues_applicable = overview.hasIssuesEnabled and not tracker_declared_external

    issue_backlog: BacklogSignal
    issue_responsiveness: ResponsivenessSignal
    pr_backlog: BacklogSignal | None = None
    pr_responsiveness: ResponsivenessSignal | None = None
    release_fallback: ReleaseFallback | None = None
    halt_reason: str | None = None

    if not issues_applicable:
        issue_backlog = BacklogSignal(state=STATE_NOT_APPLICABLE, as_of=as_of)
        issue_responsiveness = ResponsivenessSignal(state=STATE_NOT_APPLICABLE, as_of=as_of)
    else:
        issue_backlog = BacklogSignal(state=STATE_INCOMPLETE, as_of=as_of)
        issue_responsiveness = ResponsivenessSignal(state=STATE_INCOMPLETE, as_of=as_of)

    try:
        if issues_applicable:
            issue_backlog = await collector.collect_open_ages(pulls=False)

        pr_backlog = await collector.collect_open_ages(pulls=True)

        if issues_applicable:
            issue_responsiveness = await collector.collect_issue_cohort()

        pr_responsiveness = await collector.collect_pull_cohort()

        if need_release_fallback:
            release_fallback = collector.collect_release_fallback()

    except _CollectionHalt as halt:
        halt_reason = halt.reason
        logger.warning(
            f"maintenance-collect halted: repo={owner}/{repo} reason={halt.reason} "
            f"detail={halt.detail} requests_used={collector.requests_used}"
        )

    if halt_reason is not None:
        if issues_applicable and issue_backlog.state == STATE_INCOMPLETE and issue_backlog.incomplete_reason is None:
            issue_backlog = BacklogSignal(state=STATE_INCOMPLETE, as_of=as_of, incomplete_reason=halt_reason)

        if (
            issues_applicable
            and issue_responsiveness.state == STATE_INCOMPLETE
            and issue_responsiveness.incomplete_reason is None
        ):
            issue_responsiveness = ResponsivenessSignal(state=STATE_INCOMPLETE, as_of=as_of, incomplete_reason=halt_reason)

        if pr_backlog is None:
            pr_backlog = BacklogSignal(state=STATE_INCOMPLETE, as_of=as_of, incomplete_reason=halt_reason)

        if pr_responsiveness is None:
            pr_responsiveness = ResponsivenessSignal(state=STATE_INCOMPLETE, as_of=as_of, incomplete_reason=halt_reason)

        if need_release_fallback and release_fallback is None:
            release_fallback = ReleaseFallback(state=STATE_INCOMPLETE, as_of=as_of, incomplete_reason=halt_reason)

    signals = MaintenanceSignals(
        schema_version=MAINTENANCE_SIGNALS_SCHEMA_VERSION,
        collected_at=as_of,
        window_days=settings.MAINTENANCE_WINDOW_DAYS,
        response_interval_days=settings.MAINTENANCE_RESPONSE_INTERVAL_DAYS,
        requests_used=collector.requests_used,
        issues_enabled=overview.hasIssuesEnabled,
        external_tracker_declared=tracker_declared_external,
        archived=overview.isArchived,
        issue_responsiveness=issue_responsiveness,
        issue_backlog=issue_backlog,
        pr_responsiveness=pr_responsiveness,
        pr_backlog=pr_backlog,
        release_fallback=release_fallback,
    )

    return CollectedMaintenance(signals=signals, pushed_at=pushed_at)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _keep_better_signal[SignalT: (ResponsivenessSignal, BacklogSignal, ReleaseFallback)](
    old: SignalT | None,
    new: SignalT | None,
) -> SignalT | None:
    """
    Per-signal merge: an incomplete result never overwrites a complete one.

    The kept signal retains its own ``as_of``, so the materializer's
    freshness policy — not this merge — decides how long it stays ranked.
    """

    if new is None:
        return old

    if old is not None and new.state == STATE_INCOMPLETE and old.state == STATE_OK:
        return old

    return new


def merge_collected_signals(old: MaintenanceSignals | None, new: MaintenanceSignals) -> MaintenanceSignals:
    """Merge one fresh collection into the previously stored payload."""

    if old is None:
        return new

    return MaintenanceSignals(
        schema_version=new.schema_version,
        collected_at=new.collected_at,
        window_days=new.window_days,
        response_interval_days=new.response_interval_days,
        requests_used=new.requests_used,
        issues_enabled=new.issues_enabled,
        external_tracker_declared=new.external_tracker_declared,
        archived=new.archived,
        issue_responsiveness=_keep_better_signal(old.issue_responsiveness, new.issue_responsiveness),
        issue_backlog=_keep_better_signal(old.issue_backlog, new.issue_backlog),
        pr_responsiveness=_keep_better_signal(old.pr_responsiveness, new.pr_responsiveness),
        pr_backlog=_keep_better_signal(old.pr_backlog, new.pr_backlog),
        release_fallback=_keep_better_signal(old.release_fallback, new.release_fallback),
    )


@dataclass(frozen=True)
class _ResolvedRepo:
    """Identity and stored inputs for one tracked repository."""

    repo_id: int
    canonical_id: str
    releases: list[Release] | None
    repo_metadata: dict[str, Any] | None


async def _resolve_repo(session: AsyncSession, owner: str, repo: str) -> _ResolvedRepo:
    """
    Resolve the tracked ``Repo`` by case-insensitive canonical identity.

    Zero matches raise ``MaintenanceSourceRepoNotFound``; more than one raise
    ``AmbiguousMaintenanceSourceRepo`` — the collector never silently picks
    one.
    """

    wanted = f"pkg:github/{owner}/{repo}".lower()
    rows = (
        await session.execute(
            select(Repo.id, Repo.canonical_id, Repo.releases, Repo.repo_metadata).where(
                func.lower(Repo.canonical_id) == wanted
            )
        )
    ).all()

    if not rows:
        raise MaintenanceSourceRepoNotFound(f"No tracked repo for {owner}/{repo}")
    if len(rows) > 1:
        logger.warning(f"maintenance-collect ambiguous source identity: repo={owner}/{repo} matches={len(rows)}")
        raise AmbiguousMaintenanceSourceRepo(f"{len(rows)} tracked repos match {owner}/{repo}")

    repo_id, canonical_id, releases, repo_metadata = rows[0]

    return _ResolvedRepo(repo_id=repo_id, canonical_id=canonical_id, releases=releases, repo_metadata=repo_metadata)


async def _persist_collected(
    session: AsyncSession,
    resolved: _ResolvedRepo,
    collected: CollectedMaintenance,
) -> MaintenanceSignals:
    """Merge with the stored payload and write signals plus ``pushed_at``."""

    previous = signals_from_metadata(resolved.repo_metadata, repo_canonical_id=resolved.canonical_id)
    merged = merge_collected_signals(previous, collected.signals)
    payload = {MAINTENANCE_SIGNALS_KEY: signals_to_payload(merged)}

    await session.execute(
        update(Repo)
        .where(Repo.id == resolved.repo_id)
        .values(
            repo_metadata=repo_metadata_merge_expression(payload),
            pushed_at=collected.pushed_at,
        )
        .execution_options(synchronize_session=False)
    )

    return merged


@dataclass(frozen=True)
class MaintenanceCollectionOutcome:
    """Summary of one persisted collection run for logging."""

    owner: str
    repo: str
    requests_used: int
    signal_states: dict[str, str]


def _signal_states(signals: MaintenanceSignals) -> dict[str, str]:
    """Reduce a signals payload to its per-signal states for summary logs."""

    states: dict[str, str] = {}
    for name, signal in (
        ("issue_responsiveness", signals.issue_responsiveness),
        ("issue_backlog", signals.issue_backlog),
        ("pr_responsiveness", signals.pr_responsiveness),
        ("pr_backlog", signals.pr_backlog),
        ("release_fallback", signals.release_fallback),
    ):
        if signal is not None:
            states[name] = signal.state

    return states


async def run_maintenance_collection(owner: str, repo: str) -> MaintenanceCollectionOutcome:
    """
    Collect and persist maintenance signals for one repository.

    Resolution and persistence run in short separate sessions around the
    network phase, so no transaction spans GitHub calls.
    """

    from pg_atlas.db_models.session import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        resolved = await _resolve_repo(session, owner, repo)

    depsdev_date_count = len(distinct_release_dates(resolved.releases))
    need_release_fallback = depsdev_date_count < settings.MAINTENANCE_CADENCE_MIN_EVENTS

    collected = await collect_maintenance_for_repo(owner, repo, need_release_fallback=need_release_fallback)

    async with factory() as session:
        merged = await _persist_collected(session, resolved, collected)
        await session.commit()

    outcome = MaintenanceCollectionOutcome(
        owner=owner,
        repo=repo,
        requests_used=collected.signals.requests_used,
        signal_states=_signal_states(merged),
    )

    states_text = " ".join(f"{name}={state}" for name, state in sorted(outcome.signal_states.items()))
    logger.info(
        f"maintenance-collect: repo={owner}/{repo} requests_used={outcome.requests_used} "
        f"depsdev_release_dates={depsdev_date_count} fallback_fetched={need_release_fallback} {states_text}"
    )

    return outcome


# ---------------------------------------------------------------------------
# CLI (explicit runs bypass the gate)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect GitHub maintenance signals for tracked repositories.")
    parser.add_argument("repos", nargs="+", help="Repositories as owner/repo")

    return parser


async def _cli_main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    if not settings.DATABASE_URL:
        logger.error("PG_ATLAS_DATABASE_URL is required for maintenance collection")
        raise SystemExit(1)

    if not os.environ.get("GITHUB_TOKEN"):
        logger.warning(
            "GITHUB_TOKEN is not set; unauthenticated collection is not viable for issue/PR volumes "
            "and will record rate-limited signals as incomplete"
        )

    failures = 0
    for spec in args.repos:
        parts = spec.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            logger.error(f"Invalid repository argument (expected owner/repo): {spec!r}")
            failures += 1
            continue

        try:
            await run_maintenance_collection(parts[0], parts[1])
        except (MaintenanceSourceRepoNotFound, AmbiguousMaintenanceSourceRepo, MaintenanceCollectionFailed) as exc:
            logger.error(f"maintenance-collect failed: repo={spec} error={exc}")
            failures += 1

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(_cli_main())
