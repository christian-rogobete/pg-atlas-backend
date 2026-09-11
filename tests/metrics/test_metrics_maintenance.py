"""
Unit tests for maintenance-metric signal computation.

Covers the plan's semantics exactly: interval-aged response cohorts where
unanswered work counts against the fraction, backlog medians as order
statistics that survive capped listings, release cadence over distinct
publication dates, artifact-gated commit activity, and inverted percentile
ranking.

SPDX-FileCopyrightText: 2026 PG Atlas contributors
SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

import datetime as dt

from pg_atlas.db_models.release import Release
from pg_atlas.gitlog.parser import CommitRecord
from pg_atlas.metrics.maintenance import (
    CADENCE_SOURCE_DEPSDEV,
    CADENCE_SOURCE_GITHUB_RELEASES,
    REASON_PAGE_CAP,
    REASON_STALE_ARTIFACT,
    REASON_WINDOW_NOT_COVERED,
    STATE_INCOMPLETE,
    STATE_OK,
    STATE_UNAVAILABLE,
    CadenceResult,
    CommitActivityResult,
    MaintenanceSignals,
    ResponseItem,
    ResponsivenessSignal,
    compute_backlog,
    compute_commit_activity,
    compute_release_cadence,
    compute_responsiveness,
    distinct_release_dates,
    rank_scalar_values,
    signals_from_metadata,
    signals_to_payload,
)

NOW = dt.datetime(2026, 9, 9, 12, 0, tzinfo=dt.UTC)
WINDOW_DAYS = 180
INTERVAL_DAYS = 7


def _days_ago(days: float) -> dt.datetime:
    return NOW - dt.timedelta(days=days)


# ---------------------------------------------------------------------------
# Responsiveness (signals 2 and 4)
# ---------------------------------------------------------------------------


class TestComputeResponsiveness:
    def _compute(self, items: list[ResponseItem]) -> ResponsivenessSignal:
        return compute_responsiveness(
            items,
            now=NOW,
            window_days=WINDOW_DAYS,
            interval_days=INTERVAL_DAYS,
            as_of=NOW.isoformat(),
        )

    def test_answered_within_interval_counts(self) -> None:
        items = [ResponseItem(created_at=_days_ago(30), first_response_at=_days_ago(29))]
        signal = self._compute(items)
        assert signal.state == STATE_OK
        assert signal.eligible_size == 1
        assert signal.responded_within_interval == 1
        assert signal.response_fraction == 1.0

    def test_answered_after_interval_stays_in_denominator(self) -> None:
        """A late answer counts against the fraction but feeds the context median."""
        items = [ResponseItem(created_at=_days_ago(30), first_response_at=_days_ago(10))]
        signal = self._compute(items)
        assert signal.response_fraction == 0.0
        assert signal.median_response_days_context == 20.0

    def test_unanswered_eligible_counts_against_fraction(self) -> None:
        """Unanswered work must not vanish from the measurement."""
        items = [
            ResponseItem(created_at=_days_ago(30), first_response_at=_days_ago(29.5)),
            ResponseItem(created_at=_days_ago(40), first_response_at=None),
        ]
        signal = self._compute(items)
        assert signal.eligible_size == 2
        assert signal.responded_within_interval == 1
        assert signal.response_fraction == 0.5

    def test_item_younger_than_interval_not_yet_eligible(self) -> None:
        """An item without the full response interval cannot be judged yet."""
        items = [
            ResponseItem(created_at=_days_ago(2), first_response_at=None),
            ResponseItem(created_at=_days_ago(30), first_response_at=_days_ago(29)),
        ]
        signal = self._compute(items)
        assert signal.cohort_size == 2
        assert signal.eligible_size == 1
        assert signal.response_fraction == 1.0

    def test_item_outside_window_excluded(self) -> None:
        items = [
            ResponseItem(created_at=_days_ago(WINDOW_DAYS + 1), first_response_at=None),
            ResponseItem(created_at=_days_ago(30), first_response_at=_days_ago(29)),
        ]
        signal = self._compute(items)
        assert signal.cohort_size == 1

    def test_indeterminate_excluded_and_counted(self) -> None:
        """Unknown attribution is excluded from the fraction, never guessed."""
        items = [
            ResponseItem(created_at=_days_ago(30), indeterminate=True),
            ResponseItem(created_at=_days_ago(40), first_response_at=None),
        ]
        signal = self._compute(items)
        assert signal.unknown_attribution_count == 1
        assert signal.eligible_size == 1
        assert signal.response_fraction == 0.0

    def test_empty_eligible_cohort_has_no_fraction(self) -> None:
        """Nothing to rank is distinct from a real fraction of zero."""
        signal = self._compute([ResponseItem(created_at=_days_ago(1), first_response_at=None)])
        assert signal.state == STATE_OK
        assert signal.eligible_size == 0
        assert signal.response_fraction is None

    def test_all_unanswered_is_a_real_zero(self) -> None:
        items = [ResponseItem(created_at=_days_ago(60), first_response_at=None)]
        signal = self._compute(items)
        assert signal.response_fraction == 0.0

    def test_median_context_over_all_answered(self) -> None:
        items = [
            ResponseItem(created_at=_days_ago(30), first_response_at=_days_ago(29)),
            ResponseItem(created_at=_days_ago(40), first_response_at=_days_ago(37)),
            ResponseItem(created_at=_days_ago(50), first_response_at=None),
        ]
        signal = self._compute(items)
        assert signal.median_response_days_context == 2.0

    def test_incomplete_reason_marks_signal_incomplete(self) -> None:
        signal = compute_responsiveness(
            [],
            now=NOW,
            window_days=WINDOW_DAYS,
            interval_days=INTERVAL_DAYS,
            incomplete_reason=REASON_PAGE_CAP,
        )
        assert signal.state == STATE_INCOMPLETE
        assert signal.incomplete_reason == REASON_PAGE_CAP


# ---------------------------------------------------------------------------
# Backlog snapshot (signal 3 and signal 4's snapshot)
# ---------------------------------------------------------------------------


class TestComputeBacklog:
    def test_complete_listing_exact_median(self) -> None:
        created = [_days_ago(100), _days_ago(50), _days_ago(10)]
        signal = compute_backlog(created, 3, now=NOW, listing_complete=True)
        assert signal.state == STATE_OK
        assert signal.open_count == 3
        assert signal.median_open_age_days == 50.0
        assert signal.median_age_exact is True

    def test_zero_open_is_a_real_zero(self) -> None:
        signal = compute_backlog([], 0, now=NOW, listing_complete=True)
        assert signal.state == STATE_OK
        assert signal.open_count == 0
        assert signal.median_open_age_days is None
        assert signal.median_age_exact is True

    def test_capped_prefix_reaching_middle_yields_exact_median_odd(self) -> None:
        """Oldest-first pagination makes the median an order statistic we may already hold."""
        created = [_days_ago(100), _days_ago(80), _days_ago(60)]  # oldest 3 of 5 open
        signal = compute_backlog(created, 5, now=NOW, listing_complete=False, incomplete_reason=REASON_PAGE_CAP)
        assert signal.state == STATE_OK
        assert signal.open_count == 5
        assert signal.median_open_age_days == 60.0

    def test_capped_prefix_reaching_middle_yields_exact_median_even(self) -> None:
        created = [_days_ago(100), _days_ago(80), _days_ago(60)]  # oldest 3 of 4 open
        signal = compute_backlog(created, 4, now=NOW, listing_complete=False, incomplete_reason=REASON_PAGE_CAP)
        assert signal.median_open_age_days == 70.0

    def test_capped_prefix_short_of_middle_is_incomplete(self) -> None:
        created = [_days_ago(100), _days_ago(80)]  # oldest 2 of 10 open
        signal = compute_backlog(created, 10, now=NOW, listing_complete=False, incomplete_reason=REASON_PAGE_CAP)
        assert signal.state == STATE_INCOMPLETE
        assert signal.incomplete_reason == REASON_PAGE_CAP
        assert signal.median_open_age_days is None
        assert signal.open_count == 10
        assert signal.open_count_exact is True


# ---------------------------------------------------------------------------
# Release cadence (signal 1)
# ---------------------------------------------------------------------------


def _release(date: str, purl: str = "pkg:npm/a", version: str = "1.0.0") -> Release:
    return Release(purl=purl, version=version, release_date=date)


class TestDistinctReleaseDates:
    def test_multi_package_same_date_is_one_shipping_event(self) -> None:
        releases = [
            _release("2026-06-01T10:00:00Z", purl="pkg:npm/a"),
            _release("2026-06-01T10:05:00Z", purl="pkg:npm/b"),
            _release("2026-07-01T09:00:00Z", purl="pkg:npm/a"),
        ]
        assert distinct_release_dates(releases) == [dt.date(2026, 6, 1), dt.date(2026, 7, 1)]

    def test_undated_records_are_skipped(self) -> None:
        releases = [_release(""), _release("not-a-date")]
        assert distinct_release_dates(releases) == []

    def test_none_input(self) -> None:
        assert distinct_release_dates(None) == []


class TestComputeReleaseCadence:
    def _cadence(
        self, depsdev: list[dt.date], fallback: list[dt.date], *, last_n: int = 10, min_events: int = 3
    ) -> CadenceResult:
        return compute_release_cadence(depsdev, fallback, now=NOW, last_n_events=last_n, min_events=min_events)

    def test_interleaved_two_package_cycle_yields_true_gap(self) -> None:
        """Two packages shipped together every 30 days must not produce zero gaps."""
        dates = [dt.date(2026, 5, 1), dt.date(2026, 5, 31), dt.date(2026, 6, 30), dt.date(2026, 7, 30)]
        result = self._cadence(dates, [])
        assert result.state == STATE_OK
        assert result.source == CADENCE_SOURCE_DEPSDEV
        assert result.median_gap_days == 30.0
        assert result.days_since_last_release == float((NOW.date() - dt.date(2026, 7, 30)).days)

    def test_below_min_events_gives_no_median_but_recency(self) -> None:
        result = self._cadence([dt.date(2026, 7, 1), dt.date(2026, 8, 1)], [])
        assert result.state == STATE_OK
        assert result.median_gap_days is None
        assert result.days_since_last_release is not None

    def test_no_dates_anywhere_is_unavailable(self) -> None:
        """Tag-only or release-free repos are unsupported for cadence, never zero."""
        result = self._cadence([], [])
        assert result.state == STATE_UNAVAILABLE
        assert result.source is None
        assert result.median_gap_days is None
        assert result.days_since_last_release is None

    def test_fallback_wins_with_more_dates(self) -> None:
        fallback = [dt.date(2026, 1, 1), dt.date(2026, 2, 1), dt.date(2026, 3, 1)]
        result = self._cadence([dt.date(2026, 3, 1)], fallback)
        assert result.source == CADENCE_SOURCE_GITHUB_RELEASES
        assert result.shipping_events == 3

    def test_tie_keeps_depsdev(self) -> None:
        result = self._cadence([dt.date(2026, 3, 1)], [dt.date(2026, 4, 1)])
        assert result.source == CADENCE_SOURCE_DEPSDEV

    def test_median_uses_only_last_n_events(self) -> None:
        """Ancient gaps outside the last-N window must not drag the median."""
        old = [dt.date(2020, 1, 1), dt.date(2021, 1, 1)]
        recent = [dt.date(2026, 5, 1), dt.date(2026, 5, 31), dt.date(2026, 6, 30), dt.date(2026, 7, 30)]
        result = self._cadence(old + recent, [], last_n=4)
        assert result.median_gap_days == 30.0


# ---------------------------------------------------------------------------
# Commit activity (signal 5)
# ---------------------------------------------------------------------------


def _commit(days_ago: float, name: str = "Alice", email: str = "alice@example.org") -> CommitRecord:
    return CommitRecord(author_name=name, author_email=email, timestamp=_days_ago(days_ago), commit_hash="a" * 40)


class TestComputeCommitActivity:
    def _compute(
        self, commits: list[CommitRecord], *, submitted_days_ago: float = 1.0, since_months: int = 24
    ) -> CommitActivityResult:
        return compute_commit_activity(
            commits,
            artifact_submitted_at=_days_ago(submitted_days_ago),
            artifact_since_months=since_months,
            now=NOW,
            window_days=WINDOW_DAYS,
            max_artifact_age_days=21,
        )

    def test_window_and_bot_filtering(self) -> None:
        commits = [
            _commit(10),
            _commit(170),
            _commit(WINDOW_DAYS + 5),  # outside the window
            _commit(20, name="dependabot[bot]", email="1+dependabot[bot]@users.noreply.github.com"),
        ]
        result = self._compute(commits)
        assert result.state == STATE_OK
        assert result.commit_count == 2

    def test_zero_commits_fresh_artifact_is_a_real_zero(self) -> None:
        result = self._compute([])
        assert result.state == STATE_OK
        assert result.commit_count == 0

    def test_stale_artifact_yields_context_not_a_ranked_value(self) -> None:
        """Dormant repos are re-logged slowly; their counts must not rank."""
        result = self._compute([_commit(10)], submitted_days_ago=40.0)
        assert result.state == STATE_INCOMPLETE
        assert result.reason == REASON_STALE_ARTIFACT
        assert result.commit_count == 1
        assert result.artifact_as_of == _days_ago(40.0).isoformat()

    def test_artifact_not_covering_window_is_incomplete(self) -> None:
        result = self._compute([_commit(10)], since_months=3)
        assert result.state == STATE_INCOMPLETE
        assert result.reason == REASON_WINDOW_NOT_COVERED

    def test_missing_artifact_is_unavailable(self) -> None:
        result = compute_commit_activity(
            None,
            artifact_submitted_at=None,
            artifact_since_months=None,
            now=NOW,
            window_days=WINDOW_DAYS,
            max_artifact_age_days=21,
        )
        assert result.state == STATE_UNAVAILABLE
        assert result.commit_count is None


# ---------------------------------------------------------------------------
# Percentile ranking
# ---------------------------------------------------------------------------


class TestRankScalarValues:
    def test_higher_is_better_ranks_ascending(self) -> None:
        percentiles, pool_size = rank_scalar_values([1.0, 2.0, 3.0, 4.0], higher_is_better=True)
        assert pool_size == 4
        assert percentiles == [0.0, 25.0, 50.0, 75.0]

    def test_lower_is_better_inverts(self) -> None:
        """A higher percentile must always mean better upkeep."""
        percentiles, _ = rank_scalar_values([1.0, 2.0, 3.0, 4.0], higher_is_better=False)
        assert percentiles == [75.0, 50.0, 25.0, 0.0]

    def test_ties_share_the_lowest_percentile(self) -> None:
        percentiles, _ = rank_scalar_values([5.0, 5.0, 1.0], higher_is_better=True)
        assert percentiles[0] == percentiles[1]
        assert percentiles[2] == 0.0

    def test_none_excluded_from_pool(self) -> None:
        percentiles, pool_size = rank_scalar_values([None, 2.0, None, 1.0], higher_is_better=True)
        assert pool_size == 2
        assert percentiles[0] is None
        assert percentiles[2] is None
        assert percentiles[1] == 50.0
        assert percentiles[3] == 0.0

    def test_singleton_pool_gets_zeroth_percentile(self) -> None:
        """No element is unconditionally top-ranked, even alone."""
        percentiles, pool_size = rank_scalar_values([7.0], higher_is_better=True)
        assert pool_size == 1
        assert percentiles == [0.0]

    def test_empty_pool(self) -> None:
        percentiles, pool_size = rank_scalar_values([None, None], higher_is_better=True)
        assert pool_size == 0
        assert percentiles == [None, None]


# ---------------------------------------------------------------------------
# Payload round trip
# ---------------------------------------------------------------------------


class TestSignalsPayloadRoundTrip:
    def test_round_trip_through_metadata(self) -> None:
        signals = MaintenanceSignals(
            schema_version=1,
            collected_at=NOW.isoformat(),
            window_days=WINDOW_DAYS,
            response_interval_days=INTERVAL_DAYS,
            requests_used=7,
            issues_enabled=True,
        )
        metadata = {"maintenance_signals": signals_to_payload(signals)}
        restored = signals_from_metadata(metadata)
        assert restored == signals

    def test_missing_key_returns_none(self) -> None:
        assert signals_from_metadata({}) is None
        assert signals_from_metadata(None) is None

    def test_invalid_shape_is_logged_and_skipped(self) -> None:
        assert signals_from_metadata({"maintenance_signals": {"schema_version": "not-an-int"}}) is None
