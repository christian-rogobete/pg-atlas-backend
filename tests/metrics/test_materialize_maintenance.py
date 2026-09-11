"""
DB-backed tests for maintenance profile materialization and persistence.

Seeds deterministic repos inside a rolled-back transaction, runs the
materializer, and asserts per-repo values, percentile ranks, coverage
states, freshness downgrades, idempotent re-runs, the execution-time gate,
stale-profile cleanup, and the collector's per-signal persistence merge.

SPDX-FileCopyrightText: 2026 PG Atlas contributors
SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncGenerator, Callable
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pg_atlas.config import settings
from pg_atlas.db_models.base import ActivityStatus, ProjectType, SubmissionStatus, Visibility
from pg_atlas.db_models.gitlog_artifact import GitLogArtifact
from pg_atlas.db_models.project import Project
from pg_atlas.db_models.release import Release
from pg_atlas.db_models.repo_vertex import Repo
from pg_atlas.metrics.maintenance import (
    MAINTENANCE_PROFILE_KEY,
    MAINTENANCE_SIGNALS_KEY,
    REASON_STALE_COLLECTION,
    STATE_INCOMPLETE,
    STATE_OK,
    STATE_UNAVAILABLE,
    BacklogSignal,
    MaintenanceSignals,
    ResponsivenessSignal,
    signals_to_payload,
)
from pg_atlas.metrics.materialize_maintenance import materialize_maintenance_profiles
from pg_atlas.procrastinate.github_maintenance import (
    AmbiguousMaintenanceSourceRepo,
    CollectedMaintenance,
    MaintenanceSourceRepoNotFound,
    _persist_collected,
    _resolve_repo,
)


@pytest.fixture
async def rollback_db_session(db_session: AsyncSession) -> AsyncGenerator[AsyncSession, None]:
    """Run each maintenance materialization test inside a rolled-back transaction."""

    transaction = await db_session.begin()
    try:
        yield db_session
    finally:
        if transaction.is_active:
            await transaction.rollback()


def _gitlog_content(commits: list[tuple[str, str, dt.datetime]]) -> bytes:
    """Render commits in the stored ``%aN%x00%aE%x00%aI%x00%H`` line format."""

    lines = [f"{name}\x00{email}\x00{ts.isoformat()}\x00{'a' * 40}" for name, email, ts in commits]

    return "\n".join(lines).encode()


def _ok_signals(now: dt.datetime, *, as_of: dt.datetime | None = None) -> MaintenanceSignals:
    """A fully collected signals payload with known values."""

    stamp = (as_of or now).isoformat()

    return MaintenanceSignals(
        schema_version=1,
        collected_at=stamp,
        window_days=180,
        response_interval_days=7,
        requests_used=9,
        issues_enabled=True,
        archived=False,
        issue_responsiveness=ResponsivenessSignal(
            state=STATE_OK,
            as_of=stamp,
            window_days=180,
            interval_days=7,
            cohort_size=5,
            eligible_size=5,
            responded_within_interval=4,
            response_fraction=0.8,
            median_response_days_context=1.5,
        ),
        issue_backlog=BacklogSignal(state=STATE_OK, as_of=stamp, open_count=3, median_open_age_days=40.0),
        pr_responsiveness=ResponsivenessSignal(
            state=STATE_OK,
            as_of=stamp,
            window_days=180,
            interval_days=7,
            include_drafts=True,
            cohort_size=4,
            eligible_size=4,
            responded_within_interval=2,
            response_fraction=0.5,
        ),
        pr_backlog=BacklogSignal(state=STATE_OK, as_of=stamp, open_count=2, median_open_age_days=10.0),
    )


class SeededMaintenanceFixture:
    """Seeded IDs for one deterministic maintenance materialization test."""

    def __init__(self) -> None:
        self.now = dt.datetime.now(dt.UTC)
        self.repo_full_id = 0
        self.repo_sparse_id = 0
        self.repo_stale_id = 0
        self.repo_ineligible_id = 0
        self.last_release_date = dt.date.min


async def _seed_maintenance_fixture(
    session: AsyncSession,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> SeededMaintenanceFixture:
    """Insert deterministic repos, one project, and one git-log artifact."""

    fixture = SeededMaintenanceFixture()
    suffix = uuid4().hex[:8]
    now = fixture.now

    monkeypatch.setattr(settings, "MAINTENANCE_WINDOW_DAYS", 180)
    monkeypatch.setattr(settings, "MAINTENANCE_RESPONSE_INTERVAL_DAYS", 7)
    monkeypatch.setattr(settings, "MAINTENANCE_INCLUDE_DRAFT_PRS", True)
    monkeypatch.setattr(settings, "MAINTENANCE_CADENCE_LAST_N_EVENTS", 10)
    monkeypatch.setattr(settings, "MAINTENANCE_CADENCE_MIN_EVENTS", 3)
    monkeypatch.setattr(settings, "MAINTENANCE_SIGNALS_MAX_AGE_DAYS", 14)
    monkeypatch.setattr(settings, "MAINTENANCE_GITLOG_MAX_AGE_DAYS", 21)
    monkeypatch.setattr(settings, "MAINTENANCE_DECLARED_MAINTAINERS", "")

    # Neutralize pre-existing rows inside the rollback-only transaction so
    # the percentile pools are deterministic for this test.
    await session.execute(update(Repo).values(releases=None, repo_metadata=None, pushed_at=None))
    await session.execute(update(GitLogArtifact).values(status=SubmissionStatus.failed))
    await session.flush()

    project = Project(
        canonical_id=f"daoip-5:stellar:project:maintenance-{suffix}",
        display_name=f"Maintenance {suffix}",
        project_type=ProjectType.public_good,
        activity_status=ActivityStatus.live,
    )
    session.add(project)
    await session.flush()

    # Six shipping events 30 days apart, two packages published together on
    # the last date (one shipping event, not two).
    release_dates = [now.date() - dt.timedelta(days=15 + 30 * i) for i in range(6)]
    fixture.last_release_date = release_dates[0]
    releases = [
        Release(purl="pkg:npm/a", version=f"1.{i}.0", release_date=f"{day.isoformat()}T10:00:00Z")
        for i, day in enumerate(release_dates)
    ]
    releases.append(Release(purl="pkg:npm/b", version="2.0.0", release_date=f"{release_dates[0].isoformat()}T10:05:00Z"))

    repo_full = Repo(
        canonical_id=f"pkg:github/test/maintenance-full-{suffix}",
        display_name=f"maintenance-full-{suffix}",
        visibility=Visibility.public,
        latest_version="1.5.0",
        project_id=project.id,
        repo_url=f"https://github.com/test/maintenance-full-{suffix}",
        releases=releases,
        repo_metadata={MAINTENANCE_SIGNALS_KEY: signals_to_payload(_ok_signals(now))},
        pushed_at=now - dt.timedelta(days=2),
    )
    repo_sparse = Repo(
        canonical_id=f"pkg:github/test/maintenance-sparse-{suffix}",
        display_name=f"maintenance-sparse-{suffix}",
        visibility=Visibility.public,
        latest_version="0.1.0",
        project_id=project.id,
    )
    stale_signals = _ok_signals(now, as_of=now - dt.timedelta(days=60))
    repo_stale = Repo(
        canonical_id=f"pkg:github/test/maintenance-stale-{suffix}",
        display_name=f"maintenance-stale-{suffix}",
        visibility=Visibility.public,
        latest_version="0.2.0",
        project_id=project.id,
        repo_metadata={MAINTENANCE_SIGNALS_KEY: signals_to_payload(stale_signals)},
        pushed_at=now - dt.timedelta(days=5),
    )
    repo_ineligible = Repo(
        canonical_id=f"pkg:github/test/maintenance-ineligible-{suffix}",
        display_name=f"maintenance-ineligible-{suffix}",
        visibility=Visibility.public,
        latest_version="0.0.1",
        repo_metadata={MAINTENANCE_PROFILE_KEY: {"schema_version": 1}, "other_key": "kept"},
    )
    session.add_all([repo_full, repo_sparse, repo_stale, repo_ineligible])
    await session.flush()

    fixture.repo_full_id = repo_full.id
    fixture.repo_sparse_id = repo_sparse.id
    fixture.repo_stale_id = repo_stale.id
    fixture.repo_ineligible_id = repo_ineligible.id

    # Fresh, window-covering git-log artifact for the full repo: three human
    # commits in the window, one bot commit, one commit outside the window.
    artifact_dir = tmp_path_factory.mktemp("maintenance-artifacts")
    monkeypatch.setattr(settings, "ARTIFACT_STORE_PATH", artifact_dir)
    artifact_path = f"git-logs/test/maintenance-full-{suffix}.gitlog"
    full_path = artifact_dir / artifact_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_bytes(
        _gitlog_content(
            [
                ("Alice", "alice@example.org", now - dt.timedelta(days=10)),
                ("Alice", "alice@example.org", now - dt.timedelta(days=50)),
                ("Bob", "bob@example.org", now - dt.timedelta(days=170)),
                ("dependabot[bot]", "1+dependabot[bot]@users.noreply.github.com", now - dt.timedelta(days=20)),
                ("Alice", "alice@example.org", now - dt.timedelta(days=200)),
            ]
        )
    )
    session.add(
        GitLogArtifact(
            repo_id=repo_full.id,
            since_months=24,
            artifact_path=artifact_path,
            status=SubmissionStatus.processed,
        )
    )
    await session.flush()

    return fixture


async def _load_profile(session: AsyncSession, repo_id: int) -> dict[str, Any] | None:
    metadata = (await session.execute(select(Repo.repo_metadata).where(Repo.id == repo_id))).scalar_one()
    if metadata is None:
        return None

    profile = metadata.get(MAINTENANCE_PROFILE_KEY)

    return dict(profile) if profile is not None else None


class TestMaterializeMaintenanceProfiles:
    async def test_full_pass_values_ranks_and_states(
        self,
        rollback_db_session: AsyncSession,
        tmp_path_factory: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
        assert_no_uow: Callable[[AsyncSession], None],
    ) -> None:
        session = rollback_db_session
        fixture = await _seed_maintenance_fixture(session, tmp_path_factory, monkeypatch)

        stats, payloads = await materialize_maintenance_profiles(session, now=fixture.now)
        assert_no_uow(session)

        assert stats.gate_skipped is False
        assert stats.repos_eligible >= 3
        assert stats.profiles_written == stats.repos_eligible
        assert stats.stale_profiles_cleared >= 1
        assert stats.artifacts_read >= 1

        # --- full repo: every signal present, correct, and ranked ---
        profile = await _load_profile(session, fixture.repo_full_id)
        assert profile is not None
        signals = profile["signals"]

        cadence = signals["release_cadence"]
        assert cadence["source"] == "depsdev"
        assert cadence["shipping_events"] == 6
        assert cadence["median_gap_days"]["state"] == STATE_OK
        assert cadence["median_gap_days"]["value"] == 30.0
        expected_days_since = float((fixture.now.date() - fixture.last_release_date).days)
        assert cadence["days_since_last_release"]["value"] == expected_days_since

        issue_resp = signals["issue_responsiveness"]
        assert issue_resp["response_fraction"]["state"] == STATE_OK
        assert issue_resp["response_fraction"]["value"] == 0.8
        assert issue_resp["response_fraction"]["percentile"] is not None
        assert issue_resp["response_fraction"]["pool_size"] == 1
        assert issue_resp["cohort_size"] == 5

        issue_backlog = signals["issue_backlog"]
        assert issue_backlog["open_count"]["value"] == 3
        assert issue_backlog["median_open_age_days"]["value"] == 40.0

        pr_resp = signals["pr_responsiveness"]
        assert pr_resp["response_fraction"]["value"] == 0.5
        assert pr_resp["open_count"]["value"] == 2
        assert pr_resp["median_open_age_days"]["value"] == 10.0

        activity = signals["activity_recency"]
        assert activity["days_since_push"]["state"] == STATE_OK
        assert activity["days_since_push"]["value"] == 2.0
        assert activity["commit_count_window"]["state"] == STATE_OK
        assert activity["commit_count_window"]["value"] == 3

        # The collected-signals payload must survive the profile write.
        metadata = (await session.execute(select(Repo.repo_metadata).where(Repo.id == fixture.repo_full_id))).scalar_one()
        assert metadata is not None
        assert MAINTENANCE_SIGNALS_KEY in metadata

        # --- sparse repo: everything unavailable, nothing coerced to zero ---
        sparse = await _load_profile(session, fixture.repo_sparse_id)
        assert sparse is not None
        sparse_signals = sparse["signals"]
        assert sparse_signals["release_cadence"]["median_gap_days"]["state"] == STATE_UNAVAILABLE
        assert sparse_signals["issue_responsiveness"]["response_fraction"]["state"] == STATE_UNAVAILABLE
        assert sparse_signals["activity_recency"]["days_since_push"]["state"] == STATE_UNAVAILABLE
        assert sparse_signals["activity_recency"]["commit_count_window"]["state"] == STATE_UNAVAILABLE
        assert "percentile" not in sparse_signals["issue_responsiveness"]["response_fraction"]

        # --- stale repo: context only, excluded from every ranked pool ---
        stale = await _load_profile(session, fixture.repo_stale_id)
        assert stale is not None
        stale_fraction = stale["signals"]["issue_responsiveness"]["response_fraction"]
        assert stale_fraction["state"] == STATE_INCOMPLETE
        assert stale_fraction["reason"] == REASON_STALE_COLLECTION
        assert stale_fraction["value"] == 0.8
        assert "percentile" not in stale_fraction
        stale_push = stale["signals"]["activity_recency"]["days_since_push"]
        assert stale_push["state"] == STATE_INCOMPLETE
        assert stale_push["reason"] == REASON_STALE_COLLECTION

        # --- ineligible repo: profile removed, other metadata preserved ---
        ineligible_metadata = (
            await session.execute(select(Repo.repo_metadata).where(Repo.id == fixture.repo_ineligible_id))
        ).scalar_one()
        assert ineligible_metadata is not None
        assert MAINTENANCE_PROFILE_KEY not in ineligible_metadata
        assert ineligible_metadata["other_key"] == "kept"

        # --- export payloads mirror what was written ---
        full_canonical = (await session.execute(select(Repo.canonical_id).where(Repo.id == fixture.repo_full_id))).scalar_one()
        assert payloads[full_canonical]["signals"]["issue_responsiveness"]["response_fraction"]["value"] == 0.8

    async def test_rerun_is_idempotent(
        self,
        rollback_db_session: AsyncSession,
        tmp_path_factory: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = rollback_db_session
        fixture = await _seed_maintenance_fixture(session, tmp_path_factory, monkeypatch)

        stats_first, _ = await materialize_maintenance_profiles(session, now=fixture.now)
        first = await _load_profile(session, fixture.repo_full_id)

        stats_second, _ = await materialize_maintenance_profiles(session, now=fixture.now)
        second = await _load_profile(session, fixture.repo_full_id)

        assert stats_second.repos_eligible == stats_first.repos_eligible
        assert first == second

    async def test_gated_run_skips_when_disabled(
        self,
        rollback_db_session: AsyncSession,
        tmp_path_factory: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The execution-time half of the fail-closed gate: nothing is written."""

        session = rollback_db_session
        fixture = await _seed_maintenance_fixture(session, tmp_path_factory, monkeypatch)
        monkeypatch.setattr(settings, "MAINTENANCE_METRIC_ENABLED", False)

        stats, payloads = await materialize_maintenance_profiles(session, enforce_gate=True, now=fixture.now)

        assert stats.gate_skipped is True
        assert stats.profiles_written == 0
        assert payloads == {}
        assert await _load_profile(session, fixture.repo_full_id) is None

        # The ineligible repo's stale profile also stays untouched by a gated run.
        ineligible_metadata = (
            await session.execute(select(Repo.repo_metadata).where(Repo.id == fixture.repo_ineligible_id))
        ).scalar_one()
        assert ineligible_metadata is not None
        assert MAINTENANCE_PROFILE_KEY in ineligible_metadata

    async def test_no_write_computes_without_persisting(
        self,
        rollback_db_session: AsyncSession,
        tmp_path_factory: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = rollback_db_session
        fixture = await _seed_maintenance_fixture(session, tmp_path_factory, monkeypatch)

        stats, payloads = await materialize_maintenance_profiles(session, write=False, now=fixture.now)

        assert stats.profiles_written == 0
        assert payloads
        assert await _load_profile(session, fixture.repo_full_id) is None


# ---------------------------------------------------------------------------
# Collector persistence (resolution and per-signal merge round trip)
# ---------------------------------------------------------------------------


class TestPersistCollected:
    async def test_merge_preserves_foreign_metadata_and_complete_signals(
        self,
        rollback_db_session: AsyncSession,
    ) -> None:
        session = rollback_db_session
        suffix = uuid4().hex[:8]
        now = dt.datetime.now(dt.UTC)

        old = _ok_signals(now, as_of=now - dt.timedelta(days=3))
        repo = Repo(
            canonical_id=f"pkg:github/Test/persist-{suffix}",
            display_name=f"persist-{suffix}",
            visibility=Visibility.public,
            latest_version="1.0.0",
            repo_metadata={
                "adoption_downloads_by_purl": {"pkg:npm/a": 42},
                MAINTENANCE_SIGNALS_KEY: signals_to_payload(old),
            },
        )
        session.add(repo)
        await session.flush()

        stamp = now.isoformat()
        new = MaintenanceSignals(
            schema_version=1,
            collected_at=stamp,
            window_days=180,
            response_interval_days=7,
            requests_used=4,
            issues_enabled=True,
            issue_responsiveness=ResponsivenessSignal(state=STATE_INCOMPLETE, as_of=stamp, incomplete_reason="rate-limited"),
            issue_backlog=BacklogSignal(state=STATE_OK, as_of=stamp, open_count=7, median_open_age_days=12.0),
        )
        pushed_at = now - dt.timedelta(days=1)

        resolved = await _resolve_repo(session, "test", f"PERSIST-{suffix}")
        assert resolved.repo_id == repo.id
        await _persist_collected(session, resolved, CollectedMaintenance(signals=new, pushed_at=pushed_at))

        row = (await session.execute(select(Repo.repo_metadata, Repo.pushed_at).where(Repo.id == repo.id))).one()
        metadata, stored_pushed_at = row

        assert metadata["adoption_downloads_by_purl"] == {"pkg:npm/a": 42}
        assert stored_pushed_at == pushed_at

        stored = metadata[MAINTENANCE_SIGNALS_KEY]
        # The incomplete responsiveness result must not clobber the stored
        # complete signal; the fresh complete backlog replaces the old one.
        assert stored["issue_responsiveness"]["state"] == STATE_OK
        assert stored["issue_responsiveness"]["response_fraction"] == 0.8
        assert stored["issue_backlog"]["open_count"] == 7
        assert stored["collected_at"] == stamp

    async def test_resolution_errors(self, rollback_db_session: AsyncSession) -> None:
        session = rollback_db_session
        suffix = uuid4().hex[:8]

        with pytest.raises(MaintenanceSourceRepoNotFound):
            await _resolve_repo(session, "nobody", f"missing-{suffix}")

        for casing in (f"Amb-{suffix}", f"amb-{suffix}"):
            session.add(
                Repo(
                    canonical_id=f"pkg:github/test/{casing}",
                    display_name=casing,
                    visibility=Visibility.public,
                    latest_version="1.0.0",
                )
            )
        await session.flush()

        with pytest.raises(AmbiguousMaintenanceSourceRepo):
            await _resolve_repo(session, "test", f"amb-{suffix}")


# ---------------------------------------------------------------------------
# Artifact read failures
# ---------------------------------------------------------------------------


class TestCommitActivityArtifactStates:
    async def test_unreadable_artifact_is_incomplete_not_unavailable(
        self,
        tmp_path_factory: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Failed collection and absent data are different states."""

        from pg_atlas.metrics.maintenance import REASON_ARTIFACT_UNREADABLE
        from pg_atlas.metrics.materialize_maintenance import _ArtifactRef, _commit_activity_for_repo

        monkeypatch.setattr(settings, "ARTIFACT_STORE_PATH", tmp_path_factory.mktemp("empty-artifact-store"))
        now = dt.datetime.now(dt.UTC)
        submitted_at = now - dt.timedelta(days=1)
        artifact = _ArtifactRef(artifact_path="git-logs/missing.gitlog", since_months=24, submitted_at=submitted_at)

        result, was_read = await _commit_activity_for_repo("pkg:github/test/missing", artifact, now=now)

        assert was_read is False
        assert result.state == STATE_INCOMPLETE
        assert result.reason == REASON_ARTIFACT_UNREADABLE
        assert result.commit_count is None
        assert result.artifact_as_of == submitted_at.isoformat()

    async def test_absent_artifact_is_unavailable(self) -> None:
        from pg_atlas.metrics.materialize_maintenance import _commit_activity_for_repo

        result, was_read = await _commit_activity_for_repo("pkg:github/test/none", None, now=dt.datetime.now(dt.UTC))

        assert was_read is False
        assert result.state == STATE_UNAVAILABLE
        assert result.reason is None


# ---------------------------------------------------------------------------
# Cadence fallback states and the parameter gate (no database needed)
# ---------------------------------------------------------------------------


def _unavailable_activity() -> Any:
    from pg_atlas.metrics.maintenance import CommitActivityResult

    return CommitActivityResult(state=STATE_UNAVAILABLE, commit_count=None, artifact_as_of=None, reason=None)


def _signals_with_fallback(fallback: Any, now: dt.datetime) -> MaintenanceSignals:
    return MaintenanceSignals(
        schema_version=1,
        collected_at=now.isoformat(),
        window_days=180,
        response_interval_days=7,
        release_fallback=fallback,
    )


@pytest.fixture
def pinned_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MAINTENANCE_WINDOW_DAYS", 180)
    monkeypatch.setattr(settings, "MAINTENANCE_RESPONSE_INTERVAL_DAYS", 7)
    monkeypatch.setattr(settings, "MAINTENANCE_INCLUDE_DRAFT_PRS", True)
    monkeypatch.setattr(settings, "MAINTENANCE_INCLUDE_PRERELEASES", True)
    monkeypatch.setattr(settings, "MAINTENANCE_CADENCE_LAST_N_EVENTS", 10)
    monkeypatch.setattr(settings, "MAINTENANCE_CADENCE_MIN_EVENTS", 3)
    monkeypatch.setattr(settings, "MAINTENANCE_SIGNALS_MAX_AGE_DAYS", 14)
    monkeypatch.setattr(settings, "MAINTENANCE_GITLOG_MAX_AGE_DAYS", 21)
    monkeypatch.setattr(settings, "MAINTENANCE_DECLARED_MAINTAINERS", "")


class TestCadenceFallbackStates:
    def _profile_signals(self, releases: Any, fallback: Any, now: dt.datetime) -> dict[str, Any]:
        from pg_atlas.metrics.materialize_maintenance import _build_repo_profile

        profile = _build_repo_profile(
            1,
            "pkg:github/test/cadence",
            releases,
            _signals_with_fallback(fallback, now),
            None,
            _unavailable_activity(),
            now=now,
        )

        return profile.signals["release_cadence"]

    def test_incomplete_fallback_is_context_not_ranked(self, pinned_settings: None) -> None:
        from pg_atlas.metrics.maintenance import ReleaseFallback

        now = dt.datetime.now(dt.UTC)
        dates = [(now.date() - dt.timedelta(days=30 * i)).isoformat() for i in range(3)]
        fallback = ReleaseFallback(
            state=STATE_INCOMPLETE,
            as_of=now.isoformat(),
            publication_dates=sorted(dates),
            complete=False,
            incomplete_reason="page-cap",
        )

        block = self._profile_signals(None, fallback, now)

        assert block["days_since_last_release"]["state"] == STATE_INCOMPLETE
        assert block["days_since_last_release"]["reason"] == "page-cap"
        assert block["days_since_last_release"]["value"] is not None
        assert block["median_gap_days"]["state"] == STATE_INCOMPLETE
        assert "percentile" not in block["days_since_last_release"]

    def test_failed_empty_fallback_keeps_its_reason(self, pinned_settings: None) -> None:
        from pg_atlas.metrics.maintenance import ReleaseFallback

        now = dt.datetime.now(dt.UTC)
        fallback = ReleaseFallback(
            state=STATE_INCOMPLETE, as_of=now.isoformat(), complete=False, incomplete_reason="rate-limited"
        )

        block = self._profile_signals(None, fallback, now)

        assert block["days_since_last_release"]["state"] == STATE_INCOMPLETE
        assert block["days_since_last_release"]["reason"] == "rate-limited"
        assert "value" not in block["days_since_last_release"]

    def test_stale_fallback_is_context(self, pinned_settings: None) -> None:
        from pg_atlas.metrics.maintenance import ReleaseFallback

        now = dt.datetime.now(dt.UTC)
        old = (now - dt.timedelta(days=60)).isoformat()
        fallback = ReleaseFallback(state=STATE_OK, as_of=old, publication_dates=["2026-01-01", "2026-02-01", "2026-03-01"])

        block = self._profile_signals(None, fallback, now)

        assert block["days_since_last_release"]["state"] == STATE_INCOMPLETE
        assert block["days_since_last_release"]["reason"] == REASON_STALE_COLLECTION

    def test_prerelease_rule_mismatch_is_context(self, pinned_settings: None) -> None:
        from pg_atlas.metrics.maintenance import ReleaseFallback

        now = dt.datetime.now(dt.UTC)
        fallback = ReleaseFallback(
            state=STATE_OK,
            as_of=now.isoformat(),
            publication_dates=["2026-01-01", "2026-02-01", "2026-03-01"],
            include_prereleases=False,
        )

        block = self._profile_signals(None, fallback, now)

        assert block["days_since_last_release"]["state"] == STATE_INCOMPLETE
        assert block["days_since_last_release"]["reason"] == "parameter-mismatch"

    def test_usable_depsdev_source_ignores_blocked_fallback(self, pinned_settings: None) -> None:
        from pg_atlas.metrics.maintenance import ReleaseFallback

        now = dt.datetime.now(dt.UTC)
        releases = [
            Release(
                purl="pkg:npm/a",
                version=f"1.{i}.0",
                release_date=f"{(now.date() - dt.timedelta(days=15 + 30 * i)).isoformat()}T10:00:00Z",
            )
            for i in range(6)
        ]
        fallback = ReleaseFallback(state=STATE_INCOMPLETE, as_of=now.isoformat(), complete=False, incomplete_reason="page-cap")

        block = self._profile_signals(releases, fallback, now)

        assert block["source"] == "depsdev"
        assert block["days_since_last_release"]["state"] == STATE_OK
        assert block["median_gap_days"]["state"] == STATE_OK

    def test_complete_empty_fallback_with_no_depsdev_dates_is_unavailable(self, pinned_settings: None) -> None:
        from pg_atlas.metrics.maintenance import ReleaseFallback

        now = dt.datetime.now(dt.UTC)
        fallback = ReleaseFallback(state=STATE_OK, as_of=now.isoformat())

        block = self._profile_signals(None, fallback, now)

        assert block["days_since_last_release"]["state"] == STATE_UNAVAILABLE
        assert block["median_gap_days"]["state"] == STATE_UNAVAILABLE


class TestResponsivenessParameterGate:
    def test_mismatched_window_is_context(self, pinned_settings: None) -> None:
        from pg_atlas.metrics.materialize_maintenance import _responsiveness_signal_dict

        now = dt.datetime.now(dt.UTC)
        signal = ResponsivenessSignal(
            state=STATE_OK,
            as_of=now.isoformat(),
            window_days=90,
            interval_days=7,
            response_fraction=0.9,
        )

        block, rankable = _responsiveness_signal_dict(
            signal, now=now, expected_include_drafts=None, expected_declared_maintainers=None
        )

        assert rankable is None
        assert block["response_fraction"]["state"] == STATE_INCOMPLETE
        assert block["response_fraction"]["reason"] == "parameter-mismatch"
        assert block["response_fraction"]["value"] == 0.9
        assert block["window_days"] == 90

    def test_mismatched_draft_rule_is_context(self, pinned_settings: None) -> None:
        from pg_atlas.metrics.materialize_maintenance import _responsiveness_signal_dict

        now = dt.datetime.now(dt.UTC)
        signal = ResponsivenessSignal(
            state=STATE_OK,
            as_of=now.isoformat(),
            window_days=180,
            interval_days=7,
            include_drafts=False,
            response_fraction=0.9,
        )

        block, rankable = _responsiveness_signal_dict(
            signal, now=now, expected_include_drafts=True, expected_declared_maintainers=None
        )

        assert rankable is None
        assert block["response_fraction"]["reason"] == "parameter-mismatch"

    def test_matching_parameters_rank(self, pinned_settings: None) -> None:
        from pg_atlas.metrics.materialize_maintenance import _responsiveness_signal_dict

        now = dt.datetime.now(dt.UTC)
        signal = ResponsivenessSignal(
            state=STATE_OK,
            as_of=now.isoformat(),
            window_days=180,
            interval_days=7,
            include_drafts=True,
            response_fraction=0.9,
        )

        block, rankable = _responsiveness_signal_dict(
            signal, now=now, expected_include_drafts=True, expected_declared_maintainers=None
        )

        assert rankable == 0.9
        assert block["response_fraction"]["state"] == STATE_OK

    def test_mismatched_declared_maintainers_is_context(self, pinned_settings: None) -> None:
        from pg_atlas.metrics.materialize_maintenance import _responsiveness_signal_dict

        now = dt.datetime.now(dt.UTC)
        signal = ResponsivenessSignal(
            state=STATE_OK,
            as_of=now.isoformat(),
            window_days=180,
            interval_days=7,
            declared_maintainers=["erin"],
            response_fraction=0.9,
        )

        block, rankable = _responsiveness_signal_dict(
            signal, now=now, expected_include_drafts=None, expected_declared_maintainers=None
        )

        assert rankable is None
        assert block["response_fraction"]["reason"] == "parameter-mismatch"
        assert block["declared_maintainers"] == ["erin"]

    def test_matching_declared_maintainers_rank(self, pinned_settings: None) -> None:
        from pg_atlas.metrics.materialize_maintenance import _responsiveness_signal_dict

        now = dt.datetime.now(dt.UTC)
        signal = ResponsivenessSignal(
            state=STATE_OK,
            as_of=now.isoformat(),
            window_days=180,
            interval_days=7,
            declared_maintainers=["erin", "rando"],
            response_fraction=0.9,
        )

        block, rankable = _responsiveness_signal_dict(
            signal, now=now, expected_include_drafts=None, expected_declared_maintainers=["erin", "rando"]
        )

        assert rankable == 0.9
        assert block["response_fraction"]["state"] == STATE_OK


class TestDeclaredMaintainersExpectedList:
    def _profile(self, canonical_id: str, signal: ResponsivenessSignal, now: dt.datetime) -> Any:
        from pg_atlas.metrics.maintenance import MaintenanceSignals
        from pg_atlas.metrics.materialize_maintenance import _build_repo_profile

        signals = MaintenanceSignals(
            schema_version=1,
            collected_at=now.isoformat(),
            window_days=180,
            response_interval_days=7,
            issue_responsiveness=signal,
        )

        return _build_repo_profile(1, canonical_id, None, signals, None, _unavailable_activity(), now=now)

    def test_declared_repo_ranks_under_matching_list(self, pinned_settings: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """The expected list is derived from the canonical id: prefix stripped, lowercased."""
        monkeypatch.setattr(settings, "MAINTENANCE_DECLARED_MAINTAINERS", "Owner/Repo=Rando|ERIN")
        now = dt.datetime.now(dt.UTC)
        signal = ResponsivenessSignal(
            state=STATE_OK,
            as_of=now.isoformat(),
            window_days=180,
            interval_days=7,
            declared_maintainers=["erin", "rando"],
            response_fraction=0.5,
        )

        profile = self._profile("pkg:github/Owner/Repo", signal, now)

        assert profile.signals["issue_responsiveness"]["response_fraction"]["state"] == STATE_OK
        assert profile.rankable["issue_responsiveness.response_fraction"] == 0.5

    def test_declared_payload_against_empty_setting_is_context(self, pinned_settings: None) -> None:
        now = dt.datetime.now(dt.UTC)
        signal = ResponsivenessSignal(
            state=STATE_OK,
            as_of=now.isoformat(),
            window_days=180,
            interval_days=7,
            declared_maintainers=["erin"],
            response_fraction=0.5,
        )

        profile = self._profile("pkg:github/owner/repo", signal, now)

        block = profile.signals["issue_responsiveness"]["response_fraction"]
        assert block["state"] == STATE_INCOMPLETE
        assert block["reason"] == "parameter-mismatch"
        assert "issue_responsiveness.response_fraction" not in profile.rankable


class TestArtifactTransportAndDamage:
    async def test_transport_timeout_is_contained_to_the_repo(
        self, monkeypatch: pytest.MonkeyPatch, pinned_settings: None
    ) -> None:
        """An IPFS gateway timeout marks one repo incomplete; the pass survives."""

        import httpx

        import pg_atlas.metrics.materialize_maintenance as mm
        from pg_atlas.metrics.materialize_maintenance import _ArtifactRef, _commit_activity_for_repo

        async def _timeout(path: str) -> bytes:
            raise httpx.ReadTimeout("gateway timeout")

        monkeypatch.setattr(mm, "read_artifact", _timeout)
        now = dt.datetime.now(dt.UTC)
        artifact = _ArtifactRef(artifact_path="git-logs/x.gitlog", since_months=24, submitted_at=now - dt.timedelta(days=1))

        result, was_read = await _commit_activity_for_repo("pkg:github/test/timeout", artifact, now=now)

        assert was_read is False
        assert result.state == STATE_INCOMPLETE
        assert result.reason == "artifact-unreadable"

    async def test_malformed_artifact_is_context_not_a_ranked_zero(
        self, monkeypatch: pytest.MonkeyPatch, pinned_settings: None
    ) -> None:
        import pg_atlas.metrics.materialize_maintenance as mm
        from pg_atlas.metrics.materialize_maintenance import _ArtifactRef, _commit_activity_for_repo

        async def _corrupt(path: str) -> bytes:
            return b"corrupted nonempty artifact\n"

        monkeypatch.setattr(mm, "read_artifact", _corrupt)
        now = dt.datetime.now(dt.UTC)
        artifact = _ArtifactRef(artifact_path="git-logs/x.gitlog", since_months=24, submitted_at=now - dt.timedelta(days=1))

        result, was_read = await _commit_activity_for_repo("pkg:github/test/corrupt", artifact, now=now)

        assert was_read is True
        assert result.state == STATE_INCOMPLETE
        assert result.reason == "artifact-malformed"
        assert result.commit_count == 0

    async def test_partially_malformed_artifact_reports_floor_count(
        self, monkeypatch: pytest.MonkeyPatch, pinned_settings: None
    ) -> None:
        import pg_atlas.metrics.materialize_maintenance as mm
        from pg_atlas.metrics.materialize_maintenance import _ArtifactRef, _commit_activity_for_repo

        now = dt.datetime.now(dt.UTC)
        eligible = f"Alice\x00alice@example.org\x00{(now - dt.timedelta(days=10)).isoformat()}\x00{'a' * 40}"
        bot = (
            f"dependabot[bot]\x001+dependabot[bot]@users.noreply.github.com"
            f"\x00{(now - dt.timedelta(days=5)).isoformat()}\x00{'b' * 40}"
        )
        out_of_window = f"Bob\x00bob@example.org\x00{(now - dt.timedelta(days=400)).isoformat()}\x00{'c' * 40}"

        async def _partial(path: str) -> bytes:
            return f"{eligible}\n{bot}\n{out_of_window}\ntruncated garbage\n".encode()

        monkeypatch.setattr(mm, "read_artifact", _partial)
        artifact = _ArtifactRef(artifact_path="git-logs/x.gitlog", since_months=24, submitted_at=now - dt.timedelta(days=1))

        result, was_read = await _commit_activity_for_repo("pkg:github/test/partial", artifact, now=now)

        # The context value is the ordinary filtered window count: the bot
        # and the 400-day-old commit are excluded, exactly as when ranking.
        assert was_read is True
        assert result.state == STATE_INCOMPLETE
        assert result.reason == "artifact-malformed"
        assert result.commit_count == 1


class TestConcurrentPersistence:
    async def test_stale_snapshot_cannot_overwrite_a_concurrent_complete_signal(
        self,
        rollback_db_session: AsyncSession,
    ) -> None:
        """The persist transaction merges against the locked current row."""

        from pg_atlas.metrics.maintenance import signals_from_metadata

        session = rollback_db_session
        suffix = uuid4().hex[:8]
        now = dt.datetime.now(dt.UTC)

        repo = Repo(
            canonical_id=f"pkg:github/test/race-{suffix}",
            display_name=f"race-{suffix}",
            visibility=Visibility.public,
            latest_version="1.0.0",
        )
        session.add(repo)
        await session.flush()

        # Both runs resolve while the row has no payload.
        resolved = await _resolve_repo(session, "test", f"race-{suffix}")

        # Run A persists a complete backlog signal first.
        complete = MaintenanceSignals(
            schema_version=1,
            collected_at=now.isoformat(),
            window_days=180,
            response_interval_days=7,
            issue_backlog=BacklogSignal(state=STATE_OK, as_of=now.isoformat(), open_count=4, median_open_age_days=12.0),
        )
        await _persist_collected(session, resolved, CollectedMaintenance(signals=complete, pushed_at=None))

        # Run B, still holding the empty snapshot, persists an incomplete one.
        incomplete = MaintenanceSignals(
            schema_version=1,
            collected_at=(now + dt.timedelta(minutes=1)).isoformat(),
            window_days=180,
            response_interval_days=7,
            issue_backlog=BacklogSignal(
                state=STATE_INCOMPLETE, as_of=(now + dt.timedelta(minutes=1)).isoformat(), incomplete_reason="rate-limited"
            ),
        )
        await _persist_collected(session, resolved, CollectedMaintenance(signals=incomplete, pushed_at=None))

        metadata = (await session.execute(select(Repo.repo_metadata).where(Repo.id == repo.id))).scalar_one()
        stored = signals_from_metadata(metadata)
        assert stored is not None
        assert stored.issue_backlog is not None
        assert stored.issue_backlog.state == STATE_OK
        assert stored.issue_backlog.open_count == 4

    async def test_merge_survives_sql_null_and_json_null_metadata(
        self,
        rollback_db_session: AsyncSession,
    ) -> None:
        """Both storage forms of an absent payload merge into a proper object."""

        from sqlalchemy import text

        from pg_atlas.metrics.maintenance import MAINTENANCE_SIGNALS_KEY

        session = rollback_db_session
        suffix = uuid4().hex[:8]
        now = dt.datetime.now(dt.UTC)

        repos: list[Repo] = []
        for marker in ("sqlnull", "jsonnull"):
            repo = Repo(
                canonical_id=f"pkg:github/test/{marker}-{suffix}",
                display_name=f"{marker}-{suffix}",
                visibility=Visibility.public,
                latest_version="1.0.0",
            )
            session.add(repo)
            repos.append(repo)
        await session.flush()

        await session.execute(text("UPDATE repos SET metadata = NULL WHERE id = :id"), {"id": repos[0].id})
        await session.execute(sqlalchemy_update_json_null(repos[1].id))

        signals = MaintenanceSignals(
            schema_version=1,
            collected_at=now.isoformat(),
            window_days=180,
            response_interval_days=7,
            issue_backlog=BacklogSignal(state=STATE_OK, as_of=now.isoformat(), open_count=1, median_open_age_days=2.0),
        )

        for marker, repo in zip(("sqlnull", "jsonnull"), repos, strict=True):
            resolved = await _resolve_repo(session, "test", f"{marker}-{suffix}")
            await _persist_collected(session, resolved, CollectedMaintenance(signals=signals, pushed_at=None))
            metadata = (await session.execute(select(Repo.repo_metadata).where(Repo.id == repo.id))).scalar_one()
            assert isinstance(metadata, dict), marker
            assert MAINTENANCE_SIGNALS_KEY in metadata, marker


def sqlalchemy_update_json_null(repo_id: int) -> Any:
    """An ORM ``None`` write, which this JSONB column stores as JSON null."""

    from sqlalchemy import update as _update

    return _update(Repo).where(Repo.id == repo_id).values(repo_metadata=None)


class TestBlockedFallbackAgainstUsableDepsdev:
    def test_out_counting_blocked_fallback_cannot_win_source_selection(self, pinned_settings: None) -> None:
        """Blocked fallback data never ranks, even when it holds more dates."""

        from pg_atlas.metrics.maintenance import ReleaseFallback
        from pg_atlas.metrics.materialize_maintenance import _build_repo_profile

        now = dt.datetime.now(dt.UTC)
        depsdev_newest = now.date() - dt.timedelta(days=20)
        releases = [
            Release(
                purl="pkg:npm/a",
                version=f"1.{i}.0",
                release_date=f"{(depsdev_newest - dt.timedelta(days=30 * i)).isoformat()}T10:00:00Z",
            )
            for i in range(3)
        ]
        fallback = ReleaseFallback(
            state=STATE_INCOMPLETE,
            as_of=now.isoformat(),
            publication_dates=sorted((now.date() - dt.timedelta(days=2 + 10 * i)).isoformat() for i in range(6)),
            complete=False,
            incomplete_reason="page-cap",
        )

        profile = _build_repo_profile(
            1,
            "pkg:github/test/outcount",
            releases,
            _signals_with_fallback(fallback, now),
            None,
            _unavailable_activity(),
            now=now,
        )
        block = profile.signals["release_cadence"]

        assert block["source"] == "depsdev"
        assert block["days_since_last_release"]["state"] == STATE_OK
        assert block["days_since_last_release"]["value"] == float((now.date() - depsdev_newest).days)
        assert block["median_gap_days"]["state"] == STATE_OK
        assert profile.rankable["release_cadence.days_since_last_release"] == float((now.date() - depsdev_newest).days)


class TestLateRunPersistence:
    async def test_older_run_finishing_late_keeps_the_newer_push_observation(
        self,
        rollback_db_session: AsyncSession,
    ) -> None:
        """pushed_at and run-level fields pair with the newest overview."""

        from pg_atlas.metrics.maintenance import signals_from_metadata

        session = rollback_db_session
        suffix = uuid4().hex[:8]
        t1 = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=10)
        t2 = dt.datetime.now(dt.UTC)
        push_old = t1 - dt.timedelta(days=3)
        push_new = t2 - dt.timedelta(days=1)

        repo = Repo(
            canonical_id=f"pkg:github/test/lateold-{suffix}",
            display_name=f"lateold-{suffix}",
            visibility=Visibility.public,
            latest_version="1.0.0",
        )
        session.add(repo)
        await session.flush()

        resolved = await _resolve_repo(session, "test", f"lateold-{suffix}")

        newer_run = MaintenanceSignals(
            schema_version=1,
            collected_at=t2.isoformat(),
            window_days=180,
            response_interval_days=7,
            issues_enabled=True,
            issue_backlog=BacklogSignal(state=STATE_OK, as_of=t2.isoformat(), open_count=2, median_open_age_days=4.0),
        )
        await _persist_collected(session, resolved, CollectedMaintenance(signals=newer_run, pushed_at=push_new))

        older_run = MaintenanceSignals(
            schema_version=1,
            collected_at=t1.isoformat(),
            window_days=180,
            response_interval_days=7,
            issues_enabled=False,
            issue_backlog=BacklogSignal(state=STATE_OK, as_of=t1.isoformat(), open_count=9, median_open_age_days=90.0),
            pr_backlog=BacklogSignal(state=STATE_OK, as_of=t1.isoformat(), open_count=1, median_open_age_days=5.0),
        )
        await _persist_collected(session, resolved, CollectedMaintenance(signals=older_run, pushed_at=push_old))

        row = (await session.execute(select(Repo.repo_metadata, Repo.pushed_at).where(Repo.id == repo.id))).one()
        metadata, stored_pushed_at = row
        stored = signals_from_metadata(metadata)

        assert stored_pushed_at == push_new
        assert stored is not None
        assert stored.collected_at == t2.isoformat()
        assert stored.issues_enabled is True
        assert stored.issue_backlog is not None
        assert stored.issue_backlog.open_count == 2
        # A slot the newer run did not observe still accepts the older data.
        assert stored.pr_backlog is not None
        assert stored.pr_backlog.open_count == 1
