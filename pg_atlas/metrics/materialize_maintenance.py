"""
Maintenance profile materialization from stored signals.

Reads the eligible population (repos associated with a registered project),
computes release cadence from stored releases, commit activity from the
latest git-log artifact, and the collected issue/PR signals from
``repo_metadata["maintenance_signals"]``, percentile-ranks every scalar over
its non-NULL pool, and writes the inspectable profile to
``repo_metadata["maintenance_profile"]``. There is no combined score: the
profile is per-signal raw values, percentile ranks, and coverage states.

Percentile direction is normalized: lower-is-better scalars (gaps, ages,
counts, days-since) are inverted before ranking, so a higher percentile
always means better upkeep. Percentile pools deliberately cover only the
eligible population — funded public goods compare against each other — and
every ranked scalar carries its pool size; the profile carries its as-of.

Collected values rank only while fresh (``MAINTENANCE_SIGNALS_MAX_AGE_DAYS``)
and git-log-based counts only from a fresh, window-covering artifact
(``MAINTENANCE_GITLOG_MAX_AGE_DAYS``); anything staler is kept as context
with an ``incomplete`` state, never silently mixed into ranking pools.

Delivery in the pilot phase is this CLI plus its private JSON export — the
public API is unchanged. The ``--gated`` flag makes a scheduled invocation
re-check ``MAINTENANCE_METRIC_ENABLED`` at execution time and no-op when
disabled; explicit runs without it are deliberate operator actions.

Usage::

    uv run python -m pg_atlas.metrics.materialize_maintenance
    uv run python -m pg_atlas.metrics.materialize_maintenance --export profiles.json --no-write
    uv run python -m pg_atlas.metrics.materialize_maintenance --gated --tee=maintenance.log

SPDX-FileCopyrightText: 2026 PG Atlas contributors
SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import literal, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pg_atlas.config import settings
from pg_atlas.db_models.base import SubmissionStatus
from pg_atlas.db_models.gitlog_artifact import GitLogArtifact
from pg_atlas.db_models.repo_vertex import Repo
from pg_atlas.db_models.session import get_session_factory
from pg_atlas.gitlog.parser import CommitRecord, parse_log_bytes
from pg_atlas.instruments.tee import run_with_tee
from pg_atlas.metrics.maintenance import (
    MAINTENANCE_PROFILE_KEY,
    MAINTENANCE_PROFILE_SCHEMA_VERSION,
    REASON_ARTIFACT_UNREADABLE,
    REASON_STALE_COLLECTION,
    STATE_INCOMPLETE,
    STATE_NOT_APPLICABLE,
    STATE_OK,
    STATE_UNAVAILABLE,
    BacklogSignal,
    CommitActivityResult,
    MaintenanceSignals,
    ResponsivenessSignal,
    compute_commit_activity,
    compute_release_cadence,
    distinct_release_dates,
    rank_scalar_values,
    repo_metadata_merge_expression,
    signals_from_metadata,
)
from pg_atlas.storage.artifacts import read_artifact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scalar registry — one row per ranked value in the profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ScalarSpec:
    """One rankable scalar: its profile location and ranking direction."""

    signal: str
    field: str
    higher_is_better: bool


_SCALAR_SPECS: tuple[_ScalarSpec, ...] = (
    _ScalarSpec("release_cadence", "median_gap_days", higher_is_better=False),
    _ScalarSpec("release_cadence", "days_since_last_release", higher_is_better=False),
    _ScalarSpec("issue_responsiveness", "response_fraction", higher_is_better=True),
    _ScalarSpec("issue_backlog", "open_count", higher_is_better=False),
    _ScalarSpec("issue_backlog", "median_open_age_days", higher_is_better=False),
    _ScalarSpec("pr_responsiveness", "response_fraction", higher_is_better=True),
    _ScalarSpec("pr_responsiveness", "open_count", higher_is_better=False),
    _ScalarSpec("pr_responsiveness", "median_open_age_days", higher_is_better=False),
    _ScalarSpec("activity_recency", "days_since_push", higher_is_better=False),
    _ScalarSpec("activity_recency", "commit_count_window", higher_is_better=True),
)


def _scalar_entry(
    state: str,
    *,
    value: float | int | None = None,
    reason: str | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Build one profile scalar entry; percentile and pool size come later."""

    entry: dict[str, Any] = {"state": state}
    if value is not None:
        entry["value"] = value
    if reason is not None:
        entry["reason"] = reason
    if as_of:
        entry["as_of"] = as_of

    return entry


def _is_stale(as_of: str, now: dt.datetime, max_age_days: int) -> bool:
    """Return whether a per-signal as-of timestamp is too old to rank."""

    if not as_of:
        return True

    try:
        observed = dt.datetime.fromisoformat(as_of)
    except ValueError:
        return True

    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=dt.UTC)

    return (now - observed) > dt.timedelta(days=max_age_days)


# ---------------------------------------------------------------------------
# Per-repo signal extraction
# ---------------------------------------------------------------------------


@dataclass
class _RepoProfile:
    """One repo's profile under construction: signal dicts plus rankability."""

    repo_id: int
    canonical_id: str
    signals: dict[str, dict[str, Any]]
    #: Raw values feeding the percentile pools, keyed by ``signal.field``.
    rankable: dict[str, float]


def _responsiveness_signal_dict(
    signal: ResponsivenessSignal | None,
    *,
    now: dt.datetime,
) -> tuple[dict[str, Any], float | None]:
    """Build the responsiveness profile block; return the rankable fraction."""

    if signal is None:
        return {"response_fraction": _scalar_entry(STATE_UNAVAILABLE)}, None

    if signal.state == STATE_NOT_APPLICABLE:
        return {"response_fraction": _scalar_entry(STATE_NOT_APPLICABLE, as_of=signal.as_of)}, None

    stale = _is_stale(signal.as_of, now, settings.MAINTENANCE_SIGNALS_MAX_AGE_DAYS)

    block: dict[str, Any] = {
        "cohort_size": signal.cohort_size,
        "eligible_size": signal.eligible_size,
        "responded_within_interval": signal.responded_within_interval,
        "median_response_days_context": signal.median_response_days_context,
        "unknown_attribution_count": signal.unknown_attribution_count,
        "maintainer_authored_count": signal.maintainer_authored_count,
    }

    if signal.state == STATE_INCOMPLETE:
        block["response_fraction"] = _scalar_entry(STATE_INCOMPLETE, reason=signal.incomplete_reason, as_of=signal.as_of)
        return block, None

    if stale:
        block["response_fraction"] = _scalar_entry(
            STATE_INCOMPLETE, value=signal.response_fraction, reason=REASON_STALE_COLLECTION, as_of=signal.as_of
        )
        return block, None

    if signal.response_fraction is None:
        block["response_fraction"] = _scalar_entry(STATE_UNAVAILABLE, as_of=signal.as_of)
        return block, None

    block["response_fraction"] = _scalar_entry(STATE_OK, value=signal.response_fraction, as_of=signal.as_of)

    return block, signal.response_fraction


def _backlog_scalar_dicts(
    signal: BacklogSignal | None,
    *,
    now: dt.datetime,
) -> tuple[dict[str, Any], float | None, float | None]:
    """
    Build backlog scalar entries; return rankable (count, median age).

    The open count stays exact under a capped listing, so it can rank even
    when the median cannot; a zero backlog is a real zero with no ages to
    take a median of.
    """

    if signal is None:
        return (
            {
                "open_count": _scalar_entry(STATE_UNAVAILABLE),
                "median_open_age_days": _scalar_entry(STATE_UNAVAILABLE),
            },
            None,
            None,
        )

    if signal.state == STATE_NOT_APPLICABLE:
        return (
            {
                "open_count": _scalar_entry(STATE_NOT_APPLICABLE, as_of=signal.as_of),
                "median_open_age_days": _scalar_entry(STATE_NOT_APPLICABLE, as_of=signal.as_of),
            },
            None,
            None,
        )

    if signal.open_count is None:
        reason = signal.incomplete_reason
        return (
            {
                "open_count": _scalar_entry(STATE_INCOMPLETE, reason=reason, as_of=signal.as_of),
                "median_open_age_days": _scalar_entry(STATE_INCOMPLETE, reason=reason, as_of=signal.as_of),
            },
            None,
            None,
        )

    if _is_stale(signal.as_of, now, settings.MAINTENANCE_SIGNALS_MAX_AGE_DAYS):
        return (
            {
                "open_count": _scalar_entry(
                    STATE_INCOMPLETE, value=signal.open_count, reason=REASON_STALE_COLLECTION, as_of=signal.as_of
                ),
                "median_open_age_days": _scalar_entry(
                    STATE_INCOMPLETE, value=signal.median_open_age_days, reason=REASON_STALE_COLLECTION, as_of=signal.as_of
                ),
            },
            None,
            None,
        )

    count_entry = _scalar_entry(STATE_OK, value=signal.open_count, as_of=signal.as_of)

    if signal.open_count == 0:
        median_entry = _scalar_entry(STATE_NOT_APPLICABLE, as_of=signal.as_of)
        median_value: float | None = None
    elif signal.median_age_exact and signal.median_open_age_days is not None:
        median_entry = _scalar_entry(STATE_OK, value=signal.median_open_age_days, as_of=signal.as_of)
        median_value = signal.median_open_age_days
    else:
        median_entry = _scalar_entry(STATE_INCOMPLETE, reason=signal.incomplete_reason, as_of=signal.as_of)
        median_value = None

    return {"open_count": count_entry, "median_open_age_days": median_entry}, float(signal.open_count), median_value


def _build_repo_profile(
    repo_id: int,
    canonical_id: str,
    releases: list[Any] | None,
    collected: MaintenanceSignals | None,
    pushed_at: dt.datetime | None,
    commit_activity: CommitActivityResult,
    *,
    now: dt.datetime,
) -> _RepoProfile:
    """Assemble one repo's profile blocks and its rankable raw values."""

    signals: dict[str, dict[str, Any]] = {}
    rankable: dict[str, float] = {}

    # --- release cadence (signal 1) ---
    fallback_dates: list[dt.date] = []
    fallback = collected.release_fallback if collected is not None else None
    if fallback is not None and not _is_stale(fallback.as_of, now, settings.MAINTENANCE_SIGNALS_MAX_AGE_DAYS):
        for raw_date in fallback.publication_dates:
            try:
                fallback_dates.append(dt.date.fromisoformat(raw_date))
            except ValueError:
                logger.warning(f"Invalid release fallback date for {canonical_id}: {raw_date!r}")

    cadence = compute_release_cadence(
        distinct_release_dates(releases),
        fallback_dates,
        now=now,
        last_n_events=settings.MAINTENANCE_CADENCE_LAST_N_EVENTS,
        min_events=settings.MAINTENANCE_CADENCE_MIN_EVENTS,
    )

    cadence_block: dict[str, Any] = {"source": cadence.source, "shipping_events": cadence.shipping_events}

    if cadence.state == STATE_OK and cadence.days_since_last_release is not None:
        cadence_block["days_since_last_release"] = _scalar_entry(STATE_OK, value=cadence.days_since_last_release)
        rankable["release_cadence.days_since_last_release"] = cadence.days_since_last_release
    else:
        cadence_block["days_since_last_release"] = _scalar_entry(STATE_UNAVAILABLE)

    if cadence.state == STATE_OK and cadence.median_gap_days is not None:
        cadence_block["median_gap_days"] = _scalar_entry(STATE_OK, value=cadence.median_gap_days)
        rankable["release_cadence.median_gap_days"] = cadence.median_gap_days
    else:
        cadence_block["median_gap_days"] = _scalar_entry(STATE_UNAVAILABLE)

    signals["release_cadence"] = cadence_block

    # --- issue responsiveness (signal 2) ---
    issue_resp_block, issue_fraction = _responsiveness_signal_dict(
        collected.issue_responsiveness if collected is not None else None, now=now
    )
    signals["issue_responsiveness"] = issue_resp_block
    if issue_fraction is not None:
        rankable["issue_responsiveness.response_fraction"] = issue_fraction

    # --- issue backlog (signal 3) ---
    issue_backlog_block, issue_count, issue_median = _backlog_scalar_dicts(
        collected.issue_backlog if collected is not None else None, now=now
    )
    signals["issue_backlog"] = issue_backlog_block
    if issue_count is not None:
        rankable["issue_backlog.open_count"] = issue_count
    if issue_median is not None:
        rankable["issue_backlog.median_open_age_days"] = issue_median

    # --- PR responsiveness plus its backlog snapshot (signal 4) ---
    pr_resp_block, pr_fraction = _responsiveness_signal_dict(
        collected.pr_responsiveness if collected is not None else None, now=now
    )
    pr_backlog_block, pr_count, pr_median = _backlog_scalar_dicts(
        collected.pr_backlog if collected is not None else None, now=now
    )
    pr_resp_block.update(pr_backlog_block)
    signals["pr_responsiveness"] = pr_resp_block
    if pr_fraction is not None:
        rankable["pr_responsiveness.response_fraction"] = pr_fraction
    if pr_count is not None:
        rankable["pr_responsiveness.open_count"] = pr_count
    if pr_median is not None:
        rankable["pr_responsiveness.median_open_age_days"] = pr_median

    # --- activity recency (signal 5) ---
    activity_block: dict[str, Any] = {}

    collected_at = collected.collected_at if collected is not None else ""
    if pushed_at is None:
        activity_block["days_since_push"] = _scalar_entry(STATE_UNAVAILABLE)
    else:
        days_since_push = round((now - pushed_at).total_seconds() / 86400.0, 2)
        if _is_stale(collected_at, now, settings.MAINTENANCE_SIGNALS_MAX_AGE_DAYS):
            activity_block["days_since_push"] = _scalar_entry(
                STATE_INCOMPLETE, value=days_since_push, reason=REASON_STALE_COLLECTION, as_of=collected_at
            )
        else:
            activity_block["days_since_push"] = _scalar_entry(STATE_OK, value=days_since_push, as_of=collected_at)
            rankable["activity_recency.days_since_push"] = days_since_push

    if commit_activity.state == STATE_OK and commit_activity.commit_count is not None:
        activity_block["commit_count_window"] = _scalar_entry(
            STATE_OK, value=commit_activity.commit_count, as_of=commit_activity.artifact_as_of
        )
        rankable["activity_recency.commit_count_window"] = float(commit_activity.commit_count)
    else:
        activity_block["commit_count_window"] = _scalar_entry(
            commit_activity.state,
            value=commit_activity.commit_count,
            reason=commit_activity.reason,
            as_of=commit_activity.artifact_as_of,
        )

    signals["activity_recency"] = activity_block

    return _RepoProfile(repo_id=repo_id, canonical_id=canonical_id, signals=signals, rankable=rankable)


# ---------------------------------------------------------------------------
# Artifact-backed commit activity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ArtifactRef:
    """Latest processed git-log artifact for one repo."""

    artifact_path: str
    since_months: int
    submitted_at: dt.datetime


async def _load_latest_artifacts(session: AsyncSession, repo_ids: list[int]) -> dict[int, _ArtifactRef]:
    """Load each repo's newest processed artifact reference."""

    if not repo_ids:
        return {}

    stmt = (
        select(
            GitLogArtifact.repo_id,
            GitLogArtifact.artifact_path,
            GitLogArtifact.since_months,
            GitLogArtifact.submitted_at,
        )
        .where(GitLogArtifact.repo_id.in_(repo_ids))
        .where(GitLogArtifact.status == SubmissionStatus.processed)
        .where(GitLogArtifact.artifact_path.is_not(None))
        .order_by(GitLogArtifact.repo_id, GitLogArtifact.submitted_at.desc(), GitLogArtifact.id.desc())
        .distinct(GitLogArtifact.repo_id)
    )
    rows = (await session.execute(stmt)).all()

    return {
        int(repo_id): _ArtifactRef(artifact_path=str(path), since_months=int(months), submitted_at=submitted)
        for repo_id, path, months, submitted in rows
    }


async def _commit_activity_for_repo(
    canonical_id: str,
    artifact: _ArtifactRef | None,
    *,
    now: dt.datetime,
) -> tuple[CommitActivityResult, bool]:
    """
    Read and parse one artifact; return the activity and whether it was read.

    A repo without any processed artifact is ``unavailable``; an artifact
    that exists but cannot be read or parsed is failed collection —
    ``incomplete`` with its own reason, keeping the artifact's observation
    time — the four states stay distinct.
    """

    if artifact is None:
        return (
            compute_commit_activity(
                None,
                artifact_submitted_at=None,
                artifact_since_months=None,
                now=now,
                window_days=settings.MAINTENANCE_WINDOW_DAYS,
                max_artifact_age_days=settings.MAINTENANCE_GITLOG_MAX_AGE_DAYS,
            ),
            False,
        )

    commits: list[CommitRecord]
    try:
        raw = await read_artifact(artifact.artifact_path)
        commits = parse_log_bytes(raw)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(f"Git-log artifact unreadable for {canonical_id}: {type(exc).__name__}: {exc}")

        return (
            CommitActivityResult(
                state=STATE_INCOMPLETE,
                commit_count=None,
                artifact_as_of=artifact.submitted_at.isoformat(),
                reason=REASON_ARTIFACT_UNREADABLE,
            ),
            False,
        )

    return (
        compute_commit_activity(
            commits,
            artifact_submitted_at=artifact.submitted_at,
            artifact_since_months=artifact.since_months,
            now=now,
            window_days=settings.MAINTENANCE_WINDOW_DAYS,
            max_artifact_age_days=settings.MAINTENANCE_GITLOG_MAX_AGE_DAYS,
        ),
        True,
    )


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaintenanceMaterializationStats:
    """Summarize one maintenance profile materialization pass."""

    gate_skipped: bool
    repos_eligible: int
    profiles_written: int
    stale_profiles_cleared: int
    artifacts_read: int
    duration_seconds: float
    pool_sizes: dict[str, int]


async def materialize_maintenance_profiles(
    session: AsyncSession,
    *,
    enforce_gate: bool = False,
    write: bool = True,
    now: dt.datetime | None = None,
) -> tuple[MaintenanceMaterializationStats, dict[str, dict[str, Any]]]:
    """
    Recompute maintenance profiles over the eligible population.

    Returns the stats plus the assembled profiles keyed by repo canonical id
    (for the CLI export). With ``enforce_gate`` the pass no-ops while
    ``MAINTENANCE_METRIC_ENABLED`` is false — the execution-time half of the
    fail-closed rollout gate. All writes are bulk DML on ``repo_metadata``;
    profiles on repos that left the eligible population are removed.
    """

    started_at = time.perf_counter()
    observed_now = now if now is not None else dt.datetime.now(dt.UTC)

    if enforce_gate and not settings.MAINTENANCE_METRIC_ENABLED:
        logger.info("materialize_maintenance_profiles: MAINTENANCE_METRIC_ENABLED is false, skipping gated run")
        stats = MaintenanceMaterializationStats(
            gate_skipped=True,
            repos_eligible=0,
            profiles_written=0,
            stale_profiles_cleared=0,
            artifacts_read=0,
            duration_seconds=time.perf_counter() - started_at,
            pool_sizes={},
        )

        return stats, {}

    rows = (
        await session.execute(
            select(Repo.id, Repo.canonical_id, Repo.releases, Repo.repo_metadata, Repo.pushed_at)
            .where(Repo.project_id.is_not(None))
            .order_by(Repo.id)
        )
    ).all()

    repo_ids = [int(row[0]) for row in rows]
    artifacts = await _load_latest_artifacts(session, repo_ids)

    profiles: list[_RepoProfile] = []
    artifacts_read = 0
    for repo_id, canonical_id, releases, repo_metadata, pushed_at in rows:
        collected = signals_from_metadata(repo_metadata, repo_canonical_id=canonical_id)
        commit_activity, artifact_was_read = await _commit_activity_for_repo(
            canonical_id, artifacts.get(int(repo_id)), now=observed_now
        )
        if artifact_was_read:
            artifacts_read += 1

        profiles.append(
            _build_repo_profile(
                int(repo_id),
                str(canonical_id),
                releases,
                collected,
                pushed_at,
                commit_activity,
                now=observed_now,
            )
        )

    # --- rank every scalar over its pool and stamp percentiles in place ---
    pool_sizes: dict[str, int] = {}
    for spec in _SCALAR_SPECS:
        key = f"{spec.signal}.{spec.field}"
        values = [profile.rankable.get(key) for profile in profiles]
        percentiles, pool_size = rank_scalar_values(values, higher_is_better=spec.higher_is_better)
        pool_sizes[key] = pool_size

        for profile, percentile in zip(profiles, percentiles, strict=True):
            entry = profile.signals[spec.signal][spec.field]
            if percentile is not None:
                entry["percentile"] = percentile
                entry["pool_size"] = pool_size

    # --- assemble full profile payloads ---
    payloads: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        payloads[profile.canonical_id] = {
            "schema_version": MAINTENANCE_PROFILE_SCHEMA_VERSION,
            "as_of": observed_now.isoformat(),
            "window_days": settings.MAINTENANCE_WINDOW_DAYS,
            "response_interval_days": settings.MAINTENANCE_RESPONSE_INTERVAL_DAYS,
            "eligible_population": len(profiles),
            "signals": profile.signals,
        }

    profiles_written = 0
    stale_profiles_cleared = 0
    if write:
        for profile in profiles:
            payload = {MAINTENANCE_PROFILE_KEY: payloads[profile.canonical_id]}
            await session.execute(
                update(Repo)
                .where(Repo.id == profile.repo_id)
                .values(repo_metadata=repo_metadata_merge_expression(payload))
                .execution_options(synchronize_session=False)
            )
            profiles_written += 1

        # --- remove profiles from repos that left the eligible population ---
        stale_result = await session.execute(
            update(Repo)
            .where(Repo.project_id.is_(None))
            .where(Repo.repo_metadata.has_key(MAINTENANCE_PROFILE_KEY))
            .values(repo_metadata=Repo.repo_metadata.op("-")(literal(MAINTENANCE_PROFILE_KEY)))
            .execution_options(synchronize_session=False)
        )
        stale_rowcount = getattr(stale_result, "rowcount", 0)
        stale_profiles_cleared = stale_rowcount if isinstance(stale_rowcount, int) and stale_rowcount > 0 else 0

    duration_seconds = time.perf_counter() - started_at
    stats = MaintenanceMaterializationStats(
        gate_skipped=False,
        repos_eligible=len(profiles),
        profiles_written=profiles_written,
        stale_profiles_cleared=stale_profiles_cleared,
        artifacts_read=artifacts_read,
        duration_seconds=duration_seconds,
        pool_sizes=pool_sizes,
    )

    pools_text = " ".join(f"{key}={size}" for key, size in sorted(pool_sizes.items()))
    logger.info(
        "materialize_maintenance_profiles: "
        f"repos_eligible={stats.repos_eligible} "
        f"profiles_written={stats.profiles_written} "
        f"stale_profiles_cleared={stats.stale_profiles_cleared} "
        f"artifacts_read={stats.artifacts_read} "
        f"duration_seconds={stats.duration_seconds:.3f} "
        f"pools: {pools_text}"
    )

    return stats, payloads


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for maintenance profile materialization."""

    parser = argparse.ArgumentParser(description="Materialize per-repo maintenance profiles.")
    parser.add_argument(
        "--gated",
        action="store_true",
        help="Re-check MAINTENANCE_METRIC_ENABLED at execution time and no-op when disabled (for scheduled runs).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Compute profiles without writing them (use with --export for inspection).",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Write the computed profiles as JSON to this path (private export).",
    )
    parser.add_argument(
        "--tee",
        type=Path,
        default=None,
        help="Optional path to mirror stdout/stderr logs while preserving console output.",
    )

    return parser


async def _async_main(args: argparse.Namespace) -> None:
    """Run one materialization pass, commit, and export when requested."""

    factory = get_session_factory()
    async with factory() as session:
        stats, payloads = await materialize_maintenance_profiles(
            session,
            enforce_gate=args.gated,
            write=not args.no_write,
        )
        await session.commit()

    if args.export is not None and not stats.gate_skipped:
        export_payload = {
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            "eligible_population": stats.repos_eligible,
            "pool_sizes": stats.pool_sizes,
            "profiles": payloads,
        }
        args.export.write_text(json.dumps(export_payload, indent=2, sort_keys=True) + "\n")
        logger.info(f"maintenance profiles exported: path={args.export} repos={len(payloads)}")

    logger.info(
        "maintenance materialization finished: "
        f"gate_skipped={stats.gate_skipped} "
        f"repos_eligible={stats.repos_eligible} "
        f"profiles_written={stats.profiles_written} "
        f"stale_profiles_cleared={stats.stale_profiles_cleared} "
        f"duration_seconds={stats.duration_seconds:.3f}"
    )


def main() -> None:
    """Parse CLI arguments and run the materialization pass."""

    args = _build_parser().parse_args()

    def _run() -> None:
        asyncio.run(_async_main(args))

    run_with_tee(args.tee, _run)


if __name__ == "__main__":
    main()
