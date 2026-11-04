import type { ActiveRun, DocFailure, OpsRun, Params } from '../api'
import { ageSeconds } from '../format'
import type { Drawer } from '../hashState'

/**
 * Run history (FR-17) and run-drawer helpers. Pure, so the paging, the
 * failure summary and the "what to do" choice are tested without a DOM.
 */

/** §3.5 LIGHTRAG_OPS_RUNS.kind, in the order the filter offers them. */
export const RUN_KINDS = ['process', 'email_batch', 'delete', 'clear', 'email_delete', 'scan', 'startup_resume', 'admin'] as const
/** §3.5 LIGHTRAG_OPS_RUNS.status. */
export const RUN_STATUSES = ['running', 'succeeded', 'failed', 'cancelled', 'paused', 'orphaned'] as const

export type RunsFilter = {
  /** Exact workspace ID (the route filters by equality). */
  workspace: string
  kind: (typeof RUN_KINDS)[number] | 'all'
  status: (typeof RUN_STATUSES)[number] | 'all'
}

export const DEFAULT_RUNS_FILTER: RunsFilter = { workspace: '', kind: 'all', status: 'all' }

/** The longest workspace ID the routes accept. */
export const MAX_WORKSPACE_LENGTH = 255

/** The /runs query for a filter and a keyset cursor; empty filters are left out so equal views share a cache entry. */
export function runsQuery(filter: RunsFilter, before: string | null = null): Params {
  const ws = filter.workspace.trim().slice(0, MAX_WORKSPACE_LENGTH)
  return {
    workspace: ws || undefined,
    kind: filter.kind === 'all' ? undefined : filter.kind,
    status: filter.status === 'all' ? undefined : filter.status,
    before: before || undefined
  }
}

/**
 * The first page (polled) followed by the older pages the user loaded. The
 * first page wins for a run that appears in both (its state is fresher), and
 * order is kept: newest first, then each older page as it came.
 */
export function mergeRunPages(firstPage: OpsRun[], older: OpsRun[][]): OpsRun[] {
  const seen = new Set<string>()
  const out: OpsRun[] = []
  for (const run of [firstPage, ...older].flat()) {
    if (seen.has(run.run_id)) continue
    seen.add(run.run_id)
    out.push(run)
  }
  return out
}

/** How long a run took (finished), or has been going (measured against when the data was confirmed). */
export function runDurationSeconds(run: Pick<OpsRun, 'duration_s' | 'started_at' | 'finished_at' | 'status'>, now: number | null): number | null {
  if (typeof run.duration_s === 'number') return run.duration_s
  if (!run.started_at) return null
  if (run.finished_at) {
    const end = Date.parse(run.finished_at)
    return Number.isFinite(end) ? ageSeconds(run.started_at, end) : null
  }
  return run.status === 'running' && now !== null ? ageSeconds(run.started_at, now) : null
}

/** Failures of a run counted by error kind, most frequent first (the drawer's one-line summary). */
export function failureKindCounts(failures: DocFailure[]): { key: string; count: number }[] {
  const counts = new Map<string, number>()
  for (const f of failures) {
    const key = f.error_kind || 'unknown'
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  return [...counts.entries()].map(([key, count]) => ({ key, count })).sort((a, b) => b.count - a.count || a.key.localeCompare(b.key))
}

/**
 * The "what to do" note for a run that needs a person, as an i18n key under
 * operations.runDrawer.guide.*, or null. Orphaned runs only exist under
 * Gunicorn; Release busy arrives with admin actions, so until then the safe
 * step for a dead owner is a container restart.
 */
export function runGuidance(state: string | null | undefined, multiWorker: boolean): string | null {
  switch (state) {
    case 'hung':
      return 'hung'
    case 'unresponsive':
      return 'unresponsive'
    case 'orphaned':
      return multiWorker ? 'orphaned' : null
    case 'paused':
      return 'paused'
    default:
      return null
  }
}

/** Where an active-run row drills to: its Run drawer once run tracking names the run, else the workspace. */
export const runDrawerOf = (run: Pick<ActiveRun, 'run_id' | 'workspace'>): Drawer =>
  run.run_id ? { kind: 'run', id: run.run_id } : { kind: 'workspace', id: run.workspace }
