import type { AttentionItem, OpsOverview, OverviewKpis } from './api'
import type { HashTarget } from './hashState'
import { RingCandidate, isWorsening } from './motion'
import { OpsState, rankOf, toState, workersState } from './severity'

/**
 * Pure read-model helpers for the Overview (spec §2.3, FR-1–FR-4): tile
 * order and drill targets, a KPI's severity, the header chip's wording and
 * which rings a new poll may ask for.
 */

/** The overview (header state chip and Overview tab) refreshes every 15 s (§3.10). */
export const OVERVIEW_REFRESH_SECONDS = 15
/** Timeline markers refresh every 60 s over the last 6 hours (§2.3). */
export const TIMELINE_REFRESH_SECONDS = 60
export const TIMELINE_HOURS = 6

export const KPI_IDS = ['active', 'backlog', 'failed_24h', 'stuck', 'email_queue', 'llm_15m', 'deps', 'workers'] as const
export type KpiId = (typeof KPI_IDS)[number]

/** Where each tile drills down to (§2.3 "Tile targets"). */
export const KPI_TARGETS: Record<KpiId, HashTarget> = {
  active: { tab: 'processing', sub: 'active' },
  backlog: { tab: 'processing', sub: 'backlog' },
  failed_24h: { tab: 'processing', sub: 'failures' },
  stuck: { tab: 'processing', sub: 'stuck' },
  email_queue: { tab: 'processing', sub: 'email' },
  llm_15m: { tab: 'performance', sub: 'llm' },
  deps: { tab: 'health', sub: 'deps' },
  workers: { tab: 'health', sub: 'workers' }
}

/** Where a rule's items drill to when the server sends no explicit target. */
export const RULE_TARGETS: Record<string, HashTarget> = {
  stale_docs: { tab: 'processing', sub: 'stuck' },
  run_hung: { tab: 'processing', sub: 'stuck' },
  run_orphaned: { tab: 'processing', sub: 'stuck' },
  busy_without_run: { tab: 'processing', sub: 'stuck' },
  reservation_without_owner: { tab: 'processing', sub: 'stuck' },
  oldest_pending: { tab: 'processing', sub: 'backlog' },
  failure_spike: { tab: 'processing', sub: 'failures' },
  email_queue_stalled: { tab: 'processing', sub: 'email' },
  email_queue_bad: { tab: 'processing', sub: 'email' },
  provider_paused: { tab: 'processing', sub: 'pauses' },
  llm_error_rate_15m: { tab: 'performance', sub: 'llm' },
  llm_p95_15m: { tab: 'performance', sub: 'llm' },
  embed_rerank_error_rate_15m: { tab: 'performance', sub: 'deps' },
  http_5xx_rate_5m: { tab: 'performance', sub: 'http' },
  workers_below_expected: { tab: 'health', sub: 'workers' },
  worker_heartbeat: { tab: 'health', sub: 'workers' },
  worker_unresponsive: { tab: 'health', sub: 'workers' },
  dependency_down: { tab: 'health', sub: 'deps' },
  pg_probe: { tab: 'health', sub: 'deps', dep: 'postgres' },
  smtp_probe: { tab: 'health', sub: 'deps', dep: 'smtp' },
  tls_expiry: { tab: 'health', sub: 'deps' },
  recorder_lost_rows: { tab: 'health', sub: 'deps' },
  manager_rtt: { tab: 'health', sub: 'workers' },
  pg_pool_wait: { tab: 'health', sub: 'workers' },
  signin_failures_15m: { tab: 'health', sub: 'signin' },
  disk_free: { tab: 'health', sub: 'host' }
}

/** An attention item's drill target: the server's, else its workspace, dependency or worker on the rule's view. */
export function attentionTarget(item: AttentionItem): HashTarget {
  if (item.target) return item.target
  const base = RULE_TARGETS[item.rule] ?? { tab: 'health', sub: 'deps' }
  if (item.workspace) return { ...base, workspace: item.workspace }
  if (item.dep) return { ...base, dep: item.dep }
  if (item.worker_id) return { ...base, worker_id: item.worker_id }
  return base
}

const num = (n: unknown): number | null => (typeof n === 'number' && Number.isFinite(n) ? n : null)

/** §1.7 defaults, used only when the server sends no state for a tile. */
const LLM_MIN_CALLS = 20
const LLM_WARN_RATE = 0.05
const LLM_SERIOUS_RATE = 0.2
const OLDEST_WARN_S = 2 * 3600
const OLDEST_SERIOUS_S = 6 * 3600

/**
 * A tile's severity. The server's state wins (it evaluates the rules under
 * lease); otherwise the §1.7 default thresholds apply. A tile with no data is
 * Unknown, never OK.
 */
export function kpiState(id: KpiId, kpis: OverviewKpis | null | undefined): OpsState {
  const k = kpis?.[id] as (Record<string, unknown> & { state?: string | null }) | undefined
  if (!k) return 'unknown'
  if (k.state) return toState(k.state)
  switch (id) {
    case 'active':
      return num(k.count) === null ? 'unknown' : 'ok'
    case 'backlog': {
      if (num(k.inflight) === null) return 'unknown'
      const oldest = num(k.oldest_s)
      if (oldest !== null && oldest > OLDEST_SERIOUS_S) return 'serious'
      if (oldest !== null && oldest > OLDEST_WARN_S) return 'warning'
      return 'ok'
    }
    case 'failed_24h':
      return num(k.real) === null ? 'unknown' : 'ok'
    case 'stuck': {
      const parts = [num(k.runs), num(k.docs), num(k.reservations)]
      if (parts.every((p) => p === null)) return 'unknown'
      return parts.some((p) => (p ?? 0) > 0) ? 'warning' : 'ok'
    }
    case 'email_queue':
      if (num(k.records) === null) return 'unknown'
      return (num(k.bad) ?? 0) > 0 ? 'warning' : 'ok'
    case 'llm_15m': {
      const rate = num(k.error_rate)
      const calls = num(k.calls)
      if (rate === null) return 'unknown'
      if (calls !== null && calls < LLM_MIN_CALLS) return 'ok'
      if (rate > LLM_SERIOUS_RATE) return 'serious'
      if (rate > LLM_WARN_RATE) return 'warning'
      return 'ok'
    }
    case 'deps': {
      const ok = num(k.ok)
      const total = num(k.total)
      if (ok === null || total === null || total <= 0) return 'unknown'
      return ok >= total ? 'ok' : 'warning'
    }
    case 'workers':
      return workersState(num(k.expected), num(k.live))
  }
}

/** Sum of the stuck parts; null when none is known yet. */
export function stuckTotal(kpis: OverviewKpis | null | undefined): number | null {
  const s = kpis?.stuck
  if (!s) return null
  const parts = [num(s?.runs), num(s?.docs), num(s?.reservations)]
  if (parts.every((p) => p === null)) return null
  return parts.reduce<number>((a, p) => a + (p ?? 0), 0)
}

/** Header wording: Healthy / Degraded / Down / Unknown / Maintenance (US-1). */
export type OverallWord = 'healthy' | 'degraded' | 'down' | 'unknown' | 'maintenance'

export function overallWord(raw: string | null | undefined, maintenance?: unknown): OverallWord {
  if (maintenance) return 'maintenance'
  const key = (raw ?? '').toLowerCase()
  if (key === 'down') return 'down'
  if (key === 'maintenance') return 'maintenance'
  if (key === 'ok' || key === 'healthy' || key === 'up') return 'healthy'
  if (!key || key === 'unknown' || key === 'stale') return 'unknown'
  return rankOf(key) > 0 || key === 'degraded' ? 'degraded' : 'unknown'
}

/** The severity the chip shows: the worst open severity, or the state itself. */
export function overallState(o: Pick<OpsOverview, 'state' | 'worst' | 'maintenance'> | null | undefined): OpsState {
  if (!o) return 'unknown'
  if (o.maintenance) return 'maintenance'
  if ((o.state ?? '').toLowerCase() === 'down') return 'critical'
  return toState(o.worst ?? o.state)
}

export type SeverityCounts = { critical: number; serious: number; warning: number }

export function attentionCounts(items: AttentionItem[] | null | undefined): SeverityCounts {
  const c: SeverityCounts = { critical: 0, serious: 0, warning: 0 }
  for (const a of items ?? []) {
    const s = toState(a.severity)
    if (s === 'critical' || s === 'serious' || s === 'warning') c[s] += 1
  }
  return c
}

/** Worst first, then the oldest; the server order breaks ties. */
export function sortAttention(items: AttentionItem[] | null | undefined): AttentionItem[] {
  return [...(items ?? [])]
    .map((a, i) => ({ a, i }))
    .sort((x, y) => rankOf(y.a.severity) - rankOf(x.a.severity) || x.i - y.i)
    .map(({ a }) => a)
}

/**
 * The rings a new (non-baseline) overview may ask for: the state chip when
 * the overall severity rises, and every critical attention item that was not
 * in the previous poll. The attention slot grants at most one.
 */
export function ringCandidates(prev: OpsOverview | null | undefined, next: OpsOverview | null | undefined): RingCandidate[] {
  if (!prev || !next) return []
  const out: RingCandidate[] = []
  const before = overallState(prev)
  const after = overallState(next)
  if (isWorsening(before, after)) out.push({ key: 'chip', rank: rankOf(after), source: 'chip' })
  const seen = new Set((prev.attention ?? []).map((a) => a.id))
  for (const a of sortAttention(next.attention)) {
    if (!seen.has(a.id) && toState(a.severity) === 'critical') out.push({ key: `att:${a.id}`, rank: rankOf(a.severity), source: 'attention' })
  }
  return out
}

/** A marker's position along the timeline window, 0..100 (%), or null when outside it. */
export function markerOffset(ts: string | number, from: number, to: number): number | null {
  const at = typeof ts === 'number' ? ts : Date.parse(ts)
  if (!Number.isFinite(at) || to <= from || at < from || at > to) return null
  return ((at - from) / (to - from)) * 100
}
