/**
 * Performance read models (spec §2.7, §3.7 rows tagged PR-6, FR-40 to FR-46).
 *
 * Server answers → the shapes the Performance tab, the Sign-in card and the
 * topology edge flow read. Every normaliser accepts the spellings a route may
 * use (`p95` or `p95_ms`, `t` as ISO text, epoch seconds or epoch ms, a count
 * per bucket or a per-second rate), keeps going on a partial payload and
 * never throws, so a server one release ahead or behind still renders.
 * The shaping helpers (pivots, top lists, rates) are pure and covered by
 * performance.test.ts.
 */

import { DEFAULT_OPS_HASH, OpsHash } from '../hashState'
import { LogWindow, encodeLogFilter } from '../logs'

type Loose = Record<string, any>

const obj = (v: unknown): Loose => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Loose) : {})
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const str = (v: unknown): string | null => (typeof v === 'string' && v ? v : null)
const arr = (v: unknown): Loose[] => (Array.isArray(v) ? v.filter((x) => x && typeof x === 'object' && !Array.isArray(x)) : [])
const first = (...values: unknown[]): number | null => {
  for (const v of values) {
    const n = num(v)
    if (n !== null) return n
  }
  return null
}

// ---------- windows ----------

export const PERF_WINDOWS = ['15m', '1h', '24h', '7d'] as const
export type PerfWindow = (typeof PERF_WINDOWS)[number]
export const DEFAULT_PERF_WINDOW: PerfWindow = '1h'

export const PERF_WINDOW_SECONDS: Record<PerfWindow, number> = {
  '15m': 15 * 60,
  '1h': 3600,
  '24h': 24 * 3600,
  '7d': 7 * 24 * 3600
}

/** Refresh every 60 s for 15 m and 1 h; manual for 24 h and 7 d (§2.7). */
export const perfRefreshSeconds = (w: PerfWindow): number | undefined => (w === '15m' || w === '1h' ? 60 : undefined)

/**
 * How long a window's answer stays fresh (for the stale chip): one refresh
 * period for the polled windows; the server's cache period for the manual
 * ones, so a 24 h chart does not read "stale" a minute after it loaded.
 */
export const perfFreshnessSeconds = (w: PerfWindow): number => (w === '7d' ? 3 * 3600 : w === '24h' ? 30 * 60 : 60)

/** Percentiles need raw calls; the 7-day window reads hourly rollups (average and max only, FR-40). */
export const isRollupWindow = (w: PerfWindow) => w === '7d'

/** The bucket a server most likely used for a window when it does not say. */
const DEFAULT_BUCKET_S: Record<PerfWindow, number> = { '15m': 60, '1h': 60, '24h': 300, '7d': 3600 }

export const isPerfWindow = (v: unknown): v is PerfWindow => typeof v === 'string' && (PERF_WINDOWS as readonly string[]).includes(v)

/** `gpu` is offered only while the server reads a GPU exporter (`features.gpu`, D5). */
export const PERF_SECTIONS = ['llm', 'deps', 'http', 'throughput', 'concurrency', 'top', 'gpu'] as const
export type PerfSection = (typeof PERF_SECTIONS)[number]

// ---------- time ----------

/** Epoch ms from ISO text, epoch seconds or epoch ms; null when unreadable. */
export function toMs(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v < 1e11 ? v * 1000 : v
  if (typeof v === 'string' && v) {
    const t = Date.parse(v)
    return Number.isFinite(t) ? t : null
  }
  return null
}

/** The bucket width in seconds: what the server says, else the smallest gap between points, else the window's default. */
export function bucketSeconds(raw: Loose, times: number[], window?: PerfWindow): number {
  const said = first(raw.bucket_s, raw.step_s, raw.bucket_seconds, raw.step)
  if (said !== null && said > 0) return said
  const sorted = [...new Set(times)].sort((a, b) => a - b)
  let gap = Infinity
  for (let i = 1; i < sorted.length; i += 1) gap = Math.min(gap, sorted[i] - sorted[i - 1])
  if (Number.isFinite(gap) && gap > 0) return Math.round(gap / 1000)
  return window ? DEFAULT_BUCKET_S[window] : 60
}

// ---------- shared point shapes ----------

/** One latency bucket: counts plus percentiles (≤24 h) or average and max (7 d rollups), all in ms. */
export type LatencyPoint = {
  t: number
  calls: number | null
  errors: number | null
  p50: number | null
  p95: number | null
  p99: number | null
  avg: number | null
  max: number | null
}

const latencyOf = (p: Loose) => ({
  p50: first(p.p50_ms, p.p50),
  p95: first(p.p95_ms, p.p95),
  p99: first(p.p99_ms, p.p99),
  avg: first(p.avg_ms, p.avg, p.mean_ms, p.mean),
  max: first(p.max_ms, p.max)
})

/** Calls in a bucket: a count when sent, else a per-second or per-minute rate times the bucket. */
const callsOf = (p: Loose, bucketS: number | null): number | null => {
  const n = first(p.calls, p.count, p.requests, p.n)
  if (n !== null) return n
  const rps = num(p.rps)
  if (rps !== null && bucketS) return rps * bucketS
  const perMin = first(p.per_min, p.rate_per_min, p.calls_per_min)
  if (perMin !== null && bucketS) return (perMin * bucketS) / 60
  return null
}

function latencyPoints(raw: unknown, bucketS: number | null): LatencyPoint[] {
  return arr(raw)
    .map((p) => {
      const t = toMs(p.t ?? p.ts ?? p.time ?? p.bucket)
      if (t === null) return null
      return { t, calls: callsOf(p, bucketS), errors: first(p.errors, p.error_count, p.err, p.errors_5xx, p.err5xx_count), ...latencyOf(p) }
    })
    .filter((p): p is LatencyPoint => p !== null)
    .sort((a, b) => a.t - b.t)
}

const timesOf = (raw: unknown) => arr(raw).map((p) => toMs(p.t ?? p.ts ?? p.time ?? p.bucket)).filter((t): t is number => t !== null)

/**
 * A rate given as a ratio (0..1) or as a percentage (0..100). Values above 1
 * read as percentages. When only counts arrive, errors / calls.
 */
export function ratioOf(rate: unknown, errors?: unknown, calls?: unknown): number | null {
  const r = num(rate)
  if (r !== null) return r > 1 ? Math.min(r / 100, 1) : Math.max(r, 0)
  const e = num(errors)
  const c = num(calls)
  return e !== null && c !== null && c > 0 ? Math.min(e / c, 1) : null
}

type Generated = { generated_at?: string; available?: boolean; enabled?: boolean }

const meta = (d: Loose): Generated => ({
  generated_at: str(d.generated_at) ?? undefined,
  available: typeof d.available === 'boolean' ? d.available : undefined,
  // OPS_METRICS_ENABLED=false: the hot-path counters are off, the routes say so.
  enabled: typeof d.enabled === 'boolean' ? d.enabled : typeof d.metrics_enabled === 'boolean' ? d.metrics_enabled : undefined
})

// ---------- LLM (Q5 on LIGHTRAG_USAGE_CALLS, FR-40) ----------

export type LlmGroup = 'purpose' | 'model'

export type LlmSeriesPoint = LatencyPoint & { key: string }

export type LlmRow = {
  key: string
  calls: number | null
  p50: number | null
  p95: number | null
  p99: number | null
  avg: number | null
  max: number | null
  error_rate: number | null
  truncated_rate: number | null
  running: number | null
  queued: number | null
}

export type QueueNow = { running: number | null; queued: number | null; in_flight: number | null }
export type QueuePoint = { t: number; running: number | null; queued: number | null }

export type PerfLlm = Generated & {
  /** False when the usage recorder (Token Insights) is off: no LLM latency source. */
  insights: boolean
  window: PerfWindow | null
  group: LlmGroup
  bucket_s: number
  /** The 7-day window: hourly average and max only. */
  approx_7d: boolean
  series: LlmSeriesPoint[]
  table: LlmRow[]
  /** Summed running and queued LLM calls across every loaded instance, now. */
  queue: QueueNow | null
  /** The same over time, when the server keeps it. */
  queue_series: QueuePoint[]
}

const queueNow = (raw: unknown): QueueNow | null => {
  const q = obj(raw)
  if (!Object.keys(q).length) return null
  return { running: num(q.running), queued: num(q.queued), in_flight: num(q.in_flight) }
}

/** One series row per (bucket, key); a server may also nest points under each key. */
function llmSeries(raw: unknown, bucketS: number): LlmSeriesPoint[] {
  const out: LlmSeriesPoint[] = []
  for (const s of arr(raw)) {
    const key = str(s.key) ?? str(s.purpose) ?? str(s.model) ?? str(s.name)
    if (Array.isArray(s.points)) {
      for (const p of latencyPoints(s.points, bucketS)) out.push({ ...p, key: key ?? '–' })
      continue
    }
    const [p] = latencyPoints([s], bucketS)
    if (p) out.push({ ...p, key: key ?? '–' })
  }
  return out.sort((a, b) => a.t - b.t || a.key.localeCompare(b.key))
}

export function normalizePerfLlm(raw: unknown, window?: PerfWindow, group: LlmGroup = 'purpose'): PerfLlm {
  const d = obj(raw)
  const flat = arr(d.series).flatMap((s) => (Array.isArray(s.points) ? timesOf(s.points) : timesOf([s])))
  const bucket_s = bucketSeconds(d, flat, window)
  const w = isPerfWindow(d.window) ? d.window : (window ?? null)
  return {
    ...meta(d),
    insights: d.insights !== false,
    window: w,
    group: d.group === 'model' || d.group === 'purpose' ? d.group : group,
    bucket_s,
    approx_7d: typeof d.approx_7d === 'boolean' ? d.approx_7d : w === '7d',
    series: llmSeries(d.series, bucket_s),
    table: arr(d.table ?? d.rows).map((r) => ({
      key: str(r.key) ?? str(r.purpose) ?? str(r.model) ?? '–',
      calls: first(r.calls, r.count),
      ...latencyOf(r),
      error_rate: ratioOf(r.error_rate ?? r.err_rate, r.errors, r.calls),
      truncated_rate: ratioOf(r.truncated_rate ?? r.trunc_rate, r.truncated, r.calls),
      running: num(r.running),
      queued: num(r.queued)
    })),
    queue: queueNow(d.queue),
    queue_series: arr(d.queue_series)
      .map((p) => ({ t: toMs(p.t ?? p.ts), running: num(p.running), queued: num(p.queued) }))
      .filter((p): p is QueuePoint => p.t !== null)
      .sort((a, b) => a.t - b.t)
  }
}

/** The keys charted as small multiples: the busiest first, at most `n` (the table lists them all). */
export function topKeys(data: Pick<PerfLlm, 'table' | 'series'>, n = 4): string[] {
  const calls = new Map<string, number>()
  for (const r of data.table) calls.set(r.key, (calls.get(r.key) ?? 0) + (r.calls ?? 0))
  for (const p of data.series) if (!calls.has(p.key)) calls.set(p.key, 0)
  for (const p of data.series) if (!data.table.length) calls.set(p.key, (calls.get(p.key) ?? 0) + (p.calls ?? 0))
  return [...calls.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, n)
    .map(([k]) => k)
}

/** One key's series as chart rows. */
export const seriesFor = (series: LlmSeriesPoint[], key: string): LatencyPoint[] => series.filter((p) => p.key === key)

/** Calls per bucket summed over every key (the LLM edge of the topology). */
export function sumCalls(series: { t: number; calls: number | null }[]): { t: number; calls: number }[] {
  const by = new Map<number, number>()
  for (const p of series) if (p.calls !== null) by.set(p.t, (by.get(p.t) ?? 0) + p.calls)
  return [...by.entries()].sort((a, b) => a[0] - b[0]).map(([t, calls]) => ({ t, calls }))
}

// ---------- embed and rerank (_record_dep_latency, FR-41) ----------

export const DEP_ROLES = ['embed', 'rerank'] as const
export type DepRole = (typeof DEP_ROLES)[number]

export type DepPoint = LatencyPoint & { waiting: number | null; running: number | null; kv_cache: number | null }

/** vLLM /metrics from the probe: requests running and waiting, KV cache use (0..1). */
export type VllmNow = { running: number | null; waiting: number | null; kv_cache: number | null }

export type DepSummary = {
  calls: number | null
  error_rate: number | null
  p50: number | null
  p95: number | null
  p99: number | null
}

export type PerfDeps = Generated & {
  bucket_s: number
  series: Record<DepRole, DepPoint[]>
  vllm: Record<DepRole, VllmNow | null>
  summary: Record<DepRole, DepSummary>
}

const ROLE_ALIASES: Record<DepRole, string[]> = { embed: ['embed', 'embedding', 'vllm_embed'], rerank: ['rerank', 'vllm_rerank'] }

const roleValue = (o: Loose, role: DepRole) => {
  for (const k of ROLE_ALIASES[role]) if (o[k] !== undefined) return o[k]
  return undefined
}

const vllmNow = (raw: unknown): VllmNow | null => {
  const v = obj(raw)
  if (!Object.keys(v).length) return null
  // The probe sends `kv_cache_pct` as 0..100 (0.5 means half a percent); other spellings are a 0..1 fraction.
  const pct = num(v.kv_cache_pct)
  const frac = first(v.kv_cache, v.kv_cache_usage, v.gpu_cache_usage_perc)
  return {
    running: first(v.running, v.num_requests_running),
    waiting: first(v.waiting, v.num_requests_waiting),
    kv_cache: pct !== null ? Math.min(Math.max(pct / 100, 0), 1) : frac === null ? null : frac > 1 ? frac / 100 : frac
  }
}

/** Calls, error rate and percentiles over the window: the server's summary, else totals and the worst bucket. */
function depSummary(raw: unknown, points: DepPoint[]): DepSummary {
  const s = obj(raw)
  if (Object.keys(s).length) {
    return {
      calls: first(s.calls, s.count),
      error_rate: ratioOf(s.error_rate, s.errors, s.calls),
      p50: first(s.p50_ms, s.p50),
      p95: first(s.p95_ms, s.p95),
      p99: first(s.p99_ms, s.p99)
    }
  }
  const calls = points.reduce((a, p) => a + (p.calls ?? 0), 0)
  const errors = points.reduce((a, p) => a + (p.errors ?? 0), 0)
  // Without a window summary, the worst bucket is the honest upper bound.
  const worst = (k: 'p50' | 'p95' | 'p99') => points.reduce<number | null>((m, p) => (p[k] !== null && (m === null || p[k]! > m) ? p[k] : m), null)
  return { calls: points.length ? calls : null, error_rate: calls ? errors / calls : null, p50: worst('p50'), p95: worst('p95'), p99: worst('p99') }
}

export function normalizePerfDeps(raw: unknown, window?: PerfWindow): PerfDeps {
  const d = obj(raw)
  const series = obj(d.series)
  const all = DEP_ROLES.flatMap((r) => timesOf(roleValue(series, r)))
  const bucket_s = bucketSeconds(d, all, window)
  const vllm = obj(d.vllm)
  const summary = obj(d.summary ?? d.table)
  const pointsFor = (role: DepRole): DepPoint[] =>
    arr(roleValue(series, role))
      .map((row) => {
        const [p] = latencyPoints([row], bucket_s)
        if (!p) return null
        const kv = first(row.kv_cache, row.vllm_kv_cache)
        return {
          ...p,
          waiting: first(row.waiting, row.vllm_waiting),
          running: first(row.running, row.vllm_running),
          kv_cache: kv === null ? null : kv > 1 ? kv / 100 : kv
        }
      })
      .filter((p): p is DepPoint => p !== null)
      .sort((a, b) => a.t - b.t)
  const embed = pointsFor('embed')
  const rerank = pointsFor('rerank')
  return {
    ...meta(d),
    bucket_s,
    series: { embed, rerank },
    vllm: { embed: vllmNow(roleValue(vllm, 'embed')), rerank: vllmNow(roleValue(vllm, 'rerank')) },
    summary: { embed: depSummary(roleValue(summary, 'embed'), embed), rerank: depSummary(roleValue(summary, 'rerank'), rerank) }
  }
}

// ---------- HTTP (RequestIdMiddleware, FR-42) ----------

/** Requests matching no route are counted under this one key (FR-42). */
export const UNMATCHED_ROUTE = '<unmatched>'
/** The table shows the ten slowest routes by p95 (§2.7). */
export const TOP_ROUTES = 10

export type HttpRoute = {
  /** The server's key, unique per row: "GET /documents/{doc_id}" or `<unmatched>`. */
  key: string
  /** The path template without the method ("/documents/{doc_id}"), or `<unmatched>`. */
  route: string
  method: string | null
  calls: number | null
  rps: number | null
  p50: number | null
  p95: number | null
  p99: number | null
  /** 5xx share of the route's requests, 0..1. */
  err5xx: number | null
  unmatched: boolean
}

export type HttpPoint = { t: number; calls: number | null; errors: number | null; p50: number | null; p95: number | null; p99: number | null }

export type PerfHttp = Generated & {
  bucket_s: number
  routes: HttpRoute[]
  series: HttpPoint[]
  /** The ≤200 distinct keys per minute cap was hit at least once in the window. */
  capped: boolean
}

export function normalizePerfHttp(raw: unknown, window?: PerfWindow): PerfHttp {
  const d = obj(raw)
  const bucket_s = bucketSeconds(d, timesOf(d.series), window)
  const seconds = window ? PERF_WINDOW_SECONDS[window] : null
  return {
    ...meta(d),
    bucket_s,
    routes: arr(d.routes).map((r) => {
      const key = str(r.route) ?? str(r.key) ?? str(r.path) ?? UNMATCHED_ROUTE
      const method = str(r.method)
      // The server keys a route as "METHOD /template" and sends the parts too; older shapes sent the template alone.
      const route = str(r.path) ?? (method && key.startsWith(`${method} `) ? key.slice(method.length + 1) : key)
      const calls = first(r.calls, r.count, r.requests)
      const rps = first(r.rps, calls !== null && seconds ? calls / seconds : null)
      return {
        key,
        route,
        method,
        calls,
        rps,
        ...(({ p50, p95, p99 }) => ({ p50, p95, p99 }))(latencyOf(r)),
        err5xx: ratioOf(r.err5xx ?? r.err5xx_rate ?? r.error_rate, r.errors_5xx ?? r.err5xx_count ?? r.errors, calls),
        unmatched: route === UNMATCHED_ROUTE || key === UNMATCHED_ROUTE
      }
    }),
    series: latencyPoints(d.series, bucket_s).map(({ t, calls, errors, p50, p95, p99 }) => ({ t, calls, errors, p50, p95, p99 })),
    capped: d.capped === true || d.keys_capped === true
  }
}

/**
 * The routes table: the slowest `n` by p95, plus `<unmatched>` as exactly one
 * row whenever it carries traffic (it is where scanners and typos land), in
 * its p95 place.
 */
export function topRoutes(routes: HttpRoute[], n = TOP_ROUTES): HttpRoute[] {
  const byP95 = (a: HttpRoute, b: HttpRoute) => (b.p95 ?? -1) - (a.p95 ?? -1) || (b.calls ?? 0) - (a.calls ?? 0) || a.key.localeCompare(b.key)
  const top = routes.filter((r) => !r.unmatched).sort(byP95).slice(0, n)
  const unmatched = routes.filter((r) => r.unmatched)
  if (!unmatched.length) return top
  const sum = (k: 'calls' | 'rps') => (unmatched.some((r) => r[k] !== null) ? unmatched.reduce((a, r) => a + (r[k] ?? 0), 0) : null)
  const worst = (k: 'p50' | 'p95' | 'p99' | 'err5xx') => unmatched.reduce<number | null>((m, r) => (r[k] !== null && (m === null || r[k]! > m) ? r[k] : m), null)
  const one: HttpRoute = { ...unmatched[0], key: UNMATCHED_ROUTE, route: UNMATCHED_ROUTE, method: null, calls: sum('calls'), rps: sum('rps'), p50: worst('p50'), p95: worst('p95'), p99: worst('p99'), err5xx: worst('err5xx') }
  return [...top, one].sort(byP95)
}

/**
 * The fixed text that opens Logs for a route: the template up to its first
 * path parameter ("/documents/{doc_id}" → "/documents/"), since log lines
 * carry real paths, not templates. Null for `<unmatched>`.
 */
export function routeLogText(route: string): string | null {
  if (!route || route === UNMATCHED_ROUTE) return null
  const brace = route.indexOf('{')
  const fixed = (brace >= 0 ? route.slice(0, brace) : route).trim()
  return fixed && fixed !== '/' ? fixed.slice(0, 200) : null
}

/** Logs for a route: level ≥ ERROR, the route's fixed path as text, over the same window (text search caps at 24 h). */
export function routeLogsView(route: string, window: PerfWindow): OpsHash {
  const q = routeLogText(route)
  const logQuery = encodeLogFilter({ level: 'ERROR', window: window as LogWindow, q: q ?? '' })
  return { ...DEFAULT_OPS_HASH, tab: 'logs', sub: 'stream', logQuery }
}

// ---------- concurrency (heartbeat queue sums, FR-44) ----------

export const QUEUE_ROLES = ['llm', 'embed', 'rerank'] as const
export type QueueRole = (typeof QUEUE_ROLES)[number]

export type RoleQueue = { running: number | null; in_flight: number | null; queued: number | null }

export type TopInstance = {
  workspace: string
  worker_id: string | null
  pid: number | null
  role: string | null
  running: number | null
  queued: number | null
  in_flight: number | null
}

/** OPS_PROVIDER_CONCURRENCY_BUDGET: the ratio is shown only when it is set (FR-44). */
export type Budget = { limit: number; used: number | null; ratio: number | null }

export type PerfConcurrency = Generated & {
  roles: Record<QueueRole, RoleQueue | null>
  top: TopInstance[]
  budget: Budget | null
  instances: number | null
  workers: number | null
  approximate: boolean
  /** False when no worker heartbeat could be read: the numbers are unknown, not zero. */
  heartbeats: boolean
  at: string | null
}

export function normalizePerfConcurrency(raw: unknown): PerfConcurrency {
  const d = obj(raw)
  const roles = obj(d.roles ?? d.queues)
  const role = (k: QueueRole): RoleQueue | null => {
    const r = roles[k] ?? (k === 'embed' ? roles.embedding : undefined)
    if (!r || typeof r !== 'object') return null
    return { running: num(r.running), in_flight: num(r.in_flight), queued: num(r.queued) }
  }
  const out = { llm: role('llm'), embed: role('embed'), rerank: role('rerank') }
  let budget: Budget | null = null
  const b = d.budget
  const limit = typeof b === 'number' ? b : first(obj(b).limit, obj(b).budget, obj(b).max)
  if (limit !== null && limit > 0) {
    // The provider sees calls in flight (running plus those mid-retry), not just running.
    const used = first(obj(b).in_flight, obj(b).used, out.llm?.in_flight, out.llm?.running)
    budget = { limit, used, ratio: first(obj(b).ratio, used !== null ? used / limit : null) }
  }
  return {
    ...meta(d),
    roles: out,
    top: arr(d.top ?? d.top_instances)
      .map((r) => ({
        workspace: str(r.workspace) ?? '',
        worker_id: str(r.worker_id),
        pid: num(r.pid),
        role: str(r.role),
        running: num(r.running),
        queued: num(r.queued),
        in_flight: num(r.in_flight)
      }))
      .sort((a, b) => (b.queued ?? 0) - (a.queued ?? 0) || (b.running ?? 0) - (a.running ?? 0) || a.workspace.localeCompare(b.workspace))
      .slice(0, 10),
    budget,
    instances: first(d.instances, d.loaded_instances),
    workers: first(d.workers, d.workers_reporting),
    approximate: d.approximate === true,
    heartbeats: d.heartbeats !== false,
    at: str(d.at) ?? str(d.as_of) ?? str(d.generated_at)
  }
}

// ---------- throughput (FR-43) ----------

export type CountPoint = { t: number; value: number | null; failed: number | null }

export type PerfThroughput = Generated & {
  days: number
  /** Documents processed and failed per hour (from runs). */
  docs: CountPoint[]
  /** LLM calls per minute (≤ 24 h). */
  llm_calls_per_min: CountPoint[]
  /** Email batches per hour. */
  email_batches: CountPoint[]
}

const countPoints = (raw: unknown, keys: string[], failedKeys: string[] = ['failed']): CountPoint[] =>
  arr(raw)
    .map((p) => ({
      t: toMs(p.t ?? p.ts ?? p.hour ?? p.minute),
      value: first(...keys.map((k) => p[k])),
      failed: first(...failedKeys.map((k) => p[k]))
    }))
    .filter((p): p is CountPoint => p.t !== null)
    .sort((a, b) => a.t - b.t)

export function normalizePerfThroughput(raw: unknown, days = 7): PerfThroughput {
  const d = obj(raw)
  return {
    ...meta(d),
    days: first(d.days) ?? days,
    docs: countPoints(d.docs ?? d.docs_per_hour, ['processed', 'done', 'value', 'count']),
    llm_calls_per_min: countPoints(d.llm_calls_per_min ?? d.llm_calls, ['value', 'calls', 'count'], ['errors', 'failed']),
    email_batches: countPoints(d.email_batches ?? d.email_batches_per_hour, ['value', 'batches', 'count'])
  }
}

/** Throughput asks for days, not a window: 1 day up to 24 h, 7 days for the 7 d window. */
export const throughputDays = (w: PerfWindow) => (w === '7d' ? 7 : 1)

// ---------- top workspaces (FR-45) ----------

export const TOP_METRICS = ['llm_calls', 'processing_minutes', 'failures'] as const
export type TopMetric = (typeof TOP_METRICS)[number]

export type TopWorkspaces = Generated & { metric: TopMetric; items: { workspace: string; value: number }[] }

export function normalizeTopWorkspaces(raw: unknown, metric: TopMetric): TopWorkspaces {
  const d = obj(raw)
  return {
    ...meta(d),
    metric: (TOP_METRICS as readonly string[]).includes(d.metric) ? d.metric : metric,
    items: arr(d.items)
      .map((r) => ({ workspace: str(r.workspace) ?? '', value: first(r.value, r.count, r.calls, r.minutes) ?? 0 }))
      .filter((r) => r.workspace)
      .sort((a, b) => b.value - a.value || a.workspace.localeCompare(b.workspace))
      .slice(0, 20)
  }
}

// ---------- sign-in health (auth counters, FR-35) ----------

export type SigninPoint = { t: number; sso_errors: number; token_401: number; lockouts: number }

export type SigninCounts = { sso_errors: number | null; token_401: number | null; lockouts: number | null }

export type Signin = Generated &
  SigninCounts & {
    window: string | null
    /** The last 15 minutes, sent with every window so one request fills the card; null when absent. */
    last_15m: SigninCounts | null
    /** signin_failures_15m as the server judged it, with its KV thresholds; null from an older server. */
    state: SigninRuleState | null
    series: SigninPoint[]
  }

export type SigninRuleState = 'ok' | 'warning' | 'serious'
const SIGNIN_RULE_STATES = new Set<string>(['ok', 'warning', 'serious'])

const signinCounts = (d: Loose): SigninCounts => ({
  sso_errors: first(d.sso_errors, d.sso_callback_errors),
  token_401: first(d.token_401, d.unauthorized, d.token_failures),
  lockouts: first(d.lockouts, d.login_lockouts)
})

export function normalizeSignin(raw: unknown, window?: string): Signin {
  const d = obj(raw)
  const w = str(d.window) ?? window ?? null
  const recent = d.last_15m && typeof d.last_15m === 'object' ? signinCounts(obj(d.last_15m)) : null
  const own = signinCounts(d)
  return {
    ...meta(d),
    window: w,
    ...own,
    // A 15-minute answer is its own last 15 minutes.
    last_15m: recent ?? (w === '15m' ? own : null),
    state: typeof d.state === 'string' && SIGNIN_RULE_STATES.has(d.state) ? (d.state as SigninRuleState) : null,
    series: arr(d.series)
      .map((p) => ({
        t: toMs(p.t ?? p.ts),
        sso_errors: first(p.sso_errors, p.sso_callback_errors) ?? 0,
        token_401: first(p.token_401, p.unauthorized) ?? 0,
        lockouts: first(p.lockouts, p.login_lockouts) ?? 0
      }))
      .filter((p): p is SigninPoint => p.t !== null)
      .sort((a, b) => a.t - b.t)
  }
}

/**
 * The sign-in state from the 15-minute counts, by the default rule
 * `signin_failures_15m` (§1.7): warning above 5 SSO errors or 3 lockouts,
 * serious above 20 of either. 401s alone are normal (expired tokens).
 */
export function signinState(s: Pick<Signin, 'sso_errors' | 'lockouts'> | null | undefined): 'ok' | 'warning' | 'serious' | 'unknown' {
  if (!s || (s.sso_errors === null && s.lockouts === null)) return 'unknown'
  const sso = s.sso_errors ?? 0
  const lock = s.lockouts ?? 0
  if (sso > 20 || lock > 20) return 'serious'
  if (sso > 5 || lock > 3) return 'warning'
  return 'ok'
}

// ---------- rates (topology flow, §2.12) ----------

/** A traffic rate per minute plus the edge's own typical (p50) and busy (p90) rates over 24 h. */
export type EdgeRate = { perMin: number; p50: number | null; p90: number | null }

/**
 * Calls per minute in the latest complete bucket: the newest bucket that
 * ended by `now`. A source that sent nothing for longer than two buckets
 * reads 0 (servers leave empty buckets out).
 */
export function latestPerMin(points: { t: number; calls: number | null }[], bucketS: number, now: number): number {
  const ms = Math.max(1, bucketS) * 1000
  let pick: { t: number; calls: number | null } | null = null
  for (const p of points) if (p.t + ms <= now + 1000 && (!pick || p.t > pick.t)) pick = p
  if (!pick || pick.calls === null) return 0
  if (now - (pick.t + ms) > 2 * ms) return 0
  return (pick.calls * 60) / Math.max(1, bucketS)
}

/** The q-quantile (0..1) of a list, by linear interpolation; null for an empty list. */
export function quantile(values: number[], q: number): number | null {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b)
  if (!v.length) return null
  const at = Math.min(Math.max(q, 0), 1) * (v.length - 1)
  const lo = Math.floor(at)
  const hi = Math.ceil(at)
  return v[lo] + (v[hi] - v[lo]) * (at - lo)
}

/**
 * The edge's p50 and p90 per-minute rate over 24 h, from the buckets that
 * carried traffic: an edge idle most of the night would otherwise have a
 * median of 0 and read "high" at the first request of the morning.
 */
export function ratePercentiles(points: { calls: number | null }[], bucketS: number): { p50: number | null; p90: number | null } {
  const rates = points.filter((p) => p.calls !== null && p.calls > 0).map((p) => (p.calls! * 60) / Math.max(1, bucketS))
  return { p50: quantile(rates, 0.5), p90: quantile(rates, 0.9) }
}

export type FlowSource = 'http' | 'embed' | 'rerank' | 'llm'
export type FlowRates = Partial<Record<FlowSource, EdgeRate>>

type RateInputs = {
  http?: PerfHttp | null
  deps?: PerfDeps | null
  llm?: PerfLlm | null
}

/**
 * Per-source rates for the topology edges: the current minute from the
 * 15-minute answers, the thresholds from the 24-hour ones. A source whose
 * current answer is missing is left out (its edges stay static).
 */
export function flowRatesFrom(current: RateInputs, baseline: RateInputs, now: number): FlowRates {
  const out: FlowRates = {}
  const add = (k: FlowSource, cur: { t: number; calls: number | null }[] | null, curBucket: number, base: { calls: number | null }[] | null, baseBucket: number) => {
    if (!cur) return
    const p = base ? ratePercentiles(base, baseBucket) : { p50: null, p90: null }
    out[k] = { perMin: latestPerMin(cur, curBucket, now), ...p }
  }
  const usable = <T extends Generated>(d: T | null | undefined) => (d && d.available !== false && d.enabled !== false ? d : null)
  const http = usable(current.http)
  const httpBase = usable(baseline.http)
  add('http', http?.series ?? null, http?.bucket_s ?? 60, httpBase?.series ?? null, httpBase?.bucket_s ?? 300)
  const deps = usable(current.deps)
  const depsBase = usable(baseline.deps)
  for (const role of DEP_ROLES) {
    add(role, deps ? deps.series[role] : null, deps?.bucket_s ?? 60, depsBase ? depsBase.series[role] : null, depsBase?.bucket_s ?? 300)
  }
  // Without the usage recorder (insights: false) there is no LLM source: the edge stays static, not "0/min".
  const llmUsable = (d: PerfLlm | null | undefined) => (d && d.available !== false && d.insights !== false ? d : null)
  const llm = llmUsable(current.llm)
  const llmBase = llmUsable(baseline.llm)
  add('llm', llm ? sumCalls(llm.series) : null, llm?.bucket_s ?? 60, llmBase ? sumCalls(llmBase.series) : null, llmBase?.bucket_s ?? 300)
  return out
}

// ---------- chart rows ----------

/** Rows for a latency chart: percentiles (≤24 h) or average and max (7 d). */
export function latencyRows(points: LatencyPoint[], rollup: boolean): Record<string, number | null>[] {
  return points.map((p): Record<string, number | null> => (rollup ? { t: p.t, avg: p.avg, max: p.max } : { t: p.t, p50: p.p50, p95: p.p95, p99: p.p99 }))
}

/** Requests (or calls) per minute per bucket. */
export const perMinRows = (points: { t: number; calls: number | null }[], bucketS: number) =>
  points.map((p) => ({ t: p.t, rate: p.calls === null ? null : (p.calls * 60) / Math.max(1, bucketS) }))

/** 5xx share per bucket (0..1), null when the bucket had no requests. */
export const errorShareRows = (points: { t: number; calls: number | null; errors: number | null }[]) =>
  points.map((p) => ({ t: p.t, share: p.calls && p.errors !== null ? p.errors / p.calls : p.calls === 0 ? 0 : null }))

/** "≈ 75/min" for an edge label: whole numbers from 10, one decimal below, "<0.1" for a trickle. */
export function fmtPerMin(perMin: number | null | undefined): string {
  if (typeof perMin !== 'number' || !Number.isFinite(perMin) || perMin <= 0) return '0/min'
  if (perMin < 0.1) return '<0.1/min'
  if (perMin < 10) return `${(Math.round(perMin * 10) / 10).toFixed(1)}/min`
  return `${Math.round(perMin).toLocaleString('en-US')}/min`
}
