import type {
  DepDetail,
  DepHealth,
  HostInfo,
  OpsConfig,
  OpsHealth,
  OpsWorker,
  OpsWorkers,
  PgBlocking,
  ProbeResult,
  ProbeStatus,
  QueueCounts,
  SmtpTestResult,
  StripCell,
  WorkerSample,
  WorkerSnapshot
} from '../api'
import type { LightragStatus } from '@/api/lightrag'
import { ageSeconds, fmtBytes, fmtClock, fmtCount, fmtLatency } from '../format'
import { OpsState, isStale } from '../severity'
import { gpuLine, normalizeGpus } from './gpu'

/**
 * Read models for the Health tab (spec §2.6, §3.7 PR-3): the dependency grid
 * with its 48-cell availability strip, the workers table, the dependency and
 * worker drawers and the redacted config. As in ../normalize.ts, every
 * normaliser accepts the spellings the server may use, keeps unknown fields
 * out of the views' way and never throws on a partial payload.
 */

type Loose = Record<string, any>

const obj = (v: unknown): Loose => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Loose) : {})
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const str = (v: unknown): string | null => (typeof v === 'string' && v ? v : null)
const arr = <T = Loose>(v: unknown): T[] => (Array.isArray(v) ? (v as T[]) : [])
const first = <T>(...values: (T | null | undefined)[]): T | null => {
  for (const v of values) if (v !== null && v !== undefined) return v
  return null
}
const bool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null)
/** Epoch ms from an ISO string or a number; numbers below 1e11 are epoch seconds. */
const ms = (iso: unknown): number | null => {
  if (typeof iso === 'number' && Number.isFinite(iso)) return iso < 1e11 ? iso * 1000 : iso
  const s = str(iso)
  if (!s) return null
  const t = Date.parse(s)
  return Number.isFinite(t) ? t : null
}

// ---------- probe status ----------

const STATUS_ALIASES: Record<string, ProbeStatus> = {
  ok: 'ok',
  healthy: 'ok',
  up: 'ok',
  pass: 'ok',
  passing: 'ok',
  warn: 'warn',
  warning: 'warn',
  slow: 'warn',
  degraded: 'warn',
  serious: 'warn',
  down: 'down',
  fail: 'down',
  failed: 'down',
  error: 'down',
  critical: 'down',
  crit: 'down',
  timeout: 'down',
  unknown: 'unknown',
  stale: 'unknown',
  unprobed: 'unprobed',
  not_probed: 'unprobed',
  disabled: 'unprobed',
  skipped: 'unprobed'
}

export const toProbeStatus = (raw: unknown): ProbeStatus => STATUS_ALIASES[(str(raw) ?? '').toLowerCase()] ?? 'unknown'

/** A probe status in the page's state vocabulary (§2.2): down is Critical, "not probed" is Unknown. */
export const depState = (status: ProbeStatus | string | null | undefined): OpsState => {
  switch (toProbeStatus(status)) {
    case 'ok':
      return 'ok'
    case 'warn':
      return 'warning'
    case 'down':
      return 'critical'
    default:
      return 'unknown'
  }
}

// ---------- the availability strip ----------

export const STRIP_CELLS = 48
export const SLOT_MS = 30 * 60 * 1000

/** Start of the half-hour slot that holds `at` (epoch ms). */
export const slotStart = (at: number) => Math.floor(at / SLOT_MS) * SLOT_MS

const cellOf = (raw: unknown): Omit<StripCell, 'start'> & { start: number | null } => {
  if (raw === null || raw === undefined) return { start: null, status: 'unknown', total: null, failed: null, slow: null }
  if (typeof raw === 'string') {
    const s = toProbeStatus(raw)
    return { start: null, status: s === 'unprobed' ? 'unknown' : s, total: null, failed: null, slow: null }
  }
  if (typeof raw === 'number') {
    // A ratio of good results in the slot: 1 all good, 0 all failed.
    if (!Number.isFinite(raw) || raw < 0) return { start: null, status: 'unknown', total: null, failed: null, slow: null }
    return { start: null, status: raw >= 1 ? 'ok' : raw <= 0 ? 'down' : 'warn', total: null, failed: null, slow: null }
  }
  const c = obj(raw)
  const total = first(num(c.total), num(c.count), num(c.n))
  const failed = first(num(c.failed), num(c.down), num(c.fail), num(c.errors))
  const slow = first(num(c.slow), num(c.warn))
  const okCount = num(c.ok)
  let status: StripCell['status']
  const given = str(c.status) ?? str(c.state) ?? str(c.s)
  if (given) {
    const s = toProbeStatus(given)
    status = s === 'unprobed' ? 'unknown' : s
  } else {
    const n = total ?? (okCount !== null ? okCount + (failed ?? 0) + (slow ?? 0) : null)
    if (!n) status = 'unknown'
    else if ((failed ?? 0) >= n) status = 'down'
    else if ((failed ?? 0) > 0 || (slow ?? 0) > 0) status = 'warn'
    else status = 'ok'
  }
  const n = total ?? (okCount !== null ? okCount + (failed ?? 0) + (slow ?? 0) : null)
  return { start: first(ms(c.t), ms(c.start), ms(c.ts), ms(c.slot)), status, total: n, failed, slow }
}

/**
 * The 48 half-hour cells, oldest first. Missing cells at the start (a new
 * dependency, a short history) are Unknown. Cells without their own start
 * time count back from the slot that holds `anchor` (the latest cycle), so
 * keys only change when a new half hour begins.
 */
export function normalizeStrip(raw: unknown, anchor: number | null): StripCell[] {
  const list = arr<unknown>(raw).slice(-STRIP_CELLS)
  const cells = list.map(cellOf)
  const pad = STRIP_CELLS - cells.length
  const padded = [...Array.from({ length: pad }, () => cellOf(null)), ...cells]
  const last = anchor !== null && Number.isFinite(anchor) ? slotStart(anchor) : null
  return padded.map((c, i) => ({ ...c, start: c.start ?? (last === null ? null : last - (STRIP_CELLS - 1 - i) * SLOT_MS) }))
}

/** "11:00–11:30" for a slot, or null when its time is not known. */
export function slotRange(start: number | null): string | null {
  if (start === null || !Number.isFinite(start)) return null
  return `${fmtClock(start).slice(0, 5)}–${fmtClock(start + SLOT_MS).slice(0, 5)}`
}

/** The cell's text for screen readers and the tooltip: "11:00–11:30: 2 failed of 60". */
export function cellLabel(cell: StripCell, t: (key: string, opts?: Record<string, unknown>) => string): string {
  const range = slotRange(cell.start) ?? t('operations.strip.slot')
  if (cell.status === 'unknown') return t('operations.strip.cell.unknown', { range })
  if (cell.status === 'down' || (cell.failed ?? 0) > 0) {
    return cell.total !== null && cell.failed !== null
      ? t('operations.strip.cell.failed', { range, failed: cell.failed, total: cell.total })
      : t('operations.strip.cell.down', { range })
  }
  if (cell.status === 'warn') {
    return cell.total !== null && cell.slow !== null
      ? t('operations.strip.cell.slowOf', { range, slow: cell.slow, total: cell.total })
      : t('operations.strip.cell.warn', { range })
  }
  return cell.total !== null ? t('operations.strip.cell.okOf', { range, total: cell.total }) : t('operations.strip.cell.ok', { range })
}

/** Availability over the strip: share of cells with data that were OK (null without data). */
export function stripUptime(cells: StripCell[]): number | null {
  const known = cells.filter((c) => c.status !== 'unknown')
  if (!known.length) return null
  return known.filter((c) => c.status === 'ok').length / known.length
}

// ---------- dependencies ----------

/** Seconds between checks by dependency (§2.13): SMTP and TLS every 15 min, the IdP every 5 min, the rest every 30 s. */
const DEFAULT_INTERVAL: Record<string, number> = { smtp: 900, tls: 900, tls_proxy: 900, proxy: 900, idp: 300, keycloak: 300, oidc: 300 }
export const PROBE_INTERVAL_SECONDS = 30

export const depInterval = (d: Pick<DepHealth, 'id' | 'interval_s'>) => d.interval_s ?? DEFAULT_INTERVAL[d.id] ?? PROBE_INTERVAL_SECONDS

/** Ids whose health is read from real traffic or local state: there is no probe to re-run. */
const PASSIVE_IDS = new Set(['llm', 'llm_provider', 'recorder', 'insights', 'log_sink', 'manager', 'disk', 'signin'])

export function normalizeDep(raw: unknown, anchor: number | null = null): DepHealth | null {
  const d = obj(raw)
  const id = str(d.id) ?? str(d.dependency) ?? str(d.name)
  if (!id) return null
  const detail = obj(d.detail)
  const checked = first(str(d.checked_at), str(d.ts), str(d.at), str(d.last_at))
  return {
    id,
    name: str(d.name) && d.name !== id ? str(d.name) : str(d.label),
    kind: str(d.kind) ?? str(d.role) ?? str(detail.kind),
    target: str(d.target) ?? str(detail.target) ?? str(detail.host),
    // A dependency nothing checks (etcd, MinIO behind Milvus) reads "not probed", whatever its status says.
    status: d.probed === false ? 'unprobed' : toProbeStatus(d.status ?? d.state),
    stale: bool(d.stale) ?? undefined,
    latency_ms: first(num(d.latency_ms), num(d.latency)),
    checked_at: checked,
    last_ok_at: str(d.last_ok_at),
    consecutive_failures: first(num(d.consecutive_failures), num(d.failures)),
    error_class: str(d.error_class),
    detail,
    strip: Array.isArray(d.strip) ? normalizeStrip(d.strip, ms(checked) ?? anchor) : [],
    interval_s: first(num(d.interval_s), num(d.every_s)),
    passive: bool(d.passive) ?? PASSIVE_IDS.has(id)
  }
}

export function normalizeDeps(raw: unknown, anchor: number | null = null): DepHealth[] {
  const list = Array.isArray(raw)
    ? raw
    : raw && typeof raw === 'object'
      ? // `{postgres: {...}, milvus: {...}}`
      Object.entries(raw as Loose).map(([id, v]) => ({ id, ...obj(v) }))
      : []
  return list.map((d) => normalizeDep(d, anchor)).filter((d): d is DepHealth => d !== null)
}

export function normalizeHealth(raw: unknown): OpsHealth {
  const h = obj(raw)
  const c = h.cycle ? obj(h.cycle) : null
  const anchor = first(ms(c?.at), ms(h.generated_at))
  const deps = normalizeDeps(h.deps ?? h.dependencies, anchor)
  return {
    ...h,
    // Probes run without PostgreSQL KV too (NFR-8): a payload with dependencies is never "unavailable".
    available: deps.length ? true : h.available,
    cycle: c
      ? {
        at: str(c.at),
        duration_ms: first(num(c.duration_ms), num(c.duration_s) === null ? null : (num(c.duration_s) as number) * 1000),
        holder: str(c.holder) ?? str(c.worker_id),
        holder_pid: first(num(c.holder_pid), num(c.pid)),
        interval_s: num(c.interval_s)
      }
      : null,
    deps,
    enabled: bool(h.enabled) ?? bool(h.probes_enabled) ?? undefined
  } as OpsHealth
}

/** POST /health/{dep}/probe answers with the new result; any shape becomes a card. */
export function normalizeProbeResult(dep: string, raw: unknown): DepHealth {
  const r = obj(raw)
  const inner = r.dep ?? r.result ?? r
  return normalizeDep({ id: dep, ...obj(inner) }) ?? { id: dep, status: 'unknown', detail: {}, strip: [] }
}

export function normalizeSmtpTest(raw: unknown): SmtpTestResult {
  const r = obj(raw)
  const ok = bool(r.ok) ?? toProbeStatus(r.status) === 'ok'
  return { ok, error_class: str(r.error_class), latency_ms: first(num(r.latency_ms), num(r.latency)), detail: r.detail ? obj(r.detail) : null }
}

const BLOCKING_KEYS = ['blocking', 'blocked', 'blocking_sessions', 'locks']

export function normalizeBlocking(detail: Loose): PgBlocking[] {
  const key = BLOCKING_KEYS.find((k) => Array.isArray(detail[k]))
  if (!key) return []
  return arr(detail[key]).map((b) => {
    const x = obj(b)
    const by = x.blocked_by ?? x.blocking_pids ?? x.blockers
    return {
      pid: num(x.pid),
      blocked_by: Array.isArray(by) ? by.filter((p): p is number => typeof p === 'number') : typeof by === 'number' ? [by] : [],
      duration_s: first(num(x.duration_s), num(x.wait_s), num(x.age_s), num(x.duration_ms) === null ? null : (num(x.duration_ms) as number) / 1000),
      state: str(x.state),
      statement_type: str(x.statement_type) ?? str(x.statement) ?? str(x.command),
      fingerprint: str(x.fingerprint) ?? str(x.query_fingerprint) ?? str(x.query_id),
      query_length: first(num(x.query_length), num(x.query_len)),
      wait_event: str(x.wait_event) ?? str(x.wait_event_type)
    }
  })
}

export function normalizeDepDetail(dep: string, raw: unknown): DepDetail {
  const d = obj(raw)
  const recent: ProbeResult[] = arr(d.recent ?? d.results)
    .map((r) => obj(r))
    .filter((r) => str(r.ts) ?? str(r.at))
    .map((r) => ({
      ts: (str(r.ts) ?? str(r.at)) as string,
      status: toProbeStatus(r.status),
      latency_ms: first(num(r.latency_ms), num(r.latency)),
      error_class: str(r.error_class),
      detail: r.detail ? obj(r.detail) : null
    }))
    .sort((a, b) => Date.parse(b.ts) - Date.parse(a.ts))
  const series = arr(d.latency_series ?? d.series)
    .map((p) => obj(p))
    .map((p) => ({ t: first(ms(p.t), ms(p.ts)), latency_ms: first(num(p.latency_ms), num(p.latency), num(p.ms)), status: toProbeStatus(p.status ?? 'ok') }))
    .filter((p): p is { t: number; latency_ms: number | null; status: ProbeStatus } => p.t !== null)
    .sort((a, b) => a.t - b.t)
  const detail = obj(d.detail)
  const classes = new Map<string, number>()
  if (Array.isArray(d.error_classes)) {
    for (const e of d.error_classes) {
      const x = obj(e)
      const key = str(x.key) ?? str(x.error_class)
      if (key) classes.set(key, num(x.count) ?? 0)
    }
  } else {
    for (const r of recent) if (r.error_class) classes.set(r.error_class, (classes.get(r.error_class) ?? 0) + 1)
  }
  const card = d.dep ? normalizeDep({ id: dep, ...obj(d.dep) }) : d.status ? normalizeDep({ id: dep, ...d, strip: undefined }) : null
  return {
    ...d,
    id: dep,
    dep: card,
    latency_series: series,
    recent: recent.slice(0, 50),
    detail,
    blocking: normalizeBlocking({ ...detail, ...d }),
    error_classes: [...classes.entries()].map(([key, count]) => ({ key, count })).sort((a, b) => b.count - a.count)
  } as DepDetail
}

/** The status the Dependency drawer shows: its own card, else the newest probe result. */
export function drawerStatus(d: DepDetail | null | undefined): ProbeStatus {
  if (!d) return 'unknown'
  if (d.dep) return d.dep.status
  return d.recent[0]?.status ?? 'unknown'
}

/** A dependency card is stale once its last check is older than twice its own cadence (§2.12). */
export function depIsStale(d: DepHealth, now: number, fallbackCheckedAt: number | null): boolean {
  if (d.status === 'unprobed') return false
  const at = ms(d.checked_at)
  // The server marks a dependency it has no result for yet as stale.
  if (at === null && d.stale) return true
  return isStale(at ?? fallbackCheckedAt, depInterval(d), now)
}

/** Grid order: storage first, then the model services, external services and the local health checks. */
const DEP_ORDER = [
  'postgres',
  'graph',
  'neo4j',
  'memgraph',
  'mongo',
  'vector',
  'milvus',
  'etcd',
  'minio',
  'qdrant',
  'redis',
  'vllm_embed',
  'embed',
  'vllm_rerank',
  'rerank',
  'docling',
  'llm',
  'smtp',
  'idp',
  'tls',
  'signin',
  'manager',
  'recorder',
  'log_sink',
  'disk'
]

export function sortDeps(deps: DepHealth[]): DepHealth[] {
  const rank = (id: string) => {
    const i = DEP_ORDER.indexOf(id)
    return i < 0 ? DEP_ORDER.length : i
  }
  return [...deps].sort((a, b) => rank(a.id) - rank(b.id) || a.id.localeCompare(b.id))
}

/** "OK of total" over probed dependencies (the Overview Deps tile). */
export function depCounts(deps: DepHealth[]): { ok: number; total: number } {
  const probed = deps.filter((d) => d.status !== 'unprobed')
  return { ok: probed.filter((d) => d.status === 'ok').length, total: probed.length }
}

/** Dependencies a "Re-check" can run for (passive and unprobed ones have no probe). */
export const canRecheck = (d: DepHealth) => !d.passive && d.status !== 'unprobed'

/**
 * Dependencies "Re-check all" includes: the slow-cadence probes (SMTP, TLS, IdP) talk to
 * outside servers and are re-checked one at a time only (the server allows one a minute).
 */
export const inRecheckAll = (d: DepHealth) => canRecheck(d) && depInterval(d) <= PROBE_INTERVAL_SECONDS

// ---------- workers ----------

const MB = 1024 * 1024

const queueOf = (raw: unknown): QueueCounts | null => {
  if (!raw || typeof raw !== 'object') return null
  const q = obj(raw)
  const running = first(num(q.running), num(q.run))
  const queued = first(num(q.queued), num(q.waiting), num(q.queue))
  if (running === null && queued === null) return null
  return { running, queued, in_flight: num(q.in_flight) }
}

const drainersOf = (raw: unknown): WorkerSnapshot['drainers'] => {
  if (Array.isArray(raw)) {
    return raw
      .map((d) => (typeof d === 'string' ? { workspace: d } : { ...obj(d), workspace: str(obj(d).workspace) ?? '' }))
      .filter((d) => d.workspace)
      .map((d) => ({ workspace: d.workspace, state: str((d as Loose).state), backoff_until: str((d as Loose).backoff_until) }))
  }
  const o = obj(raw)
  const backoff = str(o.backoff_until)
  return arr<unknown>(o.workspaces)
    .filter((w): w is string => typeof w === 'string' && !!w)
    .map((workspace) => ({ workspace, state: null, backoff_until: backoff }))
}

export function normalizeSnapshot(raw: unknown): WorkerSnapshot {
  const s = obj(raw)
  const pool = obj(s.pool ?? s.pg_pool)
  const queues = obj(s.queues)
  const rec = s.recorder ?? s.insights_recorder
  const sink = s.log_sink ?? s.sink
  const r = obj(rec)
  const k = obj(sink)
  const rssMb = num(s.rss_mb)
  const hasPool = Object.keys(pool).length > 0 || num(s.pool_wait_ms) !== null
  return {
    ...s,
    rss: first(num(s.rss), num(s.rss_bytes), rssMb === null ? null : rssMb * MB),
    cpu_pct: first(num(s.cpu_pct), num(s.cpu), num(s.cpu_percent)),
    fds: first(num(s.fds), num(s.num_fds), num(s.open_fds)),
    threads: first(num(s.threads), num(s.num_threads)),
    instances: first(num(s.instances), num(s.loaded_instances), num(s.loaded)),
    loop_lag_ms: first(num(s.loop_lag_ms), num(s.loop_lag)),
    manager_rtt_ms: first(num(s.manager_rtt_ms), num(s.manager_rtt), num(s.mgr_rtt_ms)),
    manager_state: str(s.manager_state),
    pool: hasPool
      ? {
        size: first(num(pool.size), num(pool.get_size)),
        idle: first(num(pool.idle), num(pool.idle_size)),
        max: first(num(pool.max), num(pool.max_size)),
        wait_ms: first(num(pool.wait_ms), num(pool.acquire_wait_ms), num(s.pool_wait_ms))
      }
      : null,
    queues: {
      llm: queueOf(queues.llm ?? s.llm_queue),
      embed: queueOf(queues.embed ?? queues.embedding ?? s.embed_queue),
      rerank: queueOf(queues.rerank ?? s.rerank_queue)
    },
    recorder: rec
      ? {
        buffer: first(num(r.buffer), num(r.buffer_len)),
        last_flush_ok_at: str(r.last_flush_ok_at),
        flush_failures: num(r.flush_failures),
        lost_rows: first(num(r.lost_rows), num(r.lost)),
        dropped_events: num(r.dropped_events)
      }
      : null,
    log_sink: sink ? { buffer: first(num(k.buffer), num(k.buffer_len)), dropped: num(k.dropped), rate_limited: num(k.rate_limited) } : null,
    drainers: drainersOf(s.drainers ?? s.resync_drainers)
  } as WorkerSnapshot
}

/** Liveness rule (§3.4): a worker is alive while its heartbeat is under 45 s old. */
export const ALIVE_SECONDS = 45

export function normalizeWorker(raw: unknown, generatedAt: number | null = null): OpsWorker | null {
  const w = obj(raw)
  const id = str(w.worker_id) ?? str(w.id)
  if (!id) return null
  const age = first(num(w.heartbeat_age_s), str(w.heartbeat_at) && generatedAt !== null ? ageSeconds(w.heartbeat_at, generatedAt) : null)
  const exited = str(w.exited_at)
  const alive = bool(w.alive) ?? (exited ? false : age === null ? null : age < ALIVE_SECONDS)
  const snapshot = normalizeSnapshot(w.snapshot ?? w)
  const leases = w.leases_held ?? obj(w.snapshot).leases
  return {
    ...w,
    worker_id: id,
    pid: num(w.pid),
    host: str(w.host),
    mode: str(w.mode),
    version: str(w.version),
    started_at: str(w.started_at),
    heartbeat_age_s: age,
    alive,
    superseded: !!w.superseded,
    exited_at: exited,
    leases_held: arr<unknown>(leases).filter((l): l is string => typeof l === 'string'),
    self: bool(w.self) ?? bool(w.is_self) ?? undefined,
    snapshot
  } as OpsWorker
}

/** Alive first, then the newest start; superseded and exited rows last. */
export function sortWorkers(items: OpsWorker[]): OpsWorker[] {
  const group = (w: OpsWorker) => (w.superseded || w.exited_at ? 2 : w.alive === false ? 1 : 0)
  return [...items].sort(
    (a, b) => group(a) - group(b) || (ms(b.started_at) ?? 0) - (ms(a.started_at) ?? 0) || a.worker_id.localeCompare(b.worker_id)
  )
}

export function normalizeHost(raw: unknown): HostInfo | null {
  if (!raw || typeof raw !== 'object') return null
  const h = obj(raw)
  const mem = obj(h.memory ?? h.mem)
  const load = h.load ?? h.loadavg ?? h.load_avg
  const disksRaw = h.disks ?? h.disk
  const disks = (Array.isArray(disksRaw) ? disksRaw : disksRaw ? Object.entries(obj(disksRaw)).map(([path, v]) => ({ path, ...obj(v) })) : [])
    .map((d) => obj(d))
    .filter((d) => str(d.path) ?? str(d.name))
    .map((d) => ({ path: (str(d.path) ?? str(d.name)) as string, used: num(d.used), total: first(num(d.total), num(d.size)) }))
  return {
    mem_used: first(num(h.mem_used), num(mem.used), num(mem.current)),
    mem_limit: first(num(h.mem_limit), num(mem.limit), num(mem.max)),
    cpu_pct: first(num(h.cpu_pct), num(h.container_cpu_pct), num(h.cpu)),
    load: Array.isArray(load) ? load.filter((n): n is number => typeof n === 'number').slice(0, 3) : null,
    disks,
    sampled_at: str(h.sampled_at) ?? str(h.at)
  }
}

export function normalizeWorkers(raw: unknown): OpsWorkers {
  const r = Array.isArray(raw) ? { items: raw } : obj(raw)
  const at = ms(r.generated_at)
  const items = arr(r.items ?? r.workers)
    .map((w) => normalizeWorker(w, at))
    .filter((w): w is OpsWorker => w !== null)
  return {
    ...r,
    items: sortWorkers(items),
    host: normalizeHost(r.host),
    mode: str(r.mode),
    expected: first(num(r.expected), num(r.expected_workers)),
    live: first(num(r.live), num(r.workers_live))
  } as OpsWorkers
}

export function normalizeWorkerHistory(raw: unknown): WorkerSample[] {
  const list = Array.isArray(raw) ? raw : arr(obj(raw).items ?? obj(raw).samples ?? obj(raw).series)
  return list
    .map((p) => obj(p))
    .map((p) => ({
      t: first(ms(p.t), ms(p.minute), ms(p.ts)),
      rss: first(num(p.rss), num(p.rss_bytes)),
      cpu: first(num(p.cpu), num(p.cpu_pct)),
      loop_lag: first(num(p.loop_lag), num(p.loop_lag_ms)),
      pool_wait: first(num(p.pool_wait), num(p.pool_wait_ms))
    }))
    .filter((p): p is WorkerSample => p.t !== null)
    .sort((a, b) => a.t - b.t)
}

export type WorkerLook = { state: OpsState; label: string }

/**
 * A worker row's state: exited and superseded rows are history (Unknown, with
 * their own words), a stale heartbeat on a current row is Critical, a live
 * one OK. A heartbeat age that is not known yet is Unknown, never OK.
 */
export function workerLook(w: OpsWorker): WorkerLook {
  if (w.exited_at) return { state: 'unknown', label: 'operations.worker.state.exited' }
  if (w.superseded) return { state: 'unknown', label: 'operations.worker.state.superseded' }
  if (w.alive === false) return { state: 'critical', label: 'operations.worker.state.stale' }
  if (w.alive === true) return { state: 'ok', label: 'operations.worker.state.alive' }
  return { state: 'unknown', label: 'operations.state.unknown' }
}

/** Uptime in seconds from the start time, against when the data was confirmed. */
export const uptimeSeconds = (w: OpsWorker, now: number) => (w.exited_at || w.alive === false ? null : ageSeconds(w.started_at, now))

/** "12/50 (3 ms)": pool size against its maximum, with the sampled acquire wait. */
export const poolText = (p: WorkerSnapshot['pool']) => {
  if (!p) return '–'
  const size = p.size !== null && p.max !== null ? `${fmtCount(p.size)}/${fmtCount(p.max)}` : fmtCount(p.size ?? null)
  return typeof p.wait_ms === 'number' ? `${size} (${fmtLatency(p.wait_ms)})` : size
}

/** Workers that count toward "live vs expected" (superseded and exited rows do not). */
export const currentWorkers = (items: OpsWorker[]) => items.filter((w) => !w.superseded && !w.exited_at)

// ---------- config ----------

export function normalizeConfig(raw: unknown): OpsConfig {
  const c = obj(raw)
  const configuration = c.configuration ? obj(c.configuration) : Object.fromEntries(Object.entries(c).filter(([k]) => k !== 'generated_at' && k !== 'available'))
  return { ...c, configuration } as OpsConfig
}

/**
 * The redacted config plus one worker's queues in the shape StatusCard reads,
 * so the Worker drawer can embed it (§2.6). Null when the config lacks a field
 * StatusCard dereferences, so a partial answer never breaks the drawer.
 */
export function toLightragStatus(config: OpsConfig | null | undefined, snapshot?: WorkerSnapshot | null): LightragStatus | null {
  const c = config?.configuration
  if (!c || typeof c !== 'object' || c.force_llm_summary_on_merge === undefined || c.force_llm_summary_on_merge === null) return null
  const status = {
    status: 'healthy',
    working_directory: config?.working_directory ?? '',
    input_directory: config?.input_directory ?? '',
    configuration: c,
    core_version: config?.core_version ?? undefined,
    api_version: config?.api_version ?? undefined,
    pipeline_busy: false,
    llm_queue_status: snapshot?.llm_queue_status ?? undefined,
    embedding_queue_status: snapshot?.embedding_queue_status ?? undefined,
    rerank_queue_status: snapshot?.rerank_queue_status ?? undefined
  }
  return status as unknown as LightragStatus
}

/** Config keys grouped for the viewer: LLM, embedding, rerank, storage, pipeline, query; the rest under "other". */
export const CONFIG_GROUPS: { id: string; match: (key: string) => boolean }[] = [
  { id: 'llm', match: (k) => k.startsWith('llm_') || k === 'max_async' || k === 'role_llm_config' || k === 'enable_llm_cache' || k.startsWith('enable_llm_cache') },
  { id: 'embedding', match: (k) => k.startsWith('embedding_') },
  { id: 'rerank', match: (k) => k.startsWith('rerank_') || k === 'enable_rerank' || k === 'min_rerank_score' },
  { id: 'storage', match: (k) => k.endsWith('_storage') || k === 'workspace' || k === 'storage_workspaces' },
  {
    id: 'pipeline',
    match: (k) => k.startsWith('max_parallel') || k.startsWith('summary_') || k.startsWith('force_llm') || k.startsWith('chunk') || k.includes('docling') || k.includes('document')
  },
  { id: 'query', match: (k) => k.startsWith('top_k') || k.includes('threshold') || k.includes('token') || k === 'related_chunk_number' || k === 'max_graph_nodes' || k.startsWith('cosine') }
]

export function groupConfig(configuration: Record<string, unknown>): { id: string; entries: [string, unknown][] }[] {
  const groups = new Map<string, [string, unknown][]>()
  for (const g of CONFIG_GROUPS) groups.set(g.id, [])
  groups.set('other', [])
  for (const [k, v] of Object.entries(configuration).sort(([a], [b]) => a.localeCompare(b))) {
    const g = CONFIG_GROUPS.find((x) => x.match(k))
    groups.get(g ? g.id : 'other')!.push([k, v])
  }
  return [...groups.entries()].filter(([, entries]) => entries.length).map(([id, entries]) => ({ id, entries }))
}

// ---------- detail lines ----------

const pct = (n: number) => `${Math.round(n <= 1 ? n * 100 : n)}%`

/** The soonest certificate expiry over TLS targets (`{targets: [{days_left, self_signed}]}`). */
function tlsSoonest(targets: unknown): { days: number; selfSigned: boolean } | null {
  let out: { days: number; selfSigned: boolean } | null = null
  for (const row of arr(targets)) {
    const r = obj(row)
    const days = num(r.days_left)
    if (days === null) continue
    if (!out || days < out.days) out = { days, selfSigned: !!r.self_signed }
  }
  return out
}

/**
 * The one-line summary under a dependency card, from its whitelisted detail
 * (§2.6): connections for Postgres, queue depth for vLLM, certificate days
 * for TLS and SMTP, free space for disks, buffer and drops for the recorder.
 * Returned as [i18n key, params] pairs so the view translates them; unknown
 * dependencies show nothing extra.
 */
/** The Postgres numbers the Host view shows (FR-33), from the postgres card's 5-minute extras. */
export type PgHostFacts = {
  dbSizeBytes: number | null
  connections: number | null
  maxConnections: number | null
  connectionsPct: number | null
  blocked: number | null
  blockedMaxS: number | null
  delta: { windowS: number; commits: number | null; rollbacks: number | null; deadlocks: number | null; tempBytes: number | null; cacheHitPct: number | null } | null
  at: string | null
}

export function pgHostFacts(deps: DepHealth[] | null | undefined): PgHostFacts | null {
  const pg = (deps ?? []).find((d) => d.id === 'postgres')
  if (!pg) return null
  const x = pg.detail ?? {}
  const n = (k: string) => num(x[k])
  const connections = n('connections')
  const maxConnections = n('max_connections')
  const pct = n('connections_pct') ?? (connections !== null && maxConnections ? (connections / maxConnections) * 100 : null)
  const d = obj(x.db_delta)
  const windowS = num(d.window_s)
  const facts: PgHostFacts = {
    dbSizeBytes: n('db_size_bytes'),
    connections,
    maxConnections,
    connectionsPct: pct,
    blocked: n('blocked'),
    blockedMaxS: n('blocked_max_s'),
    delta:
      windowS !== null && windowS > 0
        ? { windowS, commits: num(d.commits), rollbacks: num(d.rollbacks), deadlocks: num(d.deadlocks), tempBytes: num(d.temp_bytes), cacheHitPct: num(d.cache_hit_pct) }
        : null,
    at: str(x.extras_at)
  }
  const empty = facts.dbSizeBytes === null && connections === null && facts.blocked === null && !facts.delta
  return empty ? null : facts
}

/** Why a graph or vector store card is not probed (the server's `detail.reason`). */
const UNPROBED_REASONS = new Set(['covered_by_postgres', 'local_storage', 'not_configured', 'no_probe'])
/** Why the GPU card warns (ops/gpu.py gpu_status). */
const GPU_REASONS = new Set(['hot', 'memory_full', 'no_gpus'])

export function detailParts(d: DepHealth): [string, Record<string, unknown>][] {
  const x = d.detail ?? {}
  const out: [string, Record<string, unknown>][] = []
  const n = (k: string) => num(x[k])
  const conns = first(n('connections'), n('conns'), n('active_connections'))
  const maxConns = first(n('max_connections'), n('max_conns'))
  if (conns !== null && maxConns !== null) out.push(['operations.health.detail.conns', { used: conns, max: maxConns }])
  const blocked = first(n('blocked'), n('blocked_count'))
  if (blocked !== null && d.id === 'postgres') out.push(['operations.health.detail.blocked', { count: blocked }])
  const dbSize = n('db_size_bytes')
  if (dbSize !== null) out.push(['operations.health.detail.dbSize', { size: fmtBytes(dbSize) }])
  if (x.metrics_available === false || x.metrics === 'unavailable') out.push(['operations.health.detail.metricsUnavailable', {}])
  if (x.health_endpoint === 'auth_required') out.push(['operations.health.detail.authRequired', {}])
  const running = first(n('running'), n('num_requests_running'))
  const waiting = first(n('waiting'), n('num_requests_waiting'))
  if (running !== null || waiting !== null) out.push(['operations.health.detail.queue', { running: running ?? '–', waiting: waiting ?? '–' }])
  const kvPct = n('kv_cache_pct')
  const kv = first(n('kv_cache'), n('kv_cache_usage'), n('gpu_cache_usage'))
  if (kvPct !== null) out.push(['operations.health.detail.kvCache', { pct: `${Math.round(kvPct)}%` }])
  else if (kv !== null) out.push(['operations.health.detail.kvCache', { pct: pct(kv) }])
  const tls = tlsSoonest(x.targets)
  const days = tls?.days ?? first(n('cert_days_left'), n('days_left'), n('expires_in_days'))
  if (days !== null) out.push([(tls?.selfSigned ?? x.self_signed) ? 'operations.health.detail.certSelfSigned' : 'operations.health.detail.certDays', { days: Math.floor(days) }])
  if (x.starttls === true) out.push(['operations.health.detail.starttls', {}])
  const version = str(x.version) ?? str(x.server_version)
  if (version) out.push(['operations.health.detail.version', { version: version.split(' ')[0] }])
  const errRate = first(n('error_rate'), n('err_rate'))
  if (errRate !== null) out.push(['operations.health.detail.errorRate', { rate: pct(errRate) }])
  if (x.low_volume === true) out.push(['operations.health.detail.lowVolume', {}])
  const free = n('free_pct')
  if (free !== null) out.push(['operations.health.detail.diskFree', { pct: `${Math.round(free)}%` }])
  const used = first(n('used_pct'), n('disk_used_pct'))
  if (used !== null) out.push(['operations.health.detail.diskUsed', { pct: pct(used) }])
  const rtt = n('rtt_ms')
  if (rtt !== null) out.push(['operations.health.detail.rtt', { rtt: fmtLatency(rtt) }])
  // The GPU exporter card (D5): one line per GPU, the first two, then a count.
  const gpus = normalizeGpus(x)
  for (const g of gpus.slice(0, 2)) {
    const line = gpuLine(g)
    out.push(['operations.health.detail.gpu', { index: g.index, util: line.util, mem: line.mem, temp: line.temp }])
  }
  if (gpus.length > 2) out.push(['operations.health.detail.gpuMore', { count: gpus.length - 2 }])
  const gpuReason = d.id === 'gpu' ? str(x.reason) : null
  if (gpuReason && GPU_REASONS.has(gpuReason)) out.push([`operations.health.detail.gpuReason.${gpuReason}`, {}])
  const reason = str(x.reason)
  if (reason && UNPROBED_REASONS.has(reason)) out.push([`operations.health.detail.reason.${reason}`, {}])
  if (d.id === 'signin' && (n('sso_errors') !== null || n('lockouts') !== null)) {
    // The last 15 minutes of the auth counters (counts only, never who).
    out.push(['operations.health.detail.signin', { sso: n('sso_errors') ?? 0, token401: n('token_401') ?? 0, lockouts: n('lockouts') ?? 0 }])
  }
  if (d.id === 'recorder' && n('buffer') !== null) {
    out.push(['operations.health.detail.recorder', { buffer: n('buffer'), lost: n('lost_5m') ?? 0, dropped: n('dropped_5m') ?? 0 }])
  }
  return out
}
