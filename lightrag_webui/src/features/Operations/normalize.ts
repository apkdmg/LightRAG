import type {
  ActiveRun,
  ActiveRuns,
  AttentionItem,
  AuditPage,
  AuditRow,
  DocFailure,
  DocFailureState,
  DocRow,
  DocStage,
  DocumentDetail,
  EmailQueue,
  FailureDoc,
  Failures,
  Feed,
  OpsEvent,
  OpsOverview,
  OpsRun,
  OpsRuns,
  OpsWorker,
  OverviewKpis,
  Page,
  PauseItem,
  ProcessingSummary,
  RunDetail,
  Stuck,
  StuckRun,
  WorkspaceDetail,
  WorkspaceLive
} from './api'
import { ageSeconds } from './format'
import { readOpsHash } from './hashState'
import { depCounts, normalizeDeps, normalizeWorker, sortWorkers } from './health/model'

/**
 * Server answers → the shapes the views read. The routes follow §3.7 but
 * spell some fields their own way (a tile's `value`, a drill `link` as a
 * hash, nested status counts, stage `id`/`duration_s`, threshold minutes);
 * every normaliser accepts both spellings, keeps unknown fields, and never
 * throws on a partial payload, so a server one release ahead or behind
 * still renders.
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

/** A drill target from a server `link` ("#/?ops=1&otab=processing&osub=stuck"). */
function targetOfLink(link: unknown): AttentionItem['target'] {
  const hash = str(link)
  if (!hash) return null
  const view = readOpsHash(hash.startsWith('#') ? hash : `#/?${hash.replace(/^[/?#]+/, '')}`)
  if (!view) return null
  const d = view.drawer
  return {
    tab: view.tab,
    sub: view.sub,
    workspace: d?.kind === 'workspace' ? d.id : d?.kind === 'document' ? d.workspace : null,
    doc_id: d?.kind === 'document' ? d.id : null,
    run_id: d?.kind === 'run' ? d.id : null,
    dep: d?.kind === 'dependency' ? d.id : null,
    worker_id: d?.kind === 'worker' ? d.id : null
  }
}

export function normalizeAttention(raw: unknown): AttentionItem[] {
  return arr(raw)
    .map((a) => obj(a))
    .filter((a) => str(a.id) || str(a.rule))
    .map((a) => ({
      ...a,
      id: String(a.id ?? a.rule),
      rule: String(a.rule ?? a.id),
      severity: String(a.severity ?? 'warning'),
      title: str(a.title),
      detail: str(a.detail),
      runbook: str(a.runbook),
      since: str(a.since),
      count: num(a.count) ?? num(obj(a.params).count),
      workspace: str(a.workspace) ?? str(obj(a.params).workspace),
      run_id: str(a.run_id) ?? str(obj(a.params).run_id),
      dep: str(a.dep) ?? str(obj(a.params).dep) ?? str(obj(a.params).dependency),
      worker_id: str(a.worker_id) ?? str(obj(a.params).worker_id),
      target: a.target ? obj(a.target) : targetOfLink(a.link)
    }))
}

/** Tiles: `value` or the named field, `llm` or `llm_15m`; an unavailable source is no tile data. */
export function normalizeKpis(raw: unknown, generatedAt: string | null): OverviewKpis {
  const k = obj(raw)
  const at = generatedAt ? Date.parse(generatedAt) : Number.NaN
  const ageOf = (iso: unknown) => (str(iso) && Number.isFinite(at) ? ageSeconds(iso as string, at) : null)
  const meta = (t: Loose) => ({ state: str(t.state), scope: str(t.scope) })
  const out: OverviewKpis = {}
  if (k.active) {
    const t = obj(k.active)
    out.active = { ...meta(t), count: first(num(t.count), num(t.value)), longest_s: num(t.longest_s), approximate: !!t.approximate }
  }
  if (k.backlog) {
    const t = obj(k.backlog)
    out.backlog = {
      ...meta(t),
      inflight: first(num(t.inflight), num(t.value)),
      delta_1h: num(t.delta_1h),
      oldest_s: first(num(t.oldest_s), ageOf(t.oldest_inflight_at))
    }
  }
  if (k.failed_24h) {
    const t = obj(k.failed_24h)
    out.failed_24h = { ...meta(t), real: first(num(t.real), num(t.value)), cancelled: num(t.cancelled), dup: num(t.dup) }
  }
  if (k.stuck) {
    const t = obj(k.stuck)
    // A null value is "not known" whatever the parts say (a failed read is not "nothing stuck").
    const unknown = t.value === null || (t.value === undefined && t.runs === undefined && t.docs === undefined)
    out.stuck = unknown ? { ...meta(t) } : { ...meta(t), runs: num(t.runs), docs: first(num(t.docs), num(t.value)), reservations: num(t.reservations) }
  }
  if (k.email_queue) {
    const t = obj(k.email_queue)
    out.email_queue = {
      ...meta(t),
      records: first(num(t.records), num(t.value)),
      workspaces: num(t.workspaces),
      longest_pending_s: num(t.longest_pending_s),
      bad: num(t.bad)
    }
  }
  const llm = obj(k.llm_15m ?? k.llm)
  if (Object.keys(llm).length) {
    if (llm.available === false) out.llm_15m = undefined
    else out.llm_15m = { ...meta(llm), calls: num(llm.calls), error_rate: num(llm.error_rate), p95_ms: num(llm.p95_ms) }
  }
  if (k.deps) {
    const t = obj(k.deps)
    out.deps = t.available === false ? { ...meta(t), ok: null, total: null } : { ...meta(t), ok: num(t.ok), total: num(t.total) }
  }
  if (k.workers) {
    const t = obj(k.workers)
    out.workers = { ...meta(t), live: num(t.live), expected: num(t.expected) }
  }
  return out
}

export function normalizeEvents(raw: unknown): OpsEvent[] {
  const list = Array.isArray(raw) ? raw : arr(obj(raw).items)
  return list
    .map((e) => obj(e))
    .filter((e) => str(e.ts) && str(e.kind))
    .map((e) => ({ ...e, ts: e.ts, kind: e.kind, workspace: str(e.workspace), detail: e.detail ? obj(e.detail) : null, source: str(e.source) }))
}

export function normalizeOverview(raw: unknown): OpsOverview {
  const o = obj(raw)
  const parsedAt = str(o.generated_at) ? Date.parse(o.generated_at) : Number.NaN
  const generatedAt = Number.isFinite(parsedAt) ? parsedAt : null
  const c = obj(o.checked)
  const timeline = obj(o.timeline)
  const deps = normalizeDeps(o.deps, generatedAt)
  const kpis = normalizeKpis(o.kpis, str(o.generated_at))
  // An older server may list dependencies without the summary tile: count them here.
  if (!kpis.deps && deps.length) kpis.deps = depCounts(deps)
  return {
    ...o,
    state: str(o.state) ?? 'unknown',
    worst: str(o.worst),
    kpis,
    attention: normalizeAttention(o.attention),
    checked: o.checked
      ? {
        workspaces: first(num(c.workspaces), num(c.workspaces_with_backlog)),
        active: first(num(c.active), num(c.active_candidates)),
        email_queue: c.email_queue === undefined ? undefined : !!c.email_queue,
        at: str(c.at)
      }
      : null,
    timeline: o.timeline ? { buckets: arr(timeline.buckets), events: Array.isArray(timeline.events) ? normalizeEvents(timeline.events) : undefined } : null,
    active_top: arr(o.active_top),
    deps,
    workers: sortWorkers(arr(o.workers).map((w) => normalizeWorker(w, generatedAt)).filter((w): w is OpsWorker => w !== null))
  } as OpsOverview
}

export function normalizeSummary(raw: unknown): ProcessingSummary {
  const s = obj(raw)
  return {
    ...s,
    backlog: obj(s.backlog),
    active: arr(s.active),
    stuck: obj(s.stuck),
    email_queue: s.email_queue ? obj(s.email_queue) : null,
    pauses: s.pauses ? obj(s.pauses) : null
  } as ProcessingSummary
}

// ---------- runs ----------

/** Outcomes a finished run row can have (§3.5 LIGHTRAG_OPS_RUNS.status). */
export const RUN_OUTCOMES = ['succeeded', 'failed', 'cancelled', 'paused', 'orphaned'] as const

const STATUS_ALIASES: Record<string, string> = {
  ok: 'succeeded',
  success: 'succeeded',
  completed: 'succeeded',
  done: 'succeeded',
  error: 'failed',
  canceled: 'cancelled'
}

const canonicalStatus = (raw: unknown): string | null => {
  const s = str(raw)?.toLowerCase() ?? null
  return s ? (STATUS_ALIASES[s] ?? s) : null
}

/**
 * What a run shows as: a finished run shows its outcome; a running one shows
 * the server's live classification (running, hung, paused, unresponsive,
 * orphaned), or "running" when the server sent none.
 */
export function runDisplayState(status: unknown, state: unknown): string {
  const st = canonicalStatus(status)
  const live = canonicalStatus(state)
  if (st && st !== 'running') return st
  return live ?? st ?? 'unknown'
}

const seconds = (from: unknown, to: unknown): number | null => {
  const a = str(from) ? Date.parse(from as string) : Number.NaN
  const b = str(to) ? Date.parse(to as string) : Number.NaN
  return Number.isFinite(a) && Number.isFinite(b) ? Math.max(0, (b - a) / 1000) : null
}

/** One run row; null when it has no id (nothing to open). A missing workspace takes `workspace`. */
export function normalizeRun(raw: unknown, workspace?: string): OpsRun | null {
  const r = obj(raw)
  const runId = str(r.run_id) ?? str(r.id)
  const ws = str(r.workspace) ?? workspace ?? null
  if (!runId || !ws) return null
  const status = canonicalStatus(r.status) ?? (r.finished_at ? 'unknown' : 'running')
  return {
    ...r,
    run_id: runId,
    workspace: ws,
    kind: str(r.kind),
    trigger: str(r.trigger),
    status,
    state: runDisplayState(status, r.state),
    was_orphaned: !!r.was_orphaned,
    worker_id: str(r.worker_id),
    pid: num(r.pid),
    worker_alive: typeof r.worker_alive === 'boolean' ? r.worker_alive : null,
    started_at: str(r.started_at),
    heartbeat_at: str(r.heartbeat_at),
    progress_at: str(r.progress_at),
    finished_at: str(r.finished_at),
    heartbeat_age_s: num(r.heartbeat_age_s),
    progress_age_s: num(r.progress_age_s),
    duration_s: first(num(r.duration_s), num(r.duration_ms) === null ? null : (num(r.duration_ms) as number) / 1000, seconds(r.started_at, r.finished_at)),
    docs_total: num(r.docs_total),
    docs_done: num(r.docs_done),
    docs_failed: num(r.docs_failed),
    stage: str(r.stage),
    label: str(r.label),
    error_kind: str(r.error_kind),
    error_summary: str(r.error_summary),
    request_id: str(r.request_id),
    live: r.live ? (obj(r.live) as OpsRun['live']) : null,
    pause: r.pause ? (obj(r.pause) as OpsRun['pause']) : null
  }
}

/** A keyset cursor as the server sent it: a string (or number), or `{ts|started_at}` from an object. */
const cursorOf = (v: unknown): string | null => {
  if (typeof v === 'number' && Number.isFinite(v)) return String(v)
  if (str(v)) return v as string
  const o = obj(v)
  return str(o.ts) ?? str(o.started_at) ?? null
}

export function normalizeRuns(raw: unknown): OpsRuns {
  const r = obj(raw)
  const list = Array.isArray(raw) ? raw : arr(r.items ?? r.runs)
  return {
    ...r,
    items: list.map((i) => normalizeRun(i)).filter((i): i is OpsRun => i !== null),
    next_before: cursorOf(r.next_before)
  } as OpsRuns
}

export function normalizeDocFailure(raw: unknown): DocFailure | null {
  const f = obj(raw)
  const ts = str(f.ts) ?? str(f.failed_at) ?? str(f.at)
  if (!ts) return null
  return {
    ...f,
    ts,
    workspace: str(f.workspace),
    doc_id: str(f.doc_id),
    track_id: str(f.track_id),
    run_id: str(f.run_id),
    stage: str(f.stage),
    error_kind: str(f.error_kind),
    http_status: num(f.http_status),
    exc_type: str(f.exc_type),
    message: str(f.message) ?? str(f.message_redacted) ?? str(f.error),
    attempt: num(f.attempt),
    model: str(f.model),
    file_path_masked: str(f.file_path_masked) ?? str(f.file)
  }
}

const docFailures = (v: unknown) => arr(v).map(normalizeDocFailure).filter((f): f is DocFailure => f !== null)

/** `metadata.failure`; null when it carries nothing worth showing. */
export function normalizeFailureState(raw: unknown): DocFailureState | null {
  const f = obj(raw)
  if (!Object.keys(f).length) return null
  const out: DocFailureState = {
    stage: str(f.stage),
    error_kind: str(f.error_kind),
    http_status: num(f.http_status),
    attempt: num(f.attempt),
    first_failed_at: str(f.first_failed_at),
    last_failed_at: str(f.last_failed_at) ?? str(f.failed_at),
    last_run_id: str(f.last_run_id) ?? str(f.run_id),
    model: str(f.model)
  }
  return out.error_kind || out.stage || out.attempt !== null ? out : null
}

export function normalizeRunDetail(raw: unknown): RunDetail {
  const d = obj(raw)
  // Some servers return the run's fields at the top level.
  const run = normalizeRun(d.run ?? d)
  return {
    ...d,
    run,
    failures: docFailures(d.failures),
    log_count: num(d.log_count)
  } as RunDetail
}

/** /processing/active: durable run rows (or interim rows) with one display state each. */
export function normalizeActive(raw: unknown): ActiveRuns {
  const a = obj(raw)
  const runs = arr(a.runs).map((r): ActiveRun => {
    const row = obj(r)
    return {
      ...row,
      workspace: String(row.workspace ?? ''),
      run_id: str(row.run_id),
      status: canonicalStatus(row.status),
      state: row.status ? runDisplayState(row.status, row.state) : (canonicalStatus(row.state) ?? 'unknown')
    } as ActiveRun
  })
  return { ...a, runs: runs.filter((r) => r.workspace) } as ActiveRuns
}

/**
 * One stuck item as the server sends it. Reservations name their flag by
 * `kind` (scanning | pending_enqueues) and `count` on older servers; the
 * reason is a code (`reason_code`, `reason` on older servers) that the view
 * translates, never text to print.
 */
const stuckRuns = (v: unknown): StuckRun[] =>
  arr(v)
    .map((r) => obj(r))
    .filter((r) => str(r.workspace))
    .map((r) => {
      const kind = str(r.kind)
      const state = str(r.state)
      return {
        ...r,
        workspace: r.workspace,
        run_id: str(r.run_id),
        pid: first(num(r.pid), num(r.owner_pid)),
        worker_id: str(r.worker_id) ?? str(r.owner_worker_id),
        age_s: num(r.age_s),
        reason_code: str(r.reason_code) ?? str(r.reason),
        since: str(r.since) ?? str(r.started_at),
        kind,
        label: str(r.label),
        state,
        owner_unresponsive: r.owner_unresponsive === true || state === 'unresponsive',
        heartbeat_age_s: num(r.heartbeat_age_s),
        progress_age_s: num(r.progress_age_s),
        worker_heartbeat_age_s: num(r.worker_heartbeat_age_s),
        scanning: typeof r.scanning === 'boolean' ? r.scanning : kind === 'scanning' ? true : kind === 'pending_enqueues' ? false : null,
        pending_enqueues: first(num(r.pending_enqueues), kind === 'pending_enqueues' ? num(r.count) : null),
        confirmed_by: str(r.confirmed_by)
      }
    })

export function normalizeStuck(raw: unknown): Stuck {
  const s = obj(raw)
  const th = obj(s.thresholds)
  const docs = obj(s.stale_docs)
  return {
    ...s,
    orphaned: stuckRuns(s.orphaned),
    hung: stuckRuns(s.hung),
    busy_without_run: stuckRuns(s.busy_without_run),
    reservations: stuckRuns(s.reservations),
    stale_docs: {
      ...docs,
      items: arr(docs.items).map((d) => ({ ...d, updated_at: str(d.updated_at) ?? str(d.since), file_path_masked: str(d.file_path_masked) ?? str(d.file) })),
      total: num(docs.total) ?? arr(docs.items).length,
      page: num(docs.page) ?? undefined,
      page_size: num(docs.page_size) ?? undefined
    },
    thresholds: {
      orphan_s: first(num(th.orphan_s), num(th.orphan_seconds)),
      hung_min: first(num(th.hung_min), num(th.hung_minutes)),
      doc_min: first(num(th.doc_min), num(th.doc_minutes))
    },
    checked_classes: Array.isArray(s.checked_classes) ? s.checked_classes.filter((c: unknown) => typeof c === 'string') : undefined,
    checked: s.checked ? obj(s.checked) : null
  } as Stuck
}

/** Durable groups may name their key after the group (`error_kind`, `stage`, `model`, `workspace`). */
const groupRows = (v: unknown) =>
  arr(v)
    .map((g) => obj(g))
    .map((g) => ({ ...g, key: String(g.key ?? g.error_kind ?? g.stage ?? g.model ?? g.workspace ?? ''), count: num(g.count) ?? 0 }))

export function normalizeFailures(raw: unknown): Failures {
  const f = obj(raw)
  return { ...f, groups: groupRows(f.groups), legacy: groupRows(f.legacy), excluded: obj(f.excluded) } as Failures
}

const docFields = <T extends Loose>(d: T) => ({
  ...d,
  file_path_masked: str(d.file_path_masked) ?? str(d.file),
  error_redacted: str(d.error_redacted) ?? str(d.error) ?? str(d.message)
})

export function normalizePage<T>(raw: unknown, mapItem: (item: Loose) => T = (i) => i as T): Page<T> {
  const p = obj(raw)
  const items = arr(p.items).map((i) => mapItem(obj(i)))
  return { ...p, items, page: num(p.page) ?? 1, page_size: num(p.page_size) ?? Math.max(items.length, 1), total: num(p.total) ?? items.length } as Page<T>
}

export const normalizeFailureDocs = (raw: unknown) =>
  normalizePage<FailureDoc>(raw, (d) => ({ ...docFields(d), updated_at: str(d.updated_at) ?? str(d.ts), http_status: num(d.http_status), attempt: num(d.attempt) }) as FailureDoc)
export const normalizeDocuments = (raw: unknown) => normalizePage<DocRow>(raw, (d) => docFields(d) as DocRow)

/** One workspace's queue state; the backoff deadline (FR-32) may be spelled after the drainer's sleep. */
const drains = (raw: unknown) => {
  if (!raw || typeof raw !== 'object') return null
  const d = obj(raw)
  return { ok: num(d.ok) ?? 0, failed: first(num(d.failed), num(d.error)) ?? 0, slot_busy: num(d.slot_busy) ?? 0 }
}

const queueItem = (raw: unknown) => {
  const i = obj(raw)
  return {
    ...i,
    backoff_until: str(i.backoff_until) ?? str(i.next_attempt_at) ?? str(i.backoff_deadline),
    last_attempt_at: str(i.last_attempt_at),
    drains_1h: drains(i.drains_1h),
    drainer: i.drainer ? obj(i.drainer) : null
  }
}

export function normalizeEmailQueue(raw: unknown): EmailQueue {
  const q = obj(raw)
  return {
    ...q,
    total: num(q.total) ?? 0,
    workspaces: num(q.workspaces) ?? 0,
    items: arr(q.items).map(queueItem),
    // The pager counts workspaces with records; the routes call it total_items.
    total_items: first(num(q.total_items), num(q.workspaces)) ?? 0
  } as EmailQueue
}

/** The live pause on a workspace (`provider_rejection`) as a pause row. */
function pauseOfLive(workspace: string, rejection: Loose): PauseItem | null {
  if (!rejection || rejection.active === false) return null
  if (!Object.keys(rejection).length) return null
  return {
    workspace,
    slot: str(rejection.slot) ?? str(rejection.role),
    kind: str(rejection.kind),
    status_code: num(rejection.status_code),
    since: str(rejection.since),
    retry_at: str(rejection.retry_at),
    byo: typeof rejection.byo === 'boolean' ? rejection.byo : null
  }
}

export function normalizeWorkspace(raw: unknown): WorkspaceDetail {
  const w = obj(raw)
  const workspace = String(w.workspace ?? '')
  let live: WorkspaceLive | null = null
  if (w.live) {
    const l = obj(w.live)
    const rejection = l.provider_rejection ? obj(l.provider_rejection) : null
    live = {
      ...l,
      run_id: str(l.run_id),
      owner_worker_id: str(l.owner_worker_id),
      owner_pid: num(l.owner_pid),
      job_name: str(l.job_name) ?? str(l.job),
      latest_message: str(l.latest_message) ?? str(l.latest_message_redacted),
      paused: typeof l.paused === 'boolean' ? l.paused : !!rejection && rejection.active !== false,
      provider_rejection: rejection
        ? { ...rejection, slot: str(rejection.slot) ?? str(rejection.role), status_code: num(rejection.status_code), since: str(rejection.since) }
        : null
    } as WorkspaceLive
  }
  const c = obj(w.counts)
  const counts = w.counts ? { ...obj(c.statuses), ...c, statuses: undefined } : undefined
  const fromLive = live?.provider_rejection ? pauseOfLive(workspace, live.provider_rejection as Loose) : null
  return {
    ...w,
    workspace,
    live,
    live_status: str(w.live_status) ?? (w.manager === 'unresponsive' ? 'unresponsive' : w.idle ? 'idle' : 'ok'),
    counts,
    email_queue: w.email_queue ? queueItem(w.email_queue) : null,
    pause: Array.isArray(w.pause) ? w.pause : fromLive ? [fromLive] : [],
    recent_runs: arr(w.recent_runs)
      .map((r) => normalizeRun(r, workspace))
      .filter((r): r is OpsRun => r !== null),
    failures_by_kind: arr(w.failures_by_kind)
      .map((f) => obj(f))
      .map((f) => ({ key: String(f.key ?? f.error_kind ?? 'unknown'), count: num(f.count) ?? 0, http_status: num(f.http_status) }))
      .filter((f) => f.count > 0)
  } as WorkspaceDetail
}

export function normalizeFeed(raw: unknown): Feed {
  const f = obj(raw)
  const lines = arr<unknown>(f.lines).filter((l): l is string => typeof l === 'string')
  return { ...f, lines, total: num(f.total) ?? lines.length, reset: !!f.reset, next_offset: num(f.next_offset) } as Feed
}

/** Stages: `{id, duration_s}` or `{stage, duration_ms}`. */
export function normalizeStage(raw: Loose): DocStage {
  const s = obj(raw)
  const seconds = num(s.duration_s)
  return {
    stage: String(s.stage ?? s.id ?? 'unknown'),
    started_at: str(s.started_at),
    ended_at: str(s.ended_at),
    duration_ms: first(num(s.duration_ms), seconds === null ? null : seconds * 1000),
    state: str(s.state)
  }
}

export function normalizeDocument(raw: unknown): DocumentDetail {
  const d = obj(raw)
  const doc = obj(d.doc)
  return {
    ...d,
    doc: {
      ...docFields(doc),
      status: String(doc.status ?? 'unknown'),
      stages: arr(doc.stages).map(normalizeStage),
      failure: normalizeFailureState(doc.failure)
    },
    failures: docFailures(d.failures),
    calls: arr(d.calls),
    calls_available: typeof d.calls_available === 'boolean' ? d.calls_available : undefined
  } as DocumentDetail
}

/** An id as text: the audit table's bigint id may arrive as a number. */
const idText = (v: unknown): string | null => (typeof v === 'number' && Number.isFinite(v) ? String(v) : str(v))

/**
 * One audit row (§3.5 LIGHTRAG_OPS_AUDIT). The target is the row's own
 * `target`, else its workspace; a `params.target` from an older writer is
 * accepted too. `params` itself is never shown (it may name fields).
 */
export function normalizeAuditRow(raw: unknown): AuditRow | null {
  const r = obj(raw)
  const id = idText(r.id)
  const action = str(r.action)
  if (!id || !action) return null
  return {
    id,
    ts: str(r.ts),
    actor: str(r.actor),
    auth_kind: str(r.auth_kind),
    impersonated_by: str(r.impersonated_by),
    action,
    workspace: str(r.workspace),
    target: str(r.target) ?? str(obj(r.params).target),
    reason: str(r.reason),
    phase: str(r.phase),
    outcome: str(r.outcome),
    request_id: str(r.request_id)
  }
}

/** GET /audit: offset pages or a keyset cursor, whichever the server sends. */
export function normalizeAuditPage(raw: unknown): AuditPage {
  const p = obj(raw)
  const items = arr(p.items ?? p.rows)
    .map(normalizeAuditRow)
    .filter((r): r is AuditRow => r !== null)
  return {
    ...p,
    items,
    page: num(p.page),
    page_size: num(p.page_size),
    total: num(p.total),
    next_before: idText(p.next_before)
  } as AuditPage
}
