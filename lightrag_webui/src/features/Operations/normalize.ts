import type {
  AttentionItem,
  DocRow,
  DocStage,
  DocumentDetail,
  EmailQueue,
  FailureDoc,
  Failures,
  Feed,
  OpsEvent,
  OpsOverview,
  OpsWorker,
  OverviewKpis,
  Page,
  PauseItem,
  ProcessingSummary,
  Stuck,
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

export function normalizeStuck(raw: unknown): Stuck {
  const s = obj(raw)
  const th = obj(s.thresholds)
  const docs = obj(s.stale_docs)
  return {
    ...s,
    orphaned: arr(s.orphaned),
    hung: arr(s.hung),
    busy_without_run: arr(s.busy_without_run),
    reservations: arr(s.reservations),
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
    }
  } as Stuck
}

export function normalizeFailures(raw: unknown): Failures {
  const f = obj(raw)
  return { ...f, groups: arr(f.groups), legacy: arr(f.legacy), excluded: obj(f.excluded) } as Failures
}

const docFields = <T extends Loose>(d: T) => ({
  ...d,
  file_path_masked: str(d.file_path_masked) ?? str(d.file),
  error_redacted: str(d.error_redacted) ?? str(d.error)
})

export function normalizePage<T>(raw: unknown, mapItem: (item: Loose) => T = (i) => i as T): Page<T> {
  const p = obj(raw)
  const items = arr(p.items).map((i) => mapItem(obj(i)))
  return { ...p, items, page: num(p.page) ?? 1, page_size: num(p.page_size) ?? Math.max(items.length, 1), total: num(p.total) ?? items.length } as Page<T>
}

export const normalizeFailureDocs = (raw: unknown) => normalizePage<FailureDoc>(raw, (d) => docFields(d) as FailureDoc)
export const normalizeDocuments = (raw: unknown) => normalizePage<DocRow>(raw, (d) => docFields(d) as DocRow)

export function normalizeEmailQueue(raw: unknown): EmailQueue {
  const q = obj(raw)
  return {
    ...q,
    total: num(q.total) ?? 0,
    workspaces: num(q.workspaces) ?? 0,
    items: arr(q.items),
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
    email_queue: w.email_queue ? obj(w.email_queue) : null,
    pause: Array.isArray(w.pause) ? w.pause : fromLive ? [fromLive] : [],
    recent_runs: arr(w.recent_runs),
    failures_by_kind: arr(w.failures_by_kind)
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
    doc: { ...docFields(doc), status: String(doc.status ?? 'unknown'), stages: arr(doc.stages).map(normalizeStage) },
    failures: arr(d.failures),
    calls: arr(d.calls),
    calls_available: typeof d.calls_available === 'boolean' ? d.calls_available : undefined
  } as DocumentDetail
}
