import { axiosInstance } from '@/api/lightrag'
import {
  normalizeDocument,
  normalizeDocuments,
  normalizeEmailQueue,
  normalizeEvents,
  normalizeFailureDocs,
  normalizeFailures,
  normalizeFeed,
  normalizeOverview,
  normalizePage,
  normalizeStuck,
  normalizeSummary,
  normalizeWorkspace
} from './normalize'

/**
 * Operations API client. Every route lives under /admin/ops behind the
 * server's router-wide admin check; responses carry `generated_at` and errors
 * are `{detail, request_id}` (spec §3.7). Requests go through the shared
 * axios instance so auth headers, token renewal and the error mapping apply.
 *
 * The response types below follow the §3.7 catalogue. Fields that arrive in
 * later releases (run tracking, heartbeats, probes, logs) are optional, and
 * every view renders a missing field as "–" or an honest note rather than
 * failing.
 */

export const OPS_PREFIX = '/admin/ops'

export type Params = Record<string, string | number | boolean | undefined | null>

export const clean = (params: Params = {}) =>
  Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ''))

export const get = async <T>(path: string, params?: Params, signal?: AbortSignal): Promise<T> =>
  (await axiosInstance.get(`${OPS_PREFIX}${path}`, { params: clean(params), signal })).data

export const post = async <T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> =>
  (await axiosInstance.post(`${OPS_PREFIX}${path}`, body ?? {}, { signal })).data

/** Feature switches reported by /status; each view shows "not available" instead of failing. */
export type OpsFeatures = {
  durable: boolean
  runs: boolean
  logs: boolean
  probes: boolean
  alerts: boolean
  insights: boolean
  release_busy: boolean
  gpu: boolean
}

export type OpsStatus = {
  /** Ops data needs PostgreSQL KV storage; false means views stay unavailable. */
  available: boolean
  features: Partial<OpsFeatures>
  process: {
    mode: 'uvicorn' | 'gunicorn' | string
    expected_workers: number | null
    /** Live workers from heartbeats; null until worker heartbeats exist. */
    workers_live: number | null
  }
  /** May use forceful actions (release busy, reveal, export); all admins when OPS_OPERATOR_ACCOUNTS is unset. */
  operator: boolean
  version: string | null
  generated_at: string
}

type Num = number | null | undefined

/** Common to every host-wide read model. */
type Generated = {
  generated_at?: string
  /** false when the server has no PostgreSQL KV storage (NFR-8). */
  available?: boolean
}

// ---------- overview ----------

/** Every KPI may carry its own severity and scope ("host" or "worker <pid>", FR-4). */
type KpiMeta = { state?: string | null; scope?: string | null }

export type OverviewKpis = {
  active?: KpiMeta & { count: Num; longest_s?: Num; approximate?: boolean }
  backlog?: KpiMeta & { inflight: Num; delta_1h?: Num; oldest_s?: Num }
  failed_24h?: KpiMeta & { real: Num; cancelled?: Num; dup?: Num }
  stuck?: KpiMeta & { runs?: Num; docs?: Num; reservations?: Num }
  email_queue?: KpiMeta & { records: Num; workspaces?: Num; longest_pending_s?: Num; bad?: Num }
  llm_15m?: KpiMeta & { calls?: Num; error_rate?: Num; p95_ms?: Num }
  deps?: KpiMeta & { ok?: Num; total?: Num }
  workers?: KpiMeta & { live?: Num; expected?: Num }
}

/** Where an attention item or tile drills down to (rendered as a real #… link). */
export type OpsTarget = {
  tab?: string | null
  sub?: string | null
  workspace?: string | null
  doc_id?: string | null
  run_id?: string | null
}

export type AttentionItem = {
  /** Stable across polls: a new id is a new item (motion keys on the id set). */
  id: string
  rule: string
  severity: 'warning' | 'serious' | 'critical' | string
  /** Server text (counts, workspace); the rule's i18n title is the fallback. */
  title?: string | null
  detail?: string | null
  runbook?: string | null
  since?: string | null
  count?: Num
  workspace?: string | null
  target?: OpsTarget | null
}

export type OpsEvent = {
  ts: string
  kind: string
  workspace?: string | null
  detail?: Record<string, unknown> | null
  /** Where the marker came from: usage_events, audit or runs. */
  source?: string | null
}

export type TimelineBucket = { t: string; processed?: Num; failed?: Num }

export type OpsOverview = Generated & {
  /** ok | warning | serious | critical | down | unknown | maintenance */
  state: string
  worst?: string | null
  maintenance?: { until?: string | null; by?: string | null; reason?: string | null } | null
  kpis: OverviewKpis
  attention: AttentionItem[]
  /** What the attention check covered, for the "No issues. Checked …" sentence. */
  checked?: { workspaces?: Num; active?: Num; email_queue?: boolean; at?: string | null } | null
  workers?: unknown[]
  timeline?: { buckets?: TimelineBucket[]; events?: OpsEvent[] } | null
  active_top?: ActiveRun[]
  deps?: { id: string; status: string }[]
  alert_channel?: { ok: boolean | null; reason?: string | null } | null
  approximate?: boolean
}

// ---------- processing ----------

export type LiveFlags = {
  busy?: boolean | null
  destructive_busy?: boolean | null
  scanning?: boolean | null
  pending_enqueues?: Num
  request_pending?: boolean | null
  cancellation_requested?: boolean | null
  paused?: boolean | null
}

export type ActiveRun = {
  /** null for interim (approximate) rows, before run tracking. */
  run_id?: string | null
  workspace: string
  kind?: string | null
  trigger?: string | null
  /** Redacted, path-masked job name. */
  label?: string | null
  worker_id?: string | null
  pid?: Num
  worker_alive?: boolean | null
  started_at?: string | null
  heartbeat_age_s?: Num
  progress_age_s?: Num
  docs_done?: Num
  docs_total?: Num
  stage?: string | null
  /** running | paused | hung | orphaned | … (severity.ts RUN_STATES) */
  state: string
  live?: LiveFlags | null
  pause?: { status_code?: Num; slot?: string | null; role?: string | null; kind?: string | null; active?: boolean } | null
  /** Redacted last pipeline message (interim rows). */
  latest_message?: string | null
}

export type BacklogCounts = {
  pending?: Num
  parsing?: Num
  analyzing?: Num
  processing?: Num
  failed?: Num
  cancelled?: Num
  dup?: Num
}

export type ProcessingSummary = Generated & {
  backlog: BacklogCounts
  active: ActiveRun[]
  stuck: { runs?: Num; docs?: Num; reservations?: Num }
  /** Runs currently running or paused (`active` holds only the first few). */
  active_count?: Num
  email_queue?: { total?: Num; workspaces?: Num; longest_pending_s?: Num; bad?: Num; scanned_at?: string | null } | null
  pauses?: { count?: Num; by_slot_kind?: SlotKindCounts } | null
  approximate?: boolean
}

export type ActiveRuns = Generated & { runs: ActiveRun[]; approximate?: boolean; truncated?: boolean }

export type StaleDoc = {
  workspace: string
  doc_id: string
  status: string
  file_path_masked?: string | null
  updated_at?: string | null
  age_s?: Num
}

export type StuckRun = {
  workspace: string
  run_id?: string | null
  pid?: Num
  age_s?: Num
  reason?: string | null
  since?: string | null
}

export type StuckThresholds = { orphan_s?: Num; hung_min?: Num; doc_min?: Num }

export type Stuck = Generated & {
  orphaned: StuckRun[]
  hung: StuckRun[]
  busy_without_run: StuckRun[]
  reservations: StuckRun[]
  stale_docs: { items: StaleDoc[]; total: number; page?: number; page_size?: number }
  thresholds: StuckThresholds
  /** Which classes the server actually checked (only stale docs before run tracking). */
  checked_classes?: string[]
}

export type WorkspaceRow = {
  workspace: string
  owner?: string | null
  pending?: Num
  parsing?: Num
  analyzing?: Num
  processing?: Num
  failed?: Num
  cancelled?: Num
  oldest_inflight_at?: string | null
  queue_pending?: Num
  paused?: boolean | null
  last_run?: { state?: string | null; started_at?: string | null; finished_at?: string | null } | null
  state?: string | null
}

export type Page<T> = Generated & { items: T[]; page: number; page_size: number; total: number }

export type FailureGroup = {
  key: string
  count: number
  workspaces?: Num
  first_seen?: string | null
  last_seen?: string | null
  /** Redacted, path-masked sample message. */
  example?: string | null
  /** Legacy rows: what `key` is (signature, workspace or cancelled); the docs query uses it as the group. */
  kind?: string | null
}

export type Failures = Generated & {
  groups: FailureGroup[]
  excluded: { dup?: Num; cancelled?: Num }
  /** Older rows without an error kind, grouped by a normalised-message signature. */
  legacy: FailureGroup[]
  /** How legacy rows are grouped for the chosen group (signature or workspace). */
  legacy_group?: string | null
  /** False until failure kinds are recorded at the source. */
  durable?: boolean
}

export type FailureDoc = {
  workspace: string
  doc_id: string
  status?: string | null
  file_path_masked?: string | null
  updated_at?: string | null
  error_redacted?: string | null
}

export type EmailQueueItem = {
  workspace: string
  pending: number
  pending_since?: string | null
  bad?: Num
  attempts?: Num
  last_result_class?: string | null
  reason?: string | null
  last_ok_at?: string | null
  drainer?: { worker_id?: string | null; state?: string | null; backoff_until?: string | null } | null
}

export type EmailQueue = Generated & {
  total: number
  workspaces: number
  longest_pending_s?: Num
  bad?: Num
  scanned_at?: string | null
  items: EmailQueueItem[]
  page?: number
  page_size?: number
  /** Workspaces with records: what the pager counts. */
  total_items?: number
}

/** `{ "extract:budget": 3 }` or `{ extract: { budget: 3 } }`; normalised by slotKindRows(). */
export type SlotKindCounts = Record<string, number | Record<string, number>>

export type PauseItem = {
  workspace: string
  slot?: string | null
  kind?: string | null
  status_code?: Num
  byo?: boolean | null
  since?: string | null
  retry_at?: string | null
}

export type Pauses = Generated & {
  by_slot_kind: SlotKindCounts
  items: PauseItem[]
  approximate?: boolean
  page?: number
  page_size?: number
  total?: number
}

// ---------- drawers ----------

export type WorkspaceLive = LiveFlags & {
  job_name?: string | null
  job_start?: string | null
  docs?: Num
  batchs?: Num
  cur_batch?: Num
  latest_message?: string | null
  provider_rejection?: { status_code?: Num; slot?: string | null; message?: string | null; since?: string | null } | null
}

export type WorkspaceDetail = Generated & {
  workspace: string
  owner?: string | null
  loaded_in?: string[] | null
  /** null when the workspace has no live state (idle) or the Manager did not answer. */
  live?: WorkspaceLive | null
  /** "unresponsive" when the Manager timed out: SQL sections still render. */
  live_status?: 'ok' | 'idle' | 'unresponsive' | string | null
  action_in_progress?: { actor?: string | null; action?: string | null; since?: string | null } | null
  counts?: BacklogCounts & { processed?: Num }
  email_queue?: Omit<EmailQueueItem, 'workspace'> | null
  pause?: PauseItem[] | null
  recent_runs?: { run_id?: string | null; kind?: string | null; state?: string | null; started_at?: string | null; docs_done?: Num; docs_total?: Num }[]
  failures_by_kind?: { key: string; count: number }[]
  errors_1h?: Num
}

export type Feed = Generated & {
  lines: string[]
  total: number
  reset: boolean
  /** Offset to ask for next; derived from the request when the server leaves it out. */
  next_offset?: number | null
}

export type DocRow = {
  doc_id: string
  status: string
  /** A FAILED row whose error is a user cancellation (retried by the next run). */
  cancelled?: boolean
  file_path_masked?: string | null
  updated_at?: string | null
  created_at?: string | null
  chunks_count?: Num
  error_redacted?: string | null
}

export type DocStage = {
  stage: string
  started_at?: string | null
  ended_at?: string | null
  duration_ms?: Num
  /** ok | failed | running | skipped */
  state?: string | null
}

export type LlmCall = {
  ts: string
  purpose?: string | null
  operation?: string | null
  model?: string | null
  prompt_tokens?: Num
  completion_tokens?: Num
  latency_ms?: Num
  status?: string | null
  cache_hit?: boolean | null
  billing?: string | null
}

export type DocumentDetail = Generated & {
  doc: {
    doc_id?: string
    status: string
    cancelled?: boolean
    stages: DocStage[]
    file_path_masked?: string | null
    track_id?: string | null
    chunks_count?: Num
    created_at?: string | null
    updated_at?: string | null
    error_redacted?: string | null
    failure?: Record<string, unknown> | null
  }
  failures: { ts: string; run_id?: string | null; stage?: string | null; error_kind?: string | null; attempt?: Num; model?: string | null }[]
  calls: LlmCall[]
  /** False when Token Insights is not recording, so calls cannot be listed. */
  calls_available?: boolean
  logs?: { count: number | null; first: unknown[] } | null
}

// ---------- the client ----------

const enc = encodeURIComponent

/** Every read goes through a normaliser (normalize.ts), so views see one shape. */
export const opsApi = {
  status: (signal?: AbortSignal) => get<OpsStatus>('/status', undefined, signal),
  overview: async (signal?: AbortSignal) => normalizeOverview(await get('/overview', undefined, signal)),
  summary: async (signal?: AbortSignal) => normalizeSummary(await get('/processing/summary', undefined, signal)),
  active: async (signal?: AbortSignal): Promise<ActiveRuns> => {
    const data = (await get<Partial<ActiveRuns>>('/processing/active', undefined, signal)) ?? {}
    return { ...data, runs: Array.isArray(data.runs) ? data.runs : [] }
  },
  stuck: async (params: Params, signal?: AbortSignal) => normalizeStuck(await get('/processing/stuck', params, signal)),
  workspaces: async (params: Params, signal?: AbortSignal) => normalizePage<WorkspaceRow>(await get('/processing/workspaces', params, signal)),
  failures: async (params: Params, signal?: AbortSignal) => normalizeFailures(await get('/processing/failures', params, signal)),
  failureDocs: async (params: Params, signal?: AbortSignal) => normalizeFailureDocs(await get('/processing/failures/docs', params, signal)),
  emailQueue: async (params: Params, signal?: AbortSignal) => normalizeEmailQueue(await get('/email-queue', params, signal)),
  pauses: async (params: Params, signal?: AbortSignal): Promise<Pauses> => {
    const data = (await get<Partial<Pauses>>('/pauses', params, signal)) ?? {}
    return { ...data, by_slot_kind: data.by_slot_kind ?? {}, items: Array.isArray(data.items) ? data.items : [] }
  },
  workspace: async (ws: string, signal?: AbortSignal) => normalizeWorkspace(await get(`/workspaces/${enc(ws)}`, undefined, signal)),
  feed: async (ws: string, params: Params, signal?: AbortSignal) => normalizeFeed(await get(`/workspaces/${enc(ws)}/feed`, params, signal)),
  documents: async (ws: string, params: Params, signal?: AbortSignal) =>
    normalizeDocuments(await get(`/workspaces/${enc(ws)}/documents`, params, signal)),
  document: async (ws: string, docId: string, signal?: AbortSignal) =>
    normalizeDocument(await get(`/workspaces/${enc(ws)}/documents/${enc(docId)}`, undefined, signal)),
  events: async (params: Params, signal?: AbortSignal) => normalizeEvents(await get('/events', params, signal))
}

/** Pause counts as flat rows, whichever of the two server shapes arrives. */
export function slotKindRows(counts: SlotKindCounts | null | undefined): { slot: string; kind: string; count: number }[] {
  if (!counts || typeof counts !== 'object') return []
  const rows: { slot: string; kind: string; count: number }[] = []
  for (const [key, value] of Object.entries(counts)) {
    if (typeof value === 'number') {
      const at = key.indexOf(':')
      rows.push({ slot: at > 0 ? key.slice(0, at) : key, kind: at > 0 ? key.slice(at + 1) : '', count: value })
    } else if (value && typeof value === 'object') {
      for (const [kind, n] of Object.entries(value)) if (typeof n === 'number') rows.push({ slot: key, kind, count: n })
    }
  }
  return rows.sort((a, b) => b.count - a.count || a.slot.localeCompare(b.slot) || a.kind.localeCompare(b.kind))
}
