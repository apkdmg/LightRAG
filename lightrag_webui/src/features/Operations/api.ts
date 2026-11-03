import { axiosInstance } from '@/api/lightrag'
import {
  normalizeActive,
  normalizeDocument,
  normalizeDocuments,
  normalizeEmailQueue,
  normalizeEvents,
  normalizeFailureDocs,
  normalizeFailures,
  normalizeFeed,
  normalizeOverview,
  normalizePage,
  normalizeRunDetail,
  normalizeRuns,
  normalizeStuck,
  normalizeSummary,
  normalizeWorkspace
} from './normalize'
import { normalizeConfig, normalizeDepDetail, normalizeHealth, normalizeProbeResult, normalizeSmtpTest, normalizeWorkerHistory, normalizeWorkers } from './health/model'

/**
 * Operations API client. Every route lives under /admin/ops behind the
 * server's router-wide admin check; responses carry `generated_at` and errors
 * are `{detail, request_id}` (spec §3.7). Requests go through the shared
 * axios instance so auth headers, token renewal and the error mapping apply.
 *
 * The response types below follow the §3.7 catalogue. Fields that arrive in
 * later releases (logs, alerts, actions) are optional, and fields a server
 * may leave out (run tracking turned off with OPS_RUN_TRACKING=false, a
 * server one release behind) render as "–" or an honest note rather than
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
  dep?: string | null
  worker_id?: string | null
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
  /** The run the item is about (hung and orphaned runs). */
  run_id?: string | null
  /** The dependency or worker the item is about (rules on probes and heartbeats). */
  dep?: string | null
  worker_id?: string | null
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
  /** Worker rows from heartbeats (same shape as /workers items). */
  workers?: OpsWorker[]
  timeline?: { buckets?: TimelineBucket[]; events?: OpsEvent[] } | null
  active_top?: ActiveRun[]
  /** Dependency summary from the latest probe cycle (same shape as /health deps, strip optional). */
  deps?: DepHealth[]
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
  /** null for interim (approximate) rows: run tracking off, or a server one release behind. */
  run_id?: string | null
  workspace: string
  kind?: string | null
  trigger?: string | null
  /** Redacted, path-masked job name. */
  label?: string | null
  worker_id?: string | null
  pid?: Num
  /** false once the OS confirmed the owner process is gone; null when not known. */
  worker_alive?: boolean | null
  started_at?: string | null
  heartbeat_age_s?: Num
  /** Age of the owner worker's last heartbeat (what "unresponsive" is about). */
  worker_heartbeat_age_s?: Num
  progress_age_s?: Num
  docs_done?: Num
  docs_total?: Num
  docs_failed?: Num
  /** The pipeline stage the run is in (parsing, analyzing, extract, merge, …). */
  stage?: string | null
  /** running | cancelling | paused | hung | unresponsive | orphaned | … (severity.ts RUN_STATES) */
  state: string
  /** The durable row's status (running, paused, orphaned, …); `state` is what to show. */
  status?: string | null
  live?: LiveFlags | null
  pause?: { status_code?: Num; slot?: string | null; role?: string | null; kind?: string | null; active?: boolean } | null
  /** Redacted last pipeline message (interim rows). */
  latest_message?: string | null
}

/**
 * One row of LIGHTRAG_OPS_RUNS (FR-17): a pipeline run, email batch, delete,
 * clear, scan, startup resume or admin action, with its outcome. `state` is
 * what the console shows: the outcome once finished, else the live
 * classification (running, paused, hung, unresponsive, orphaned).
 */
export type OpsRun = {
  run_id: string
  workspace: string
  kind?: string | null
  trigger?: string | null
  /** running | succeeded | failed | cancelled | paused | orphaned */
  status: string
  state: string
  /** The run was marked orphaned and its owner later finished it (FR-17). */
  was_orphaned?: boolean
  worker_id?: string | null
  pid?: Num
  worker_alive?: boolean | null
  started_at?: string | null
  heartbeat_at?: string | null
  progress_at?: string | null
  finished_at?: string | null
  heartbeat_age_s?: Num
  progress_age_s?: Num
  /** Seconds from start to finish (or to now while running). */
  duration_s?: Num
  docs_total?: Num
  docs_done?: Num
  docs_failed?: Num
  stage?: string | null
  /** Redacted, path-masked job name. */
  label?: string | null
  error_kind?: string | null
  /** Redacted. */
  error_summary?: string | null
  request_id?: string | null
  /** Live pipeline flags for a running run, when the server read them. */
  live?: LiveFlags | null
  pause?: ActiveRun['pause']
}

/** Run history, newest first, keyset-paged by `next_before` (null on the last page). */
export type OpsRuns = Generated & { items: OpsRun[]; next_before: string | null; enabled?: boolean }

/**
 * One classified document failure (LIGHTRAG_OPS_DOC_FAILURES, §3.5): which
 * stage failed, why (`error_kind`), on which model and attempt. The message
 * is redacted and path-masked on the server.
 */
export type DocFailure = {
  ts: string
  workspace?: string | null
  doc_id?: string | null
  track_id?: string | null
  run_id?: string | null
  stage?: string | null
  error_kind?: string | null
  http_status?: Num
  exc_type?: string | null
  message?: string | null
  attempt?: Num
  model?: string | null
  file_path_masked?: string | null
}

/** The doc's current failure (doc-status `metadata.failure`), carried over until it is processed. */
export type DocFailureState = {
  stage?: string | null
  error_kind?: string | null
  http_status?: Num
  attempt?: Num
  first_failed_at?: string | null
  last_failed_at?: string | null
  last_run_id?: string | null
  model?: string | null
}

export type RunDetail = Generated & {
  /** null only when the server answered without a run (treated as not found). */
  run: OpsRun | null
  /** Documents that failed in this run (at most 200), newest first. */
  failures: DocFailure[]
  /** Log lines carrying this run_id; null until the log viewer ships. */
  log_count: Num
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
  worker_id?: string | null
  age_s?: Num
  /**
   * Why the row is stuck, as a code the view translates (operations.stuck.why.*):
   * busy without a run: no_run | run_unknown | run_finished; reservations:
   * owner_dead | owner_unknown | no_owner | held. Never shown as-is.
   */
  reason_code?: string | null
  since?: string | null
  kind?: string | null
  label?: string | null
  /** A run's live classification; `unresponsive` for a hung-class run whose owner stopped beating. */
  state?: string | null
  /** The run's owner worker stopped sending heartbeats (its process still exists). */
  owner_unresponsive?: boolean | null
  heartbeat_age_s?: Num
  progress_age_s?: Num
  worker_heartbeat_age_s?: Num
  /** Reservations: which flag is held without an owner. */
  scanning?: boolean | null
  pending_enqueues?: Num
  /** Orphaned: how the owner's death was confirmed (pid_gone, create_time, exit_marker, exited, superseded, host_gone). */
  confirmed_by?: string | null
}

export type StuckThresholds = { orphan_s?: Num; hung_min?: Num; doc_min?: Num }

/** What the stuck check covered, for the "Nothing stuck. Checked …" sentence. */
export type StuckChecked = { workspaces?: Num; runs?: Num; candidates?: Num; at?: string | null }

export type Stuck = Generated & {
  orphaned: StuckRun[]
  hung: StuckRun[]
  busy_without_run: StuckRun[]
  reservations: StuckRun[]
  stale_docs: { items: StaleDoc[]; total: number; page?: number; page_size?: number }
  thresholds: StuckThresholds
  /** Which classes the server actually checked (only stale docs while run tracking is off). */
  checked_classes?: string[]
  checked?: StuckChecked | null
  /** When the reconcile pass behind the run classes last ran. */
  reconciled_at?: string | null
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
  /** Durable error_kind groups: the provider HTTP status most rows carried (402, 429, …). */
  http_status?: Num
}

export type Failures = Generated & {
  groups: FailureGroup[]
  excluded: { dup?: Num; cancelled?: Num }
  /** Older rows without an error kind, grouped by a normalised-message signature. */
  legacy: FailureGroup[]
  /** How legacy rows are grouped for the chosen group (signature or workspace). */
  legacy_group?: string | null
  /** False when failure kinds are not recorded (run tracking off, or a server one release behind). */
  durable?: boolean
}

export type FailureDoc = {
  workspace: string
  doc_id: string
  status?: string | null
  file_path_masked?: string | null
  updated_at?: string | null
  error_redacted?: string | null
  /** Durable groups: the classified failure. */
  ts?: string | null
  run_id?: string | null
  stage?: string | null
  error_kind?: string | null
  http_status?: Num
  attempt?: Num
  model?: string | null
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
  /** When the drainer last tried (KV resync:<ws>), and the backoff deadline before its next try (FR-32). */
  last_attempt_at?: string | null
  backoff_until?: string | null
  /** Drain outcomes of the last hour (FR-15); null when the server does not know. */
  drains_1h?: { ok: number; failed: number; slot_busy: number } | null
  /** The live drainer for this workspace, from worker heartbeats. */
  drainer?: { worker_id?: string | null; pid?: Num; state?: string | null; backoff_until?: string | null } | null
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
  /** Owner keys written by run tracking: which run and process hold `busy`. */
  run_id?: string | null
  owner_worker_id?: string | null
  owner_pid?: Num
  heartbeat_at?: string | null
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
  /** The latest five runs, newest first. */
  recent_runs?: OpsRun[]
  /** Classified failures in the last 7 days, most frequent first. */
  failures_by_kind?: { key: string; count: number; http_status?: Num }[]
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
    /** The current classified failure; null when the doc is not failed or failed before run tracking. */
    failure?: DocFailureState | null
  }
  /** Failure history for this doc (attempts), newest first. */
  failures: DocFailure[]
  calls: LlmCall[]
  /** False when Token Insights is not recording, so calls cannot be listed. */
  calls_available?: boolean
  logs?: { count: number | null; first: unknown[] } | null
}

// ---------- health and workers ----------

/** A probe outcome (§3.5 LIGHTRAG_OPS_PROBES.status); `unprobed` is a dependency nothing checks (etcd, MinIO). */
export type ProbeStatus = 'ok' | 'warn' | 'down' | 'unknown' | 'unprobed'

/** One half-hour cell of the 48-cell (24 h) availability strip. */
export type StripCell = {
  /** Slot start (epoch ms): from the server, or counted back from the latest cycle. */
  start: number | null
  status: Exclude<ProbeStatus, 'unprobed'>
  /** Probe results in the slot, when the server counts them. */
  total: number | null
  failed: number | null
  slow: number | null
}

export type DepHealth = {
  id: string
  /** Display name from the server ("Neo4j"); the i18n name of the id otherwise. */
  name?: string | null
  /** What it is (graph, vector, embed, …): picks the topology node for store-agnostic ids. */
  kind?: string | null
  /** Where it is checked ("neo4j:7687"), without credentials. */
  target?: string | null
  status: ProbeStatus
  /** The server has no result for it yet. */
  stale?: boolean
  latency_ms?: Num
  /** When it was last checked (the cycle that produced `status`). */
  checked_at?: string | null
  last_ok_at?: string | null
  consecutive_failures?: Num
  error_class?: string | null
  /** Dependency-specific safe detail (whitelisted on the server; URLs carry no userinfo). */
  detail: Record<string, unknown>
  /** 48 half-hour cells, oldest first; empty when the server sent none (the Overview summary). */
  strip: StripCell[]
  /** Seconds between checks: 30 for most, 900 for SMTP and TLS, 300 for the IdP (§2.13). */
  interval_s?: Num
  /** Health is read from real traffic (the LLM provider), so there is nothing to re-check. */
  passive?: boolean
}

export type OpsHealth = Generated & {
  /** The latest probe cycle: when, how long, and which worker (lease holder) ran it. */
  cycle: { at: string | null; duration_ms: Num; holder: string | null; holder_pid: Num; interval_s: Num } | null
  deps: DepHealth[]
  /** False when OPS_PROBES_ENABLED is off on the server. */
  enabled?: boolean
}

export type ProbeResult = {
  ts: string
  status: ProbeStatus
  latency_ms?: Num
  error_class?: string | null
  detail?: Record<string, unknown> | null
}

/**
 * One blocked or blocking Postgres backend: durations, statement type and a
 * query fingerprint and length. Never query text or parameters (§2.6).
 */
export type PgBlocking = {
  pid: Num
  blocked_by: number[]
  duration_s: Num
  state?: string | null
  statement_type?: string | null
  fingerprint?: string | null
  query_length?: Num
  wait_event?: string | null
}

export type DepDetail = Generated & {
  id: string
  /** The dependency's current card, when the server includes it. */
  dep?: DepHealth | null
  latency_series: { t: number; latency_ms: number | null; status: ProbeStatus }[]
  recent: ProbeResult[]
  detail: Record<string, unknown>
  /** Postgres only: blocked and blocking backends. */
  blocking: PgBlocking[]
  error_classes: { key: string; count: number }[]
}

export type QueueCounts = { running: Num; queued: Num; in_flight?: Num }

/** Whitelisted per-worker numbers from the heartbeat (§3.6). Missing ones read "–". */
export type WorkerSnapshot = {
  rss: Num
  cpu_pct: Num
  fds: Num
  threads: Num
  instances: Num
  loop_lag_ms: Num
  manager_rtt_ms: Num
  /** local | ok | unresponsive (multi-worker: the shared-state Manager). */
  manager_state?: string | null
  pool: { size: Num; idle: Num; max: Num; wait_ms: Num } | null
  queues: { llm?: QueueCounts | null; embed?: QueueCounts | null; rerank?: QueueCounts | null }
  recorder: { buffer: Num; last_flush_ok_at: string | null; flush_failures: Num; lost_rows: Num; dropped_events: Num } | null
  log_sink: { buffer: Num; dropped: Num; rate_limited: Num } | null
  drainers: { workspace: string; state?: string | null; backoff_until?: string | null }[]
  /** Per-role queues in the StatusCard shape, when the heartbeat carries them. */
  llm_queue_status?: Record<string, unknown> | null
  embedding_queue_status?: Record<string, unknown> | null
  rerank_queue_status?: Record<string, unknown> | null
}

export type OpsWorker = {
  worker_id: string
  pid: Num
  host?: string | null
  mode?: string | null
  version?: string | null
  started_at: string | null
  heartbeat_age_s: Num
  alive: boolean | null
  superseded: boolean
  exited_at: string | null
  leases_held: string[]
  /** The worker that answered this request. */
  self?: boolean
  snapshot: WorkerSnapshot
}

/** Container and host numbers (§3.6): cgroup v2 memory and CPU, load average (host-wide), disk use. */
export type HostInfo = {
  mem_used: Num
  mem_limit: Num
  cpu_pct: Num
  load: number[] | null
  disks: { path: string; used: Num; total: Num }[]
  sampled_at?: string | null
}

export type OpsWorkers = Generated & {
  items: OpsWorker[]
  host: HostInfo | null
  mode?: string | null
  expected?: Num
  live?: Num
}

export type WorkerSample = { t: number; rss: number | null; cpu: number | null; loop_lag: number | null; pool_wait: number | null }

/** `LightragStatus.configuration` with URLs stripped of userinfo and secrets redacted on the server. */
export type OpsConfig = Generated & {
  configuration: Record<string, unknown>
  working_directory?: string | null
  input_directory?: string | null
  core_version?: string | null
  api_version?: string | null
}

export type SmtpTestResult = { ok: boolean; error_class?: string | null; latency_ms?: Num; detail?: Record<string, unknown> | null }

// ---------- the client ----------

const enc = encodeURIComponent

/** Every read goes through a normaliser (normalize.ts), so views see one shape. */
export const opsApi = {
  status: (signal?: AbortSignal) => get<OpsStatus>('/status', undefined, signal),
  overview: async (signal?: AbortSignal) => normalizeOverview(await get('/overview', undefined, signal)),
  summary: async (signal?: AbortSignal) => normalizeSummary(await get('/processing/summary', undefined, signal)),
  active: async (signal?: AbortSignal): Promise<ActiveRuns> => normalizeActive(await get('/processing/active', undefined, signal)),
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
  events: async (params: Params, signal?: AbortSignal) => normalizeEvents(await get('/events', params, signal)),
  /** Run history (FR-17), newest first; pass the previous answer's `next_before` for the next page. */
  runs: async (params: Params, signal?: AbortSignal) => normalizeRuns(await get('/runs', params, signal)),
  run: async (runId: string, signal?: AbortSignal) => normalizeRunDetail(await get(`/runs/${enc(runId)}`, undefined, signal)),
  health: async (signal?: AbortSignal) => normalizeHealth(await get('/health', undefined, signal)),
  dependency: async (dep: string, hours: number, signal?: AbortSignal) =>
    normalizeDepDetail(dep, await get(`/health/${enc(dep)}`, { hours }, signal)),
  workers: async (signal?: AbortSignal) => normalizeWorkers(await get('/workers', undefined, signal)),
  workerHistory: async (workerId: string, hours: number, signal?: AbortSignal) =>
    normalizeWorkerHistory(await get(`/workers/${enc(workerId)}/history`, { hours }, signal)),
  config: async (signal?: AbortSignal) => normalizeConfig(await get('/config', undefined, signal)),
  /** Re-check one dependency now (rate-limited on the server: one per 10 s per dependency, host-wide). */
  recheck: async (dep: string) => normalizeProbeResult(dep, await post(`/health/${enc(dep)}/probe`)),
  /** EHLO, STARTTLS and NOOP against the configured SMTP server; never sends mail (1 per minute). */
  smtpTest: async () => normalizeSmtpTest(await post('/smtp/test'))
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
