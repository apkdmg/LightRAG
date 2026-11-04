/**
 * The status vocabulary of the Operations console (spec §2.2). A state always
 * renders as an icon plus text; colour is never the only signal.
 */

export const STATES = ['ok', 'warning', 'serious', 'critical', 'unknown', 'running', 'paused', 'maintenance'] as const
export type OpsState = (typeof STATES)[number]

export type Tone = 'ok' | 'warn' | 'serious' | 'crit' | 'unknown' | 'info' | 'paused'
export type IconId = 'check' | 'triangle' | 'octagon' | 'x-circle' | 'dashed-circle' | 'dot' | 'pause' | 'wrench'

export type StateLook = {
  /** i18n key under operations.state.* */
  label: string
  /** CSS token suffix: --ops-<tone> */
  tone: Tone
  icon: IconId
}

const LOOK: Record<OpsState, StateLook> = {
  ok: { label: 'operations.state.ok', tone: 'ok', icon: 'check' },
  warning: { label: 'operations.state.warning', tone: 'warn', icon: 'triangle' },
  serious: { label: 'operations.state.serious', tone: 'serious', icon: 'octagon' },
  critical: { label: 'operations.state.critical', tone: 'crit', icon: 'x-circle' },
  unknown: { label: 'operations.state.unknown', tone: 'unknown', icon: 'dashed-circle' },
  running: { label: 'operations.state.running', tone: 'info', icon: 'dot' },
  paused: { label: 'operations.state.paused', tone: 'paused', icon: 'pause' },
  maintenance: { label: 'operations.state.maintenance', tone: 'unknown', icon: 'wrench' }
}

/** Server spellings that mean the same thing as one of ours. */
const ALIASES: Record<string, OpsState> = {
  healthy: 'ok',
  up: 'ok',
  degraded: 'warning',
  warn: 'warning',
  down: 'critical',
  crit: 'critical',
  error: 'critical',
  stale: 'unknown',
  info: 'running'
}

/** Normalises any server state string; anything unrecognised is "unknown", never a crash. */
export const toState = (raw: string | null | undefined): OpsState => {
  if (!raw) return 'unknown'
  const key = raw.toLowerCase()
  if ((STATES as readonly string[]).includes(key)) return key as OpsState
  return ALIASES[key] ?? 'unknown'
}

export const lookOf = (raw: string | null | undefined): StateLook => LOOK[toState(raw)]

/**
 * Severity rank used to decide whether things got worse (OK < Warning <
 * Serious < Critical). Running, Paused, Unknown and Maintenance are not a
 * worsening on their own, so they rank with OK.
 */
export const rankOf = (raw: string | null | undefined): number => {
  switch (toState(raw)) {
    case 'warning':
      return 1
    case 'serious':
      return 2
    case 'critical':
      return 3
    default:
      return 0
  }
}

/**
 * The worst of several states, for a summary. Unknown outranks OK here: a
 * summary must not read OK while part of what it summarises is unknown.
 * An empty list is unknown.
 */
export const worstOf = (states: Array<string | null | undefined>): OpsState => {
  if (!states.length) return 'unknown'
  const weight = (s: OpsState) => (rankOf(s) > 0 ? rankOf(s) + 1 : s === 'unknown' ? 1 : 0)
  return states.map(toState).reduce((worst, s) => (weight(s) > weight(worst) ? s : worst))
}

/**
 * Data is stale once it is older than twice its source's freshness (§2.12):
 * SMTP probed every 15 min goes stale at 30 min, not at 2× the page's poll.
 */
export const isStale = (updatedAtMs: number | null | undefined, freshnessSeconds: number, now: number = Date.now()) => {
  if (updatedAtMs === null || updatedAtMs === undefined || !Number.isFinite(updatedAtMs)) return true
  return now - updatedAtMs > 2 * freshnessSeconds * 1000
}

/**
 * Run and stuck classes, each with a plain-language explanation
 * (operations.run.<id>.*): the live ones, the stuck classes, and the
 * outcomes of a finished run (succeeded, failed, cancelled). These are the
 * server's own state strings (OpsService._running_state and the stuck
 * classes); a run whose owner stopped beating is `unresponsive`.
 */
export const RUN_STATES = [
  'running',
  'cancelling',
  'paused',
  'hung',
  'orphaned',
  'busy_without_run',
  'reservation',
  'stale_doc',
  'unresponsive',
  'succeeded',
  'failed',
  'cancelled'
] as const
export type RunState = (typeof RUN_STATES)[number]

const RUN_LOOK: Record<RunState, OpsState> = {
  running: 'running',
  // Still holding the workspace until the current document finishes.
  cancelling: 'running',
  paused: 'paused',
  hung: 'serious',
  orphaned: 'critical',
  busy_without_run: 'critical',
  reservation: 'critical',
  stale_doc: 'warning',
  unresponsive: 'serious',
  succeeded: 'ok',
  // A failed run is history, not an open incident: its documents are retried.
  failed: 'serious',
  // A user's soft cancel; the cancelled documents are retried by the next run.
  cancelled: 'unknown'
}

/** States of a run that is still holding its workspace (the drawer polls faster, the dot may breathe). */
export const isLiveRunState = (raw: string | null | undefined) =>
  raw === 'running' || raw === 'cancelling' || raw === 'hung' || raw === 'unresponsive'

export type RunLook = StateLook & { state: OpsState; explain: string }

export const runLookOf = (raw: string | null | undefined): RunLook => {
  const id = (RUN_STATES as readonly string[]).includes(raw ?? '') ? (raw as RunState) : null
  if (!id) return { ...LOOK.unknown, state: 'unknown', explain: 'operations.run.unknown.explain' }
  const state = RUN_LOOK[id]
  return { ...LOOK[state], label: `operations.run.${id}.label`, state, explain: `operations.run.${id}.explain` }
}

/** Process model reported by /status. Multi-worker-only features hide in uvicorn mode. */
export type ProcessMode = 'uvicorn' | 'gunicorn' | 'unknown'

export const toProcessMode = (raw: string | null | undefined): ProcessMode =>
  raw === 'uvicorn' || raw === 'gunicorn' ? raw : 'unknown'

export const isMultiWorker = (raw: string | null | undefined) => toProcessMode(raw) === 'gunicorn'

/**
 * State of the worker count: OK when every expected worker reports, Warning
 * when some are missing, Critical when none do, Unknown before heartbeats exist.
 */
export const workersState = (expected: number | null | undefined, live: number | null | undefined): OpsState => {
  if (typeof expected !== 'number' || typeof live !== 'number' || expected <= 0) return 'unknown'
  if (live <= 0) return 'critical'
  return live < expected ? 'warning' : 'ok'
}

/** A document's doc-status value as a state: processed OK, failed Critical, in flight Running, pending Unknown. */
export const docStatusState = (status: string | null | undefined): OpsState => {
  switch ((status ?? '').toLowerCase()) {
    case 'processed':
      return 'ok'
    case 'failed':
      return 'critical'
    case 'parsing':
    case 'analyzing':
    case 'processing':
      return 'running'
    default:
      return 'unknown'
  }
}

/**
 * A backlog row's state column. Failed docs and stale docs are a warning on a
 * workspace row (hundreds of rows in red would hide what is urgent); hung is
 * serious and orphaned critical, as for runs.
 */
export const workspaceRowState = (raw: string | null | undefined): OpsState => {
  switch ((raw ?? '').toLowerCase()) {
    case 'running':
      return 'running'
    case 'paused':
      return 'paused'
    case 'idle':
      return 'ok'
    case 'stale_doc':
    case 'failed':
      return 'warning'
    case 'hung':
      return 'serious'
    case 'orphaned':
      return 'critical'
    default:
      return 'unknown'
  }
}
