import type { ActionResponse, InProgressBy, OpsStatus, WorkspaceDetail, WouldProcess } from './api'
import { REFUSALS, Refusal, toOpsError } from './errors'
import { ageSeconds } from './format'
import { isMultiWorker } from './severity'

/**
 * Admin actions (spec §2.10, §3.9, FR-55/56): what can be offered where, what
 * a confirm needs before it can be sent, and how an answer reads. Pure, so
 * the dialog gating and every result's toast are tested without a DOM. The
 * server re-checks everything; the client only avoids offering what it
 * already knows will be refused.
 */

export type ActionKind = 'cancel' | 'run' | 'drain_email' | 'release_busy'

/** The order of the Workspace drawer's buttons (§2.9 wireframe). */
export const ACTION_ORDER: ActionKind[] = ['cancel', 'release_busy', 'run', 'drain_email']

/** FR-56: every action carries a reason of 3 to 512 characters. */
export const REASON_MIN = 3
export const REASON_MAX = 512

export type ReasonState = 'empty' | 'short' | 'long' | 'ok'

export function reasonState(reason: string): ReasonState {
  const n = reason.trim().length
  if (n === 0) return 'empty'
  if (n < REASON_MIN) return 'short'
  if (n > REASON_MAX) return 'long'
  return 'ok'
}

/** The typed name must match exactly (case and all); surrounding spaces from a paste are forgiven. */
export const typedMatches = (typed: string, expected: string | null | undefined) => !!expected && typed.trim() === expected

/** What a confirm dialog collects: the reason and, for Release busy, the typed workspace name. */
export type ConfirmFields = { reason: string; typed: string }

export const EMPTY_FIELDS: ConfirmFields = { reason: '', typed: '' }

export type ConfirmGateInput = {
  reason: string
  /** The text the user must type (the workspace name), or null when no typed confirm is asked. */
  typedExpected?: string | null
  typed?: string
  /** A request is in flight: nothing more can be sent. */
  pending?: boolean
  /** Whatever else the dialog waits for (fresh live state for Release busy). */
  ready?: boolean
  /** The viewer may do this at all (operator-only actions). */
  allowed?: boolean
}

export type ConfirmGate = { canConfirm: boolean; reason: ReasonState; typedOk: boolean }

/** Whether the confirm button is enabled, and why not (the reason hint and the typed-name hint). */
export function confirmGate({ reason, typedExpected = null, typed = '', pending = false, ready = true, allowed = true }: ConfirmGateInput): ConfirmGate {
  const r = reasonState(reason)
  const typedOk = typedExpected ? typedMatches(typed, typedExpected) : true
  return { canConfirm: r === 'ok' && typedOk && !pending && ready && allowed, reason: r, typedOk }
}

// ---------- what to offer ----------

/** Actions are offered only when the server says they are on (OPS_ACTIONS_ENABLED, default off). */
export const actionsEnabled = (status: OpsStatus | null | undefined) => status?.features?.actions === true

/**
 * Release busy exists only in multi-worker mode (§2.4: "not rendered in
 * single-process mode"); a server that reports `release_busy: false` has
 * it off for another reason.
 */
export const releaseOffered = (status: OpsStatus | null | undefined) =>
  actionsEnabled(status) && isMultiWorker(status?.process.mode) && status?.features?.release_busy !== false

export type ActionContext = {
  status: OpsStatus | null | undefined
  /** The workspace's pipeline holds `busy`; null when not known (no live state). */
  busy: boolean | null
  destructiveBusy?: boolean | null
  cancellationRequested?: boolean | null
  /**
   * The owner of `busy` looks gone: the run is orphaned or unresponsive, the
   * OS confirmed the process exited, or busy has no run at all. The server
   * still confirms it before releasing.
   */
  ownerSuspect?: boolean
  /** Records waiting in the email queue; null when not known. */
  emailPending?: number | null
}

export type ActionOffer = {
  kind: ActionKind
  /** Shown with a hint why not, rather than hidden. */
  disabled: boolean
  /** operations.actions.hint.* */
  hint: string | null
}

const offer = (kind: ActionKind, hint: string | null): ActionOffer => ({ kind, disabled: hint !== null, hint })

/** One action's button state, or null when it is not offered here at all. */
export function actionOffer(kind: ActionKind, ctx: ActionContext): ActionOffer | null {
  if (!actionsEnabled(ctx.status)) return null
  if (kind === 'cancel') {
    if (ctx.busy === false) return offer(kind, 'notBusy')
    if (ctx.cancellationRequested) return offer(kind, 'cancelling')
    return offer(kind, null)
  }
  if (kind === 'run') {
    // §2.10: disabled while busy or destructive_busy.
    if (ctx.destructiveBusy) return offer(kind, 'destructive')
    if (ctx.busy) return offer(kind, 'busy')
    return offer(kind, null)
  }
  if (kind === 'drain_email') {
    if (ctx.emailPending === 0) return offer(kind, 'noEmail')
    return offer(kind, null)
  }
  // release_busy: not rendered in single-process mode; disabled with a hint for non-operators (§2.4).
  if (!releaseOffered(ctx.status)) return null
  if (!ctx.status?.operator) return offer(kind, 'operatorOnly')
  if (ctx.busy === false) return offer(kind, 'notBusy')
  if (!ctx.ownerSuspect) return offer(kind, 'ownerAlive')
  return offer(kind, null)
}

export const actionOffers = (ctx: ActionContext, kinds: ActionKind[] = ACTION_ORDER): ActionOffer[] =>
  kinds.map((k) => actionOffer(k, ctx)).filter((o): o is ActionOffer => o !== null)

/**
 * Run states whose owner may be gone, so Release busy is worth offering. An
 * unresponsive owner (live process, stale heartbeat) is not one: the server
 * refuses it as alive, and the advice is a restart.
 */
export const OWNER_SUSPECT_STATES = new Set(['orphaned', 'busy_without_run'])

/**
 * What a stuck row can be acted on with: a run that holds its workspace can
 * be cancelled, or released when its owner is gone; a stale document can be
 * re-driven by Run now. Reservations without an owner have only restart
 * guidance (OQ-12).
 */
export function stuckActionKinds(row: { cls: string }): ActionKind[] {
  if (row.cls === 'reservation') return []
  if (row.cls === 'stale_doc') return ['run']
  return ['cancel', 'release_busy']
}

// ---------- dry run ----------

export type DryRunCounts = { pending: number; failed: number; cancelled: number; inFlight: number; total: number }

const n0 = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) && v > 0 ? Math.round(v) : 0)

/**
 * "Will process 310 pending + 4 failed (including 2 user-cancelled)" (§2.10).
 * The server's `failed` already includes user cancellations (a run retries
 * both); `cancelled` says how many of them there are. Docs left mid-pipeline
 * by an interrupted run (`in_flight`) are retried too.
 */
export function dryRunCounts(w: WouldProcess | null | undefined): DryRunCounts | null {
  if (!w || typeof w !== 'object') return null
  const pending = n0(w.pending)
  const failed = n0(w.failed)
  const cancelled = Math.min(n0(w.cancelled), failed)
  const inFlight = n0(w.in_flight)
  return { pending, failed, cancelled, inFlight, total: pending + failed + inFlight }
}

// ---------- release busy ----------

/** The owner facts the release copy quotes, from the workspace's live state and its current run. */
export function releaseFacts(d: WorkspaceDetail | null, now: number) {
  const live = d?.live ?? null
  const runs = d?.recent_runs ?? []
  const run = (live?.run_id ? runs.find((r) => r.run_id === live.run_id) : undefined) ?? runs.find((r) => r.status === 'running' || r.status === 'orphaned')
  const beat = live?.heartbeat_at ? ageSeconds(live.heartbeat_at, now) : null
  return {
    busy: !!live?.busy,
    runId: live?.run_id ?? null,
    jobStart: live?.job_start ?? null,
    requestPending: !!live?.request_pending,
    worker: live?.owner_worker_id ?? run?.worker_id ?? null,
    pid: typeof live?.owner_pid === 'number' ? live.owner_pid : typeof run?.pid === 'number' ? run.pid : null,
    beatAge: beat,
    processGone: run?.worker_alive === false
  }
}

// ---------- outcomes ----------

export type OutcomeTone = 'success' | 'info' | 'refused'

export type ActionOutcome = {
  action: ActionKind | 'reveal'
  tone: OutcomeTone
  /** The result or refusal code: cancellation_requested, not_busy, lock_unavailable, … */
  code: string
  auditId: string | null
  inProgressBy: InProgressBy | null
  /** Release succeeded and a queued "process again" request was kept: offer Run now (§3.9). */
  offerRunNow: boolean
  requestId: string | null
  /** A transport or server error (not a refusal): retrying may help. */
  error: boolean
  /** Safe server detail for an unexpected error; null for known codes. */
  detail: string | null
}

/** Codes that did what was asked. */
const SUCCESS = new Set(['cancellation_requested', 'started', 'kicked', 'started_local', 'released', 'revealed'])
/** Nothing to do, or it will happen anyway: not an error, not a success. */
const NO_OP = new Set(['not_busy', 'nothing_pending', 'nothing_to_process', 'request_pending', 'nothing_to_reveal'])

const isRefusal = (code: string): code is Refusal => (REFUSALS as readonly string[]).includes(code)

const text = (v: unknown): string | null => (typeof v === 'string' && v ? v : typeof v === 'number' && Number.isFinite(v) ? String(v) : null)

/** The lease holder, however the server spells it: an object, a bare actor, or fields beside the code. */
export function inProgressOf(raw: unknown): InProgressBy | null {
  if (!raw || typeof raw !== 'object') return null
  const o = raw as Record<string, unknown>
  const by = o.in_progress_by ?? o.holder
  if (typeof by === 'string' && by) return { actor: by, action: text(o.action), since: text(o.since) }
  if (by && typeof by === 'object') {
    const b = by as Record<string, unknown>
    const actor = text(b.actor) ?? text(b.holder)
    return actor || text(b.action) ? { actor, action: text(b.action), since: text(b.since) ?? text(b.at) } : null
  }
  const actor = text(o.actor) ?? text(o.by)
  return actor ? { actor, action: text(o.action), since: text(o.since) } : null
}

const toneOf = (code: string): OutcomeTone => (SUCCESS.has(code) ? 'success' : NO_OP.has(code) ? 'info' : isRefusal(code) ? 'refused' : 'info')

/** A 2xx answer → an outcome. Release busy answers `released` plus a `reason` when it refused. */
export function outcomeOf(action: ActionKind | 'reveal', resp: ActionResponse | null | undefined, hints: { requestPending?: boolean } = {}): ActionOutcome {
  const r = resp ?? {}
  let code: string
  if (action === 'reveal') code = 'revealed'
  else if (action === 'release_busy') code = r.released === true ? 'released' : (text(r.reason) ?? text(r.result) ?? 'unknown')
  else code = text(r.result) ?? 'unknown'
  const inProgressBy = code === 'in_progress_by' ? inProgressOf(r) : null
  return {
    action,
    tone: toneOf(code),
    code,
    auditId: text(r.audit_id),
    inProgressBy,
    offerRunNow: code === 'released' && (r.run_now_offered === true || (r.run_now_offered == null && !!hints.requestPending)),
    requestId: null,
    error: false,
    detail: null
  }
}

/** A thrown error → an outcome: a refusal code when the server named one, else the mapped error. */
export function outcomeOfError(action: ActionKind | 'reveal', err: unknown): ActionOutcome {
  const e = toOpsError(err)
  const raw = (err as { detail?: unknown; response?: { data?: { detail?: unknown } } } | null) ?? null
  const detail = raw?.detail !== undefined ? raw.detail : raw?.response?.data?.detail
  const fields = detail && typeof detail === 'object' && !Array.isArray(detail) ? (detail as Record<string, unknown>) : null
  const code = e.refusal ?? e.kind
  return {
    action,
    tone: 'refused',
    code,
    // A refusal after the intent row was written may name it.
    auditId: text(fields?.audit_id),
    inProgressBy: code === 'in_progress_by' ? inProgressOf(fields) : null,
    offerRunNow: false,
    requestId: e.requestId,
    error: !e.refusal,
    detail: e.refusal ? null : e.detail
  }
}

export type OutcomeMessage = { key: string; values: Record<string, string> }

/** The sentence an outcome reads as, for the toast and the dialog's role="alert" line. */
export function outcomeMessage(o: ActionOutcome): OutcomeMessage {
  if (o.code === 'in_progress_by' && o.inProgressBy?.actor) {
    return {
      key: o.inProgressBy.action ? 'operations.actions.inProgressByAction' : 'operations.actions.inProgressBy',
      values: { actor: o.inProgressBy.actor, action: o.inProgressBy.action ?? '' }
    }
  }
  if (o.code === 'released' && o.offerRunNow) return { key: 'operations.actions.result.releasedRunPending', values: {} }
  if (o.code === 'not_busy' && o.action === 'release_busy') return { key: 'operations.actions.result.notBusyRelease', values: {} }
  // A reveal of a document or job that changed since the page was drawn.
  if (o.code === 'not_found' && o.action === 'reveal') return { key: 'operations.reveal.notFound', values: {} }
  if (SUCCESS.has(o.code) || NO_OP.has(o.code)) return { key: `operations.actions.result.${o.code}`, values: {} }
  if (isRefusal(o.code)) return { key: `operations.refusal.${o.code}`, values: {} }
  if (o.error) return { key: `operations.error.${o.code}`, values: {} }
  return { key: 'operations.actions.result.unknown', values: { code: o.code } }
}

export type ToastLevel = 'success' | 'info' | 'warning' | 'error'

export type ToastSpec = OutcomeMessage & {
  level: ToastLevel
  /** "audit 57" after the sentence (§2.10: every result toast carries the audit id). */
  auditId: string | null
  /** "Ref: …" for an error the server tagged. */
  requestId: string | null
  /** Show a "Run now…" button on the toast. */
  runNow: boolean
}

/** Every result shows as a toast with its audit id; a refusal shows its cause (§2.10). */
export function toastSpec(o: ActionOutcome): ToastSpec {
  const level: ToastLevel = o.tone === 'success' ? 'success' : o.tone === 'info' ? 'info' : o.error ? 'error' : 'warning'
  return { ...outcomeMessage(o), level, auditId: o.auditId, requestId: o.requestId, runNow: o.offerRunNow }
}
