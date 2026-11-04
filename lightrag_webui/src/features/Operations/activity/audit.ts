import type { AuditFilters, AuditRow, Params } from '../api'
import { REFUSALS } from '../errors'

/**
 * The Audit section's filters (§2.8, §3.7 `GET /audit?actor=&action=&from=&to=&before=`).
 * Without dates the server shows the last 30 days; a range may span 400.
 * Dates are picked as local calendar days and sent as UTC instants: `from`
 * is the start of its day, `to` the start of the day after, so "to 4 Nov"
 * includes all of 4 Nov.
 */

export type AuditDraft = { actor: string; action: string; from: string; to: string }

export const EMPTY_AUDIT_DRAFT: AuditDraft = { actor: '', action: '', from: '', to: '' }

const DAY = /^(\d{4})-(\d{2})-(\d{2})$/

/** "2026-11-04" (an <input type="date"> value, local) → the UTC instant of that local midnight. */
export function dayStartIso(day: string, offsetDays = 0): string | null {
  const m = DAY.exec(day.trim())
  if (!m) return null
  const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]) + offsetDays)
  return Number.isNaN(d.getTime()) ? null : d.toISOString()
}

/** The filters as the server reads them; blank fields are left out. */
export function auditFilters(draft: AuditDraft): AuditFilters {
  const f: AuditFilters = {}
  const actor = draft.actor.trim()
  const action = draft.action.trim()
  if (actor) f.actor = actor
  if (action) f.action = action
  const from = draft.from ? dayStartIso(draft.from) : null
  const to = draft.to ? dayStartIso(draft.to, 1) : null
  if (from) f.from = from
  if (to) f.to = to
  return f
}

/** A "to" day before the "from" day can match nothing: say so instead of asking. */
export const draftInvalid = (draft: AuditDraft) => !!draft.from && !!draft.to && draft.to < draft.from

/** One page of rows: the filters plus the keyset cursor (`before` = the previous page's `next_before`). */
export function auditParams(draft: AuditDraft, before: string | null): Params {
  return { ...auditFilters(draft), before: before ?? undefined }
}

/** Action names seen on the loaded page, offered as suggestions in the action filter (the server owns the names). */
export const actionSuggestions = (rows: Pick<AuditRow, 'action'>[]) => Array.from(new Set(rows.map((r) => r.action))).sort()

const OK_OUTCOMES = new Set([
  'ok',
  'success',
  'succeeded',
  'done',
  'cancellation_requested',
  'started',
  'kicked',
  'started_local',
  'released',
  'revealed',
  'exported'
])
const NO_OP_OUTCOMES = new Set(['not_busy', 'nothing_pending', 'request_pending', 'noop', 'no_op'])
const ERROR_OUTCOMES = new Set(['error', 'failed', 'failure', 'exception'])

/**
 * The state chip for a row: an intent row is "pending" (its result row
 * follows), a refusal is a warning, an error is critical. Outcome codes the
 * console does not know show neutral, as written.
 */
export function auditRowState(row: Pick<AuditRow, 'phase' | 'outcome'>): 'ok' | 'warning' | 'critical' | 'unknown' {
  if (row.phase === 'intent') return 'unknown'
  const o = (row.outcome ?? '').toLowerCase()
  if (!o) return 'unknown'
  if (OK_OUTCOMES.has(o)) return 'ok'
  if (ERROR_OUTCOMES.has(o)) return 'critical'
  if (NO_OP_OUTCOMES.has(o)) return 'unknown'
  if ((REFUSALS as readonly string[]).includes(o) || o.startsWith('refused') || o === 'denied' || o === 'forbidden') return 'warning'
  return 'unknown'
}

/** Who acted, with the impersonator when an admin acted as someone else. */
export const auditActor = (row: Pick<AuditRow, 'actor' | 'impersonated_by'>) =>
  row.impersonated_by ? `${row.actor ?? '–'} (${row.impersonated_by})` : (row.actor ?? '–')
