import { TFunction } from 'i18next'
import { ActionResponse, AlertActionResponse, AlertScope, OpsAlert, OpsStatus } from '../api'
import { ActionOutcome, AlertActionKind, outcomeOf, outcomeOfError } from '../actions'
import { useOpsShell } from '../context'
import { fmtCount } from '../format'
import { durationKey } from './alerts'

/** Alert actions are offered while the server evaluates alerts (OPS_ALERTS_ENABLED) on PostgreSQL. */
export const alertActionsEnabled = (status: OpsStatus | null | undefined) => !!status?.available && status?.features?.alerts === true

/** Whether this viewer may act on alerts at all; read views stay open to every admin. */
export function useAlertActions() {
  const { status } = useOpsShell()
  return alertActionsEnabled(status)
}

/** Sends an alert action; an answer or a thrown refusal becomes an outcome for the toast. */
export async function sendAlertAction(kind: AlertActionKind, call: () => Promise<AlertActionResponse>): Promise<ActionOutcome> {
  try {
    return outcomeOf(kind, (await call()) as ActionResponse)
  } catch (err) {
    return outcomeOfError(kind, err)
  }
}

/** "1 h", "4 h", "7 days". */
export function durationText(t: TFunction, minutes: number): string {
  const d = durationKey(minutes)
  return t(d.key, { count: d.count })
}

const countOf = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : null)

/**
 * A rule's title from the console's catalogue (with the scope's count and
 * workspace), else the server's label for it, else its name in words.
 */
export function ruleTitle(t: TFunction, rule: string, scope?: AlertScope | null, label?: string | null): string {
  const count = countOf(scope?.count)
  return t(`operations.rule.${rule}.title`, {
    count: count ?? 0,
    countText: fmtCount(count),
    workspace: scope?.workspace ?? '',
    defaultValue: label || rule.replace(/_/g, ' ')
  })
}

/** The server's title wins: it carries the counts and names the evaluator saw. */
export const alertTitle = (t: TFunction, alert: Pick<OpsAlert, 'title' | 'rule' | 'scope' | 'label'>) =>
  alert.title || ruleTitle(t, alert.rule, alert.scope, alert.label)

/** The runbook step: the alert's or rule's own text, else the catalogue's. */
export const runbookText = (t: TFunction, rule: string, own?: string | null) => own || t(`operations.rule.${rule}.runbook`, { count: 0, countText: '', workspace: '', defaultValue: '' })

/** Only plain web links are followed (the runbook base URL is server configuration, but still checked). */
export const safeHttpUrl = (url: string | null | undefined) => (url && /^https?:\/\/[^\s]+$/i.test(url) ? url : null)
