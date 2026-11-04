import { TFunction } from 'i18next'
import { toast } from 'sonner'
import { ActionOutcome, toastSpec } from './actions'

/**
 * Shows an action's result as a sonner toast carrying the audit id (§2.10).
 * A release that kept a queued "process again" request offers Run now.
 */
export function showOutcomeToast(t: TFunction, outcome: ActionOutcome, opts: { onRunNow?: () => void } = {}) {
  const spec = toastSpec(outcome)
  const parts = [t(spec.key, spec.values)]
  if (spec.auditId) parts.push(t('operations.actions.auditRef', { id: spec.auditId }))
  if (spec.requestId) parts.push(`${t('operations.error.ref')} ${spec.requestId}`)
  const action = spec.runNow && opts.onRunNow ? { label: t('operations.actions.runNowButton'), onClick: opts.onRunNow } : undefined
  toast[spec.level](parts.join(' · '), action ? { action } : undefined)
}
