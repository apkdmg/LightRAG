import { FormEvent, ReactNode, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Loader2Icon } from 'lucide-react'
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle
} from '@/components/ui/AlertDialog'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import { cn } from '@/lib/utils'
import { ActionOutcome, ConfirmFields, EMPTY_FIELDS, OutcomeAction, REASON_MAX, confirmGate, outcomeMessage, outcomeOfError } from '../actions'
import { showOutcomeToast } from '../actionToast'
import { RequestRef } from './ErrorPanel'

export type ConfirmPanelProps = {
  title: ReactNode
  /** The honest copy of §2.10: what the action does and does not do. */
  body: ReactNode
  confirmLabel: string
  cancelLabel?: string
  /** Release busy: the workspace name to type before confirming. */
  typedExpected?: string | null
  /** Shown between the copy and the reason (dry-run preview, override pause). */
  extra?: ReactNode
  /** The dialog waits for something else (fresh live state). */
  ready?: boolean
  /** The viewer may do this (operator-only actions). */
  allowed?: boolean
  notAllowedText?: string
  destructive?: boolean
  fields: ConfirmFields
  onFields: (fields: ConfirmFields) => void
  pending: boolean
  /** The last refusal: stays under the buttons, announced (role="alert"). */
  refusal: ActionOutcome | null
  onCancel: () => void
  onSubmit: () => void
}

/**
 * The inside of an action dialog, controlled: copy, optional typed confirm,
 * the reason (3–512 characters, recorded in the audit log), confirm with a
 * "Working…" state, and a refusal line that fades in under the buttons. The
 * confirm stays disabled until every gate passes (actions.ts confirmGate).
 */
export function ConfirmActionPanel({
  title,
  body,
  confirmLabel,
  cancelLabel,
  typedExpected = null,
  extra,
  ready = true,
  allowed = true,
  notAllowedText,
  destructive,
  fields,
  onFields,
  pending,
  refusal,
  onCancel,
  onSubmit
}: ConfirmPanelProps) {
  const { t } = useTranslation()
  const gate = confirmGate({ reason: fields.reason, typedExpected, typed: fields.typed, pending, ready, allowed })
  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (gate.canConfirm) onSubmit()
  }
  const reasonHint =
    gate.reason === 'short' ? t('operations.actions.reasonShort') : gate.reason === 'long' ? t('operations.actions.reasonLong') : t('operations.actions.reasonHint')
  const message = refusal ? outcomeMessage(refusal) : null
  return (
    <form onSubmit={submit} className="flex flex-col gap-4">
      <AlertDialogHeader>
        <AlertDialogTitle className="text-[17px]">{title}</AlertDialogTitle>
        <AlertDialogDescription asChild>
          <div className="text-muted-foreground flex flex-col gap-2 text-[13px] leading-relaxed">{body}</div>
        </AlertDialogDescription>
      </AlertDialogHeader>
      {extra}
      {!allowed && notAllowedText ? <p className="m-0 text-[13px] text-[color:var(--ops-warn)]">{notAllowedText}</p> : null}
      {typedExpected ? (
        <label className="flex flex-col gap-1 text-[13px]">
          <span className="font-medium">{t('operations.actions.typeToConfirm', { name: typedExpected })}</span>
          <Input
            value={fields.typed}
            onChange={(e) => onFields({ ...fields, typed: e.target.value })}
            autoComplete="off"
            spellCheck={false}
            disabled={!allowed || pending}
            aria-invalid={fields.typed.trim() !== '' && !gate.typedOk ? true : undefined}
            className="font-mono"
          />
        </label>
      ) : null}
      <label className="flex flex-col gap-1 text-[13px]">
        <span className="font-medium">{t('operations.actions.reason')}</span>
        <Textarea
          value={fields.reason}
          onChange={(e) => onFields({ ...fields, reason: e.target.value })}
          maxLength={REASON_MAX}
          rows={2}
          required
          placeholder={t('operations.actions.reasonPlaceholder')}
          disabled={!allowed || pending}
          aria-invalid={gate.reason === 'short' || gate.reason === 'long' ? true : undefined}
        />
        <span className={cn('text-xs', gate.reason === 'short' || gate.reason === 'long' ? 'text-[color:var(--ops-warn)]' : 'text-muted-foreground')}>{reasonHint}</span>
      </label>
      <AlertDialogFooter className="gap-2">
        <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={onCancel} disabled={pending}>
          {cancelLabel ?? t('operations.actions.close')}
        </button>
        <button
          type="submit"
          className={cn(
            'inline-flex min-h-9 items-center justify-center gap-1.5 rounded-lg px-3 text-[13px] hover:opacity-90 disabled:opacity-50',
            destructive ? 'bg-[color:var(--ops-crit)] text-white' : 'bg-primary text-primary-foreground'
          )}
          disabled={!gate.canConfirm}
          aria-busy={pending || undefined}
        >
          {pending ? <Loader2Icon className="size-3.5 animate-spin" aria-hidden="true" /> : null}
          {pending ? t('operations.actions.working') : confirmLabel}
        </button>
      </AlertDialogFooter>
      {refusal && message ? (
        <div role="alert" className="ops-fade-in flex flex-col gap-1 text-[13px]">
          <span className="text-[color:var(--ops-crit)]">{t(message.key, message.values)}</span>
          {refusal.detail ? <span className="text-muted-foreground text-xs break-words">{refusal.detail}</span> : null}
          {refusal.auditId ? <span className="text-muted-foreground text-xs">{t('operations.actions.auditRef', { id: refusal.auditId })}</span> : null}
          {refusal.requestId ? <RequestRef requestId={refusal.requestId} /> : null}
        </div>
      ) : null}
    </form>
  )
}

export type ConfirmActionDialogProps = Omit<ConfirmPanelProps, 'fields' | 'onFields' | 'pending' | 'refusal' | 'onCancel' | 'onSubmit'> & {
  open: boolean
  onOpenChange: (open: boolean) => void
  action: OutcomeAction
  /**
   * Sends the action. A final outcome closes the dialog (or keeps it open
   * with the cause, for a refusal) and shows the toast; null means "stay
   * open, nothing to report".
   */
  onConfirm: (fields: ConfirmFields) => Promise<ActionOutcome | null>
  /** Every final outcome, after the toast (refresh the view, open Run now…). */
  onOutcome?: (outcome: ActionOutcome) => void
  /** Run now… on the toast of a release that kept a queued request. */
  onRunNow?: () => void
}

/**
 * The action dialog (§2.10, §2.12 "Action dialog confirm"): the existing
 * AlertDialog fade and zoom, removed under reduced motion through
 * `ops-portal`. While pending the confirm shows a spinner plus "Working…";
 * success closes with the normal exit; a refusal keeps the dialog open with
 * its cause under the buttons, focus stays where it was and nothing shakes.
 * The dialog cannot be dismissed while a request is in flight.
 */
export default function ConfirmActionDialog({ open, onOpenChange, action, onConfirm, onOutcome, onRunNow, ...panel }: ConfirmActionDialogProps) {
  const { t } = useTranslation()
  const [fields, setFields] = useState<ConfirmFields>(EMPTY_FIELDS)
  const [pending, setPending] = useState(false)
  const [refusal, setRefusal] = useState<ActionOutcome | null>(null)

  const reset = () => {
    setFields(EMPTY_FIELDS)
    setRefusal(null)
  }
  const close = (next: boolean) => {
    if (pending) return
    if (!next) reset()
    onOpenChange(next)
  }
  const submit = async () => {
    if (pending) return
    setPending(true)
    setRefusal(null)
    let outcome: ActionOutcome | null
    try {
      outcome = await onConfirm({ reason: fields.reason.trim(), typed: fields.typed })
    } catch (err) {
      outcome = outcomeOfError(action, err)
    }
    setPending(false)
    if (!outcome) return
    showOutcomeToast(t, outcome, { onRunNow })
    if (outcome.tone === 'refused') setRefusal(outcome)
    else {
      reset()
      onOpenChange(false)
    }
    onOutcome?.(outcome)
  }

  return (
    <AlertDialog open={open} onOpenChange={close}>
      <AlertDialogContent className="ops-portal max-h-[90vh] max-w-lg overflow-y-auto" onEscapeKeyDown={(e) => pending && e.preventDefault()}>
        <ConfirmActionPanel
          {...panel}
          fields={fields}
          onFields={setFields}
          pending={pending}
          refusal={refusal}
          onCancel={() => close(false)}
          onSubmit={() => void submit()}
        />
      </AlertDialogContent>
    </AlertDialog>
  )
}
