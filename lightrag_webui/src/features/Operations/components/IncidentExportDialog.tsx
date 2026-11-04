import { FormEvent, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { DownloadIcon, Loader2Icon } from 'lucide-react'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/Dialog'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import { OpsAlert, opsApi } from '../api'
import { REASON_MAX, reasonState } from '../actions'
import { IncidentWindow, fromLocalInput, incidentDefaults, incidentProblem } from '../activity/alerts'
import { OpsError, toOpsError } from '../errors'
import { RequestRef } from './ErrorPanel'

/**
 * Incident timeline export (§2.8 section 4, FR-65, §3.7 `POST
 * /activity/export`): pick a window and get alerts, audit rows, runs and
 * system markers as NDJSON. No log text is included, everything is redacted
 * on the server, rows are capped, and the export is audited, so it takes a
 * reason. Operator-only (D8). A refusal stays in the dialog (role="alert").
 */
export default function IncidentExportDialog({
  open,
  onOpenChange,
  alert
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** Start the window around this alert instead of the last 6 hours. */
  alert?: Pick<OpsAlert, 'first_seen' | 'resolved_at'> | null
}) {
  const [pending, setPending] = useState(false)

  const close = (next: boolean) => {
    if (pending) return
    onOpenChange(next)
  }

  return (
    <Dialog open={open} onOpenChange={close}>
      <DialogContent className="ops-portal max-w-lg">
        {/* Mounted only while open, so every opening starts from a window
            around the current time (and a clean reason and error). */}
        <IncidentExportForm alert={alert} pending={pending} setPending={setPending} onClose={close} />
      </DialogContent>
    </Dialog>
  )
}

function IncidentExportForm({
  alert,
  pending,
  setPending,
  onClose
}: {
  alert?: Pick<OpsAlert, 'first_seen' | 'resolved_at'> | null
  pending: boolean
  setPending: (pending: boolean) => void
  onClose: (open: boolean) => void
}) {
  const { t } = useTranslation()
  const [win, setWin] = useState<IncidentWindow>(() => incidentDefaults(Date.now(), alert))
  const [reason, setReason] = useState('')
  const [error, setError] = useState<OpsError | null>(null)
  const problem = incidentProblem(win)
  const rs = reasonState(reason)
  const canSend = !problem && rs === 'ok' && !pending

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!canSend) return
    const from = fromLocalInput(win.from)
    const to = fromLocalInput(win.to)
    if (!from || !to) return
    setPending(true)
    setError(null)
    let done = false
    try {
      const result = await opsApi.exportActivity({ from, to, reason: reason.trim() })
      toast.success(
        result.auditId
          ? t('operations.export.doneAudit', { file: result.filename, audit: result.auditId })
          : t('operations.export.done', { file: result.filename })
      )
      done = true
    } catch (err) {
      setError(toOpsError(err))
    } finally {
      setPending(false)
    }
    if (done) onClose(false)
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-4">
      <DialogHeader>
        <DialogTitle>{t('operations.incident.title')}</DialogTitle>
        <DialogDescription>{t('operations.incident.body')}</DialogDescription>
      </DialogHeader>
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="flex flex-col gap-1 text-[13px]">
          <span className="font-medium">{t('operations.incident.from')}</span>
          <Input type="datetime-local" value={win.from} onChange={(e) => setWin({ ...win, from: e.target.value })} required />
        </label>
        <label className="flex flex-col gap-1 text-[13px]">
          <span className="font-medium">{t('operations.incident.to')}</span>
          <Input
            type="datetime-local"
            value={win.to}
            onChange={(e) => setWin({ ...win, to: e.target.value })}
            required
            aria-invalid={problem ? true : undefined}
          />
        </label>
        {problem ? <p className="m-0 text-xs text-[color:var(--ops-warn)] sm:col-span-2">{t(`operations.incident.problem.${problem}`)}</p> : null}
      </div>
      <label className="flex flex-col gap-1 text-[13px]">
        <span className="font-medium">{t('operations.actions.reason')}</span>
        <Textarea
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          maxLength={REASON_MAX}
          rows={2}
          required
          placeholder={t('operations.export.reasonPlaceholder')}
        />
        <span className={rs === 'short' ? 'text-xs text-[color:var(--ops-warn)]' : 'text-muted-foreground text-xs'}>
          {rs === 'short' ? t('operations.actions.reasonShort') : t('operations.actions.reasonHint')}
        </span>
      </label>
      <DialogFooter className="gap-2">
        <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => onClose(false)} disabled={pending}>
          {t('operations.export.cancel')}
        </button>
        <button
          type="submit"
          className="bg-primary text-primary-foreground inline-flex min-h-9 items-center justify-center gap-1.5 rounded-lg px-3 text-[13px] hover:opacity-90 disabled:opacity-50"
          disabled={!canSend}
          aria-busy={pending || undefined}
        >
          {pending ? <Loader2Icon className="size-3.5 animate-spin" aria-hidden="true" /> : <DownloadIcon className="size-3.5" aria-hidden="true" />}
          {pending ? t('operations.export.working') : t('operations.incident.submit')}
        </button>
      </DialogFooter>
      {error ? (
        <div role="alert" className="ops-fade-in flex flex-col gap-1 text-[13px]">
          <span className="text-[color:var(--ops-crit)]">{t(error.messageKey)}</span>
          {error.detail && !error.refusal ? <span className="text-muted-foreground text-xs break-words">{error.detail}</span> : null}
          {error.requestId ? <RequestRef requestId={error.requestId} /> : null}
        </div>
      ) : null}
    </form>
  )
}
