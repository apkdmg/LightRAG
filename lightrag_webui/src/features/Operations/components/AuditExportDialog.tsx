import { FormEvent, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { DownloadIcon, Loader2Icon } from 'lucide-react'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/Dialog'
import Textarea from '@/components/ui/Textarea'
import { AuditFilters, opsApi } from '../api'
import { REASON_MAX, reasonState } from '../actions'
import { OpsError, toOpsError } from '../errors'
import { RequestRef } from './ErrorPanel'

/**
 * "Export CSV…" for the audit log (§2.8, §3.7 `POST /audit/export`): the
 * rows for the current filters as CSV. Cells that would start a formula
 * (`= + - @`) are escaped on the server, and the export is itself audited,
 * so a reason is required. A refusal stays in the dialog (role="alert").
 */
export default function AuditExportDialog({
  open,
  onOpenChange,
  filters,
  summary
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  filters: AuditFilters
  /** The filters in words, so the reason is given for what will be exported. */
  summary: string
}) {
  const { t } = useTranslation()
  const [reason, setReason] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<OpsError | null>(null)
  const state = reasonState(reason)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (state !== 'ok' || pending) return
    setPending(true)
    setError(null)
    try {
      const result = await opsApi.exportAudit({ filters, reason: reason.trim() })
      toast.success(
        result.auditId
          ? t('operations.export.doneAudit', { file: result.filename, audit: result.auditId })
          : t('operations.export.done', { file: result.filename })
      )
      setReason('')
      onOpenChange(false)
    } catch (err) {
      setError(toOpsError(err))
    } finally {
      setPending(false)
    }
  }

  const close = (next: boolean) => {
    if (pending) return
    if (!next) setError(null)
    onOpenChange(next)
  }

  return (
    <Dialog open={open} onOpenChange={close}>
      <DialogContent className="ops-portal max-w-lg">
        <form onSubmit={submit} className="flex flex-col gap-4">
          <DialogHeader>
            <DialogTitle>{t('operations.audit.export.title')}</DialogTitle>
            <DialogDescription>{t('operations.audit.export.body')}</DialogDescription>
          </DialogHeader>
          <p className="bg-muted/50 m-0 rounded-md border px-2.5 py-1.5 text-xs break-words">{summary}</p>
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
            <span className={state === 'short' ? 'text-xs text-[color:var(--ops-warn)]' : 'text-muted-foreground text-xs'}>
              {state === 'short' ? t('operations.actions.reasonShort') : t('operations.actions.reasonHint')}
            </span>
          </label>
          <DialogFooter className="gap-2">
            <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => close(false)} disabled={pending}>
              {t('operations.export.cancel')}
            </button>
            <button
              type="submit"
              className="bg-primary text-primary-foreground inline-flex min-h-9 items-center justify-center gap-1.5 rounded-lg px-3 text-[13px] hover:opacity-90 disabled:opacity-50"
              disabled={state !== 'ok' || pending}
              aria-busy={pending || undefined}
            >
              {pending ? <Loader2Icon className="size-3.5 animate-spin" aria-hidden="true" /> : <DownloadIcon className="size-3.5" aria-hidden="true" />}
              {pending ? t('operations.export.working') : t('operations.audit.export.submit')}
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
      </DialogContent>
    </Dialog>
  )
}
