import { FormEvent, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { DownloadIcon, Loader2Icon } from 'lucide-react'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/Dialog'
import Textarea from '@/components/ui/Textarea'
import { LogExportFormat, opsApi } from '../api'
import { OpsError, toOpsError } from '../errors'
import { fmtCount } from '../format'
import { LogFilter, activeFields, effectiveWindow, exportFilters } from '../logs'
import { RequestRef } from './ErrorPanel'

/** The server caps an export at this many rows, newest first (FR-25). */
export const EXPORT_CAP = 10_000
/** Tracebacks are large: an export with them is capped lower on the server. */
export const EXPORT_CAP_WITH_EXC = 2_000
const MAX_REASON = 512
const MIN_REASON = 3

/**
 * "Export…" (§2.5, FR-25): the current filters as redacted NDJSON or CSV,
 * at most 10k rows (2k with tracebacks), tracebacks left out unless asked for. A reason is
 * required; the export is audited on the server and is operator-only. A
 * refusal stays in the dialog under the buttons (role="alert"); success
 * closes it and the file downloads.
 */
export default function ExportDialog({
  open,
  onOpenChange,
  filter,
  operator
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  filter: LogFilter
  operator: boolean
}) {
  const { t } = useTranslation()
  const [reason, setReason] = useState('')
  const [format, setFormat] = useState<LogExportFormat>('ndjson')
  const [includeExc, setIncludeExc] = useState(false)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<OpsError | null>(null)

  const trimmed = reason.trim()
  const valid = trimmed.length >= MIN_REASON && trimmed.length <= MAX_REASON
  const summary = [
    t('operations.logs.levelOption.' + filter.level),
    t(`operations.logs.window.${effectiveWindow(filter)}`),
    ...activeFields(filter).map((k) => `${t(`operations.logs.field.${k}`)}=${filter[k]}`)
  ].join(' · ')

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!valid || pending || !operator) return
    setPending(true)
    setError(null)
    try {
      const result = await opsApi.exportLogs({ filters: exportFilters(filter, Date.now()), reason: trimmed, include_exc: includeExc, format })
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
            <DialogTitle>{t('operations.export.title')}</DialogTitle>
            <DialogDescription>{t('operations.export.body', { cap: fmtCount(includeExc ? EXPORT_CAP_WITH_EXC : EXPORT_CAP) })}</DialogDescription>
          </DialogHeader>
          <p className="bg-muted/50 m-0 rounded-md border px-2.5 py-1.5 text-xs break-words">{summary}</p>
          {!operator ? <p className="m-0 text-[13px] text-[color:var(--ops-warn)]">{t('operations.export.operatorOnly')}</p> : null}
          <label className="flex flex-col gap-1 text-[13px]">
            <span className="font-medium">{t('operations.export.reason')}</span>
            <Textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              maxLength={MAX_REASON}
              rows={3}
              required
              placeholder={t('operations.export.reasonPlaceholder')}
              disabled={!operator}
            />
            <span className="text-muted-foreground text-xs">{t('operations.export.reasonHint')}</span>
          </label>
          <fieldset className="m-0 flex flex-wrap items-center gap-4 border-0 p-0 text-[13px]">
            <legend className="mb-1 font-medium">{t('operations.export.format')}</legend>
            {(['ndjson', 'csv'] as const).map((f) => (
              <label key={f} className="inline-flex items-center gap-1.5">
                <input type="radio" name="ops-export-format" value={f} checked={format === f} onChange={() => setFormat(f)} disabled={!operator} />
                {t(`operations.export.formats.${f}`)}
              </label>
            ))}
          </fieldset>
          <label className="inline-flex items-start gap-2 text-[13px]">
            <input type="checkbox" className="mt-0.5" checked={includeExc} onChange={(e) => setIncludeExc(e.target.checked)} disabled={!operator} />
            <span className="flex flex-col">
              {t('operations.export.includeExc')}
              <span className="text-muted-foreground text-xs">{t('operations.export.includeExcHint')}</span>
            </span>
          </label>
          <DialogFooter className="gap-2">
            <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => close(false)} disabled={pending}>
              {t('operations.export.cancel')}
            </button>
            <button
              type="submit"
              className="bg-primary text-primary-foreground inline-flex min-h-9 items-center justify-center gap-1.5 rounded-lg px-3 text-[13px] hover:opacity-90 disabled:opacity-50"
              disabled={!valid || pending || !operator}
              aria-busy={pending || undefined}
            >
              {pending ? <Loader2Icon className="size-3.5 animate-spin" aria-hidden="true" /> : <DownloadIcon className="size-3.5" aria-hidden="true" />}
              {pending ? t('operations.export.working') : t('operations.export.submit')}
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
