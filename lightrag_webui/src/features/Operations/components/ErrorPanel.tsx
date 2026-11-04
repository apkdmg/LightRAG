import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { CopyIcon, RefreshCwIcon } from 'lucide-react'
import { navigationService } from '@/services/navigation'
import { OpsError } from '../errors'
import { ToneChip } from './StatusChip'

/** The request ID with a copy button, so support can paste it into the server logs search. */
export function RequestRef({ requestId }: { requestId: string }) {
  const { t } = useTranslation()
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(requestId)
      toast.success(t('operations.error.refCopied'))
    } catch {
      toast.error(t('operations.error.refCopyFailed'))
    }
  }
  return (
    <span className="text-muted-foreground inline-flex items-center gap-1 text-xs">
      <span>
        {t('operations.error.ref')} <code className="font-mono">{requestId}</code>
      </span>
      <button
        type="button"
        onClick={copy}
        className="hover:bg-muted hover:text-foreground inline-flex size-6 items-center justify-center rounded"
        aria-label={t('operations.error.copyRef')}
        title={t('operations.error.copyRef')}
      >
        <CopyIcon className="size-3.5" aria-hidden="true" />
      </button>
    </span>
  )
}

/**
 * A panel's error state (§2.2): the mapped message, the server's safe detail,
 * "Ref: <request_id>" with copy, and Retry. A 403 says admin access is
 * required and offers to sign in again.
 */
export default function ErrorPanel({ error, onRetry, compact }: { error: OpsError; onRetry?: () => void; compact?: boolean }) {
  const { t } = useTranslation()
  const forbidden = error.kind === 'forbidden'
  return (
    <div role="alert" className={compact ? 'flex flex-wrap items-center gap-2' : 'flex flex-col items-start gap-2'}>
      <ToneChip tone="crit" icon="x-circle">
        {t(error.messageKey)}
      </ToneChip>
      {error.detail && !forbidden ? <p className="text-muted-foreground m-0 text-sm break-words">{error.detail}</p> : null}
      <div className="flex flex-wrap items-center gap-2">
        {error.requestId ? <RequestRef requestId={error.requestId} /> : null}
        {forbidden ? (
          <button type="button" className="hover:bg-muted min-h-8 rounded-lg border px-3 text-[13px]" onClick={() => navigationService.navigateToLogin()}>
            {t('operations.error.signInAgain')}
          </button>
        ) : null}
        {onRetry && !forbidden ? (
          <button type="button" className="hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-3 text-[13px]" onClick={onRetry}>
            <RefreshCwIcon className="size-3.5" aria-hidden="true" />
            {t('operations.error.retry')}
          </button>
        ) : null}
      </div>
    </div>
  )
}
