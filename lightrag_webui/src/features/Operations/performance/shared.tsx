import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { RefreshCwIcon } from 'lucide-react'
import { OpsPoll } from '../poll'
import { PerfWindow, perfRefreshSeconds } from './model'

/** "Refresh" for the manual windows (24 h, 7 d); the 15 m and 1 h windows refresh every 60 s by themselves. */
export function ManualRefresh({ window, poll }: { window: PerfWindow; poll: OpsPoll<unknown> }) {
  const { t } = useTranslation()
  if (perfRefreshSeconds(window)) return null
  return (
    <button
      type="button"
      onClick={poll.reload}
      disabled={poll.slow}
      aria-busy={poll.slow || undefined}
      title={t('operations.perf.manualHint')}
      className="hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-2.5 text-[13px] disabled:opacity-60"
    >
      <RefreshCwIcon className="size-3.5" aria-hidden="true" />
      {t('operations.refresh')}
    </button>
  )
}

/** OPS_METRICS_ENABLED=false: the counters behind this section are not recorded. */
export function MetricsOff() {
  const { t } = useTranslation()
  return <p className="m-0 rounded-lg border border-dashed px-3 py-2 text-[13px]">{t('operations.perf.metricsOff')}</p>
}

/** A table header cell; numbers align right. */
export function Th({ children, right }: { children: ReactNode; right?: boolean }) {
  return <th className={`p-1.5 font-medium ${right ? 'text-right' : 'text-left'}`}>{children}</th>
}
