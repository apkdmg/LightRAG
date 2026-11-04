import { useCallback, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { DownloadIcon, PauseIcon, PlayIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Seg } from '@/features/TokenInsights/ui'
import ExportDialog from './components/ExportDialog'
import LogCaptureControl from './components/LogCaptureControl'
import LogFilterBar from './components/LogFilterBar'
import { useOpsNav, useOpsShell } from './context'
import { useLogCapture } from './logCapture'
import { LogFilter, decodeLogFilter, encodeLogFilter } from './logs'
import GroupedView from './logview/GroupedView'
import StreamView from './logview/StreamView'

type Sub = 'stream' | 'grouped'

/**
 * Logs (§2.5, US-6): WARNING+ lines from every process, redacted when they
 * were written. Filters live in the URL (`olq`) so a view can be pasted into
 * chat. Stream follows new rows every 5 s; Grouped counts by call site.
 * Export is audited and operator-only. INFO is stored only during an INFO
 * capture: one module, at most 60 minutes, started here by an operator.
 */
export default function LogsTab() {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const { status } = useOpsShell()
  const filter = decodeLogFilter(nav.view.logQuery)
  const sub: Sub = nav.view.sub === 'grouped' ? 'grouped' : 'stream'
  const [follow, setFollow] = useState(true)
  const [exportOpen, setExportOpen] = useState(false)
  const [captureOpen, setCaptureOpen] = useState(false)
  const capture = useLogCapture()
  // Only a view the user switched to fades in.
  const [fadeSub, setFadeSub] = useState<Sub | null>(null)

  const { setView, view } = nav
  const setFilter = useCallback((next: LogFilter) => setView({ ...view, logQuery: encodeLogFilter(next) }), [setView, view])
  /** INFO lines of the captured module, over the last hour. */
  const showCaptured = useCallback((module: string) => setFilter({ ...filter, level: 'INFO', module, window: filter.window === '15m' ? '15m' : '1h' }), [filter, setFilter])
  const captureOffered = capture.canCapture && !capture.poll.data?.active
  const selectSub = (v: Sub) => {
    setFadeSub(v)
    nav.setSub(v)
  }

  const operator = status?.operator !== false
  // The sink (OPS_LOG_SINK), not the store: features.logs only says stored lines can be read.
  const sinkOff = status?.features?.log_sink === false
  return (
    <div className="flex flex-col gap-3">
      {sinkOff ? (
        <p className="text-muted-foreground m-0 rounded-lg border border-dashed px-3 py-2 text-[13px]" role="note">
          {t('operations.logs.sinkOff')}
        </p>
      ) : null}
      <LogCaptureControl info={capture} open={captureOpen} onOpenChange={setCaptureOpen} initialModule={filter.module || undefined} onShow={showCaptured} />
      <LogFilterBar filter={filter} onChange={setFilter} />
      <div className="flex flex-wrap items-center gap-2">
        <Seg
          value={sub}
          onChange={selectSub}
          label={t('operations.logs.viewLabel')}
          options={[
            { value: 'stream' as const, label: t('operations.logs.stream') },
            { value: 'grouped' as const, label: t('operations.logs.grouped') }
          ]}
        />
        <span className="ms-auto inline-flex flex-wrap gap-2">
          {sub === 'stream' ? (
            <button
              type="button"
              aria-pressed={follow}
              onClick={() => setFollow((f) => !f)}
              title={follow ? t('operations.logs.followOnHint') : t('operations.logs.followOffHint')}
              className={cn('hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-2.5 text-[13px]', follow && 'bg-muted font-medium')}
            >
              {follow ? <PauseIcon className="size-3.5" aria-hidden="true" /> : <PlayIcon className="size-3.5" aria-hidden="true" />}
              {follow ? t('operations.logs.followOn') : t('operations.logs.followOff')}
            </button>
          ) : null}
          <button
            type="button"
            onClick={() => setExportOpen(true)}
            disabled={!operator}
            title={operator ? undefined : t('operations.export.operatorOnly')}
            className="hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-2.5 text-[13px] disabled:cursor-not-allowed disabled:opacity-60"
          >
            <DownloadIcon className="size-3.5" aria-hidden="true" />
            {t('operations.logs.export')}
          </button>
        </span>
      </div>
      <div key={sub} className={fadeSub === sub ? 'ops-fade-in' : undefined}>
        {sub === 'stream' ? <StreamView filter={filter} follow={follow} onCapture={captureOffered ? () => setCaptureOpen(true) : undefined} /> : <GroupedView filter={filter} />}
      </div>
      <p className="text-muted-foreground m-0 text-xs">{t('operations.logs.privacy')}</p>
      <ExportDialog open={exportOpen} onOpenChange={setExportOpen} filter={filter} operator={operator} />
    </div>
  )
}
