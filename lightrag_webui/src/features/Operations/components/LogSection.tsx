import { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { LogEntry, opsApi } from '../api'
import Section from '../drawers/Section'
import { DEFAULT_LOG_FILTER, LogFilter, filterKey, logsQuery, logsView, normalizeFilter, onlyWorker } from '../logs'
import { useOpsPoll } from '../poll'
import { useOpsShell } from '../context'
import ErrorPanel from './ErrorPanel'
import { LogRows } from './LogRow'
import OpsLink from './OpsLink'
import Skeleton from './Skeleton'

/** Drawers refresh at their own 15 s pace (§3.10). */
const SECTION_SECONDS = 15
/** A drawer shows the newest lines; the Logs tab has the rest. */
export const SECTION_ROWS = 20

/** The newest rows for a drawer's filter; the window ends now. */
const fetchSection = (filter: LogFilter, signal: AbortSignal) => opsApi.logs(logsQuery(filter, Date.now(), SECTION_ROWS), signal)

/**
 * "Log lines" inside a drawer (§2.9): the newest WARNING+ rows for the
 * drawer's run, document, workspace or worker, each opening the entry drawer
 * on the breadcrumb, and "Open in Logs" with the same filter. Rows come from
 * one indexed query (run_id, doc_id, workspace) and refresh with the drawer.
 */
export default function LogSection({
  title,
  filter,
  aside,
  pid = null,
  emptyText,
  preloaded
}: {
  title: ReactNode
  filter: Partial<LogFilter>
  aside?: ReactNode
  /** The worker's pid, for the Worker drawer's own-rows check. */
  pid?: number | null
  emptyText?: string
  /** Rows the drawer's own answer already carries (the Document drawer): no second query. */
  preloaded?: LogEntry[] | null
}) {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const full = normalizeFilter({ ...DEFAULT_LOG_FILTER, ...filter })
  const own = !preloaded
  const poll = useOpsPoll((signal) => fetchSection(full, signal), [filterKey(full)], SECTION_SECONDS, { enabled: own })
  const data = own ? poll.data : { items: preloaded, available: true as const }
  const raw = data?.items ?? []
  const rows = full.worker_id ? onlyWorker(raw, full.worker_id, pid) : raw
  const sinkOff = status?.features?.log_sink === false
  return (
    <Section
      title={title}
      aside={
        <span className="flex items-center gap-2">
          {aside}
          <OpsLink to={logsView(full)} className="text-xs underline underline-offset-2">
            {t('operations.logs.openInLogs')}
          </OpsLink>
        </span>
      }
    >
      {own && poll.error && !poll.data ? <ErrorPanel error={poll.error} onRetry={poll.reload} compact /> : null}
      {!data && !(own && poll.error) ? <Skeleton className="h-16 w-full" /> : null}
      {data?.available === false ? <p className="text-muted-foreground m-0 text-xs">{t('operations.unavailableShort')}</p> : null}
      {data && data.available !== false ? (
        rows.length ? (
          <LogRows items={rows} inDrawer showWorkspace={!full.workspace} className="rounded-lg border" />
        ) : (
          <p className="text-muted-foreground m-0 text-xs">{emptyText ?? t('operations.logs.sectionEmpty', { level: full.level })}</p>
        )
      ) : null}
      {sinkOff ? <p className="text-muted-foreground m-0 text-xs">{t('operations.logs.sinkOffShort')}</p> : null}
    </Section>
  )
}
