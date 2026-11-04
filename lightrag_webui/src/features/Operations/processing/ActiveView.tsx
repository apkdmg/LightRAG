import { useTranslation } from 'react-i18next'
import { opsApi } from '../api'
import ActiveRunsTable from '../components/ActiveRunsTable'
import { Panel } from '../components/Panel'
import { useOpsShell } from '../context'
import { fmtClock } from '../format'
import { useStale } from '../hooks'
import { useOpsPoll } from '../poll'
import { isMultiWorker } from '../severity'

/** §3.10: Active refreshes every 5 s (the server caches the interim list for 15 s). */
const ACTIVE_SECONDS = 5
/** The interim list is served from a 15 s cache, so it goes stale after 30 s, not 10 s. */
const ACTIVE_FRESHNESS = 15

/**
 * Runs in progress (§2.4 Active). Until run tracking ships, the list is the
 * interim one: workspaces with in-flight docs, most recent first, at most
 * 50, enriched with a read-only snapshot of their pipeline flags, and
 * labelled "Approximate (interim)".
 */
export default function ActiveView() {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const poll = useOpsPoll((signal) => opsApi.active(signal), [], ACTIVE_SECONDS)
  const { stale } = useStale(poll.updatedAt, ACTIVE_FRESHNESS)
  return (
    <Panel
      poll={poll}
      freshness={ACTIVE_FRESHNESS}
      title={t('operations.active.title')}
      sub={poll.data?.approximate ? t('operations.active.interimNote') : undefined}
      approximate={!!poll.data?.approximate}
    >
      {(data) =>
        data.runs.length ? (
          <div className="flex flex-col gap-2">
            <ActiveRunsTable runs={data.runs} now={poll.updatedAt} stale={stale} baseline={poll.baseline} multiWorker={isMultiWorker(status?.process.mode)} />
            {data.truncated ? <p className="text-muted-foreground m-0 text-xs">{t('operations.active.truncated', { count: data.runs.length })}</p> : null}
          </div>
        ) : (
          <p className="text-muted-foreground m-0 text-sm">{t('operations.active.noneChecked', { time: fmtClock(poll.updatedAt) })}</p>
        )
      }
    </Panel>
  )
}
