import { useTranslation } from 'react-i18next'
import { opsApi } from '../api'
import ActiveRunsTable from '../components/ActiveRunsTable'
import { Panel } from '../components/Panel'
import { useOpsShell } from '../context'
import { fmtClock } from '../format'
import { useStale } from '../hooks'
import { useOpsPoll } from '../poll'
import { isMultiWorker } from '../severity'

/** §3.10: Active refreshes every 5 s. */
const ACTIVE_SECONDS = 5
/** Run rows are cached for 5 s on the server (§3.6); the interim list for 15 s, so it goes stale after 30 s. */
const DURABLE_FRESHNESS = 5
const INTERIM_FRESHNESS = 15

/**
 * Runs in progress (§2.4 Active, FR-10): the running rows of the run table,
 * with their stage, progress, owner process and live state (running, paused,
 * hung, unresponsive, orphaned). Each row opens its Run drawer. When run
 * tracking is off (OPS_RUN_TRACKING=false) the server falls back to the
 * interim list (workspaces with in-flight docs and their live flags), which
 * is labelled "Approximate (interim)".
 */
export default function ActiveView() {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const poll = useOpsPoll((signal) => opsApi.active(signal), [], ACTIVE_SECONDS)
  const approximate = !!poll.data?.approximate
  const freshness = approximate ? INTERIM_FRESHNESS : DURABLE_FRESHNESS
  const { stale } = useStale(poll.updatedAt, freshness)
  return (
    <Panel
      poll={poll}
      freshness={freshness}
      title={t('operations.active.title')}
      sub={approximate ? t('operations.active.interimNote') : poll.data ? t('operations.active.durableNote') : undefined}
      approximate={approximate}
    >
      {(data) =>
        data.runs.length ? (
          <div className="flex flex-col gap-2">
            <ActiveRunsTable runs={data.runs} now={poll.updatedAt} stale={stale} baseline={poll.baseline} multiWorker={isMultiWorker(status?.process.mode)} />
            {data.truncated ? <p className="text-muted-foreground m-0 text-xs">{t('operations.active.truncated', { count: data.runs.length })}</p> : null}
          </div>
        ) : (
          <p className="text-muted-foreground m-0 text-sm">
            {approximate ? t('operations.active.noneChecked', { time: fmtClock(poll.updatedAt) }) : t('operations.active.noneDurable', { time: fmtClock(poll.updatedAt) })}
          </p>
        )
      }
    </Panel>
  )
}
