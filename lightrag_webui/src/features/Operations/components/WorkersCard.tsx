import { useTranslation } from 'react-i18next'
import { Card, Title } from '@/features/TokenInsights/ui'
import type { OpsWorker } from '../api'
import { useOpsNav } from '../context'
import { fmtAge, fmtBytes, fmtLatency } from '../format'
import { viewForTarget } from '../hashState'
import { currentWorkers, uptimeSeconds } from '../health/model'
import OpsLink from './OpsLink'
import { WorkerName } from './WorkersTable'

/** Rows shown on the Overview; the rest are one link away on Health › Workers. */
const MAX_ROWS = 6

/**
 * The Overview's workers card (§2.3): one line per process from heartbeats,
 * "pid · up · RSS · loop lag", and a stale heartbeat in words. The Manager
 * round trip shows only in multi-worker mode. Each line opens the Worker
 * drawer.
 */
export default function WorkersCard({ workers, now, multiWorker }: { workers: OpsWorker[] | null | undefined; now: number | null; multiWorker: boolean }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const rows = workers ?? []
  const shown = rows.slice(0, MAX_ROWS)
  const rtts = currentWorkers(rows)
    .map((w) => w.snapshot.manager_rtt_ms)
    .filter((n): n is number => typeof n === 'number')
  return (
    <Card className="flex min-w-0 flex-col gap-2">
      <Title>{t('operations.worker.title')}</Title>
      {shown.length ? (
        <ul className="m-0 flex list-none flex-col gap-0.5 p-0">
          {shown.map((w) => {
            const dead = w.alive === false && !w.superseded && !w.exited_at
            return (
              <li key={w.worker_id}>
                <OpsLink
                  to={{ ...nav.view, drawer: { kind: 'worker', id: w.worker_id } }}
                  className="hover:bg-muted flex flex-wrap items-center gap-x-3 gap-y-0.5 rounded-lg px-2 py-1.5 text-xs no-underline"
                >
                  <WorkerName w={w} single={!multiWorker} />
                  {dead ? (
                    <span className="text-[color:var(--ops-crit)]">{t('operations.worker.heartbeatAge', { age: fmtAge(w.heartbeat_age_s) })}</span>
                  ) : w.superseded || w.exited_at ? null : (
                    <span className="text-muted-foreground tabular-nums">
                      {t('operations.worker.line', {
                        up: fmtAge(now === null ? null : uptimeSeconds(w, now)),
                        rss: fmtBytes(w.snapshot.rss),
                        lag: fmtLatency(w.snapshot.loop_lag_ms)
                      })}
                    </span>
                  )}
                </OpsLink>
              </li>
            )
          })}
        </ul>
      ) : (
        <p className="text-muted-foreground m-0 text-sm">{t('operations.system.workersPending')}</p>
      )}
      {multiWorker && rtts.length ? <p className="text-muted-foreground m-0 text-xs">{t('operations.worker.managerLine', { rtt: fmtLatency(Math.max(...rtts)) })}</p> : null}
      <OpsLink to={viewForTarget(nav.view, { tab: 'health', sub: 'workers' })} className="text-xs underline-offset-2 hover:underline">
        {rows.length > MAX_ROWS ? t('operations.worker.all', { count: rows.length }) : t('operations.worker.open')}
      </OpsLink>
    </Card>
  )
}
