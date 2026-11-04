import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import type { OpsWorker, QueueCounts } from '../api'
import { useOpsNav } from '../context'
import { fmtAge, fmtBytes, fmtCount, fmtLatency, utcTitle } from '../format'
import { poolText, uptimeSeconds, workerLook } from '../health/model'
import { lookOf } from '../severity'
import OpsLink from './OpsLink'
import { StateIcon } from './StatusChip'

const pct = (n: number | null | undefined) => (typeof n === 'number' && Number.isFinite(n) ? `${Math.round(n)}%` : '–')

const queue = (q: QueueCounts | null | undefined) => (q ? `${fmtCount(q.running ?? null)}/${fmtCount(q.queued ?? null)}` : '–')

/** "pid 812" or "server" (single process), with the worker's state as icon and text. */
export function WorkerName({ w, single }: { w: OpsWorker; single?: boolean }) {
  const { t } = useTranslation()
  const look = workerLook(w)
  const tone = lookOf(look.state).tone
  return (
    <span className="inline-flex items-center gap-1.5 text-[13px] font-semibold" style={{ color: `var(--ops-${tone})` }}>
      <StateIcon icon={lookOf(look.state).icon} />
      <span className="text-foreground">
        {single && !w.superseded && !w.exited_at ? t('operations.worker.server', { pid: w.pid ?? '–' }) : t('operations.worker.pid', { pid: w.pid ?? '–' })}
      </span>
      <span className="text-xs">{t(look.label)}</span>
    </span>
  )
}

/**
 * The workers table (§2.6, FR-32): one row per process from heartbeats. In
 * single-process mode there is one "server" row and no Manager column. Rows
 * from an earlier boot (superseded) and exited workers stay visible for 24 h
 * so a restart can be traced; they do not count toward live workers. Each
 * row opens the Worker drawer.
 */
export default function WorkersTable({ items, now, multiWorker }: { items: OpsWorker[]; now: number; multiWorker: boolean }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const col = (key: string) => t(`operations.worker.col.${key}`)
  const single = !multiWorker && items.filter((w) => !w.superseded && !w.exited_at).length <= 1
  return (
    <div className="overflow-x-auto">
      <table className="ops-stack w-full min-w-[1040px] text-[13px]">
        <thead>
          <tr className="text-muted-foreground border-b text-xs">
            <th className="p-2 text-left font-medium">{col('process')}</th>
            <th className="p-2 text-right font-medium">{col('up')}</th>
            <th className="p-2 text-right font-medium">{col('heartbeat')}</th>
            <th className="p-2 text-right font-medium">{col('cpu')}</th>
            <th className="p-2 text-right font-medium">{col('rss')}</th>
            <th className="p-2 text-right font-medium">{col('fds')}</th>
            <th className="p-2 text-right font-medium">{col('instances')}</th>
            <th className="p-2 text-left font-medium">{col('pool')}</th>
            <th className="p-2 text-right font-medium">{col('loopLag')}</th>
            {multiWorker ? <th className="p-2 text-right font-medium">{col('manager')}</th> : null}
            <th className="p-2 text-right font-medium">{col('llm')}</th>
            <th className="p-2 text-right font-medium">{col('embed')}</th>
            <th className="p-2 text-right font-medium">{col('rerank')}</th>
            <th className="p-2 text-left font-medium">{col('drainers')}</th>
          </tr>
        </thead>
        <tbody>
          {items.map((w) => {
            const s = w.snapshot
            const old = w.superseded || !!w.exited_at
            const dead = w.alive === false && !old
            const to = { ...nav.view, drawer: { kind: 'worker' as const, id: w.worker_id } }
            const dash = (v: string) => (dead || old ? '–' : v)
            return (
              <tr
                key={w.worker_id}
                className={cn('ops-row-link border-b align-top last:border-0', old && 'text-muted-foreground')}
                onClick={(e) => {
                  if ((e.target as HTMLElement).closest('a')) return
                  nav.openDrawer(to.drawer)
                }}
              >
                <td className="p-2" data-label={col('process')}>
                  <OpsLink to={to} className="no-underline hover:underline">
                    <WorkerName w={w} single={single} />
                  </OpsLink>
                  {w.self ? <span className="text-muted-foreground ms-1.5 text-xs">{t('operations.worker.self')}</span> : null}
                </td>
                <td className="p-2 text-right tabular-nums" data-label={col('up')} title={utcTitle(w.started_at)}>
                  {fmtAge(uptimeSeconds(w, now))}
                </td>
                <td className={cn('p-2 text-right tabular-nums', dead && 'font-semibold text-[color:var(--ops-crit)]')} data-label={col('heartbeat')}>
                  {fmtAge(w.heartbeat_age_s)}
                  {dead ? <span aria-hidden="true"> ✖</span> : null}
                </td>
                <td className="p-2 text-right tabular-nums" data-label={col('cpu')}>
                  {dash(pct(s.cpu_pct))}
                </td>
                <td className="p-2 text-right tabular-nums" data-label={col('rss')}>
                  {dash(fmtBytes(s.rss))}
                </td>
                <td className="p-2 text-right tabular-nums" data-label={col('fds')}>
                  {dash(fmtCount(s.fds))}
                </td>
                <td className="p-2 text-right tabular-nums" data-label={col('instances')}>
                  {dash(fmtCount(s.instances))}
                </td>
                <td className="p-2 font-mono text-xs" data-label={col('pool')}>
                  {dash(poolText(s.pool))}
                </td>
                <td className="p-2 text-right tabular-nums" data-label={col('loopLag')}>
                  {dash(fmtLatency(s.loop_lag_ms))}
                </td>
                {multiWorker ? (
                  <td className="p-2 text-right tabular-nums" data-label={col('manager')}>
                    {dash(s.manager_state === 'unresponsive' ? t('operations.worker.unresponsive') : fmtLatency(s.manager_rtt_ms))}
                  </td>
                ) : null}
                <td className="p-2 text-right font-mono text-xs" data-label={col('llm')}>
                  {dash(queue(s.queues.llm))}
                </td>
                <td className="p-2 text-right font-mono text-xs" data-label={col('embed')}>
                  {dash(queue(s.queues.embed))}
                </td>
                <td className="p-2 text-right font-mono text-xs" data-label={col('rerank')}>
                  {dash(queue(s.queues.rerank))}
                </td>
                <td className="max-w-[180px] truncate p-2 text-xs" data-label={col('drainers')} title={s.drainers.map((d) => d.workspace).join(', ') || undefined}>
                  {dash(s.drainers.length ? s.drainers.map((d) => d.workspace).join(', ') : '–')}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
