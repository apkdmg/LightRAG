import { useTranslation } from 'react-i18next'
import { Kv } from '@/components/admin'
import type { DepHealth, OpsWorkers } from '../api'
import { Panel } from '../components/Panel'
import WorkersTable from '../components/WorkersTable'
import { fmtBytes, fmtClock, fmtCount } from '../format'
import { OpsPoll } from '../poll'
import { workersState } from '../severity'
import { StatusChip } from '../components/StatusChip'
import { gpuDep, gpuLine, normalizeGpus } from './gpu'
import { currentWorkers, pgHostFacts } from './model'

/** Heartbeats are written every 15 s; the table refreshes with the Health tab (30 s). */
export const WORKERS_SECONDS = 30

/**
 * Health › Workers (§2.6, FR-32): "live of expected" against the process
 * model, and the per-process table from heartbeats.
 */
export function WorkersView({ poll, multiWorker, expected }: { poll: OpsPoll<OpsWorkers>; multiWorker: boolean; expected: number | null }) {
  const { t } = useTranslation()
  return (
    <Panel
      poll={poll}
      freshness={WORKERS_SECONDS}
      title={t('operations.worker.title')}
      sub={poll.updatedAt ? t('operations.worker.sub', { time: fmtClock(poll.updatedAt) }) : undefined}
      actions={
        poll.data ? (
          (() => {
            const live = currentWorkers(poll.data.items).filter((w) => w.alive).length
            const exp = poll.data.expected ?? expected
            return (
              <StatusChip state={workersState(exp, live)}>
                {multiWorker ? t('operations.worker.liveOf', { live, expected: exp ?? '–' }) : t('operations.worker.liveSingle', { live })}
              </StatusChip>
            )
          })()
        ) : null
      }
    >
      {(data) =>
        data.items.length ? (
          <div className="flex flex-col gap-2">
            <WorkersTable items={data.items} now={poll.updatedAt ?? 0} multiWorker={multiWorker} />
            {!multiWorker ? <p className="text-muted-foreground m-0 text-xs">{t('operations.worker.singleNote')}</p> : null}
          </div>
        ) : (
          <p className="text-muted-foreground m-0 text-sm">{t('operations.worker.none')}</p>
        )
      }
    </Panel>
  )
}

/** FR-33: Postgres size, connections against max_connections, blocked backends and 5-minute counter deltas. */
function PostgresFacts({ deps }: { deps: DepHealth[] | null }) {
  const { t } = useTranslation()
  const pg = pgHostFacts(deps)
  if (!pg) return <Kv k={t('operations.host.pg.title')}>{deps ? t('operations.host.pg.none') : '–'}</Kv>
  const d = pg.delta
  return (
    <>
      <Kv k={t('operations.host.pg.size')}>{pg.dbSizeBytes !== null ? fmtBytes(pg.dbSizeBytes) : '–'}</Kv>
      <Kv k={t('operations.host.pg.connections')}>
        {pg.connections !== null && pg.maxConnections !== null
          ? t('operations.host.pg.connOf', { used: fmtCount(pg.connections), max: fmtCount(pg.maxConnections), pct: Math.round(pg.connectionsPct ?? 0) })
          : '–'}
      </Kv>
      <Kv k={t('operations.host.pg.blocked')}>
        {pg.blocked === null ? '–' : pg.blocked > 0 ? t('operations.host.pg.blockedFor', { count: pg.blocked, seconds: Math.round(pg.blockedMaxS ?? 0) }) : t('operations.host.pg.noneBlocked')}
      </Kv>
      {d ? (
        <Kv k={t('operations.host.pg.activity', { minutes: Math.max(1, Math.round(d.windowS / 60)) })}>
          {t('operations.host.pg.delta', {
            commits: fmtCount(d.commits),
            rollbacks: fmtCount(d.rollbacks),
            deadlocks: fmtCount(d.deadlocks),
            temp: fmtBytes(d.tempBytes ?? 0)
          })}
          {d.cacheHitPct !== null ? ` · ${t('operations.host.pg.cacheHit', { pct: d.cacheHitPct })}` : ''}
        </Kv>
      ) : null}
      {pg.at ? <p className="text-muted-foreground m-0 text-xs">{t('operations.host.pg.sampled', { time: fmtClock(pg.at) })}</p> : null}
    </>
  )
}

/** The GPU line of the Host view: per GPU from the exporter card, else where GPU numbers come from. */
function GpuSummary({ deps }: { deps: DepHealth[] | null }) {
  const { t } = useTranslation()
  const gpus = normalizeGpus(gpuDep(deps)?.detail)
  if (!gpus.length) return <>{t('operations.host.gpuNote')}</>
  return (
    <span className="flex flex-col items-end">
      {gpus.map((g) => {
        const line = gpuLine(g)
        return <span key={g.index}>{t('operations.host.gpuLine', { index: g.index, util: line.util, mem: line.mem, temp: line.temp })}</span>
      })}
    </span>
  )
}

/**
 * Health › Host (§3.6, FR-33): container memory and CPU from cgroup v2, the
 * host load average (host-wide, labelled so), disk use and the Postgres
 * numbers from its probe. GPU numbers come from the GPU exporter card when
 * the server reads one (D5, OPS_GPU_EXPORTER_URL); otherwise only vLLM's
 * /metrics (KV cache, queues) says anything about the GPU.
 */
export function HostView({ poll, deps = null }: { poll: OpsPoll<OpsWorkers>; deps?: DepHealth[] | null }) {
  const { t } = useTranslation()
  return (
    <Panel poll={poll} freshness={60} title={t('operations.host.title')} sub={t('operations.host.sub')}>
      {(data) => {
        const h = data.host
        if (!h) return <p className="text-muted-foreground m-0 text-sm">{t('operations.host.none')}</p>
        const memPct = typeof h.mem_used === 'number' && typeof h.mem_limit === 'number' && h.mem_limit > 0 ? Math.round((h.mem_used / h.mem_limit) * 100) : null
        return (
          <div className="flex max-w-[640px] flex-col gap-1.5">
            <Kv k={t('operations.host.memory')}>
              {typeof h.mem_used === 'number'
                ? typeof h.mem_limit === 'number' && h.mem_limit > 0
                  ? t('operations.host.memOf', { used: fmtBytes(h.mem_used), limit: fmtBytes(h.mem_limit), pct: memPct })
                  : t('operations.host.memNoLimit', { used: fmtBytes(h.mem_used) })
                : '–'}
            </Kv>
            <Kv k={t('operations.host.cpu')}>{typeof h.cpu_pct === 'number' ? `${Math.round(h.cpu_pct)}%` : '–'}</Kv>
            <Kv k={t('operations.host.load')}>{h.load?.length ? h.load.map((n) => n.toFixed(2)).join(' / ') : '–'}</Kv>
            {h.disks.map((d) => (
              <Kv key={d.path} k={t('operations.host.disk', { path: d.path })}>
                {typeof d.used === 'number' && typeof d.total === 'number' && d.total > 0
                  ? t('operations.host.diskOf', { used: fmtBytes(d.used), total: fmtBytes(d.total), pct: Math.round((d.used / d.total) * 100) })
                  : fmtBytes(d.used)}
              </Kv>
            ))}
            <Kv k={t('operations.host.gpu')}>
              <GpuSummary deps={deps} />
            </Kv>
            {h.sampled_at ? <p className="text-muted-foreground m-0 text-xs">{t('operations.host.sampled', { time: fmtClock(h.sampled_at) })}</p> : null}
            <p className="text-muted-foreground m-0 text-xs">{t('operations.host.workers', { count: fmtCount(currentWorkers(data.items).length) })}</p>
            <h3 className="m-0 mt-2 text-[13px] font-semibold">{t('operations.host.pg.title')}</h3>
            <PostgresFacts deps={deps} />
          </div>
        )
      }}
    </Panel>
  )
}
