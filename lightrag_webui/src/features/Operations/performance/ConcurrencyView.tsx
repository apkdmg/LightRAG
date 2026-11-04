import { useTranslation } from 'react-i18next'
import { Meter } from '@/components/admin'
import { opsApi } from '../api'
import { Panel } from '../components/Panel'
import OpsLink from '../components/OpsLink'
import { useOpsNav } from '../context'
import { fmtClock, fmtCount, fmtPercent, utcTitle } from '../format'
import { useOpsPoll } from '../poll'
import { QUEUE_ROLES } from './model'
import { Th } from './shared'

/** "Now" values from heartbeat snapshots (every 15 s); the tab reads them once a minute (§3.10). */
const CONCURRENCY_SECONDS = 60

/**
 * Performance › Concurrency (§2.7, FR-44): running, in-flight and queued
 * calls per role summed over every loaded instance on every worker, from the
 * heartbeat snapshots (the server never loads a workspace to answer), plus
 * the ten instances with the most queued. A ratio appears only against
 * OPS_PROVIDER_CONCURRENCY_BUDGET: summed max_async would mislead, because
 * those limits are per workspace.
 */
export default function ConcurrencyView() {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const poll = useOpsPoll((signal) => opsApi.perfConcurrency(signal), [], CONCURRENCY_SECONDS)
  return (
    <Panel
      poll={poll}
      freshness={CONCURRENCY_SECONDS}
      title={t('operations.perf.concurrency.title')}
      sub={t('operations.perf.concurrency.sub')}
      approximate={!!poll.data?.approximate}
    >
      {(data) => (
        <div className="flex flex-col gap-4">
          {!data.heartbeats ? <p className="m-0 rounded-lg border border-dashed px-3 py-2 text-[13px]">{t('operations.perf.concurrency.noHeartbeats')}</p> : null}
          <div className="overflow-x-auto">
            <table className="ops-stack w-full min-w-[480px] text-[13px]">
              <caption className="sr-only">{t('operations.perf.concurrency.rolesCaption')}</caption>
              <thead>
                <tr className="text-muted-foreground border-b text-xs">
                  <Th>{t('operations.perf.concurrency.role')}</Th>
                  <Th right>{t('operations.perf.running')}</Th>
                  <Th right>{t('operations.perf.concurrency.inFlight')}</Th>
                  <Th right>{t('operations.perf.queued')}</Th>
                </tr>
              </thead>
              <tbody>
                {QUEUE_ROLES.map((role) => {
                  const r = data.roles[role]
                  return (
                    <tr key={role} className="border-b last:border-0">
                      <td className="p-1.5" data-label={t('operations.perf.concurrency.role')}>
                        {t(`operations.perf.concurrency.roles.${role}`)}
                      </td>
                      <td className="p-1.5 text-right font-mono text-xs" data-label={t('operations.perf.running')}>
                        {fmtCount(r?.running)}
                      </td>
                      <td className="p-1.5 text-right font-mono text-xs" data-label={t('operations.perf.concurrency.inFlight')}>
                        {fmtCount(r?.in_flight)}
                      </td>
                      <td className="p-1.5 text-right font-mono text-xs" data-label={t('operations.perf.queued')}>
                        {fmtCount(r?.queued)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="text-muted-foreground m-0 text-xs" title={utcTitle(data.at)}>
            {t('operations.perf.concurrency.scope', {
              instances: fmtCount(data.instances),
              workers: fmtCount(data.workers),
              time: data.at ? fmtClock(data.at) : '–'
            })}
          </p>
          {data.budget ? (
            <div className="flex max-w-[520px] flex-col gap-1">
              <span className="text-[13px]">
                {t('operations.perf.concurrency.budget', { used: fmtCount(data.budget.used), limit: fmtCount(data.budget.limit), pct: fmtPercent(data.budget.ratio) })}
              </span>
              <Meter value={Math.min(100, Math.max(0, (data.budget.ratio ?? 0) * 100))} color={(data.budget.ratio ?? 0) >= 0.9 ? 'var(--ops-warn)' : 'var(--ops-s1)'} />
            </div>
          ) : (
            <p className="text-muted-foreground m-0 text-xs">{t('operations.perf.concurrency.noBudget')}</p>
          )}
          <div className="flex flex-col gap-2">
            <h3 className="m-0 text-[13.5px] font-semibold">{t('operations.perf.concurrency.topTitle')}</h3>
            {data.top.length ? (
              <div className="overflow-x-auto">
                <table className="ops-stack w-full min-w-[480px] text-[13px]">
                  <caption className="sr-only">{t('operations.perf.concurrency.topTitle')}</caption>
                  <thead>
                    <tr className="text-muted-foreground border-b text-xs">
                      <Th>{t('operations.perf.workspace')}</Th>
                      <Th>{t('operations.perf.concurrency.worker')}</Th>
                      <Th right>{t('operations.perf.running')}</Th>
                      <Th right>{t('operations.perf.queued')}</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.top.map((r, i) => (
                      <tr key={`${r.workspace}-${r.worker_id ?? r.pid ?? i}`} className="border-b last:border-0">
                        <td className="p-1.5" data-label={t('operations.perf.workspace')}>
                          {r.workspace ? (
                            <OpsLink to={{ ...nav.view, drawer: { kind: 'workspace', id: r.workspace } }} className="break-all underline-offset-2 hover:underline">
                              {r.workspace}
                            </OpsLink>
                          ) : (
                            <span className="text-muted-foreground">{t('operations.perf.concurrency.defaultWorkspace')}</span>
                          )}
                        </td>
                        <td className="p-1.5 font-mono text-xs" data-label={t('operations.perf.concurrency.worker')}>
                          {r.pid !== null ? t('operations.perf.concurrency.pid', { pid: r.pid }) : (r.worker_id ?? '–')}
                        </td>
                        <td className="p-1.5 text-right font-mono text-xs" data-label={t('operations.perf.running')}>
                          {fmtCount(r.running)}
                        </td>
                        <td className="p-1.5 text-right font-mono text-xs" data-label={t('operations.perf.queued')}>
                          {fmtCount(r.queued)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-muted-foreground m-0 text-[13px]">{t('operations.perf.concurrency.noneQueued')}</p>
            )}
          </div>
        </div>
      )}
    </Panel>
  )
}
