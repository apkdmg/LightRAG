import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { RefreshCwIcon, SendIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { DepHealth, OpsHealth } from '../api'
import AvailabilityStrip from '../components/AvailabilityStrip'
import OpsLink from '../components/OpsLink'
import { Panel } from '../components/Panel'
import { StateIcon } from '../components/StatusChip'
import TopologyMap from '../components/TopologyMap'
import { useOpsNav } from '../context'
import { fmtClock, fmtCount, fmtDuration, fmtLatency, shortId, utcTitle } from '../format'
import Skeleton from '../components/Skeleton'
import { useNow, useStale } from '../hooks'
import { OpsPoll } from '../poll'
import { lookOf } from '../severity'
import { useProbeActions } from './actions'
import { PROBE_INTERVAL_SECONDS, depInterval, inRecheckAll, depIsStale, depState, detailParts, sortDeps, stripUptime } from './model'

/** "Probed by pid 812 (lease holder) every 30 s · last cycle 12:04:20 (1.4 s)". */
function CycleLine({ data }: { data: OpsHealth }) {
  const { t } = useTranslation()
  const c = data.cycle
  if (!c?.at) return <span>{t('operations.health.noCycle')}</span>
  const holder = typeof c.holder_pid === 'number' ? t('operations.health.holderPid', { pid: c.holder_pid }) : c.holder ? shortId(c.holder) : null
  return (
    <span title={utcTitle(c.at)}>
      {holder ? t('operations.health.probedBy', { holder, every: c.interval_s ?? PROBE_INTERVAL_SECONDS }) : t('operations.health.probedEvery', { every: c.interval_s ?? PROBE_INTERVAL_SECONDS })}
      {' · '}
      {t('operations.health.lastCycle', { time: fmtClock(c.at), duration: fmtDuration(c.duration_ms ?? null) })}
    </span>
  )
}

/**
 * One dependency card: name, state (icon and text), latency, the safe detail
 * line, and the 24 h strip. A card whose last check is older than twice its
 * own cadence reads Stale, never green. The card opens the Dependency drawer.
 */
function DepCard({ dep, now, fallbackAt, baseline }: { dep: DepHealth; now: number; fallbackAt: number | null; baseline: boolean }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const stale = depIsStale(dep, now, fallbackAt)
  const state = stale ? 'unknown' : depState(dep.status)
  const look = lookOf(state)
  const name = dep.name ?? t(`operations.dep.${dep.id}`, { defaultValue: dep.id })
  const statusText = stale
    ? t('operations.overall.stale')
    : dep.status === 'unprobed'
      ? t('operations.topology.notProbed')
      : dep.passive && dep.status === 'ok'
        ? t('operations.health.passive')
        : t(look.label)
  const parts = detailParts(dep).map(([key, params]) => t(key, params))
  const uptime = stripUptime(dep.strip)
  return (
    <OpsLink
      to={{ ...nav.view, drawer: { kind: 'dependency', id: dep.id } }}
      data-tone={look.tone}
      className={cn('ops-dep bg-card flex min-w-0 flex-col gap-1.5 rounded-lg border p-3 no-underline', stale && 'ops-stale')}
    >
      <span className="flex items-start justify-between gap-2">
        <span className="min-w-0 truncate text-[13.5px] font-semibold">{name}</span>
        <span className="inline-flex flex-none items-center gap-1 text-xs font-semibold" style={{ color: `var(--ops-${look.tone})` }}>
          <StateIcon icon={look.icon} />
          {statusText}
          {!stale && typeof dep.latency_ms === 'number' ? <span className="font-mono font-medium"> {fmtLatency(dep.latency_ms)}</span> : null}
        </span>
      </span>
      <span className="text-muted-foreground min-h-4 font-mono text-xs break-words">
        {dep.status === 'down' || dep.status === 'warn'
          ? [dep.error_class, dep.last_ok_at ? t('operations.health.lastOkAt', { time: fmtClock(dep.last_ok_at) }) : null, ...parts].filter(Boolean).join(' · ')
          : parts.join(' · ') || (dep.checked_at ? t('operations.health.checkedAt', { time: fmtClock(dep.checked_at) }) : ' ')}
      </span>
      {dep.strip.length ? (
        <span className="flex flex-col gap-0.5">
          <AvailabilityStrip cells={dep.strip} baseline={baseline} label={t('operations.strip.labelFor', { name })} />
          <span className="text-muted-foreground flex justify-between text-[11px]">
            <span>{t('operations.strip.window')}</span>
            {uptime !== null ? <span>{t('operations.strip.uptime', { pct: Math.round(uptime * 1000) / 10 })}</span> : null}
          </span>
        </span>
      ) : null}
      {stale ? <span className="text-xs">{t('operations.health.staleCard', { every: depInterval(dep) })}</span> : null}
    </OpsLink>
  )
}

/**
 * Health › Dependencies (§2.6): the topology map, who ran the last probe
 * cycle and when, Re-check all and the SMTP test, and one card per
 * dependency with its 48-cell availability strip.
 */
export default function DepsView({ poll, processLabel, probesOff }: { poll: OpsPoll<OpsHealth>; processLabel: string | null; probesOff: boolean }) {
  const { t } = useTranslation()
  const now = useNow(15_000)
  const { busy, recheckAll, smtp, smtpTest } = useProbeActions()
  // Same payload, same array: the topology is not rebuilt when the 15 s clock ticks.
  const deps = useMemo(() => sortDeps(poll.data?.deps ?? []), [poll.data])
  const hasSmtp = deps.some((d) => d.id === 'smtp')
  const recheckable = deps.filter(inRecheckAll)
  const { stale } = useStale(poll.updatedAt, PROBE_INTERVAL_SECONDS)
  return (
    <div className="flex flex-col gap-4">
      {probesOff ? (
        <p className="m-0 rounded-lg border border-dashed px-3 py-2 text-[13px]">{t('operations.health.probesOff')}</p>
      ) : null}
      {poll.data ? (
        <div className={stale || poll.error ? 'ops-stale' : undefined}>
          <TopologyMap deps={deps} processLabel={processLabel} />
        </div>
      ) : poll.error ? null : (
        <Skeleton className="h-[420px] w-full" />
      )}
      <Panel
        poll={poll}
        freshness={PROBE_INTERVAL_SECONDS}
        title={t('operations.health.depsTitle')}
        sub={poll.data ? <CycleLine data={poll.data} /> : undefined}
        actions={
          <>
            {hasSmtp ? (
              <button
                type="button"
                disabled={smtp.pending}
                aria-busy={smtp.pending || undefined}
                onClick={() => void smtpTest()}
                className="hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-2.5 text-[13px] disabled:opacity-60"
                title={t('operations.health.smtpTestHint')}
              >
                <SendIcon className="size-3.5" aria-hidden="true" />
                {smtp.pending ? t('operations.health.working') : t('operations.health.smtpTest')}
              </button>
            ) : null}
            <button
              type="button"
              disabled={busy.has('*') || !recheckable.length || probesOff}
              aria-busy={busy.has('*') || undefined}
              onClick={() => void recheckAll(deps)}
              className="hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-2.5 text-[13px] disabled:opacity-60"
              title={t('operations.health.recheckHint')}
            >
              <RefreshCwIcon className="size-3.5" aria-hidden="true" />
              {busy.has('*') ? t('operations.health.working') : t('operations.health.recheckAll')}
            </button>
          </>
        }
      >
        {(data) =>
          deps.length ? (
            <div className="flex flex-col gap-3">
              {smtp.result ? (
                <p className="m-0 text-[13px]">
                  {smtp.result.ok
                    ? t('operations.health.smtpOk', { latency: fmtLatency(smtp.result.latency_ms ?? null) })
                    : t('operations.health.smtpFailed', { error: smtp.result.error_class ?? t('operations.state.unknown') })}
                </p>
              ) : null}
              <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
                {deps.map((d) => (
                  <DepCard key={d.id} dep={d} now={now} fallbackAt={poll.updatedAt} baseline={poll.baseline} />
                ))}
              </div>
              <p className="text-muted-foreground m-0 text-xs">
                {t('operations.health.summary', {
                  ok: fmtCount(deps.filter((d) => d.status === 'ok').length),
                  total: fmtCount(deps.filter((d) => d.status !== 'unprobed').length)
                })}
                {data.cycle?.at ? null : ` · ${t('operations.health.noCycle')}`}
              </p>
            </div>
          ) : (
            <p className="text-muted-foreground m-0 text-sm">{probesOff ? t('operations.health.probesOffEmpty') : t('operations.health.noDeps')}</p>
          )
        }
      </Panel>
    </div>
  )
}
