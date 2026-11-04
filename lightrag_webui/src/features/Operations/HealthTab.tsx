import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Seg } from '@/components/admin'
import { OpsHealth, opsApi } from './api'
import { useOpsNav, useOpsShell } from './context'
import { useHealthChanged } from './health/actions'
import ConfigView from './health/ConfigView'
import DepsView from './health/DepsView'
import SigninView from './health/SigninView'
import { PROBE_INTERVAL_SECONDS, depCounts } from './health/model'
import { HostView, WORKERS_SECONDS, WorkersView } from './health/WorkersView'
import { OPS_SUBS } from './hashState'
import { useAttentionSlot, useOpsMotion } from './motion'
import { useOpsPoll } from './poll'
import { isMultiWorker, toProcessMode } from './severity'
import { nodeRingCandidates } from './topology'

type Sub = 'deps' | 'workers' | 'host' | 'config' | 'signin'

/** The process label in the topology's container box: "uvicorn · 1 process" or "gunicorn · 3 workers". */
function useProcessLabel(): string | null {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  if (!status) return null
  const mode = toProcessMode(status.process.mode)
  if (mode === 'uvicorn') return t('operations.process.uvicorn')
  if (mode === 'gunicorn' && typeof status.process.expected_workers === 'number') return t('operations.process.gunicorn', { count: status.process.expected_workers })
  return t('operations.process.unknown')
}

/**
 * Health (§2.6, §2.13): the topology map (with live edge flow) and the
 * dependency grid, sign-in health, the workers table, host numbers and the
 * redacted config. The probe results refresh
 * every 30 s, aligned to the probe cycle; only the visible view polls, and
 * the config is read once. A dependency that has just gone down offers its
 * topology node one ring to the page's attention slot (§2.12 rule 2).
 */
export default function HealthTab() {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const { status } = useOpsShell()
  const { reduced, hidden, paused } = useOpsMotion()
  const slot = useAttentionSlot()
  const processLabel = useProcessLabel()
  const sub = (nav.view.sub && OPS_SUBS.health.includes(nav.view.sub) ? nav.view.sub : 'deps') as Sub
  const [fadeSub, setFadeSub] = useState<Sub | null>(null)
  const selectSub = (v: Sub) => {
    setFadeSub(v)
    nav.setSub(v)
  }

  const probesOff = status?.features?.probes === false
  // The Host view reads the Postgres card's extras (size, connections, blocked) from the same payload.
  const health = useOpsPoll((signal) => opsApi.health(signal), [], PROBE_INTERVAL_SECONDS, { enabled: sub === 'deps' || sub === 'host' })
  const workers = useOpsPoll((signal) => opsApi.workers(signal), [], WORKERS_SECONDS, { enabled: sub === 'workers' || sub === 'host' })
  const config = useOpsPoll((signal) => opsApi.config(signal), [], undefined, { enabled: sub === 'config' })
  const reloadHealth = health.reload
  useHealthChanged(() => {
    if (sub === 'deps') reloadHealth()
  })

  const previous = useRef<OpsHealth | null>(null)
  useEffect(() => {
    const prev = previous.current
    const next = health.data
    previous.current = next
    if (!slot || !prev || !next || prev === next || sub !== 'deps' || health.baseline || reduced || hidden || paused) return
    slot.offer(nodeRingCandidates(prev.deps, next.deps), Date.now())
  }, [health.data, health.baseline, reduced, hidden, paused, slot, sub])

  const multiWorker = isMultiWorker(status?.process.mode)
  const counts = health.data ? depCounts(health.data.deps) : null
  const options: { value: Sub; label: string }[] = [
    { value: 'deps', label: counts && counts.total ? `${t('operations.health.sub.deps')} ${counts.ok}/${counts.total}` : t('operations.health.sub.deps') },
    { value: 'signin', label: t('operations.health.sub.signin') },
    { value: 'workers', label: t('operations.health.sub.workers') },
    { value: 'host', label: t('operations.health.sub.host') },
    { value: 'config', label: t('operations.health.sub.config') }
  ]

  return (
    <div className="flex flex-col gap-4">
      <div className="overflow-x-auto">
        <Seg value={sub} onChange={selectSub} options={options} label={t('operations.health.sub.label')} />
      </div>
      <div key={sub} className={fadeSub === sub ? 'ops-fade-in' : undefined}>
        {sub === 'deps' ? (
          <DepsView
            poll={health}
            processLabel={processLabel}
            probesOff={probesOff || health.data?.enabled === false}
            flow={{ enabled: status?.available === true, metrics: status?.features?.metrics !== false }}
          />
        ) : null}
        {sub === 'workers' ? <WorkersView poll={workers} multiWorker={multiWorker} expected={status?.process.expected_workers ?? null} /> : null}
        {sub === 'host' ? <HostView poll={workers} deps={health.data?.deps ?? null} /> : null}
        {sub === 'config' ? <ConfigView poll={config} /> : null}
        {sub === 'signin' ? <SigninView metricsOff={status?.features?.metrics === false} /> : null}
      </div>
    </div>
  )
}
