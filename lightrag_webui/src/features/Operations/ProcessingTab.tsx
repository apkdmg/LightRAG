import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Seg } from '@/features/TokenInsights/ui'
import { ProcessingSummary, opsApi } from './api'
import { Panel } from './components/Panel'
import { useOpsNav } from './context'
import { fmtCount } from './format'
import { OPS_SUBS } from './hashState'
import { useOpsPoll } from './poll'
import ActiveView from './processing/ActiveView'
import { BacklogState, DEFAULT_BACKLOG } from './processing/backlog'
import BacklogView from './processing/BacklogView'
import EmailQueueView from './processing/EmailQueueView'
import FailuresView from './processing/FailuresView'
import PausesView from './processing/PausesView'
import RunsView from './processing/RunsView'
import StuckView from './processing/StuckView'

/** Backlog totals come from a 10 s server cache; 15 s keeps the bar honest without extra load. */
const SUMMARY_SECONDS = 15

type Sub = 'active' | 'stuck' | 'backlog' | 'failures' | 'email' | 'pauses' | 'runs'

const MIX: { key: 'pending' | 'parsing' | 'analyzing' | 'processing' | 'failed'; color: string }[] = [
  { key: 'pending', color: 'var(--ops-unknown)' },
  { key: 'parsing', color: 'color-mix(in srgb, var(--ops-info) 55%, var(--background))' },
  { key: 'analyzing', color: 'var(--ops-paused)' },
  { key: 'processing', color: 'var(--ops-info)' },
  { key: 'failed', color: 'var(--ops-crit)' }
]

/** "Backlog: Pending 280 · Parsing 12 · … (+5 cancelled · 812 dup excluded)" and its clickable mix bar. */
function BacklogBar({ data, onPick }: { data: ProcessingSummary; onPick: (key: string) => void }) {
  const { t } = useTranslation()
  const b = data.backlog ?? {}
  const parts = MIX.map((m) => ({ ...m, n: typeof b[m.key] === 'number' ? (b[m.key] as number) : 0 }))
  const total = parts.reduce((a, p) => a + p.n, 0)
  return (
    <div className="flex flex-col gap-2">
      <p className="m-0 flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px]">
        <strong>{t('operations.backlog.title')}</strong>
        {parts.map((p) => (
          <span key={p.key} className="inline-flex items-center gap-1.5">
            <span aria-hidden="true" className="inline-block size-2.5 rounded-[3px]" style={{ background: p.color }} />
            {t(`operations.docStatus.${p.key}`)} {fmtCount(p.n)}
          </span>
        ))}
        <span className="text-muted-foreground">
          {t('operations.backlog.excluded', { cancelled: fmtCount(b.cancelled ?? 0), dup: fmtCount(b.dup ?? 0) })}
        </span>
      </p>
      {total > 0 ? (
        <div className="ops-dim flex h-3 gap-0.5 overflow-hidden rounded-[3px]" role="group" aria-label={t('operations.backlog.mixLabel')}>
          {parts
            .filter((p) => p.n > 0)
            .map((p) => (
              <button
                key={p.key}
                type="button"
                className="h-full min-w-[3px] hover:opacity-80"
                style={{ flexGrow: p.n, flexBasis: 0, background: p.color }}
                title={`${t(`operations.docStatus.${p.key}`)} ${fmtCount(p.n)}`}
                aria-label={t('operations.backlog.segment', { status: t(`operations.docStatus.${p.key}`), count: fmtCount(p.n) })}
                onClick={() => onPick(p.key)}
              />
            ))}
        </div>
      ) : null}
    </div>
  )
}

/**
 * Processing across all workspaces (§2.4): the backlog bar, then one of
 * Active · Stuck · Backlog · Failures · Email queue · Pauses · Runs. Only the
 * visible view polls. Everything here is read-only.
 */
export default function ProcessingTab() {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const sub = (nav.view.sub && OPS_SUBS.processing.includes(nav.view.sub) ? nav.view.sub : 'active') as Sub
  const summary = useOpsPoll((signal) => opsApi.summary(signal), [], SUMMARY_SECONDS)
  // The backlog view's filters live here, so a click on the mix bar can set them.
  const [backlog, setBacklog] = useState<BacklogState>(DEFAULT_BACKLOG)
  // Only a view the user switched to fades in; the first paint and deep links do not.
  const [fadeSub, setFadeSub] = useState<Sub | null>(null)
  const selectSub = (v: Sub) => {
    setFadeSub(v)
    nav.setSub(v)
  }

  const s = summary.data
  const count = (n: number | null | undefined) => (typeof n === 'number' ? ` ${fmtCount(n)}` : '')
  const stuckCount = s ? (s.stuck?.runs ?? 0) + (s.stuck?.docs ?? 0) + (s.stuck?.reservations ?? 0) : null
  const options: { value: Sub; label: string }[] = [
    { value: 'active', label: `${t('operations.sub.active')}${count(s ? (s.active_count ?? s.active?.length ?? 0) : null)}` },
    { value: 'stuck', label: `${t('operations.sub.stuck')}${count(stuckCount)}` },
    { value: 'backlog', label: t('operations.sub.backlog') },
    { value: 'failures', label: `${t('operations.sub.failures')}${count(s?.backlog?.failed)}` },
    { value: 'email', label: `${t('operations.sub.email')}${count(s?.email_queue?.total)}` },
    { value: 'pauses', label: `${t('operations.sub.pauses')}${count(s?.pauses?.count)}` },
    { value: 'runs', label: t('operations.sub.runs') }
  ]

  const pickSegment = (key: string) => {
    setBacklog((b) => ({ ...b, has: key === 'failed' ? ['failed'] : b.has.filter((f) => f !== 'failed'), sort: key === 'failed' ? 'failed' : 'oldest', dir: key === 'failed' ? 'desc' : 'asc', page: 1 }))
    selectSub('backlog')
  }

  return (
    <div className="flex flex-col gap-4">
      <Panel poll={summary} freshness={SUMMARY_SECONDS} approximate={!!s?.approximate} skeleton={<span className="bg-muted block h-12 w-full rounded-md" />}>
        {(data) => <BacklogBar data={data} onPick={pickSegment} />}
      </Panel>
      <div className="overflow-x-auto">
        <Seg value={sub} onChange={selectSub} options={options} label={t('operations.sub.label')} />
      </div>
      <div key={sub} className={fadeSub === sub ? 'ops-fade-in' : undefined}>
        {sub === 'active' ? <ActiveView /> : null}
        {sub === 'stuck' ? <StuckView /> : null}
        {sub === 'backlog' ? <BacklogView state={backlog} onChange={setBacklog} /> : null}
        {sub === 'failures' ? <FailuresView /> : null}
        {sub === 'email' ? <EmailQueueView /> : null}
        {sub === 'pauses' ? <PausesView /> : null}
        {sub === 'runs' ? <RunsView /> : null}
      </div>
    </div>
  )
}
