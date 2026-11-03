import { AnimatePresence, motion } from 'motion/react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import { ActiveRun } from '../api'
import { useOpsNav } from '../context'
import { ageSeconds, fmtAge, fmtClock, shortId, utcTitle } from '../format'
import { TWEEN, useOpsMotion } from '../motion'
import { runLookOf } from '../severity'
import OpsLink from './OpsLink'
import ProgressBar from './ProgressBar'
import { RunStateLabel } from './RunState'

const barTone = (state: string): 'serious' | 'crit' | 'paused' | undefined => {
  const look = runLookOf(state)
  if (look.tone === 'crit') return 'crit'
  if (look.tone === 'serious') return 'serious'
  if (look.tone === 'paused') return 'paused'
  return undefined
}

/** The run's plain-language second line: why it is paused, hung or being cancelled. */
function RunNote({ run }: { run: ActiveRun }) {
  const { t } = useTranslation()
  const notes: string[] = []
  if (run.state === 'paused') {
    notes.push(
      typeof run.pause?.status_code === 'number'
        ? t('operations.active.pausedCode', { code: run.pause.status_code, slot: run.pause.slot ?? '' })
        : t('operations.active.pausedNoCode')
    )
  }
  if (typeof run.progress_age_s === 'number' && run.state !== 'running') notes.push(t('operations.active.lastProgress', { age: fmtAge(run.progress_age_s) }))
  if (run.live?.cancellation_requested) notes.push(t('operations.active.cancelling'))
  if (run.live?.destructive_busy) notes.push(t('operations.active.destructive'))
  if (!notes.length) return null
  return <div className="text-muted-foreground text-xs">{notes.join(' · ')}</div>
}

/**
 * Runs in progress across all workspaces (§2.4 Active, Overview "Processing
 * now"). Rows enter and leave with a fade and shift by FLIP (≤ 50 rows); the
 * running dot breathes only next to "Running" text. Elapsed time is measured
 * against when the data was confirmed (`now`), so it never drifts between
 * polls. Each row opens the workspace drawer; there are no action buttons.
 */
export default function ActiveRunsTable({
  runs,
  now,
  stale,
  baseline,
  compact,
  multiWorker
}: {
  runs: ActiveRun[]
  now: number | null
  stale?: boolean
  baseline?: boolean
  compact?: boolean
  multiWorker?: boolean
}) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const { reduced, hidden } = useOpsMotion()
  const instant = !!baseline || reduced || hidden
  const col = (key: string) => t(`operations.active.col.${key}`)
  return (
    <div className="overflow-x-auto">
      <table className={cn('ops-stack w-full text-[13px]', compact ? 'min-w-[420px]' : 'min-w-[900px]')}>
        <thead>
          <tr className="text-muted-foreground border-b text-xs">
            {!compact ? <th className="p-2 text-left font-medium">{col('state')}</th> : null}
            <th className="p-2 text-left font-medium">{col('workspace')}</th>
            <th className="p-2 text-left font-medium">{col('kind')}</th>
            {compact ? <th className="p-2 text-left font-medium">{col('state')}</th> : null}
            {!compact ? <th className="p-2 text-left font-medium">{col('label')}</th> : null}
            {!compact ? <th className="p-2 text-left font-medium">{col('trigger')}</th> : null}
            {!compact ? <th className="p-2 text-left font-medium">{col('started')}</th> : null}
            <th className="p-2 text-right font-medium">{col('elapsed')}</th>
            <th className="p-2 text-left font-medium">{col('progress')}</th>
            {multiWorker && !compact ? <th className="p-2 text-left font-medium">{col('worker')}</th> : null}
          </tr>
        </thead>
        <tbody>
          <AnimatePresence initial={false}>
            {runs.map((run) => {
              const key = run.run_id ?? `ws:${run.workspace}`
              const elapsed = run.started_at ? ageSeconds(run.started_at, now ?? undefined) : null
              const open = () => nav.openDrawer({ kind: 'workspace', id: run.workspace })
              return (
                <motion.tr
                  key={key}
                  layout={instant ? false : 'position'}
                  initial={instant ? false : { opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={instant ? { opacity: 0, transition: { duration: 0 } } : { opacity: 0, transition: TWEEN.baseExit }}
                  transition={TWEEN.base}
                  className="ops-row-link border-b align-top last:border-0"
                  onClick={(e) => {
                    if ((e.target as HTMLElement).closest('a')) return
                    open()
                  }}
                >
                  {!compact ? (
                    <td className="p-2" data-label={col('state')}>
                      <span className="flex flex-col items-end gap-0.5 md:items-start">
                        <RunStateLabel state={run.state} stale={stale} />
                        <RunNote run={run} />
                      </span>
                    </td>
                  ) : null}
                  <td className="p-2" data-label={col('workspace')}>
                    <OpsLink to={{ ...nav.view, drawer: { kind: 'workspace', id: run.workspace } }} onNavigate={open} className="break-all underline-offset-2 hover:underline">
                      {run.workspace}
                    </OpsLink>
                  </td>
                  <td className="p-2 font-mono text-xs" data-label={col('kind')}>
                    {run.kind ? t(`operations.kind.${run.kind}`, { defaultValue: run.kind }) : '–'}
                  </td>
                  {compact ? (
                    <td className="p-2" data-label={col('state')}>
                      <RunStateLabel state={run.state} stale={stale} />
                    </td>
                  ) : null}
                  {!compact ? (
                    <td className="max-w-[220px] truncate p-2" data-label={col('label')} title={run.label ?? undefined}>
                      {run.label || '–'}
                    </td>
                  ) : null}
                  {!compact ? (
                    <td className="p-2" data-label={col('trigger')}>
                      {run.trigger ? t(`operations.trigger.${run.trigger}`, { defaultValue: run.trigger }) : '–'}
                    </td>
                  ) : null}
                  {!compact ? (
                    <td className="p-2 font-mono text-xs whitespace-nowrap" data-label={col('started')} title={utcTitle(run.started_at)}>
                      {fmtClock(run.started_at ?? null)}
                    </td>
                  ) : null}
                  <td className="p-2 text-right whitespace-nowrap tabular-nums" data-label={col('elapsed')}>
                    {fmtAge(elapsed)}
                  </td>
                  <td className="p-2" data-label={col('progress')}>
                    <ProgressBar done={run.docs_done} total={run.docs_total} tone={barTone(run.state)} />
                  </td>
                  {multiWorker && !compact ? (
                    <td className="p-2 font-mono text-xs" data-label={col('worker')}>
                      {typeof run.pid === 'number' ? run.pid : shortId(run.worker_id ?? null)}
                      {run.worker_alive === false ? <span className="text-[color:var(--ops-crit)]"> · {t('operations.active.workerGone')}</span> : null}
                    </td>
                  ) : null}
                </motion.tr>
              )
            })}
          </AnimatePresence>
        </tbody>
      </table>
    </div>
  )
}
