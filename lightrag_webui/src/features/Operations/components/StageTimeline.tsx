import { Fragment } from 'react'
import { useTranslation } from 'react-i18next'
import { DocStage } from '../api'
import { ageSeconds, fmtClock, fmtDuration, utcTitle } from '../format'
import { StateIcon } from './StatusChip'

/** A stage's duration: the server's, else from its start and end times. */
const durationOf = (s: DocStage): number | null => {
  if (typeof s.duration_ms === 'number') return s.duration_ms
  if (s.started_at && s.ended_at) {
    const d = ageSeconds(s.started_at, Date.parse(s.ended_at))
    return d === null ? null : d * 1000
  }
  return null
}

/**
 * The Document drawer's stage timeline (US-5): "enqueued 09:58 → parsing 0.4 s
 * → analyzing 2m 10s → processing ✖", built from the doc-status metadata
 * timings. Failed and running stages carry an icon and text, not colour alone.
 */
export default function StageTimeline({ stages }: { stages: DocStage[] }) {
  const { t } = useTranslation()
  if (!stages.length) return <p className="text-muted-foreground m-0 text-sm">{t('operations.document.noStages')}</p>
  return (
    <ol className="m-0 flex list-none flex-wrap items-center gap-x-1.5 gap-y-1 p-0 text-[13px]">
      {stages.map((s, i) => {
        const duration = durationOf(s)
        const failed = s.state === 'failed'
        const running = s.state === 'running'
        const interrupted = s.state === 'interrupted'
        const notRun = s.state === 'skipped' || s.state === 'pending' || s.state === 'unknown'
        return (
          <Fragment key={`${s.stage}-${i}`}>
            {i > 0 ? (
              <li aria-hidden="true" className="text-muted-foreground">
                →
              </li>
            ) : null}
            <li
              className="inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5"
              style={
                failed ? { color: 'var(--ops-crit)' } : running ? { color: 'var(--ops-info)' } : interrupted ? { color: 'var(--ops-warn)' } : undefined
              }
              title={utcTitle(s.started_at)}
            >
              {failed ? <StateIcon icon="x-circle" /> : running ? <StateIcon icon="dot" /> : interrupted ? <StateIcon icon="triangle" /> : null}
              <span className={notRun ? 'text-muted-foreground font-medium' : 'font-medium'}>{t(`operations.stage.${s.stage}`, { defaultValue: s.stage })}</span>
              {notRun ? (
                <span className="text-muted-foreground text-xs">{t(`operations.stage.state.${s.state}`)}</span>
              ) : (
                <span className="text-muted-foreground font-mono text-xs">
                  {duration !== null && s.stage !== 'enqueued' ? fmtDuration(duration) : fmtClock(s.started_at ?? null)}
                </span>
              )}
              {failed ? <span className="sr-only">{t('operations.stage.failedState')}</span> : null}
              {running ? <span className="sr-only">{t('operations.stage.runningState')}</span> : null}
              {interrupted ? <span className="sr-only">{t('operations.stage.state.interrupted')}</span> : null}
            </li>
          </Fragment>
        )
      })}
    </ol>
  )
}
