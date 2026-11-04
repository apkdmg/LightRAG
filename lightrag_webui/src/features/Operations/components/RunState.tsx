import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import { breathePhase, useOpsMotion } from '../motion'
import { runLookOf } from '../severity'
import { StateIcon } from './StatusChip'

/**
 * The running dot (§2.12): a CSS opacity loop on the compositor, always next
 * to the "Running" text. Its phase is taken from the document timeline when
 * the dot mounts, so all rows breathe together. It stops (static dot) under
 * reduced motion, "⏸ auto", or when the panel is stale.
 */
export function RunningDot({ stale }: { stale?: boolean }) {
  const { reduced, paused } = useOpsMotion()
  const [phase] = useState(breathePhase)
  const live = !reduced && !paused && !stale
  return (
    <span
      aria-hidden="true"
      className={cn('ops-dot', live && 'ops-breathe ops-loop')}
      style={live ? ({ '--ops-phase': phase } as React.CSSProperties) : undefined}
    />
  )
}

/** A run's state as icon (or dot) plus text, with its plain-language explanation as a tooltip. */
export function RunStateLabel({ state, stale }: { state: string | null | undefined; stale?: boolean }) {
  const { t } = useTranslation()
  const look = runLookOf(state)
  if (look.state === 'running') {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-semibold" style={{ color: 'var(--ops-info)' }} title={t(look.explain)}>
        <RunningDot stale={stale} />
        {t(look.label)}
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-semibold" style={{ color: `var(--ops-${look.tone})` }} title={t(look.explain)}>
      <StateIcon icon={look.icon} />
      {t(look.label)}
    </span>
  )
}
