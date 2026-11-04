import { ReactNode, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import { OpsHash } from '../hashState'
import { useAnimatedNumber, useTint } from '../motion'
import { fmtClock } from '../format'
import { lookOf, rankOf } from '../severity'
import OpsLink from './OpsLink'
import ScopeBadge from './ScopeBadge'
import { StateIcon } from './StatusChip'

export type KpiTileProps = {
  label: ReactNode
  /** A number tweens when its formatted text changes; text values (`display`) jump. */
  value?: number | null
  format?: (n: number) => string
  /** Unit of a formatted value (MB vs GB): a unit change jumps instead of tweening. */
  unit?: (n: number) => string | null
  display?: ReactNode
  sub?: ReactNode
  state: string
  scope?: string | null
  to: OpsHash
  /** The data came from a baseline poll: no tween, no tint, no delta. */
  baseline?: boolean
  /** When the previous value was confirmed, for "▲ +118 since 12:04:16". */
  since?: number | null
  stale?: boolean
}

const defaultFormat = (n: number) => Math.round(n).toLocaleString('en-US')

/**
 * A stale-aware KPI tile (§2.3, §2.12). Values are final on first paint. A
 * changed value tweens over 300 ms with a delta line in text; the tile is
 * tinted only when its severity changes. A stale tile is never green. The
 * whole tile is a link to its drill-down.
 */
export default function KpiTile({ label, value, format = defaultFormat, unit, display, sub, state, scope, to, baseline, since, stale }: KpiTileProps) {
  const { t } = useTranslation()
  const shownState = stale ? 'unknown' : state
  const look = lookOf(shownState)
  const rank = rankOf(shownState)
  const numeric = display === undefined
  const valueRef = useRef<HTMLSpanElement | null>(null)
  const n = useAnimatedNumber(valueRef, numeric ? value : null, { format, unit, baseline, since })
  const tint = useTint<HTMLAnchorElement>(shownState, baseline)
  return (
    <OpsLink
      ref={tint}
      to={to}
      className={cn(
        'ops-kpi bg-card hover:bg-muted flex min-w-0 flex-col gap-0.5 rounded-[10px] border p-3 no-underline',
        rank >= 2 ? 'border-[color:var(--ops-serious)]' : rank === 1 ? 'border-[color:var(--ops-warn)]' : '',
        rank === 3 && 'border-[color:var(--ops-crit)]',
        stale && 'ops-stale'
      )}
    >
      <span className="text-muted-foreground flex items-center justify-between gap-2 text-[13px]">
        <span className="truncate">{label}</span>
        {shownState !== 'ok' ? (
          <span className="inline-flex items-center gap-1 text-[11px] font-semibold" style={{ color: `var(--ops-${look.tone})` }}>
            <StateIcon icon={look.icon} className="size-3" />
            {t(stale ? 'operations.overall.stale' : look.label)}
          </span>
        ) : (
          <span className="sr-only">{t(look.label)}</span>
        )}
      </span>
      <span className="text-2xl leading-snug font-semibold tracking-tight tabular-nums">
        {numeric ? (
          <>
            <span ref={valueRef} aria-hidden="true">
              {n.text}
            </span>
            <span className="sr-only">{n.srText}</span>
          </>
        ) : (
          display
        )}
      </span>
      {sub ? <span className="text-muted-foreground truncate text-xs">{sub}</span> : null}
      {n.delta ? (
        <span className="truncate text-[11.5px] tabular-nums">
          <span aria-hidden="true">{n.delta.glyph} </span>
          {n.delta.since !== null
            ? t('operations.kpi.delta', { change: n.delta.text, time: fmtClock(n.delta.since) })
            : t('operations.kpi.deltaNoTime', { change: n.delta.text })}
        </span>
      ) : (
        <ScopeBadge scope={scope} />
      )}
    </OpsLink>
  )
}
