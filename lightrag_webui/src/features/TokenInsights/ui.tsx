import { useEffect, useRef, useState } from 'react'
import { animate, useReducedMotion } from 'motion/react'
import { useTranslation } from 'react-i18next'
import { StatusIcon } from '@/components/admin'
import { cn } from '@/lib/utils'
import { ICON_CHECK, ICON_CLOCK, PURPOSES, PURPOSE_COLOR, Purpose, SEVERITY } from './meta'

// Layout, KPI and table primitives (Card, Seg, Kv, Meter, Tile, …) live in
// '@/components/admin', shared with the Operations console. What stays here
// knows about purposes, Insights statuses and token/money formatting.

export const fmtTokens = (n: number | null | undefined) => {
  if (n === null || n === undefined) return '–'
  const a = Math.abs(n)
  if (a >= 1e9) return (n / 1e9).toFixed(2) + 'B'
  if (a >= 1e6) return (n / 1e6).toFixed(a >= 1e8 ? 0 : 1) + 'M'
  if (a >= 1e3) return (n / 1e3).toFixed(a >= 1e5 ? 0 : 1) + 'K'
  return String(Math.round(n))
}

export const fmtMoney = (n: number | null | undefined) => {
  if (n === null || n === undefined) return '–'
  if (n === 0) return '$0'
  if (n >= 1000) return '$' + Math.round(n).toLocaleString('en-US')
  if (n >= 100) return '$' + n.toFixed(0)
  if (n >= 1) return '$' + n.toFixed(2)
  return '$' + n.toFixed(3)
}

export const fmtCount = (n: number | null | undefined) =>
  n === null || n === undefined ? '–' : Math.round(n).toLocaleString('en-US')

export const fmtPct = (n: number | null | undefined) =>
  n === null || n === undefined ? '–' : Math.round(n * 100) + '%'

export const fmtChange = (n: number | null | undefined) => {
  if (n === null || n === undefined) return '–'
  const r = Math.round(n * 100)
  return r === 0 ? '0%' : (r > 0 ? '+' : '−') + Math.abs(r) + '%'
}

export const fmtDay = (iso: string, long = false) => {
  const d = new Date(iso + 'T00:00:00')
  return d.toLocaleDateString(undefined, long ? { weekday: 'short', day: 'numeric', month: 'short' } : { day: 'numeric', month: 'short' })
}

export const Swatch = ({ purpose, color, size = 11 }: { purpose?: Purpose; color?: string; size?: number }) => (
  <span
    aria-hidden="true"
    className="inline-block flex-none rounded-[3px]"
    style={{ width: size, height: size, background: color ?? (purpose ? PURPOSE_COLOR[purpose] : 'transparent') }}
  />
)

export const PurposeName = ({ id }: { id: string }) => {
  const { t } = useTranslation()
  return (
    <span className="inline-flex items-center gap-1.5">
      <Swatch purpose={id as Purpose} />
      {t(`insights.purpose.${id}`)}
    </span>
  )
}

/** Status colours always come with an icon and a label. */
export function StatusLabel({ status }: { status: { code: string; ratio?: number | null } }) {
  const { t } = useTranslation()
  const look: Record<string, { path: string; color: string }> = {
    above: SEVERITY.warning,
    unusual: SEVERITY.warning,
    below: SEVERITY.warning,
    normal: { path: ICON_CHECK, color: '#0ca30c' },
    learning: { path: ICON_CLOCK, color: 'var(--ti-axis)' },
    deleted: { path: ICON_CLOCK, color: 'var(--ti-axis)' }
  }
  const { path, color } = look[status.code] ?? look.learning
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-semibold">
      <StatusIcon path={path} color={color} />
      {t(`insights.status.${status.code}`, { ratio: status.ratio })}
    </span>
  )
}

export function SeverityLabel({ severity }: { severity: keyof typeof SEVERITY }) {
  const { t } = useTranslation()
  const { path, color } = SEVERITY[severity]
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-semibold">
      <StatusIcon path={path} color={color} />
      {t(`insights.severity.${severity}`)}
    </span>
  )
}

export function Mix({ values, tall }: { values: Record<string, number>; tall?: boolean }) {
  const total = PURPOSES.reduce((a, p) => a + (values[p] || 0), 0) || 1
  return (
    <span className={cn('flex gap-0.5', tall ? 'h-4' : 'h-2.5')}>
      {PURPOSES.filter((p) => values[p]).map((p) => (
        <span key={p} className="rounded-[2px] transition-[flex-grow] duration-300" style={{ flexGrow: values[p] / total, background: PURPOSE_COLOR[p] }} />
      ))}
    </span>
  )
}

// The count-up plays once per session; after that values move from old to new.
let introPlayed = false
export const markIntroPlayed = () => {
  introPlayed = true
}
export const isFirstOpen = () => !introPlayed

export function Counter({ value, format }: { value: number; format: (n: number) => string }) {
  const reduced = useReducedMotion()
  const [shown, setShown] = useState(reduced || !isFirstOpen() ? value : 0)
  const previous = useRef(reduced || !isFirstOpen() ? value : 0)
  useEffect(() => {
    if (reduced) return
    const controls = animate(previous.current, value, {
      duration: isFirstOpen() ? 0.6 : 0.3,
      ease: 'easeOut',
      onUpdate: setShown
    })
    previous.current = value
    return () => controls.stop()
  }, [value, reduced])
  return <>{format(reduced ? value : shown)}</>
}
