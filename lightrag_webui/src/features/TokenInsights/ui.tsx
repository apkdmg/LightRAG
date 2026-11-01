import { ReactNode, useEffect, useRef, useState } from 'react'
import { animate, useReducedMotion } from 'motion/react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import { ICON_CHECK, ICON_CLOCK, PURPOSES, PURPOSE_COLOR, Purpose, SEVERITY } from './meta'

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

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn('bg-card rounded-[10px] border p-4', className)}>{children}</div>
}

export function Title({ children, sub }: { children: ReactNode; sub?: ReactNode }) {
  return (
    <div className="mb-2">
      <h2 className="m-0 text-[15px] font-semibold">{children}</h2>
      {sub ? <p className="text-muted-foreground mt-0.5 text-[13px]">{sub}</p> : null}
    </div>
  )
}

export function Seg<T extends string | number>({
  value,
  options,
  onChange,
  label
}: {
  value: T
  options: { value: T; label: ReactNode; disabled?: boolean }[]
  onChange: (value: T) => void
  label?: string
}) {
  return (
    <div className="bg-muted inline-flex flex-wrap gap-0.5 rounded-lg p-0.5" role="group" aria-label={label}>
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          disabled={o.disabled}
          aria-pressed={value === o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            'inline-flex min-h-8 items-center gap-1.5 rounded-md px-2.5 py-1 text-[13px] transition-colors disabled:opacity-40',
            value === o.value ? 'bg-background font-medium shadow-sm' : 'text-muted-foreground hover:text-foreground'
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
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

export function StatusIcon({ path, color }: { path: string; color: string }) {
  return (
    <svg className="size-4 flex-none" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={path} />
    </svg>
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

export const Meter = ({ value, color = 'var(--muted-foreground)' }: { value: number; color?: string }) => (
  <span className="bg-muted block h-1.5 overflow-hidden rounded-full">
    <span className="block h-full rounded-full transition-[width] duration-300" style={{ width: `${Math.min(100, value)}%`, background: color }} />
  </span>
)

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

export const Kv = ({ k, children }: { k: ReactNode; children: ReactNode }) => (
  <div className="flex justify-between gap-3 text-xs">
    <span className="text-muted-foreground">{k}</span>
    <span className="text-right font-semibold">{children}</span>
  </div>
)

export const Spark = ({ values, color }: { values: number[]; color: string }) => {
  const max = Math.max(...values, 1)
  return (
    <span className="flex h-11 items-end gap-px border-b" aria-hidden="true">
      {values.map((v, i) => (
        <span key={i} className="min-w-0 flex-1 rounded-t-[1px] transition-[height] duration-300" style={{ height: `${(v / max) * 100}%`, background: color }} />
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

export function Tile({ label, value, note, hero }: { label: ReactNode; value: ReactNode; note?: ReactNode; hero?: boolean }) {
  return (
    <div className="min-w-[150px] flex-1">
      <div className="text-muted-foreground text-[13px]">{label}</div>
      <div className={cn('font-semibold tracking-tight', hero ? 'text-[34px] leading-tight' : 'text-2xl leading-snug')}>{value}</div>
      {note ? <div className="text-muted-foreground text-xs">{note}</div> : null}
    </div>
  )
}

export const Empty = ({ children }: { children: ReactNode }) => <p className="text-muted-foreground m-0 text-sm">{children}</p>

export const TextButton = ({ onClick, children, className }: { onClick: () => void; children: ReactNode; className?: string }) => (
  <button type="button" onClick={onClick} className={cn('hover:bg-muted min-h-7 rounded px-1 text-left underline underline-offset-2', className)}>
    {children}
  </button>
)

/** A row that reads like text and acts like a button. */
export const RowButton = ({ onClick, children, active }: { onClick?: () => void; children: ReactNode; active?: boolean }) => (
  <button
    type="button"
    disabled={!onClick}
    onClick={onClick}
    className={cn('block w-full rounded-lg px-2 py-2 text-left transition-colors', onClick && 'hover:bg-muted cursor-pointer', active && 'bg-muted')}
  >
    {children}
  </button>
)
