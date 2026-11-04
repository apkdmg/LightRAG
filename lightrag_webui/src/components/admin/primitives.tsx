import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

/**
 * Layout, KPI and table primitives shared by the admin consoles (Token
 * Insights and Operations, spec §2.11). They carry no feature knowledge:
 * purpose colours, severities and formatters stay with the feature that
 * owns them. Moved here from TokenInsights/ui.tsx without visual change.
 */

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

/** A segmented control: one pressed button per option, wrapping on narrow screens. */
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

/** A 24 px stroke icon from one SVG path, in the given colour. */
export function StatusIcon({ path, color }: { path: string; color: string }) {
  return (
    <svg className="size-4 flex-none" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={path} />
    </svg>
  )
}

/** A thin horizontal bar, `value` in percent (capped at 100). */
export const Meter = ({ value, color = 'var(--muted-foreground)' }: { value: number; color?: string }) => (
  <span className="bg-muted block h-1.5 overflow-hidden rounded-full">
    <span className="block h-full rounded-full transition-[width] duration-300" style={{ width: `${Math.min(100, value)}%`, background: color }} />
  </span>
)

/** A label and its value on one line. */
export const Kv = ({ k, children }: { k: ReactNode; children: ReactNode }) => (
  <div className="flex justify-between gap-3 text-xs">
    <span className="text-muted-foreground">{k}</span>
    <span className="text-right font-semibold">{children}</span>
  </div>
)

/** Bars scaled to the largest value; decorative (the numbers are shown elsewhere). */
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

/** A KPI: label, value and an optional note under it. */
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
