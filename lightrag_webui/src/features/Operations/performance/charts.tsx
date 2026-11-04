import { ReactNode, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { cn } from '@/lib/utils'
import type { OpsEvent } from '../api'
import { fmtStamp, utcTitle } from '../format'
import { OPS_CHART } from '../motion'
import { SeriesDef, axisStyle, markerColor, tickFormat } from './chartKit'

/**
 * Chart pieces for the Performance tab and the Sign-in card. Every chart
 * spreads OPS_CHART (`isAnimationActive: false`): a monitoring chart that
 * redraws on a poll or a window change must not move (§2.12 "Charts"). One
 * y-axis per chart; two measures of different scale are two charts sharing a
 * time axis (`syncId`), so the hover lines up. Every chart has a "View as
 * table" toggle, and event markers (deployments, provider rejections, admin
 * actions) are vertical rules (FR-46).
 */

/** At most this many markers are drawn per chart; the list under the charts names them all. */
const MAX_MARKERS = 50

export function ChartLegend({ series, note }: { series: SeriesDef[]; note?: ReactNode }) {
  return (
    <div className="text-muted-foreground flex flex-wrap items-center gap-x-3.5 gap-y-1 text-xs">
      {series.map((s) => (
        <span key={s.key} className="inline-flex items-center gap-1.5">
          <svg width="18" height="8" aria-hidden="true" focusable="false">
            <line x1="0" y1="4" x2="18" y2="4" stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dashed ? '4 3' : undefined} />
          </svg>
          {s.label}
        </span>
      ))}
      {note ? <span>{note}</span> : null}
    </div>
  )
}

function Markers({ events, from, to }: { events: OpsEvent[]; from: number; to: number }) {
  return (
    <>
      {events.slice(0, MAX_MARKERS).map((e, i) => {
        const at = Date.parse(e.ts)
        return Number.isFinite(at) && at >= from && at <= to ? (
          <ReferenceLine key={`${e.ts}-${i}`} x={at} stroke={markerColor(e.kind)} strokeDasharray="3 3" ifOverflow="hidden" />
        ) : null
      })}
    </>
  )
}

type ChartProps = {
  rows: Record<string, number | null>[]
  series: SeriesDef[]
  from: number
  to: number
  format: (v: number) => string
  label: string
  events?: OpsEvent[]
  height?: number
  syncId?: string
}

/** Lines over time, one y-axis. Gaps stay gaps (`connectNulls={false}`). */
export function TimeLines({ rows, series, from, to, format, label, events = [], height = 180, syncId }: ChartProps) {
  return (
    <div className="ops-dim" style={{ height }} role="img" aria-label={label}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} syncId={syncId} margin={{ top: 8, right: 10, bottom: 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--border)" />
          <XAxis dataKey="t" type="number" scale="time" domain={[from, to]} tickFormatter={tickFormat(to - from)} tick={axisStyle} tickLine={false} axisLine={false} minTickGap={36} />
          <YAxis tick={axisStyle} tickLine={false} axisLine={false} width={56} tickFormatter={(v: number) => format(v)} />
          <Tooltip
            isAnimationActive={false}
            labelFormatter={(v) => fmtStamp(Number(v))}
            formatter={(value, name) => [value === null || value === undefined ? '–' : format(Number(value)), series.find((s) => s.key === name)?.label ?? String(name)]}
          />
          {series.map((s) => (
            <Line
              key={s.key}
              dataKey={s.key}
              stroke={s.color}
              strokeWidth={s.width ?? 2}
              strokeDasharray={s.dashed ? '4 3' : undefined}
              dot={rows.length === 1 ? { r: 4, fill: s.color } : false}
              connectNulls={false}
              {...OPS_CHART}
            />
          ))}
          <Markers events={events} from={from} to={to} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

/** Counts per bucket as bars (grouped, not stacked, so each count reads from the baseline). */
export function TimeBars({ rows, series, from, to, format, label, events = [], height = 160, syncId }: ChartProps) {
  return (
    <div className="ops-dim" style={{ height }} role="img" aria-label={label}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} syncId={syncId} margin={{ top: 8, right: 10, bottom: 0, left: 0 }} barGap={2}>
          <CartesianGrid vertical={false} stroke="var(--border)" />
          <XAxis dataKey="t" type="number" scale="time" domain={[from, to]} tickFormatter={tickFormat(to - from)} tick={axisStyle} tickLine={false} axisLine={false} minTickGap={36} />
          <YAxis tick={axisStyle} tickLine={false} axisLine={false} width={48} allowDecimals={false} tickFormatter={(v: number) => format(v)} />
          <Tooltip
            isAnimationActive={false}
            cursor={{ fill: 'var(--muted)' }}
            labelFormatter={(v) => fmtStamp(Number(v))}
            formatter={(value, name) => [value === null || value === undefined ? '–' : format(Number(value)), series.find((s) => s.key === name)?.label ?? String(name)]}
          />
          {series.map((s) => (
            <Bar key={s.key} dataKey={s.key} fill={s.color} radius={[3, 3, 0, 0]} maxBarSize={18} {...OPS_CHART} />
          ))}
          <Markers events={events} from={from} to={to} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/** The rows behind a chart, newest first: the "View as table" side of every chart. */
export function SeriesTable({ rows, series, format, caption }: { rows: Record<string, number | null>[]; series: SeriesDef[]; format: (v: number) => string; caption: string }) {
  const { t } = useTranslation()
  const shown = [...rows].sort((a, b) => (b.t ?? 0) - (a.t ?? 0))
  return (
    <div className="max-h-72 overflow-auto rounded-md border">
      <table className="w-full text-[13px]">
        <caption className="sr-only">{caption}</caption>
        <thead className="bg-card sticky top-0">
          <tr className="text-muted-foreground border-b text-xs">
            <th className="p-1.5 text-left font-medium">{t('operations.perf.time')}</th>
            {series.map((s) => (
              <th key={s.key} className="p-1.5 text-right font-medium">
                {s.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((r) => (
            <tr key={r.t} className="border-b last:border-0">
              <td className="p-1.5 font-mono text-xs" title={utcTitle(r.t ?? undefined)}>
                {fmtStamp(r.t ?? null)}
              </td>
              {series.map((s) => (
                <td key={s.key} className="p-1.5 text-right font-mono text-xs">
                  {r[s.key] === null || r[s.key] === undefined ? '–' : format(r[s.key] as number)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {!shown.length ? <p className="text-muted-foreground m-0 p-2 text-xs">{t('operations.perf.noPoints')}</p> : null}
    </div>
  )
}

/**
 * A chart with its title, legend and a "View as table" toggle. The toggle
 * is per chart and per viewer session; it swaps instantly (no animation).
 */
export function ChartBlock({
  title,
  sub,
  series,
  note,
  chart,
  table,
  empty,
  className
}: {
  title: ReactNode
  sub?: ReactNode
  series: SeriesDef[]
  note?: ReactNode
  chart: ReactNode
  table: ReactNode
  /** Shown instead of the chart when there is nothing to draw. */
  empty?: ReactNode
  className?: string
}) {
  const { t } = useTranslation()
  const [asTable, setAsTable] = useState(false)
  return (
    <div className={cn('flex min-w-0 flex-col gap-2', className)}>
      <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-1">
        <div className="min-w-0">
          <h3 className="m-0 text-[13.5px] font-semibold">{title}</h3>
          {sub ? <p className="text-muted-foreground m-0 text-xs">{sub}</p> : null}
        </div>
        {empty ? null : (
          <button type="button" aria-pressed={asTable} onClick={() => setAsTable((v) => !v)} className="hover:bg-muted min-h-7 rounded-md border px-2 text-xs">
            {asTable ? t('operations.perf.viewChart') : t('operations.perf.viewTable')}
          </button>
        )}
      </div>
      {empty ? (
        <p className="text-muted-foreground m-0 text-[13px]">{empty}</p>
      ) : (
        <>
          {series.length > 1 || note ? <ChartLegend series={series} note={note} /> : null}
          {asTable ? table : chart}
        </>
      )}
    </div>
  )
}

/** The markers in the window, as a list under the charts (the rules carry no text). */
export function MarkerList({ events, from, to }: { events: OpsEvent[]; from: number; to: number }) {
  const { t } = useTranslation()
  const shown = events.filter((e) => {
    const at = Date.parse(e.ts)
    return Number.isFinite(at) && at >= from && at <= to
  })
  if (!shown.length) return null
  return (
    <ul className="m-0 flex list-none flex-wrap gap-x-4 gap-y-1 p-0 text-xs" aria-label={t('operations.perf.markers')}>
      {shown.slice(0, MAX_MARKERS).map((e, i) => (
        <li key={`${e.ts}-${i}`} className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3 border-l border-dashed" style={{ borderColor: markerColor(e.kind) }} />
          <span className="font-mono" title={utcTitle(e.ts)}>
            {fmtStamp(e.ts)}
          </span>
          {t(`operations.event.${e.kind}`, { defaultValue: e.kind.replace(/_/g, ' ') })}
          {e.workspace ? ` · ${e.workspace}` : ''}
        </li>
      ))}
      {shown.length > MAX_MARKERS ? <li className="text-muted-foreground">{t('operations.perf.moreMarkers', { count: shown.length - MAX_MARKERS })}</li> : null}
    </ul>
  )
}
