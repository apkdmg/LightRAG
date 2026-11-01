import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { useReducedMotion } from 'motion/react'
import {
  Area, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, LineChart, ReferenceDot, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis
} from 'recharts'
import { PURPOSES, PURPOSE_COLOR, Purpose, SEVERITY } from './meta'
import { fmtDay, fmtMoney, fmtTokens, isFirstOpen } from './ui'

const axisStyle = { fontSize: 11, fill: 'var(--ti-axis)' }

function TipBox({ title, rows, note }: { title: string; rows: { label: string; value: string; color?: string }[]; note?: string }) {
  return (
    <div className="bg-card min-w-[170px] rounded-lg border p-2.5 text-xs shadow-md">
      <div className="mb-1 font-semibold">{title}</div>
      {rows.map((r) => (
        <div key={r.label} className="flex items-center justify-between gap-3">
          <span className="inline-flex items-center gap-1.5">
            {r.color ? <span className="inline-block size-2.5 rounded-[3px]" style={{ background: r.color }} /> : null}
            {r.label}
          </span>
          <strong>{r.value}</strong>
        </div>
      ))}
      {note ? <div className="text-muted-foreground mt-1">{note}</div> : null}
    </div>
  )
}

export type ChartPoint = {
  key: string
  day: string
  hour?: number
  partial?: boolean
  future?: boolean
  gap?: boolean
  label: string
  values: Record<string, number>
  usual?: number | null
  anomaly?: string | null
  pins?: { kind: string }[]
}

function useAnimation() {
  const reduced = useReducedMotion()
  return { isAnimationActive: !reduced, animationDuration: isFirstOpen() ? 600 : 300 }
}

/** Combined stacked bars, or one row per purpose so a small one stays visible. */
export function UsageChart({
  points,
  purposes,
  unit,
  layout,
  onPick
}: {
  points: ChartPoint[]
  purposes: Purpose[]
  unit: 'tokens' | 'cost'
  layout: 'combined' | 'separate'
  onPick?: (day: string) => void
}) {
  const { t } = useTranslation()
  const animation = useAnimation()
  const fmt = unit === 'cost' ? fmtMoney : fmtTokens
  const data = useMemo(() => points.map((p) => ({ ...p, ...p.values, total: purposes.reduce((a, q) => a + (p.values[q] || 0), 0) })), [points, purposes])

  const tooltip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null
    const p = payload[0].payload as ChartPoint & { total: number }
    const rows: { label: string; value: string; color?: string }[] = purposes.map((q) => ({ label: t(`insights.purpose.${q}`), value: fmt(p.values[q] || 0), color: PURPOSE_COLOR[q] }))
    if (p.usual !== null && p.usual !== undefined && unit === 'tokens') rows.push({ label: t('insights.chart.usual'), value: fmtTokens(p.usual) })
    const note = p.future ? t('insights.chart.future') : p.partial ? t('insights.chart.soFar') : p.gap ? t('insights.chart.notRecording') : p.anomaly ? t(`insights.severity.${p.anomaly}`) : p.pins?.length ? t('insights.chart.events', { count: p.pins.length }) : undefined
    return <TipBox title={p.label} rows={rows} note={note} />
  }

  if (layout === 'separate') {
    return (
      <div className="flex flex-col gap-3">
        {purposes.map((q) => (
          <div key={q}>
            <div className="mb-0.5 inline-flex items-center gap-1.5 text-xs font-semibold">
              <span className="inline-block size-2.5 rounded-[3px]" style={{ background: PURPOSE_COLOR[q] }} />
              {t(`insights.purpose.${q}`)}
            </div>
            <ResponsiveContainer width="100%" height={110}>
              <BarChart data={data} syncId="ti-separate" margin={{ top: 4, right: 6, bottom: 0, left: 0 }}>
                <CartesianGrid vertical={false} stroke="var(--ti-grid)" />
                <XAxis dataKey="label" hide />
                <YAxis width={48} tick={axisStyle} axisLine={false} tickLine={false} tickFormatter={(v) => fmt(v)} />
                <Tooltip content={tooltip} cursor={{ fill: 'var(--muted)' }} />
                <Bar dataKey={q} fill={PURPOSE_COLOR[q]} radius={[3, 3, 0, 0]} {...animation} onClick={(d: any) => onPick?.(d.day ?? d.payload?.day)} cursor={onPick ? 'pointer' : undefined} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>
    )
  }

  const top = Math.max(...data.map((d) => Math.max(d.total, d.usual ?? 0)), 1)
  const flagged = data.filter((d) => d.anomaly)
  const pinned = data.filter((d) => d.pins?.length && !d.anomaly)
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 18, right: 6, bottom: 0, left: 0 }}>
        <CartesianGrid vertical={false} stroke="var(--ti-grid)" />
        <XAxis dataKey="label" tick={axisStyle} axisLine={{ stroke: 'var(--ti-axis)' }} tickLine={false} minTickGap={28} />
        <YAxis width={52} tick={axisStyle} axisLine={false} tickLine={false} tickFormatter={(v) => fmt(v)} domain={[0, top * 1.08]} />
        <Tooltip content={tooltip} cursor={{ fill: 'var(--muted)' }} />
        {purposes.map((q, i) => (
          <Bar
            key={q}
            dataKey={q}
            stackId="usage"
            fill={PURPOSE_COLOR[q]}
            stroke="var(--card)"
            strokeWidth={1}
            radius={i === purposes.length - 1 ? [3, 3, 0, 0] : undefined}
            {...animation}
            onClick={(d: any) => onPick?.(d.day ?? d.payload?.day)}
            cursor={onPick ? 'pointer' : undefined}
          >
            {data.map((d) => (
              <Cell key={d.key} fillOpacity={d.partial ? 0.55 : 1} />
            ))}
          </Bar>
        ))}
        {unit === 'tokens' ? (
          <Line dataKey="usual" stroke="var(--ti-axis)" strokeWidth={1.5} strokeDasharray="3 3" dot={false} connectNulls isAnimationActive={false} />
        ) : null}
        {flagged.map((d) => (
          <ReferenceDot
            key={d.key}
            x={d.label}
            y={top * 1.04}
            ifOverflow="extendDomain"
            shape={(props: any) => (
              <circle cx={props.cx} cy={props.cy} r={4.5} fill={SEVERITY[d.anomaly as keyof typeof SEVERITY]?.color} stroke="var(--card)" strokeWidth={2}>
                <title>{t(`insights.severity.${d.anomaly}`)}</title>
              </circle>
            )}
          />
        ))}
        {pinned.map((d) => (
          <ReferenceDot
            key={`${d.key}-pin`}
            x={d.label}
            y={top * 1.04}
            ifOverflow="extendDomain"
            shape={(props: any) => <path d={`M${props.cx} ${props.cy - 5}l5 5-5 5-5-5z`} fill="var(--foreground)" stroke="var(--card)" strokeWidth={1.5} />}
          />
        ))}
      </ComposedChart>
    </ResponsiveContainer>
  )
}

/** Work and tokens get two charts on a shared date axis, never two y-axes. */
export function WorkCharts({ days, purposes, unitLabel, perLabel }: { days: any[]; purposes: Purpose[]; unitLabel: string; perLabel: string }) {
  const { t } = useTranslation()
  const animation = useAnimation()
  const data = useMemo(() => days.map((d) => ({ ...d, ...d.tokens, label: fmtDay(d.day) })), [days])
  const tip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null
    const d = payload[0].payload
    return (
      <TipBox
        title={fmtDay(d.day, true)}
        rows={[
          ...purposes.map((p) => ({ label: t(`insights.purpose.${p}`), value: fmtTokens(d.tokens[p] || 0), color: PURPOSE_COLOR[p] })),
          { label: unitLabel, value: String(Math.round(d.units)) },
          { label: perLabel, value: fmtTokens(d.per_unit) }
        ]}
        note={d.partial ? t('insights.chart.soFar') : undefined}
      />
    )
  }
  return (
    <div className="flex flex-col gap-1">
      <div className="text-xs font-semibold">{t('insights.work.tokens')}</div>
      <ResponsiveContainer width="100%" height={170}>
        <BarChart data={data} syncId="ti-work" margin={{ top: 4, right: 6, bottom: 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--ti-grid)" />
          <XAxis dataKey="label" hide />
          <YAxis width={52} tick={axisStyle} axisLine={false} tickLine={false} tickFormatter={fmtTokens} />
          <Tooltip content={tip} cursor={{ fill: 'var(--muted)' }} />
          {purposes.map((p) => (
            <Bar key={p} dataKey={p} stackId="t" fill={PURPOSE_COLOR[p]} stroke="var(--card)" {...animation} />
          ))}
        </BarChart>
      </ResponsiveContainer>
      <div className="text-xs font-semibold">{unitLabel}</div>
      <ResponsiveContainer width="100%" height={110}>
        <BarChart data={data} syncId="ti-work" margin={{ top: 4, right: 6, bottom: 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--ti-grid)" />
          <XAxis dataKey="label" hide />
          <YAxis width={52} tick={axisStyle} axisLine={false} tickLine={false} allowDecimals={false} />
          <Tooltip content={tip} cursor={{ fill: 'var(--muted)' }} />
          <Bar dataKey="units" fill="var(--muted-foreground)" radius={[3, 3, 0, 0]} {...animation} />
        </BarChart>
      </ResponsiveContainer>
      <div className="text-xs font-semibold">{perLabel}</div>
      <ResponsiveContainer width="100%" height={110}>
        <LineChart data={data} syncId="ti-work" margin={{ top: 4, right: 6, bottom: 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--ti-grid)" />
          <XAxis dataKey="label" tick={axisStyle} axisLine={{ stroke: 'var(--ti-axis)' }} tickLine={false} minTickGap={28} />
          <YAxis width={52} tick={axisStyle} axisLine={false} tickLine={false} tickFormatter={fmtTokens} />
          <Tooltip content={tip} />
          <Line dataKey="per_unit" stroke="var(--foreground)" strokeWidth={2} dot={false} connectNulls {...animation} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

const lastOf = <T,>(items: T[]): T | undefined => items[items.length - 1]

export function ForecastChart({ forecast }: { forecast: any }) {
  const { t } = useTranslation()
  const reduced = useReducedMotion()
  const data = useMemo(() => {
    const past = (forecast.actual as number[]).map((value, i) => ({
      label: fmtDay(forecast.week_ending[i]),
      actual: value,
      system: forecast.actual_system[i],
      kind: 'actual'
    }))
    const last = past[past.length - 1]
    const end = forecast.week_ending[forecast.week_ending.length - 1]
    const future = (forecast.points as any[]).map((p) => {
      const when = new Date(end + 'T00:00:00')
      when.setDate(when.getDate() + p.week * 7)
      return {
        label: when.toLocaleDateString(undefined, { day: 'numeric', month: 'short' }),
        projected: p.value,
        range: [p.low, p.high],
        low: p.low,
        high: p.high,
        system: p.system,
        kind: 'projected',
        week: p.week
      }
    })
    if (last) Object.assign(last, { projected: last.actual, range: [last.actual, last.actual] })
    return [...past, ...future]
  }, [forecast])
  const today = lastOf(data.filter((d) => d.kind === 'actual'))?.label
  const tip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null
    const d = payload[0].payload
    return (
      <TipBox
        title={d.kind === 'actual' ? t('insights.forecast.weekEnding', { day: d.label }) : t('insights.forecast.weeksAhead', { count: d.week })}
        rows={[
          { label: t('insights.forecast.allUsage'), value: fmtTokens(d.actual ?? d.projected) },
          { label: t('insights.forecast.systemOnly'), value: fmtTokens(d.system) },
          ...(d.kind === 'projected' ? [{ label: t('insights.forecast.range'), value: `${fmtTokens(d.low)} – ${fmtTokens(d.high)}` }] : [])
        ]}
      />
    )
  }
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 8, right: 6, bottom: 0, left: 0 }}>
        <CartesianGrid vertical={false} stroke="var(--ti-grid)" />
        <XAxis dataKey="label" tick={axisStyle} axisLine={{ stroke: 'var(--ti-axis)' }} tickLine={false} minTickGap={28} />
        <YAxis width={52} tick={axisStyle} axisLine={false} tickLine={false} tickFormatter={fmtTokens} />
        <Tooltip content={tip} />
        <Area dataKey="range" stroke="none" fill="var(--muted-foreground)" fillOpacity={0.2} isAnimationActive={!reduced} animationDuration={700} />
        <Line dataKey="system" stroke="var(--muted-foreground)" strokeWidth={2} strokeDasharray="2 4" dot={false} isAnimationActive={false} />
        <Line dataKey="actual" stroke="var(--foreground)" strokeWidth={2} dot={false} isAnimationActive={false} />
        <Line dataKey="projected" stroke="var(--foreground)" strokeWidth={2} strokeDasharray="6 5" dot={false} isAnimationActive={false} />
        {today ? <ReferenceLine x={today} stroke="var(--ti-axis)" label={{ value: t('insights.forecast.today'), position: 'insideTopRight', fontSize: 11, fill: 'var(--ti-axis)' }} /> : null}
      </ComposedChart>
    </ResponsiveContainer>
  )
}

export function HoursChart({ hours, purposes }: { hours: any[]; purposes: Purpose[] }) {
  const { t } = useTranslation()
  const data = hours.map((h) => ({ ...h.tokens, usual: h.usual, label: String(h.hour).padStart(2, '0') }))
  return (
    <ResponsiveContainer width="100%" height={130}>
      <ComposedChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
        <XAxis dataKey="label" tick={axisStyle} axisLine={{ stroke: 'var(--ti-axis)' }} tickLine={false} interval={3} />
        <YAxis width={44} tick={axisStyle} axisLine={false} tickLine={false} tickFormatter={fmtTokens} />
        <Tooltip
          content={({ active, payload }: any) =>
            active && payload?.length ? (
              <TipBox title={`${payload[0].payload.label}:00`} rows={purposes.map((p) => ({ label: t(`insights.purpose.${p}`), value: fmtTokens(payload[0].payload[p] || 0), color: PURPOSE_COLOR[p] }))} />
            ) : null
          }
          cursor={{ fill: 'var(--muted)' }}
        />
        {purposes.map((p) => (
          <Bar key={p} dataKey={p} stackId="h" fill={PURPOSE_COLOR[p]} isAnimationActive={false} />
        ))}
      </ComposedChart>
    </ResponsiveContainer>
  )
}

/** Steps from the usual amount to the actual amount, one purpose at a time. */
export function WhySteps({ rows }: { rows: { id: string; tokens: number; usual: number | null }[] }) {
  const { t } = useTranslation()
  const reduced = useReducedMotion()
  const known = rows.filter((r) => r.usual !== null)
  if (!known.length) return null
  const usual = known.reduce((a, r) => a + (r.usual || 0), 0)
  const steps: { name: string; base: number; size: number; color: string; value: string }[] = [
    { name: t('insights.why.usual'), base: 0, size: usual, color: 'var(--ti-axis)', value: fmtTokens(usual) }
  ]
  let level = usual
  for (const r of known) {
    const delta = r.tokens - (r.usual || 0)
    steps.push({
      name: t(`insights.purpose.${r.id}`),
      base: delta >= 0 ? level : level + delta,
      size: Math.abs(delta),
      color: PURPOSE_COLOR[r.id as Purpose],
      value: (delta >= 0 ? '+' : '−') + fmtTokens(Math.abs(delta))
    })
    level += delta
  }
  steps.push({ name: t('insights.why.actual'), base: 0, size: level, color: 'var(--foreground)', value: fmtTokens(level) })
  const max = Math.max(...steps.map((s) => s.base + s.size), 1)
  return (
    <div className="flex h-36 items-end gap-2" role="img" aria-label={t('insights.why.stepsLabel')}>
      {steps.map((s, i) => (
        <div key={s.name} className="relative flex h-full min-w-0 flex-1 flex-col justify-end">
          <div
            className="relative rounded-[3px]"
            style={{
              height: `${Math.max((s.size / max) * 82, 1)}%`,
              marginBottom: `${(s.base / max) * 82}%`,
              background: s.color,
              animation: reduced ? undefined : `ti-rise 120ms ease-out ${i * 120}ms both`
            }}
          />
          <div className="absolute inset-x-0 bottom-[-34px] text-center text-[11px] leading-tight">
            <div className="truncate">{s.name}</div>
            <strong>{s.value}</strong>
          </div>
        </div>
      ))}
    </div>
  )
}

export { PURPOSES }
