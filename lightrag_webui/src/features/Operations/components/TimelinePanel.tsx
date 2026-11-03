import { useTranslation } from 'react-i18next'
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { OpsEvent, TimelineBucket, opsApi } from '../api'
import { fmtClock, fmtCount, utcTitle } from '../format'
import { OPS_CHART } from '../motion'
import { useOpsOverview, useOpsShell } from '../context'
import { OVERVIEW_REFRESH_SECONDS, TIMELINE_HOURS, TIMELINE_REFRESH_SECONDS, markerOffset } from '../overview'
import { OpsPoll, useOpsPoll } from '../poll'
import { Panel } from './Panel'

const HOUR_MS = 3600_000
const axisStyle = { fontSize: 11, fill: 'var(--muted-foreground)' }
/** Markers that mean something went wrong are drawn in the critical colour. */
const BAD_KINDS = new Set(['provider_rejection', 'run_orphaned'])

const markerColor = (kind: string) => (BAD_KINDS.has(kind) ? 'var(--ops-crit)' : 'var(--muted-foreground)')

function EventLabel({ e }: { e: OpsEvent }) {
  const { t } = useTranslation()
  const d = e.detail ?? {}
  const status = typeof d.status_code === 'number' ? d.status_code : typeof d.status === 'number' ? d.status : null
  const action = typeof d.action === 'string' ? d.action : null
  return (
    <>
      {t(`operations.event.${e.kind}`, { defaultValue: e.kind.replace(/_/g, ' ') })}
      {status !== null ? ` ${status}` : ''}
      {action ? ` · ${action}` : ''}
      {e.workspace ? ` · ${e.workspace}` : ''}
    </>
  )
}

function BucketChart({ buckets, events, from, to }: { buckets: TimelineBucket[]; events: OpsEvent[]; from: number; to: number }) {
  const { t } = useTranslation()
  const points = buckets
    .map((b) => ({ t: Date.parse(b.t), processed: b.processed ?? 0, failed: b.failed ?? 0 }))
    .filter((p) => Number.isFinite(p.t))
  const width = Math.max(4, Math.min(28, 600 / Math.max(points.length, 1)))
  return (
    <div className="ops-dim h-[150px]" role="img" aria-label={t('operations.timeline.chartLabel', { hours: TIMELINE_HOURS })}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={points} margin={{ top: 16, right: 8, bottom: 0, left: -18 }}>
          <CartesianGrid vertical={false} stroke="var(--border)" />
          <XAxis
            dataKey="t"
            type="number"
            scale="time"
            domain={[from, to]}
            tickFormatter={(v: number) => fmtClock(v).slice(0, 5)}
            tick={axisStyle}
            tickLine={false}
            axisLine={false}
          />
          <YAxis tick={axisStyle} tickLine={false} axisLine={false} allowDecimals={false} width={44} />
          <Tooltip
            isAnimationActive={false}
            labelFormatter={(v) => fmtClock(Number(v)).slice(0, 5)}
            formatter={(value, name) => [fmtCount(Number(value)), t(`operations.timeline.${String(name)}`)]}
          />
          <Bar dataKey="processed" fill="var(--ops-info)" barSize={width} {...OPS_CHART} />
          <Bar dataKey="failed" fill="var(--ops-crit)" barSize={Math.max(3, width / 3)} {...OPS_CHART} />
          {events.map((e, i) => {
            const at = Date.parse(e.ts)
            return Number.isFinite(at) && at >= from && at <= to ? (
              <ReferenceLine key={`${e.ts}-${i}`} x={at} stroke={markerColor(e.kind)} strokeDasharray="3 3" ifOverflow="hidden" />
            ) : null
          })}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/** Markers only, on a plain time axis (before docs-per-hour buckets exist). */
function MarkerStrip({ events, from, to }: { events: OpsEvent[]; from: number; to: number }) {
  const ticks = Array.from({ length: TIMELINE_HOURS + 1 }, (_, i) => from + i * HOUR_MS)
  return (
    <div className="ops-dim relative h-10" aria-hidden="true">
      <div className="absolute inset-x-0 top-4 border-t" />
      {ticks.map((tick) => (
        <span key={tick} className="text-muted-foreground absolute top-6 -translate-x-1/2 font-mono text-[10px]" style={{ left: `${markerOffset(tick, from, to) ?? 0}%` }}>
          {fmtClock(tick).slice(0, 5)}
        </span>
      ))}
      {events.map((e, i) => {
        const left = markerOffset(e.ts, from, to)
        return left === null ? null : (
          <span
            key={`${e.ts}-${i}`}
            className="absolute top-1 h-6 border-l border-dashed"
            style={{ left: `${left}%`, borderColor: markerColor(e.kind) }}
          />
        )
      })}
    </div>
  )
}

/**
 * "Last 6 hours" (§2.3, FR-5): docs processed and failures per hour when the
 * server sends buckets, and event markers (deployments, provider rejections,
 * admin actions) merged at read time by /events. Charts never animate.
 */
export default function TimelinePanel({ buckets, events }: { buckets: TimelineBucket[] | null | undefined; events?: OpsEvent[] | null }) {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  // The overview already carries the last 6 hours of markers; only an older server needs /events.
  const fromOverview = Array.isArray(events)
  const poll = useOpsPoll(
    async (signal) => {
      const to = Date.now()
      const from = to - TIMELINE_HOURS * HOUR_MS
      const items = await opsApi.events({ from: new Date(from).toISOString(), to: new Date(to).toISOString() }, signal)
      return { items, from, to }
    },
    [],
    TIMELINE_REFRESH_SECONDS,
    { enabled: !fromOverview }
  )
  // The window ends when the overview was confirmed, so markers do not drift between polls.
  const overview = useOpsOverview()
  const end = overview?.updatedAt ?? null
  const fromOverviewData = fromOverview && end !== null ? { items: events ?? [], from: end - TIMELINE_HOURS * HOUR_MS, to: end } : null
  const shown: OpsPoll<{ items: OpsEvent[]; from: number; to: number }> =
    fromOverview && overview ? { ...overview, data: fromOverviewData } : poll
  return (
    <Panel poll={shown} freshness={fromOverview ? OVERVIEW_REFRESH_SECONDS : TIMELINE_REFRESH_SECONDS} title={t('operations.timeline.title', { hours: TIMELINE_HOURS })}>
      {({ items, from, to }) => {
        const events = [...items].sort((a, b) => Date.parse(a.ts) - Date.parse(b.ts))
        return (
          <div className="flex flex-col gap-2">
            {buckets?.length ? <BucketChart buckets={buckets} events={events} from={from} to={to} /> : <MarkerStrip events={events} from={from} to={to} />}
            {events.length ? (
              <ul className="m-0 flex list-none flex-wrap gap-x-4 gap-y-1 p-0 text-xs">
                {events.map((e, i) => (
                  <li key={`${e.ts}-${i}`} className="inline-flex items-center gap-1.5">
                    <span aria-hidden="true" className="inline-block h-3 border-l border-dashed" style={{ borderColor: markerColor(e.kind) }} />
                    <span className="font-mono" title={utcTitle(e.ts)}>
                      {fmtClock(e.ts).slice(0, 5)}
                    </span>
                    <EventLabel e={e} />
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-muted-foreground m-0 text-xs">{t('operations.timeline.noEvents', { hours: TIMELINE_HOURS })}</p>
            )}
            {!buckets?.length ? (
              <p className="text-muted-foreground m-0 text-xs">
                {status?.features?.runs ? t('operations.timeline.noWork', { hours: TIMELINE_HOURS }) : t('operations.timeline.noBuckets')}
              </p>
            ) : null}
          </div>
        )
      }}
    </Panel>
  )
}
