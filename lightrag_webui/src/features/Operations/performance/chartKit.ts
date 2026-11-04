import type { OpsEvent } from '../api'
import { PERF_WINDOW_SECONDS, PerfWindow } from './model'

/**
 * Non-component pieces shared by the Performance charts and views: axis
 * styling, marker colours, series definitions and the time range of a
 * window (kept apart from the .tsx files so Fast Refresh keeps working).
 */

/** What every Performance section gets from the tab: the window and the event markers in it. */
export type PerfCtx = {
  window: PerfWindow
  events: OpsEvent[]
}

export type SeriesDef = {
  key: string
  label: string
  color: string
  /** A dashed line for the secondary series of a pair (max next to average). */
  dashed?: boolean
  width?: number
}

export const axisStyle = { fontSize: 11, fill: 'var(--muted-foreground)' }

/** Markers that mean something went wrong are drawn in the critical colour. */
const BAD_KINDS = new Set(['provider_rejection', 'run_orphaned', 'provider_pause'])
export const markerColor = (kind: string) => (BAD_KINDS.has(kind) ? 'var(--ops-crit)' : 'var(--muted-foreground)')

const pad = (n: number) => String(n).padStart(2, '0')

/** Axis ticks (local time): "09:14" up to a day, "11-01 09:14" beyond, so a week's ticks name their day. */
export const tickFormat = (spanMs: number) => (v: number) => {
  const d = new Date(v)
  if (Number.isNaN(d.getTime())) return '–'
  const hm = `${pad(d.getHours())}:${pad(d.getMinutes())}`
  return spanMs > 26 * 3600_000 ? `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${hm}` : hm
}

/**
 * The time axis of a section: it ends when the server answered (or when the
 * answer arrived), so the axis does not drift between polls.
 */
export function windowRange(window: PerfWindow, generatedAt: string | undefined, updatedAt: number | null): { from: number; to: number } {
  return spanEnding(PERF_WINDOW_SECONDS[window] * 1000, generatedAt, updatedAt)
}

/** A span of `ms` ending at the server's answer time, else at when it arrived. */
export function spanEnding(ms: number, generatedAt: string | undefined, updatedAt: number | null): { from: number; to: number } {
  const said = generatedAt ? Date.parse(generatedAt) : NaN
  const to = Number.isFinite(said) ? said : (updatedAt ?? Date.now())
  return { from: to - ms, to }
}

/** Latency series colours: one hue, p50 light to p99 dark (ordered, not categorical). */
export const Q_COLORS = { p50: 'var(--ops-q1)', p95: 'var(--ops-q2)', p99: 'var(--ops-q3)', avg: 'var(--ops-q2)', max: 'var(--ops-q3)' } as const

/** Latency lines: p50/p95/p99 up to 24 h, average and (dashed) max for the 7-day rollups. */
export function latencySeries(t: (k: string) => string, rollup: boolean): SeriesDef[] {
  return rollup
    ? [
      { key: 'avg', label: t('operations.perf.avg'), color: Q_COLORS.avg },
      { key: 'max', label: t('operations.perf.max'), color: Q_COLORS.max, dashed: true }
    ]
    : [
      { key: 'p50', label: 'p50', color: Q_COLORS.p50, width: 1.5 },
      { key: 'p95', label: 'p95', color: Q_COLORS.p95 },
      { key: 'p99', label: 'p99', color: Q_COLORS.p99, width: 1.5 }
    ]
}
