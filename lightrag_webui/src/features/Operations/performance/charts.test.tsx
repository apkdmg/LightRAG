import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'
import { ChartBlock, ChartLegend, MarkerList, SeriesTable } from './charts'
import type { SeriesDef } from './chartKit'
import { spanEnding, tickFormat, windowRange } from './chartKit'

const T0 = Date.parse('2026-11-03T04:00:00Z')
const SERIES: SeriesDef[] = [
  { key: 'p50', label: 'p50', color: 'var(--ops-q1)' },
  { key: 'p95', label: 'p95', color: 'var(--ops-q2)' }
]
const ROWS = [
  { t: T0, p50: 1000, p95: 2000 },
  { t: T0 + 60_000, p50: null, p95: 2500 }
]

describe('Performance chart pieces', () => {
  test('every chart offers "View as table"; the chart is the default view', () => {
    const html = renderToString(<ChartBlock title="Extract latency" series={SERIES} chart={<div id="chart" />} table={<div id="table" />} />)
    expect(html).toContain('aria-pressed="false"')
    expect(html).toContain('operations.perf.viewTable')
    expect(html).toContain('id="chart"')
    expect(html).not.toContain('id="table"')
    // Two series: a legend, so identity is never colour alone.
    expect(html).toContain('p95')
  })

  test('an empty chart says so instead of drawing axes', () => {
    const html = renderToString(<ChartBlock title="Rerank" series={SERIES} chart={<div id="chart" />} table={<div />} empty="Nothing recorded" />)
    expect(html).toContain('Nothing recorded')
    expect(html).not.toContain('id="chart"')
    expect(html).not.toContain('operations.perf.viewTable')
  })

  test('the table lists rows newest first with a dash for gaps', () => {
    const html = renderToString(<SeriesTable rows={ROWS} series={SERIES} format={(v) => `${v} ms`} caption="Latency" />)
    expect(html.indexOf('2500 ms')).toBeLessThan(html.indexOf('2000 ms'))
    expect(html).toContain('>–</td>')
    expect(html).toContain('<caption class="sr-only">Latency</caption>')
  })

  test('legend marks are static SVG lines (no animation classes)', () => {
    const html = renderToString(<ChartLegend series={[...SERIES, { key: 'max', label: 'Max', color: 'x', dashed: true }]} />)
    expect(html).toContain('stroke-dasharray="4 3"')
    expect(html).not.toMatch(/ops-(flow|breathe|ring|pulse)/)
  })

  test('markers in the window are listed in text', () => {
    const html = renderToString(
      <MarkerList
        events={[
          { ts: new Date(T0 + 30_000).toISOString(), kind: 'deployment' },
          { ts: new Date(T0 - 3600_000).toISOString(), kind: 'provider_rejection' }
        ]}
        from={T0}
        to={T0 + 60_000}
      />
    )
    // Untranslated in tests, the kind falls back to its words.
    expect(html).toContain('deployment')
    expect(html).not.toContain('provider rejection')
  })

  test('the time axis ends at the server answer time, not the client clock', () => {
    expect(windowRange('1h', new Date(T0).toISOString(), 5)).toEqual({ from: T0 - 3600_000, to: T0 })
    expect(spanEnding(1000, undefined, T0)).toEqual({ from: T0 - 1000, to: T0 })
    expect(tickFormat(3600_000)(T0)).toMatch(/^\d\d:\d\d$/)
    expect(tickFormat(7 * 86_400_000)(T0)).toMatch(/^\d\d-\d\d \d\d:\d\d$/)
  })
})
