import { describe, expect, test } from 'bun:test'
import { decodeLogFilter } from '../logs'
import {
  UNMATCHED_ROUTE,
  bucketSeconds,
  errorShareRows,
  flowRatesFrom,
  fmtPerMin,
  latencyRows,
  latestPerMin,
  normalizePerfConcurrency,
  normalizePerfDeps,
  normalizePerfHttp,
  normalizePerfLlm,
  normalizePerfThroughput,
  normalizeSignin,
  normalizeTopWorkspaces,
  perMinRows,
  perfFreshnessSeconds,
  perfRefreshSeconds,
  quantile,
  ratePercentiles,
  ratioOf,
  routeLogText,
  routeLogsView,
  seriesFor,
  signinState,
  sumCalls,
  throughputDays,
  toMs,
  topKeys,
  topRoutes
} from './model'

const T0 = Date.parse('2026-11-03T04:00:00Z')
const min = (i: number) => new Date(T0 + i * 60_000).toISOString()

describe('windows (§2.7)', () => {
  test('15 m and 1 h refresh every 60 s; 24 h and 7 d are manual', () => {
    expect(perfRefreshSeconds('15m')).toBe(60)
    expect(perfRefreshSeconds('1h')).toBe(60)
    expect(perfRefreshSeconds('24h')).toBeUndefined()
    expect(perfRefreshSeconds('7d')).toBeUndefined()
  })

  test('a manual window is not "stale" a minute after it loaded', () => {
    expect(perfFreshnessSeconds('1h')).toBe(60)
    expect(perfFreshnessSeconds('24h')).toBeGreaterThan(600)
    expect(perfFreshnessSeconds('7d')).toBeGreaterThan(perfFreshnessSeconds('24h'))
  })

  test('throughput reads whole days', () => {
    expect(throughputDays('15m')).toBe(1)
    expect(throughputDays('24h')).toBe(1)
    expect(throughputDays('7d')).toBe(7)
  })
})

describe('time and rates', () => {
  test('timestamps from ISO text, epoch seconds or epoch ms', () => {
    expect(toMs('2026-11-03T04:00:00Z')).toBe(T0)
    expect(toMs(T0 / 1000)).toBe(T0)
    expect(toMs(T0)).toBe(T0)
    expect(toMs('soon')).toBeNull()
    expect(toMs(null)).toBeNull()
  })

  test('the bucket width: as said, else the smallest gap, else the window default', () => {
    expect(bucketSeconds({ bucket_s: 300 }, [])).toBe(300)
    expect(bucketSeconds({}, [T0, T0 + 120_000, T0 + 60_000])).toBe(60)
    expect(bucketSeconds({}, [], '7d')).toBe(3600)
  })

  test('a rate as a ratio, a percentage or counts', () => {
    expect(ratioOf(0.25)).toBe(0.25)
    expect(ratioOf(25)).toBe(0.25)
    expect(ratioOf(null, 3, 12)).toBe(0.25)
    expect(ratioOf(undefined, 3, 0)).toBeNull()
  })

  test('quantiles interpolate; an empty list has none', () => {
    expect(quantile([1, 2, 3, 4, 5], 0.5)).toBe(3)
    expect(quantile([0, 10], 0.9)).toBe(9)
    expect(quantile([], 0.5)).toBeNull()
  })

  test('edge labels: whole numbers from 10, one decimal below, zero for nothing', () => {
    expect(fmtPerMin(75.4)).toBe('75/min')
    expect(fmtPerMin(1234)).toBe('1,234/min')
    expect(fmtPerMin(2.26)).toBe('2.3/min')
    expect(fmtPerMin(0.04)).toBe('<0.1/min')
    expect(fmtPerMin(0)).toBe('0/min')
    expect(fmtPerMin(null)).toBe('0/min')
  })
})

describe('LLM (Q5, FR-40)', () => {
  const raw = {
    window: '1h',
    group: 'purpose',
    series: [
      { t: min(1), key: 'extract', calls: 30, p50_ms: 4000, p95_ms: 6200, p99_ms: 11000 },
      { t: min(0), key: 'extract', calls: 20, p50: 3900, p95: 6000, p99: 10000 },
      { t: min(0), key: 'query', calls: 2, p50: 3000, p95: 8000, p99: 9000 },
      { key: 'keyword', points: [{ t: min(0), calls: 5, p95_ms: 2200 }] },
      { key: 'broken' }
    ],
    table: [
      { key: 'extract', calls: 4812, p50_ms: 4100, p95_ms: 41000, p99_ms: 63000, error_rate: 0.182, truncated: 3, running: 22, queued: 40 },
      { key: 'query', calls: 96, p50: 3400, p95: 8900, p99: 12000, errors: 1 },
      { key: 'keyword', calls: 310, error_rate: 0.3 }
    ],
    queue: { running: 14, queued: 40 }
  }

  test('both spellings, nested points, sorted by time; rows without a time are dropped', () => {
    const d = normalizePerfLlm(raw, '1h', 'purpose')
    expect(d.series.map((p) => [p.key, p.t])).toEqual([
      ['extract', T0],
      ['keyword', T0],
      ['query', T0],
      ['extract', T0 + 60_000]
    ])
    expect(seriesFor(d.series, 'extract').map((p) => p.p95)).toEqual([6000, 6200])
    expect(d.bucket_s).toBe(60)
    expect(d.approx_7d).toBe(false)
    expect(d.queue).toEqual({ running: 14, queued: 40, in_flight: null })
  })

  test('rates: ratio, percentage and counts all become 0..1', () => {
    const d = normalizePerfLlm(raw, '1h', 'purpose')
    expect(d.table[0].error_rate).toBeCloseTo(0.182)
    expect(d.table[0].truncated_rate).toBeCloseTo(3 / 4812)
    expect(d.table[1].error_rate).toBeCloseTo(1 / 96)
    // 0.3 is a ratio (30 %); only values above 1 read as percentages.
    expect(d.table[2].error_rate).toBeCloseTo(0.3)
    expect(normalizePerfLlm({ table: [{ key: 'x', error_rate: 30 }] }).table[0].error_rate).toBeCloseTo(0.3)
  })

  test('the 7-day window is labelled approximate even when the server does not say', () => {
    expect(normalizePerfLlm({ series: [] }, '7d').approx_7d).toBe(true)
    expect(normalizePerfLlm({ series: [], approx_7d: false }, '7d').approx_7d).toBe(false)
  })

  test('small multiples: the busiest keys first, at most n', () => {
    const d = normalizePerfLlm(raw, '1h', 'purpose')
    expect(topKeys(d, 2)).toEqual(['extract', 'keyword'])
    // Without a table, the calls in the series rank the keys.
    expect(topKeys({ table: [], series: d.series }, 4)).toEqual(['extract', 'keyword', 'query'])
  })

  test('chart rows: percentiles up to 24 h, average and max for rollups', () => {
    const points = normalizePerfLlm({ series: [{ t: min(0), key: 'x', p95: 5, avg_ms: 3, max_ms: 9 }] }).series
    expect(latencyRows(points, false)).toEqual([{ t: T0, p50: null, p95: 5, p99: null }])
    expect(latencyRows(points, true)).toEqual([{ t: T0, avg: 3, max: 9 }])
  })

  test('calls summed over keys per bucket (the LLM edge)', () => {
    const d = normalizePerfLlm(raw, '1h')
    expect(sumCalls(d.series)).toEqual([
      { t: T0, calls: 27 },
      { t: T0 + 60_000, calls: 30 }
    ])
  })

  test('a partial or empty payload never throws', () => {
    const d = normalizePerfLlm(null, '15m')
    expect(d.series).toEqual([])
    expect(d.table).toEqual([])
    expect(d.queue).toBeNull()
    expect(d.window).toBe('15m')
  })
})

describe('embed and rerank (FR-41)', () => {
  test('series per role with the vLLM overlay; KV cache as a ratio; summary from totals when not sent', () => {
    const d = normalizePerfDeps(
      {
        bucket_s: 60,
        series: {
          embedding: [
            { t: min(1), calls: 10, errors: 1, p50_ms: 5, p95_ms: 12, p99_ms: 30, waiting: 0, running: 1, kv_cache: 22 },
            { t: min(0), calls: 30, errors: 0, p95_ms: 20 }
          ],
          rerank: [{ t: min(0), count: 4, p95: 140 }]
        },
        vllm: { embed: { num_requests_running: 3, num_requests_waiting: 0, gpu_cache_usage_perc: 0.22 }, vllm_rerank: { running: 8, waiting: 34, kv_cache: 71 } }
      },
      '15m'
    )
    expect(d.series.embed.map((p) => p.t)).toEqual([T0, T0 + 60_000])
    expect(d.series.embed[1].kv_cache).toBeCloseTo(0.22)
    expect(d.vllm.embed).toEqual({ running: 3, waiting: 0, kv_cache: 0.22 })
    expect(d.vllm.rerank?.kv_cache).toBeCloseTo(0.71)
    expect(d.summary.embed.calls).toBe(40)
    expect(d.summary.embed.error_rate).toBeCloseTo(1 / 40)
    expect(d.summary.embed.p95).toBe(20)
    expect(d.summary.rerank.calls).toBe(4)
  })

  test('the server summary wins when sent', () => {
    const d = normalizePerfDeps({ series: {}, summary: { embed: { calls: 9, error_rate: 0.5, p95_ms: 7 } } })
    expect(d.summary.embed).toEqual({ calls: 9, error_rate: 0.5, p50: null, p95: 7, p99: null })
    expect(d.vllm.embed).toBeNull()
  })
})

describe('HTTP (FR-42)', () => {
  const routes = [
    ...Array.from({ length: 14 }, (_, i) => ({ route: `/r${i}`, method: 'GET', calls: 100, p95_ms: (i + 1) * 10, err5xx: 0 })),
    { route: UNMATCHED_ROUTE, calls: 40, p95: 5, err5xx: 0.5 }
  ]

  test('the ten slowest by p95, and <unmatched> as exactly one row in its p95 place', () => {
    const d = normalizePerfHttp({ routes, series: [] }, '1h')
    const top = topRoutes(d.routes)
    expect(top.length).toBe(11)
    expect(top[0].route).toBe('/r13')
    expect(top.filter((r) => r.unmatched).length).toBe(1)
    expect(top[top.length - 1].route).toBe(UNMATCHED_ROUTE)
    // rps from calls over the window when the server sends counts only.
    expect(d.routes[0].rps).toBeCloseTo(100 / 3600)
  })

  test('several <unmatched> rows (one per method) collapse into one', () => {
    const d = normalizePerfHttp({
      routes: [
        { route: UNMATCHED_ROUTE, method: 'GET', calls: 10, p95: 4, err5xx: 0 },
        { route: UNMATCHED_ROUTE, method: 'POST', calls: 5, p95: 9, err5xx: 0.2 },
        { route: '/query', calls: 3, p95: 900 }
      ]
    })
    const top = topRoutes(d.routes)
    expect(top.map((r) => r.route)).toEqual(['/query', UNMATCHED_ROUTE])
    expect(top[1].calls).toBe(15)
    expect(top[1].p95).toBe(9)
    expect(top[1].err5xx).toBe(0.2)
  })

  test('without <unmatched> traffic there is no such row', () => {
    expect(topRoutes(normalizePerfHttp({ routes: routes.slice(0, 3) }).routes).some((r) => r.unmatched)).toBe(false)
  })

  test('5xx as a ratio, a percentage or counts', () => {
    const d = normalizePerfHttp({ routes: [{ route: '/a', calls: 200, err5xx: 2.5 }, { route: '/b', calls: 200, errors_5xx: 50 }, { route: '/c', err5xx_rate: 0.1 }] })
    expect(d.routes.map((r) => r.err5xx)).toEqual([0.025, 0.25, 0.1])
  })

  test('series: per-minute rates and 5xx share per bucket', () => {
    const d = normalizePerfHttp({ bucket_s: 60, series: [{ t: min(0), calls: 120, errors_5xx: 6, p95_ms: 40 }, { t: min(1), rps: 3 }, { t: min(2), calls: 0 }] })
    expect(perMinRows(d.series, d.bucket_s)).toEqual([
      { t: T0, rate: 120 },
      { t: T0 + 60_000, rate: 180 },
      { t: T0 + 120_000, rate: 0 }
    ])
    expect(errorShareRows(d.series).map((r) => r.share)).toEqual([0.05, null, 0])
  })

  test('a route opens Logs at ERROR, filtered to its fixed path', () => {
    expect(routeLogText('/documents/{doc_id}/status')).toBe('/documents/')
    expect(routeLogText('/query')).toBe('/query')
    expect(routeLogText('/{path:path}')).toBeNull()
    expect(routeLogText(UNMATCHED_ROUTE)).toBeNull()
    const view = routeLogsView('/documents/{doc_id}', '1h')
    expect(view.tab).toBe('logs')
    expect(view.drawer).toBeNull()
    const filter = decodeLogFilter(view.logQuery)
    expect(filter.level).toBe('ERROR')
    expect(filter.q).toBe('/documents/')
    expect(filter.window).toBe('1h')
    expect(decodeLogFilter(routeLogsView(UNMATCHED_ROUTE, '15m').logQuery).q).toBe('')
  })
})

describe('concurrency (FR-44)', () => {
  test('roles, top ≤ 10 by queued, and a ratio only with a budget', () => {
    const top = Array.from({ length: 14 }, (_, i) => ({ workspace: `ws${i}`, running: 1, queued: i }))
    const d = normalizePerfConcurrency({ roles: { llm: { running: 14, in_flight: 54, queued: 40 }, embedding: { running: 1, queued: 0 } }, top_instances: top })
    expect(d.roles.llm).toEqual({ running: 14, in_flight: 54, queued: 40 })
    expect(d.roles.embed?.running).toBe(1)
    expect(d.roles.rerank).toBeNull()
    expect(d.top.length).toBe(10)
    expect(d.top[0].workspace).toBe('ws13')
    expect(d.budget).toBeNull()
    const b = normalizePerfConcurrency({ roles: { llm: { running: 10 } }, budget: 40 })
    expect(b.budget).toEqual({ limit: 40, used: 10, ratio: 0.25 })
    expect(normalizePerfConcurrency({ budget: { limit: 0 } }).budget).toBeNull()
  })
})

describe('throughput and top workspaces (FR-43, FR-45)', () => {
  test('docs, calls per minute and batches; sorted by time', () => {
    const d = normalizePerfThroughput({ docs: [{ t: min(60), processed: 3, failed: 1 }, { t: min(0), processed: 5 }], llm_calls_per_min: [{ t: min(0), value: 12 }], email_batches: [{ t: min(0), batches: 2 }] }, 1)
    expect(d.docs.map((p) => [p.t, p.value, p.failed])).toEqual([
      [T0, 5, null],
      [T0 + 3600_000, 3, 1]
    ])
    expect(d.llm_calls_per_min[0].value).toBe(12)
    expect(d.email_batches[0].value).toBe(2)
    expect(d.days).toBe(1)
  })

  test('top 20, largest first, blank names dropped', () => {
    const items = Array.from({ length: 25 }, (_, i) => ({ workspace: `ws${i}`, value: i }))
    const d = normalizeTopWorkspaces({ items: [...items, { workspace: '', value: 99 }] }, 'failures')
    expect(d.items.length).toBe(20)
    expect(d.items[0]).toEqual({ workspace: 'ws24', value: 24 })
    expect(d.metric).toBe('failures')
  })
})

describe('sign-in health (FR-35)', () => {
  test('counts and series; aliases accepted', () => {
    const s = normalizeSignin({ sso_callback_errors: 2, token_401: 3, login_lockouts: 0, series: [{ t: min(0), sso_errors: 1, unauthorized: 2 }] }, '15m')
    expect([s.sso_errors, s.token_401, s.lockouts]).toEqual([2, 3, 0])
    expect(s.series).toEqual([{ t: T0, sso_errors: 1, token_401: 2, lockouts: 0 }])
    expect(s.window).toBe('15m')
  })

  test('the default rule signin_failures_15m: 401s alone never alarm', () => {
    expect(signinState({ sso_errors: 0, lockouts: 0 })).toBe('ok')
    expect(signinState({ sso_errors: 5, lockouts: 3 })).toBe('ok')
    expect(signinState({ sso_errors: 6, lockouts: 0 })).toBe('warning')
    expect(signinState({ sso_errors: 0, lockouts: 4 })).toBe('warning')
    expect(signinState({ sso_errors: 21, lockouts: 0 })).toBe('serious')
    expect(signinState({ sso_errors: null, lockouts: null })).toBe('unknown')
    expect(signinState(null)).toBe('unknown')
  })
})

describe('topology rates (§2.12 edge flow)', () => {
  const NOW = T0 + 10 * 60_000 + 20_000

  test('the latest complete minute, per minute; an idle source reads 0', () => {
    const pts = [
      { t: T0 + 8 * 60_000, calls: 30 },
      { t: T0 + 9 * 60_000, calls: 60 },
      // The current minute is still filling: ignored.
      { t: T0 + 10 * 60_000, calls: 5 }
    ]
    expect(latestPerMin(pts, 60, NOW)).toBe(60)
    expect(latestPerMin([{ t: T0, calls: 30 }], 300, NOW)).toBe(6)
    // Nothing for minutes: empty buckets are left out by the server.
    expect(latestPerMin([{ t: T0, calls: 30 }], 60, NOW)).toBe(0)
    expect(latestPerMin([], 60, NOW)).toBe(0)
  })

  test('p50/p90 from the buckets that carried traffic', () => {
    const pts = [{ calls: 0 }, { calls: 0 }, { calls: 10 }, { calls: 20 }, { calls: 30 }, { calls: null }]
    const p = ratePercentiles(pts, 60)
    expect(p.p50).toBe(20)
    expect(p.p90).toBe(28)
    expect(ratePercentiles([{ calls: 0 }], 60)).toEqual({ p50: null, p90: null })
  })

  test('sources: current from 15 m, thresholds from 24 h; switched-off sources are left out', () => {
    const http = normalizePerfHttp({ bucket_s: 60, series: [{ t: T0 + 9 * 60_000, calls: 75 }] }, '15m')
    const httpBase = normalizePerfHttp({ bucket_s: 300, series: [{ t: T0, calls: 100 }, { t: T0 - 300_000, calls: 500 }] }, '24h')
    const llm = normalizePerfLlm({ bucket_s: 60, series: [{ t: T0 + 9 * 60_000, key: 'extract', calls: 300 }, { t: T0 + 9 * 60_000, key: 'query', calls: 20 }] }, '15m')
    const deps = normalizePerfDeps({ enabled: false, series: { embed: [{ t: T0 + 9 * 60_000, calls: 3 }] } }, '15m')
    const rates = flowRatesFrom({ http, llm, deps }, { http: httpBase, llm: null, deps: null }, NOW)
    expect(rates.http).toEqual({ perMin: 75, p50: 60, p90: 92 })
    expect(rates.llm).toEqual({ perMin: 320, p50: null, p90: null })
    expect(rates.embed).toBeUndefined()
    expect(rates.rerank).toBeUndefined()
    expect(flowRatesFrom({ http: { ...http, available: false } }, {}, NOW).http).toBeUndefined()
  })

  test('no usage recorder (insights: false): the LLM edge is unknown and static, never "0/min"', () => {
    const off = normalizePerfLlm({ window: '15m', insights: false, series: [], table: [], queue: { running: 0, in_flight: 0, queued: 0 } }, '15m')
    const base = normalizePerfLlm({ window: '24h', insights: false, series: [], table: [] }, '24h')
    expect(flowRatesFrom({ llm: off }, { llm: base }, NOW).llm).toBeUndefined()
    expect('llm' in flowRatesFrom({ llm: off }, {}, NOW)).toBe(false)
    // A working recorder with no calls yet is a real zero.
    const quiet = normalizePerfLlm({ window: '15m', insights: true, series: [], table: [] }, '15m')
    expect(flowRatesFrom({ llm: quiet }, {}, NOW).llm).toEqual({ perMin: 0, p50: null, p90: null })
  })
})

/**
 * The payloads below follow lightrag/api/ops/perf.py field for field (epoch
 * ms times, `count` rather than `calls`, route keys as "METHOD /template",
 * `kv_cache_pct` as 0..100, `last_15m` inside the sign-in answer), so a
 * rename on either side shows up here.
 */
describe('server shapes (perf.py)', () => {
  const ms = (i: number) => T0 + i * 60_000

  test('/performance/http: the route key splits into method and template; <unmatched> has neither', () => {
    const d = normalizePerfHttp(
      {
        window: '1h',
        bucket_s: 300,
        enabled: true,
        routes: [
          { route: 'GET /documents/{doc_id}', method: 'GET', path: '/documents/{doc_id}', rps: 0.5, err5xx: 0.02, count: 1800, errors: 36, error_rate: 0.02, p50: 12, p95: 80, p99: 300 },
          { route: 'POST /documents/{doc_id}', method: 'POST', path: '/documents/{doc_id}', rps: 0.1, err5xx: null, count: 360, errors: 0, error_rate: 0, p50: 30, p95: 90, p99: 95 },
          { route: UNMATCHED_ROUTE, method: null, path: UNMATCHED_ROUTE, rps: 0.01, err5xx: 0, count: 36, errors: 0, p95: 2 }
        ],
        top_p95: ['POST /documents/{doc_id}', 'GET /documents/{doc_id}'],
        series: [{ t: ms(5), rps: 2, err5xx: 0.01, count: 600, errors: 6, p50: 10, p95: 70, p99: 200 }]
      },
      '1h'
    )
    expect(d.routes.map((r) => [r.key, r.method, r.route])).toEqual([
      ['GET /documents/{doc_id}', 'GET', '/documents/{doc_id}'],
      ['POST /documents/{doc_id}', 'POST', '/documents/{doc_id}'],
      [UNMATCHED_ROUTE, null, UNMATCHED_ROUTE]
    ])
    expect(d.routes[0].calls).toBe(1800)
    expect(d.routes[0].err5xx).toBe(0.02)
    expect(d.routes[2].unmatched).toBe(true)
    // Same template under two methods: two rows with distinct keys, and Logs filter on the template.
    expect(new Set(topRoutes(d.routes).map((r) => r.key)).size).toBe(3)
    expect(decodeLogFilter(routeLogsView(d.routes[0].route, '1h').logQuery).q).toBe('/documents/')
    expect(d.series[0]).toEqual({ t: ms(5), calls: 600, errors: 6, p50: 10, p95: 70, p99: 200 })
    expect(perMinRows(d.series, d.bucket_s)[0].rate).toBe(120)
    expect(d.enabled).toBe(true)
  })

  test('/performance/deps: count, totals as summary, and the probe KV cache as a percentage', () => {
    const d = normalizePerfDeps(
      {
        window: '15m',
        bucket_s: 60,
        series: {
          embed: [{ t: ms(0), per_min: 30, count: 30, errors: 3, error_rate: 0.1, avg_ms: 8, p50: 6, p95: 15, p99: 40, max_ms: 41 }],
          rerank: []
        },
        models: { embed: [{ model: 'bge-m3', count: 30 }], rerank: [] },
        totals: { embed: { count: 30, errors: 3, error_rate: 0.1, p50: 6, p95: 15, p99: 40, per_min: 2 }, rerank: { count: 0, errors: 0, error_rate: null, p50: null, p95: null, p99: null } },
        summary: { embed: { count: 30, errors: 3, error_rate: 0.1, p50: 6, p95: 15, p99: 40 }, rerank: { count: 0, errors: 0, error_rate: null } },
        vllm: { embed: { status: 'ok', running: 3, waiting: 0, kv_cache_pct: 0.5 }, rerank: { status: 'ok', running: 8, waiting: 34, kv_cache_pct: 71 } },
        enabled: true
      },
      '15m'
    )
    expect(d.series.embed[0]).toMatchObject({ t: ms(0), calls: 30, errors: 3, p95: 15, avg: 8, max: 41 })
    expect(d.summary.embed).toEqual({ calls: 30, error_rate: 0.1, p50: 6, p95: 15, p99: 40 })
    expect(d.summary.rerank.calls).toBe(0)
    // 0.5 % stays half a percent; it is not read as a 50 % fraction.
    expect(d.vllm.embed?.kv_cache).toBeCloseTo(0.005)
    expect(d.vllm.rerank).toEqual({ running: 8, waiting: 34, kv_cache: 0.71 })
  })

  test('/performance/deps: vLLM running/waiting/KV cache history rides on the series points', () => {
    const d = normalizePerfDeps(
      {
        window: '15m',
        bucket_s: 60,
        series: {
          embed: [
            { t: ms(0), per_min: 4, count: 4, errors: 0, p50: 6, p95: 9, p99: 9, running: 4, waiting: 30, kv_cache: 0.555 },
            // A bucket with probe rows but no calls: a point all the same (0 calls, no latency).
            { t: ms(1), per_min: 0, count: 0, errors: 0, p50: null, p95: null, p99: null, running: 3, waiting: null, kv_cache: null }
          ],
          rerank: [{ t: ms(0), per_min: 1, count: 1, errors: 0, p50: 20, p95: 20, p99: 20 }]
        },
        totals: {},
        vllm: { embed: null, rerank: null }
      },
      '15m'
    )
    expect(d.series.embed[0]).toMatchObject({ running: 4, waiting: 30, kv_cache: 0.555, calls: 4 })
    expect(d.series.embed[1]).toMatchObject({ running: 3, waiting: null, kv_cache: null, calls: 0, p95: null })
    expect(d.series.rerank[0]).toMatchObject({ running: null, waiting: null, kv_cache: null })
  })

  test('/performance/llm: epoch-ms series, rates table, queue now; no recorder says so', () => {
    const d = normalizePerfLlm(
      {
        window: '1h',
        group: 'purpose',
        approx_7d: false,
        insights: true,
        bucket_s: 300,
        queue: { running: 4, in_flight: 6, queued: 12 },
        series: [{ t: ms(0), key: 'extract', calls: 40, errors: 1, p50: 900, p95: 4000, p99: 9000 }],
        table: [{ key: 'extract', calls: 40, calls_per_min: 0.67, errors: 1, error_rate: 0.025, truncated_rate: 0, abandoned_rate: 0, p50: 900, p95: 4000, p99: 9000, queued: 12, running: 4 }]
      },
      '1h',
      'purpose'
    )
    expect(d.insights).toBe(true)
    expect(d.series[0]).toMatchObject({ t: ms(0), key: 'extract', calls: 40, p95: 4000 })
    expect(d.table[0]).toMatchObject({ key: 'extract', calls: 40, error_rate: 0.025, truncated_rate: 0, running: 4, queued: 12 })
    expect(d.queue).toEqual({ running: 4, queued: 12, in_flight: 6 })
    const off = normalizePerfLlm({ window: '1h', group: 'purpose', approx_7d: false, queue: { running: 0, in_flight: 0, queued: 0 }, insights: false, series: [], table: [] }, '1h')
    expect(off.insights).toBe(false)
    expect(off.queue?.running).toBe(0)
  })

  test('/performance/llm: the summed running/queued history for the second axis', () => {
    const d = normalizePerfLlm(
      {
        window: '1h',
        insights: false,
        series: [],
        table: [],
        queue_series: [
          { t: ms(1), running: 10, in_flight: 12, queued: 20 },
          { t: ms(0), running: 4, in_flight: 6, queued: null }
        ]
      },
      '1h'
    )
    expect(d.queue_series).toEqual([
      { t: ms(0), running: 4, queued: null },
      { t: ms(1), running: 10, queued: 20 }
    ])
  })

  test('/performance/llm 7 d: hourly average and max only', () => {
    const d = normalizePerfLlm({ window: '7d', approx_7d: true, insights: true, bucket_s: 10800, series: [{ t: ms(0), key: 'query', calls: 9, errors: 0, avg: 1200, max: 5000 }], table: [{ key: 'query', calls: 9, avg: 1200, max: 5000 }] }, '7d')
    expect(d.approx_7d).toBe(true)
    expect(latencyRows(seriesFor(d.series, 'query'), true)).toEqual([{ t: ms(0), avg: 1200, max: 5000 }])
    expect(d.table[0]).toMatchObject({ avg: 1200, max: 5000, p95: null })
  })

  test('/performance/concurrency: the budget ratio is against calls in flight; no heartbeats is unknown, not zero', () => {
    const d = normalizePerfConcurrency({
      roles: { llm: { running: 10, in_flight: 30, queued: 5 }, embed: null, rerank: { running: 0, in_flight: 0, queued: 0 } },
      llm_roles: { extract: { running: 10, in_flight: 30, queued: 5 } },
      top: [{ workspace: 'fsktm', running: 6, queued: 5, pid: 812, worker_id: 'w-812' }],
      instances: 1204,
      workers: 1,
      heartbeats: true,
      budget: { limit: 60, in_flight: 30, ratio: 0.5 },
      generated_at: '2026-11-03T04:00:00+00:00'
    })
    expect(d.budget).toEqual({ limit: 60, used: 30, ratio: 0.5 })
    expect(d.top[0]).toMatchObject({ workspace: 'fsktm', pid: 812, worker_id: 'w-812' })
    expect(d.roles.embed).toBeNull()
    expect(d.instances).toBe(1204)
    expect(d.heartbeats).toBe(true)
    expect(d.at).toBe('2026-11-03T04:00:00+00:00')
    expect(normalizePerfConcurrency({ roles: { llm: null, embed: null, rerank: null }, top: [], heartbeats: false }).heartbeats).toBe(false)
  })

  test('/performance/throughput and /performance/top-workspaces', () => {
    const d = normalizePerfThroughput(
      {
        days: 7,
        bucket_s: 3600,
        docs: [{ t: ms(0), processed: 12, failed: 2 }],
        email_batches: [{ t: ms(0), batches: 3, failed: 1 }],
        llm_calls_per_min: [{ t: ms(0), value: 4.2, calls_per_min: 4.2, calls: 21, errors: 0 }],
        llm_calls_bucket_s: 300,
        insights: true
      },
      7
    )
    expect(d.docs[0]).toEqual({ t: ms(0), value: 12, failed: 2 })
    expect(d.email_batches[0]).toEqual({ t: ms(0), value: 3, failed: 1 })
    expect(d.llm_calls_per_min[0]).toEqual({ t: ms(0), value: 4.2, failed: 0 })
    const top = normalizeTopWorkspaces({ metric: 'processing_minutes', window: '24h', items: [{ workspace: 'a', value: 12.5 }, { workspace: 'b', value: 40 }] }, 'llm_calls')
    expect(top.metric).toBe('processing_minutes')
    expect(top.items.map((r) => r.workspace)).toEqual(['b', 'a'])
  })

  test('/signin: one 24 h answer carries the last 15 minutes for the state', () => {
    const s = normalizeSignin(
      {
        window: '24h',
        bucket_s: 3600,
        sso_errors: 9,
        sso_ok: 120,
        token_401: 40,
        lockouts: 1,
        login_failed: 3,
        last_15m: { sso_errors: 6, sso_ok: 2, token_401: 3, lockouts: 0 },
        series: [{ t: ms(0), sso_errors: 9, token_401: 40, lockouts: 1 }],
        enabled: true
      },
      '24h'
    )
    expect([s.sso_errors, s.token_401, s.lockouts]).toEqual([9, 40, 1])
    expect(s.last_15m).toEqual({ sso_errors: 6, token_401: 3, lockouts: 0 })
    expect(signinState(s.last_15m)).toBe('warning')
    // A 15-minute answer is its own last 15 minutes; an older server without the field leaves it unknown.
    expect(normalizeSignin({ window: '15m', sso_errors: 1, token_401: 0, lockouts: 0 }).last_15m).toEqual({ sso_errors: 1, token_401: 0, lockouts: 0 })
    expect(signinState(normalizeSignin({ window: '24h', sso_errors: 1 }).last_15m)).toBe('unknown')
    // The server's verdict (with KV threshold overrides) is kept; anything else is ignored.
    expect(s.state).toBeNull()
    expect(normalizeSignin({ window: '24h', state: 'warning' }).state).toBe('warning')
    expect(normalizeSignin({ window: '24h', state: 'bogus' }).state).toBeNull()
    // No usernames anywhere in the read model.
    expect(JSON.stringify(s)).not.toMatch(/user|email/i)
  })

  test('the routes answer without the ops store (available: false) still normalises', () => {
    const d = normalizePerfHttp({ available: false, window: '1h', routes: [], top_p95: [], series: [], totals: {} }, '1h')
    expect(d.available).toBe(false)
    expect(d.routes).toEqual([])
  })
})
