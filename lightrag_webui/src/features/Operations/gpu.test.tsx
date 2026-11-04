import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'
import type { DepHealth, OpsStatus } from './api'

// The stores behind these components read localStorage as they load.
const memory = new Map<string, string>()
Object.defineProperty(globalThis, 'localStorage', {
  value: {
    getItem: (k: string) => memory.get(k) ?? null,
    setItem: (k: string, v: string) => void memory.set(k, v),
    removeItem: (k: string) => void memory.delete(k)
  },
  configurable: true
})

const { gpuDep, gpuHistory, gpuKey, gpuLine, gpuWarn, normalizeGpus, normalizePerfGpu } = await import('./health/gpu')
const { detailParts, normalizeStrip } = await import('./health/model')
const { GpuCards, GpuCharts, default: GpuView } = await import('./performance/GpuView')
const { default: DepsView } = await import('./health/DepsView')
const { HostView } = await import('./health/WorkersView')
const { OpsPrefsContext } = await import('./prefs')
const { OpsShellContext } = await import('./context')
const { ServerCard } = await import('./OverviewTab')

/** The `gpu` probe's detail as ops/gpu.py writes it (parse_gpu_metrics + gpu_summary). */
const A30 = {
  target: 'nvidia-gpu-exporter:9835',
  gpu_count: 1,
  util_max_pct: 37,
  mem_max_pct: 50.4,
  temp_max_c: 61,
  power_total_w: 120.5,
  gpus: [{ index: 0, name: 'NVIDIA A30', util_pct: 37, mem_used_bytes: 12_400_000_000, mem_total_bytes: 24_600_000_000, mem_pct: 50.4, temp_c: 61, power_w: 120.5, power_limit_w: 165 }]
}

/** GET /performance/gpu (ops/gpu.py gpu_view). */
const PERF = {
  generated_at: '2026-11-04T10:00:00Z',
  window: '1h',
  enabled: true,
  configured: true,
  bucket_s: 60,
  latest: { status: 'ok', checked_at: '2026-11-04T09:59:40Z', error_class: null, reason: null, gpus: A30.gpus, summary: { gpu_count: 1 } },
  series: [
    { t: 1793786340000, gpu: 1, util_pct: 5, mem_pct: 10, temp_c: 40, power_w: 30 },
    { t: 1793786340000, gpu: 0, util_pct: 30, mem_pct: 50, temp_c: 60, power_w: 110 },
    { t: 1793786280000, gpu: 0, util_pct: 20, mem_pct: 49, temp_c: 59, power_w: 100 }
  ],
  durable: true
}

describe('GPU readings (D5)', () => {
  test('the exporter detail: one reading per GPU, in order', () => {
    const [g] = normalizeGpus(A30)
    expect(g).toMatchObject({ index: '0', name: 'NVIDIA A30', util_pct: 37, mem_pct: 50.4, temp_c: 61, power_w: 120.5, power_limit_w: 165 })
    expect(normalizeGpus({})).toEqual([])
    expect(normalizeGpus(null)).toEqual([])
  })

  test('memory share from used/total when not sent; MiB spellings; ratios as percentages', () => {
    const [g] = normalizeGpus({ gpus: [{ index: 1, memory_used_mib: 1024, memory_total_mib: 4096, util_ratio: 0.25 }] })
    expect(g.mem_used_bytes).toBe(1024 * 1024 * 1024)
    expect(g.mem_pct).toBe(25)
    expect(g.util_pct).toBe(25)
  })

  test('card text and warnings', () => {
    const [g] = normalizeGpus(A30)
    const l = gpuLine(g)
    expect(l.util).toBe('37%')
    expect(l.mem).toContain('/')
    expect(l.temp).toBe('61 °C')
    expect(l.power).toBe('121 / 165 W')
    expect(gpuWarn(g)).toBe(false)
    expect(gpuWarn({ ...g, temp_c: 88 })).toBe(true)
    expect(gpuWarn({ ...g, mem_pct: 96 })).toBe(true)
    expect(gpuLine({ ...g, util_pct: null, temp_c: null, power_w: null, mem_used_bytes: null, mem_total_bytes: null, mem_pct: null })).toEqual({ util: '–', mem: '–', temp: '–', power: '–' })
  })

  test('history from the flat server rows: one row per time, one series per GPU, oldest first', () => {
    const h = gpuHistory(PERF.series)
    expect(h.gpus).toEqual(['0', '1'])
    expect(h.rows.util.map((r) => r.t)).toEqual([1793786280000, 1793786340000])
    // GPU 1 is missing from the first bucket: a gap, not zero.
    expect(h.rows.util[0]).toEqual({ t: 1793786280000, [gpuKey('0')]: 20, [gpuKey('1')]: null })
    expect(h.rows.power[1]).toEqual({ t: 1793786340000, g0: 110, g1: 30 })
  })

  test('history from nested samples and probe results reads the same', () => {
    const nested = gpuHistory([{ t: 1793786280000, gpus: [{ index: 0, util_pct: 20 }] }])
    const probes = gpuHistory([{ ts: '2026-11-04T09:58:00Z', detail: { gpus: [{ index: 0, util_pct: 20 }] } }])
    expect(nested.rows.util[0].g0).toBe(20)
    expect(probes.rows.util[0].g0).toBe(20)
    expect(gpuHistory(null)).toEqual({ gpus: [], rows: { util: [], mem: [], temp: [], power: [] } })
  })

  test('GET /performance/gpu normalises; off when the exporter is not configured', () => {
    const p = normalizePerfGpu(PERF)
    expect(p.enabled).toBe(true)
    expect(p.latest?.gpus[0].index).toBe('0')
    expect(p.history.gpus).toEqual(['0', '1'])
    expect(normalizePerfGpu({ enabled: false, configured: false, latest: null, series: [] })).toMatchObject({ enabled: false, latest: null })
  })
})

const AT = Date.now() - 5000
const dep = (id: string, extra: Partial<DepHealth> = {}): DepHealth => ({
  id,
  status: 'ok',
  detail: {},
  strip: normalizeStrip(Array(48).fill('ok'), AT),
  checked_at: new Date(AT).toISOString(),
  latency_ms: 4,
  ...extra
})
const prefs = { paused: false, setPaused: () => {}, reduceMotion: false, setReduceMotion: () => {} }
const status = (features: Record<string, boolean>): OpsStatus => ({
  available: true,
  features,
  process: { mode: 'uvicorn', expected_workers: 1, workers_live: 1 },
  operator: true,
  version: null,
  generated_at: new Date(AT).toISOString()
})
const wrap = (node: React.ReactNode, s: OpsStatus = status({ gpu: true })) =>
  renderToString(
    <OpsShellContext.Provider value={{ status: s, openInsights: () => {} }}>
      <OpsPrefsContext.Provider value={prefs}>{node}</OpsPrefsContext.Provider>
    </OpsShellContext.Provider>
  )

describe('GPU dependency card and section', () => {
  test('the card says per GPU what it reads; reasons in words', () => {
    const parts = detailParts(dep('gpu', { detail: A30 }))
    expect(parts.find(([k]) => k === 'operations.health.detail.gpu')?.[1]).toMatchObject({ index: '0', util: '37%', temp: '61 °C' })
    const hot = detailParts(dep('gpu', { status: 'warn', detail: { ...A30, reason: 'hot' } }))
    expect(hot.map(([k]) => k)).toContain('operations.health.detail.gpuReason.hot')
    const many = detailParts(dep('gpu', { detail: { gpus: [0, 1, 2, 3].map((index) => ({ index, util_pct: 1 })) } }))
    expect(many.filter(([k]) => k === 'operations.health.detail.gpu').length).toBe(2)
    expect(many.find(([k]) => k === 'operations.health.detail.gpuMore')?.[1]).toEqual({ count: 2 })
  })

  test('the Health grid renders the GPU card with its lines', () => {
    const html = wrap(
      <DepsView
        poll={{ data: { cycle: null, deps: [dep('postgres'), dep('gpu', { detail: A30 })] } as any, error: null, loading: false, slow: false, updatedAt: AT, previousAt: null, baseline: true, reload: () => {} }}
        processLabel={null}
        probesOff={false}
        flow={{ enabled: false, metrics: false }}
      />
    )
    expect(html).toContain('operations.health.detail.gpu')
    expect(html).toContain('odep=gpu')
  })

  test('GPU tiles: name, bars with numbers, temperature and power; a warning in words', () => {
    const [g] = normalizeGpus(A30)
    const html = renderToString(<GpuCards gpus={[g, { ...g, index: '1', temp_c: 90 }]} />)
    expect(html).toContain('NVIDIA A30')
    expect(html).toContain('data-gpu="0"')
    expect(html).toContain('37%')
    expect(html).toContain('operations.gpu.hot')
    expect(renderToString(<GpuCards gpus={[]} />)).toContain('operations.gpu.noGpus')
  })

  test('the charts never animate and each has a table view', () => {
    const html = renderToString(<GpuCharts data={normalizePerfGpu(PERF)} from={1793786200000} to={1793786400000} ctx={{ window: '1h', events: [] }} />)
    expect(html).toContain('operations.perf.viewTable')
    expect(html).toContain('operations.gpu.chart.util')
    expect(html).not.toMatch(/ops-fade-in|transform[:=]/i)
  })

  test('the Host view quotes the GPUs from the exporter card, else where GPU numbers come from', () => {
    const poll = { data: { items: [], host: { mem_used: 1, mem_limit: 2, cpu_pct: 1, load: [1, 1, 1], disks: [], sampled_at: null } } as any, error: null, loading: false, slow: false, updatedAt: AT, previousAt: null, baseline: true, reload: () => {} }
    expect(wrap(<HostView poll={poll} deps={[dep('gpu', { detail: A30 })]} />)).toContain('operations.host.gpuLine')
    expect(wrap(<HostView poll={poll} deps={[dep('postgres')]} />)).toContain('operations.host.gpuNote')
  })

  test('the section waits for its first answer; the server card lists the feature', () => {
    expect(wrap(<GpuView ctx={{ window: '1h', events: [] }} />)).toContain('operations.gpu.now')
    expect(wrap(<ServerCard />, status({ gpu: true }))).toContain('operations.feature.gpu')
    expect(wrap(<ServerCard />, status({ gpu: false }))).toContain('operations.feature.off')
  })

  test('the dependency card is found by id', () => {
    expect(gpuDep([dep('postgres'), dep('gpu')])?.id).toBe('gpu')
    expect(gpuDep(null)).toBeNull()
  })
})
