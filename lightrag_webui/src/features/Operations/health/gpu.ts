import type { DepHealth } from '../api'
import { fmtBytes } from '../format'

/**
 * GPU numbers (D5, Phase 2): the API reads the nvidia_gpu_exporter sidecar's
 * /metrics in its probe cycle (OPS_GPU_EXPORTER_URL; off when unset) and
 * reports one `gpu` dependency whose detail lists each GPU's utilisation,
 * memory, temperature and power. History comes from the stored probe rows.
 * vLLM's own /metrics (running, waiting, KV cache) stays on the embed and
 * rerank cards. Pure, so the card text and the charts' rows are tested
 * without a DOM; every reader accepts the spellings the server may use.
 */

export const GPU_DEP = 'gpu'

export type GpuReading = {
  /** "0", "1", … (the exporter's index), or the position in the list. */
  index: string
  name: string | null
  /** Utilisation, 0–100. */
  util_pct: number | null
  mem_used_bytes: number | null
  mem_total_bytes: number | null
  /** Memory in use, 0–100 (from used/total when the server does not send it). */
  mem_pct: number | null
  temp_c: number | null
  power_w: number | null
  power_limit_w: number | null
}

type Loose = Record<string, any>

const obj = (v: unknown): Loose => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Loose) : {})
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const str = (v: unknown): string | null => (typeof v === 'string' && v ? v : null)
const first = (...values: (number | null)[]): number | null => {
  for (const v of values) if (v !== null) return v
  return null
}
const MIB = 1024 * 1024
const mib = (v: unknown) => {
  const n = num(v)
  return n === null ? null : n * MIB
}
/** A percentage from a 0–100 value, or from a 0–1 ratio under a key that says "ratio". */
const pct = (percent: unknown, ratio?: unknown): number | null => {
  const p = num(percent)
  if (p !== null) return Math.max(0, Math.min(100, p))
  const r = num(ratio)
  return r === null ? null : Math.max(0, Math.min(100, r * 100))
}

export function normalizeGpu(raw: unknown, position: number): GpuReading {
  const g = obj(raw)
  const used = first(num(g.mem_used_bytes), num(g.memory_used_bytes), mib(g.mem_used_mib), mib(g.memory_used_mib))
  const total = first(num(g.mem_total_bytes), num(g.memory_total_bytes), mib(g.mem_total_mib), mib(g.memory_total_mib))
  const memPct = first(pct(g.mem_pct ?? g.memory_pct ?? g.memory_used_pct, g.mem_ratio ?? g.memory_ratio), used !== null && total ? Math.min(100, (used / total) * 100) : null)
  const index = g.index ?? g.gpu ?? g.id
  return {
    index: typeof index === 'number' && Number.isFinite(index) ? String(index) : (str(index) ?? String(position)),
    name: str(g.name) ?? str(g.model) ?? str(g.product),
    util_pct: pct(g.util_pct ?? g.utilization_pct ?? g.gpu_util_pct, g.util_ratio ?? g.utilization_ratio),
    mem_used_bytes: used,
    mem_total_bytes: total,
    mem_pct: memPct,
    temp_c: first(num(g.temp_c), num(g.temperature_c), num(g.temperature)),
    power_w: first(num(g.power_w), num(g.power_draw_w)),
    power_limit_w: first(num(g.power_limit_w), num(g.enforced_power_limit_w))
  }
}

/** The GPUs in a probe's detail (`gpus`), in the exporter's order; empty when there are none. */
export function normalizeGpus(detail: unknown): GpuReading[] {
  const d = obj(detail)
  const list = Array.isArray(d.gpus) ? d.gpus : Array.isArray(d.devices) ? d.devices : []
  return list.map((g, i) => normalizeGpu(g, i))
}

/** The `gpu` card among the dependencies, if the server reports one. */
export const gpuDep = (deps: DepHealth[] | null | undefined): DepHealth | null => deps?.find((d) => d.id === GPU_DEP) ?? null

const round = (n: number | null) => (n === null ? '–' : String(Math.round(n)))

/** "37% · 12.1 GB / 24 GB · 61 °C · 120 W" for one GPU (card lines and the Host view). */
export function gpuLine(g: GpuReading): { util: string; mem: string; temp: string; power: string } {
  return {
    util: g.util_pct === null ? '–' : `${Math.round(g.util_pct)}%`,
    mem:
      g.mem_used_bytes !== null && g.mem_total_bytes !== null
        ? `${fmtBytes(g.mem_used_bytes)} / ${fmtBytes(g.mem_total_bytes)}`
        : g.mem_pct !== null
          ? `${Math.round(g.mem_pct)}%`
          : '–',
    temp: g.temp_c === null ? '–' : `${round(g.temp_c)} °C`,
    power: g.power_w === null ? '–' : g.power_limit_w !== null ? `${round(g.power_w)} / ${round(g.power_limit_w)} W` : `${round(g.power_w)} W`
  }
}

/** A GPU's state for its tile: hot or nearly out of memory is a warning; the probe's own status decides the card. */
export function gpuWarn(g: GpuReading): boolean {
  return (g.temp_c !== null && g.temp_c >= 85) || (g.mem_pct !== null && g.mem_pct >= 95)
}

export type GpuMetric = 'util' | 'mem' | 'temp' | 'power'

const METRIC_OF: Record<GpuMetric, (g: GpuReading) => number | null> = {
  util: (g) => g.util_pct,
  mem: (g) => g.mem_pct,
  temp: (g) => g.temp_c,
  power: (g) => g.power_w
}

const tOf = (raw: unknown): number | null => {
  if (typeof raw === 'number' && Number.isFinite(raw)) return raw < 1e11 ? raw * 1000 : raw
  const s = str(raw)
  if (!s) return null
  const t = Date.parse(s)
  return Number.isFinite(t) ? t : null
}

export type GpuHistory = {
  /** GPU indexes, in first-seen order: each is a series. */
  gpus: string[]
  /** One row per probe time, oldest first: `{t, g0, g1, …}` per metric. */
  rows: Record<GpuMetric, Record<string, number | null>[]>
}

/** Series key of a GPU in the history rows. */
export const gpuKey = (index: string) => `g${index}`

/**
 * Per-GPU history, oldest first, one row per time with one series per GPU
 * (`g0`, `g1`, …). Accepts the server's flat rows (`[{t, gpu, util_pct,
 * mem_pct, temp_c, power_w}]`, one per GPU and bucket, from GET
 * /performance/gpu) and nested samples (`[{t, gpus:[…]}]`, or probe results
 * with `{ts, detail:{gpus}}`). A GPU missing from a bucket is a gap, not zero.
 */
export function gpuHistory(points: unknown): GpuHistory {
  const byT = new Map<number, GpuReading[]>()
  for (const p of Array.isArray(points) ? points : []) {
    const x = obj(p)
    const t = tOf(x.t ?? x.ts ?? x.at)
    if (t === null) continue
    const list = byT.get(t) ?? []
    if (Array.isArray(x.gpus) || x.detail) list.push(...normalizeGpus(x.detail ?? x))
    else list.push(normalizeGpu({ ...x, index: x.gpu ?? x.index }, list.length))
    byT.set(t, list)
  }
  const times = [...byT.keys()].sort((a, b) => a - b)
  const gpus: string[] = []
  for (const t of times) for (const g of byT.get(t) ?? []) if (!gpus.includes(g.index)) gpus.push(g.index)
  gpus.sort((a, b) => Number(a) - Number(b) || a.localeCompare(b))
  const rows = { util: [], mem: [], temp: [], power: [] } as GpuHistory['rows']
  for (const t of times) {
    const sample = byT.get(t) ?? []
    for (const metric of Object.keys(rows) as GpuMetric[]) {
      const row: Record<string, number | null> = { t }
      for (const id of gpus) {
        const g = sample.find((x) => x.index === id)
        row[gpuKey(id)] = g ? METRIC_OF[metric](g) : null
      }
      rows[metric].push(row)
    }
  }
  return { gpus, rows }
}

export type GpuLatest = {
  status: string | null
  checked_at: string | null
  error_class: string | null
  /** Why the probe warned: hot | memory_full | no_gpus. */
  reason: string | null
  gpus: GpuReading[]
}

/** GET /performance/gpu (D5): the latest probe's GPUs and the history by bucket. */
export type PerfGpu = {
  generated_at?: string
  available?: boolean
  /** The exporter is configured and probed here; false: GPU metrics are off. */
  enabled: boolean | null
  bucket_s: number | null
  latest: GpuLatest | null
  history: GpuHistory
}

export function normalizePerfGpu(raw: unknown): PerfGpu {
  const o = obj(raw)
  const l = o.latest && typeof o.latest === 'object' ? obj(o.latest) : null
  return {
    generated_at: str(o.generated_at) ?? undefined,
    available: o.available === false ? false : undefined,
    enabled: typeof o.enabled === 'boolean' ? o.enabled : null,
    bucket_s: num(o.bucket_s),
    latest: l
      ? { status: str(l.status), checked_at: str(l.checked_at), error_class: str(l.error_class), reason: str(l.reason), gpus: normalizeGpus(l) }
      : null,
    history: gpuHistory(o.series)
  }
}

/** Series colours for up to three GPUs, then repeating (the A30 host has one). */
export const GPU_COLORS = ['var(--ops-s1)', 'var(--ops-s2)', 'var(--ops-s3)']
export const gpuColor = (i: number) => GPU_COLORS[i % GPU_COLORS.length]
