/**
 * Formatters for the Operations console. Every one returns '–' for a missing
 * value, so a panel never prints "NaN" or "undefined" during an incident.
 */

export const DASH = '–'

const isNum = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n)

/** 1234567 → "1,234,567". */
export const fmtCount = (n: number | null | undefined) => (isNum(n) ? Math.round(n).toLocaleString('en-US') : DASH)

/**
 * An age or elapsed time in seconds, two units at most: "now", "12s", "4m 05s",
 * "2h 10m", "3d 4h". Negative ages (clock skew) read as "now".
 */
export const fmtAge = (seconds: number | null | undefined) => {
  if (!isNum(seconds)) return DASH
  const s = Math.floor(Math.max(0, seconds))
  if (s < 1) return 'now'
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) {
    const rest = s % 60
    return rest ? `${m}m ${String(rest).padStart(2, '0')}s` : `${m}m`
  }
  const h = Math.floor(m / 60)
  if (h < 24) {
    const rest = m % 60
    return rest ? `${h}h ${rest}m` : `${h}h`
  }
  const d = Math.floor(h / 24)
  const rest = h % 24
  return rest ? `${d}d ${rest}h` : `${d}d`
}

/** Seconds since an ISO timestamp (or epoch ms), measured against `now` (epoch ms). */
export const ageSeconds = (at: string | number | null | undefined, now: number = Date.now()) => {
  if (at === null || at === undefined || at === '') return null
  const t = typeof at === 'number' ? at : Date.parse(at)
  return Number.isFinite(t) ? (now - t) / 1000 : null
}

/** A duration in milliseconds: "850 ms", "1.4 s", "42 s", "2m 10s". */
export const fmtDuration = (ms: number | null | undefined) => {
  if (!isNum(ms)) return DASH
  const v = Math.max(0, ms)
  if (v < 1000) return `${Math.round(v)} ms`
  if (v < 10_000) return `${(Math.floor(v / 100) / 10).toFixed(1)} s`
  if (v < 60_000) return `${Math.floor(v / 1000)} s`
  return fmtAge(v / 1000)
}

/** A latency in milliseconds, with sub-millisecond values shown honestly: "<1 ms", "41 ms", "1.2 s". */
export const fmtLatency = (ms: number | null | undefined) => {
  if (!isNum(ms)) return DASH
  if (ms < 1) return '<1 ms'
  if (ms < 1000) return `${Math.round(ms)} ms`
  if (ms < 10_000) return `${(Math.round(ms / 100) / 10).toFixed(1)} s`
  return `${Math.round(ms / 1000)} s`
}

const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB'] as const

/** Binary byte sizes: "512 B", "1.5 KB", "2.1 GB". One decimal below 10, none above. */
export const fmtBytes = (n: number | null | undefined) => {
  if (!isNum(n)) return DASH
  let v = Math.max(0, n)
  let unit = 0
  while (v >= 1024 && unit < BYTE_UNITS.length - 1) {
    v /= 1024
    unit += 1
  }
  if (unit === 0) return `${Math.round(v)} B`
  const text = v < 10 ? (Math.round(v * 10) / 10).toFixed(1) : String(Math.round(v))
  return `${text} ${BYTE_UNITS[unit]}`
}

/** The unit fmtBytes would pick, so a tween can jump instead of sliding from MB to GB. */
export const bytesUnit = (n: number | null | undefined) => {
  if (!isNum(n)) return null
  let v = Math.max(0, n)
  let unit = 0
  while (v >= 1024 && unit < BYTE_UNITS.length - 1) {
    v /= 1024
    unit += 1
  }
  return BYTE_UNITS[unit]
}

/** A ratio 0..1 as a whole percentage: 0.183 → "18%". Values above 0 but under 1 % read "<1%". */
export const fmtPercent = (ratio: number | null | undefined) => {
  if (!isNum(ratio)) return DASH
  const pct = ratio * 100
  if (pct > 0 && pct < 1) return '<1%'
  return `${Math.round(pct)}%`
}

/** Local wall-clock time "12:04:31" for an ISO timestamp or epoch ms. */
export const fmtClock = (at: string | number | null | undefined) => {
  if (at === null || at === undefined || at === '') return DASH
  const d = new Date(at)
  if (Number.isNaN(d.getTime())) return DASH
  const pad = (v: number) => String(v).padStart(2, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

/** A shortened identifier for tables: "7f3a9c2e1b…" keeps the first `keep` characters. */
export const shortId = (id: string | null | undefined, keep = 8) => {
  if (!id) return DASH
  return id.length > keep + 1 ? `${id.slice(0, keep)}…` : id
}
