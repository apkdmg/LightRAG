import { pointsAttr } from '../topology'

const W = 240
const H = 40
const PAD = 2

/**
 * SVG polyline points for a series scaled into the sparkline box; gaps (null)
 * split the line.
 */
function sparkSegments(values: (number | null)[], width = W, height = H): [number, number][][] {
  const finite = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  if (!finite.length) return []
  const min = Math.min(...finite, 0)
  const max = Math.max(...finite)
  const span = max - min || 1
  const step = values.length > 1 ? (width - 2 * PAD) / (values.length - 1) : 0
  const segments: [number, number][][] = []
  let current: [number, number][] = []
  values.forEach((v, i) => {
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      if (current.length) segments.push(current)
      current = []
      return
    }
    const x = Math.round((PAD + i * step) * 10) / 10
    const y = Math.round((height - PAD - ((v - min) / span) * (height - 2 * PAD)) * 10) / 10
    current.push([x, y])
  })
  if (current.length) segments.push(current)
  return segments
}

/**
 * A static 24 h sparkline (the Worker drawer's RSS, CPU, loop lag and pool
 * wait). It never animates (§2.12 "Charts"); it dims with its panel when the
 * data is stale. The numbers it summarises are in the text beside it, so the
 * drawing is aria-hidden.
 */
export default function Sparkline({ values, color = 'var(--ops-info)' }: { values: (number | null)[]; color?: string }) {
  const segments = sparkSegments(values)
  return (
    <svg className="ops-dim ops-spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true" focusable="false">
      <line x1={0} y1={H - 0.5} x2={W} y2={H - 0.5} stroke="var(--border)" strokeWidth={1} />
      {segments.map((seg, i) =>
        seg.length === 1 ? (
          <circle key={i} cx={seg[0][0]} cy={seg[0][1]} r={1.5} fill={color} />
        ) : (
          <polyline key={i} points={pointsAttr(seg)} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
        )
      )}
    </svg>
  )
}
