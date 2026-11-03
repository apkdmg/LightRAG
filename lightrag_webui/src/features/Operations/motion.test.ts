import { describe, expect, test } from 'bun:test'
import { bytesUnit, fmtBytes, fmtCount, fmtPercent } from './format'
import { AttentionSlot, RING_COOLDOWN_MS, flowBucket, isBaseline, isWorsening, kpiDelta, shouldTint, shouldTween } from './motion'

const live = { format: fmtCount, baseline: false, reduced: false, hidden: false }

describe('shouldTween (§2.12 KPI value change)', () => {
  test('a changed formatted value tweens', () => {
    expect(shouldTween(412, 530, live)).toBe(true)
  })

  test('equal formatted values give no tween, even when the raw numbers differ', () => {
    expect(shouldTween(412, 412, live)).toBe(false)
    expect(shouldTween(0.1831, 0.1849, { ...live, format: fmtPercent })).toBe(false)
  })

  test('a baseline poll, reduced motion and a hidden tab jump instead', () => {
    expect(shouldTween(412, 530, { ...live, baseline: true })).toBe(false)
    expect(shouldTween(412, 530, { ...live, reduced: true })).toBe(false)
    expect(shouldTween(412, 530, { ...live, hidden: true })).toBe(false)
  })

  test('first paint has nothing to tween from', () => {
    expect(shouldTween(null, 530, live)).toBe(false)
    expect(shouldTween(412, null, live)).toBe(false)
    expect(shouldTween(412, 530, { ...live, hasPrevious: false })).toBe(false)
  })

  test('a unit change (MB → GB) jumps', () => {
    const mb = 900 * 1024 * 1024
    const gb = 1.2 * 1024 * 1024 * 1024
    expect(shouldTween(mb, gb, { ...live, format: fmtBytes, unit: bytesUnit })).toBe(false)
    expect(shouldTween(mb, mb * 1.05, { ...live, format: fmtBytes, unit: bytesUnit })).toBe(true)
  })
})

describe('shouldTint', () => {
  test('a tint appears only on a severity change', () => {
    expect(shouldTint('ok', 'ok')).toBe(false)
    expect(shouldTint('warning', 'warning')).toBe(false)
    expect(shouldTint('ok', 'warning')).toBe(true)
    expect(shouldTint('serious', 'warning')).toBe(true)
  })

  test('states that rank the same do not tint', () => {
    expect(shouldTint('ok', 'running')).toBe(false)
    expect(shouldTint('unknown', 'ok')).toBe(false)
  })

  test('a baseline context gives no tint', () => {
    expect(shouldTint('ok', 'critical', { baseline: true })).toBe(false)
    expect(shouldTint('ok', 'critical', { hidden: true })).toBe(false)
  })
})

describe('isWorsening', () => {
  test('a rank rise is a worsening, a drop is not', () => {
    expect(isWorsening('ok', 'warning')).toBe(true)
    expect(isWorsening('warning', 'critical')).toBe(true)
    expect(isWorsening('critical', 'serious')).toBe(false)
    expect(isWorsening('serious', 'serious')).toBe(false)
  })

  test('Down counts as critical; Unknown and Maintenance never ring', () => {
    expect(isWorsening('warning', 'down')).toBe(true)
    expect(isWorsening('ok', 'unknown')).toBe(false)
    expect(isWorsening('ok', 'maintenance')).toBe(false)
  })
})

describe('isBaseline', () => {
  test('baseline, hidden and first paint replay nothing', () => {
    expect(isBaseline({ baseline: true })).toBe(true)
    expect(isBaseline({ hidden: true })).toBe(true)
    expect(isBaseline({ hasPrevious: false })).toBe(true)
    expect(isBaseline({})).toBe(false)
  })
})

describe('kpiDelta', () => {
  test('says the change in text with a glyph', () => {
    expect(kpiDelta(412, 530, fmtCount)).toEqual({ glyph: '▲', text: '+118' })
    expect(kpiDelta(530, 412, fmtCount)).toEqual({ glyph: '▼', text: '−118' })
  })

  test('nothing for an unchanged formatted value or a missing side', () => {
    expect(kpiDelta(412, 412, fmtCount)).toBeNull()
    expect(kpiDelta(null, 412, fmtCount)).toBeNull()
  })

  test('formats like the tile', () => {
    expect(kpiDelta(1000, 2500, fmtCount)).toEqual({ glyph: '▲', text: '+1,500' })
  })
})

describe('AttentionSlot (rule 2)', () => {
  const chip = { key: 'chip', rank: 3, source: 'chip' as const }
  const item = { key: 'att:milvus', rank: 3, source: 'attention' as const }
  const node = { key: 'node:milvus', rank: 3, source: 'node' as const }

  test('same-poll priority: the chip first, then an attention item, then a node', () => {
    expect(new AttentionSlot().offer([node, item, chip], 0)).toEqual(chip)
    expect(new AttentionSlot().offer([node, item], 0)).toEqual(item)
    expect(new AttentionSlot().offer([node], 0)).toEqual(node)
  })

  test('one grant at a time: a ring that cannot get the slot is dropped, not queued', () => {
    const slot = new AttentionSlot()
    expect(slot.offer([chip], 0)).toEqual(chip)
    expect(slot.offer([item], 100)).toBeNull()
    // After the ring the slot is free again, and the dropped ring did not wait in line.
    expect(slot.token('att:milvus')).toBe(0)
    expect(slot.offer([item], 1000)).toEqual(item)
  })

  test('the cooldown blocks a second ring for the same target and rank', () => {
    const slot = new AttentionSlot()
    expect(slot.offer([chip], 0)).not.toBeNull()
    expect(slot.offer([chip], 60_000)).toBeNull()
    expect(slot.offer([chip], RING_COOLDOWN_MS - 1)).toBeNull()
    expect(slot.offer([chip], RING_COOLDOWN_MS)).toEqual(chip)
  })

  test('a higher rank is a new target for the cooldown', () => {
    const slot = new AttentionSlot()
    slot.offer([{ key: 'chip', rank: 1, source: 'chip' }], 0)
    expect(slot.offer([{ key: 'chip', rank: 3, source: 'chip' }], 5_000)).not.toBeNull()
  })

  test('a candidate in cooldown passes the slot to the next one', () => {
    const slot = new AttentionSlot()
    slot.offer([chip], 0)
    expect(slot.offer([chip, item], 5_000)).toEqual(item)
  })

  test('tokens count grants and notify subscribers', () => {
    const slot = new AttentionSlot()
    let calls = 0
    const off = slot.subscribe(() => {
      calls += 1
    })
    slot.offer([chip], 0)
    slot.offer([chip], RING_COOLDOWN_MS)
    off()
    slot.offer([chip], RING_COOLDOWN_MS * 2)
    expect(slot.token('chip')).toBe(3)
    expect(calls).toBe(2)
  })

  test('nothing offered, nothing granted', () => {
    expect(new AttentionSlot().offer([], 0)).toBeNull()
  })
})

describe('flowBucket (bucket edges per edge)', () => {
  const edge = { p50: 10, p90: 40 }

  test('no traffic is idle', () => {
    expect(flowBucket(0, edge)).toBe('idle')
    expect(flowBucket(null, edge)).toBe('idle')
    expect(flowBucket(-1, edge)).toBe('idle')
  })

  test('the edge’s own p50 and p90 are the bucket boundaries', () => {
    expect(flowBucket(9.99, edge)).toBe('low')
    expect(flowBucket(10, edge)).toBe('medium')
    expect(flowBucket(39.9, edge)).toBe('medium')
    expect(flowBucket(40, edge)).toBe('high')
  })

  test('each edge uses its own baseline', () => {
    expect(flowBucket(20, { p50: 10, p90: 40 })).toBe('medium')
    expect(flowBucket(20, { p50: 100, p90: 400 })).toBe('low')
  })

  test('without a baseline a busy edge flows slowly', () => {
    expect(flowBucket(5)).toBe('low')
    expect(flowBucket(5, { p50: 2 })).toBe('medium')
  })
})
