import { describe, expect, test } from 'bun:test'
import { DASH, ageSeconds, bytesUnit, fmtAge, fmtBytes, fmtClock, fmtCount, fmtDuration, fmtLatency, fmtPercent, fmtRatePct, shortId } from './format'

describe('format', () => {
  test('every formatter prints a dash for missing values', () => {
    for (const fn of [fmtAge, fmtDuration, fmtLatency, fmtBytes, fmtPercent, fmtCount]) {
      expect(fn(null)).toBe(DASH)
      expect(fn(undefined)).toBe(DASH)
      expect(fn(Number.NaN)).toBe(DASH)
    }
    expect(fmtClock(null)).toBe(DASH)
    expect(fmtClock('not a date')).toBe(DASH)
    expect(shortId(null)).toBe(DASH)
  })

  test('fmtAge uses at most two units', () => {
    expect(fmtAge(-5)).toBe('now')
    expect(fmtAge(0.4)).toBe('now')
    expect(fmtAge(12)).toBe('12s')
    expect(fmtAge(60)).toBe('1m')
    expect(fmtAge(245)).toBe('4m 05s')
    expect(fmtAge(3600)).toBe('1h')
    expect(fmtAge(7800)).toBe('2h 10m')
    expect(fmtAge(86400)).toBe('1d')
    expect(fmtAge(3 * 86400 + 4 * 3600 + 59)).toBe('3d 4h')
  })

  test('ageSeconds accepts ISO strings and epoch ms', () => {
    const now = Date.parse('2026-11-03T12:00:00Z')
    expect(ageSeconds('2026-11-03T11:59:00Z', now)).toBe(60)
    expect(ageSeconds(now - 1500, now)).toBe(1.5)
    expect(ageSeconds('garbage', now)).toBeNull()
    expect(ageSeconds(null, now)).toBeNull()
  })

  test('fmtDuration', () => {
    expect(fmtDuration(850)).toBe('850 ms')
    expect(fmtDuration(1450)).toBe('1.4 s')
    expect(fmtDuration(42_000)).toBe('42 s')
    expect(fmtDuration(130_000)).toBe('2m 10s')
  })

  test('fmtLatency shows sub-millisecond values honestly', () => {
    expect(fmtLatency(0.3)).toBe('<1 ms')
    expect(fmtLatency(41.4)).toBe('41 ms')
    expect(fmtLatency(1234)).toBe('1.2 s')
    expect(fmtLatency(41_000)).toBe('41 s')
  })

  test('fmtBytes uses binary units', () => {
    expect(fmtBytes(512)).toBe('512 B')
    expect(fmtBytes(1536)).toBe('1.5 KB')
    expect(fmtBytes(2.1 * 1024 ** 3)).toBe('2.1 GB')
    expect(fmtBytes(41 * 1024 ** 3)).toBe('41 GB')
    expect(bytesUnit(900 * 1024 ** 2)).toBe('MB')
    expect(bytesUnit(1100 * 1024 ** 2)).toBe('GB')
    expect(bytesUnit(null)).toBeNull()
  })

  test('fmtPercent and fmtCount', () => {
    expect(fmtPercent(0.183)).toBe('18%')
    expect(fmtPercent(0.004)).toBe('<1%')
    expect(fmtPercent(0)).toBe('0%')
    expect(fmtCount(1234567)).toBe('1,234,567')
  })

  test('fmtClock prints local wall-clock time', () => {
    const d = new Date(2026, 10, 3, 9, 4, 31)
    expect(fmtClock(d.getTime())).toBe('09:04:31')
  })

  test('shortId keeps short ids whole', () => {
    expect(shortId('7f3a')).toBe('7f3a')
    expect(shortId('7f3a9c2e1b44')).toBe('7f3a9c2e…')
  })

  test('fmtRatePct keeps one decimal below 10 % so 0.4 % and 0 % differ', () => {
    expect(fmtRatePct(0.004)).toBe('0.4%')
    expect(fmtRatePct(0)).toBe('0%')
    expect(fmtRatePct(0.0004)).toBe('<0.1%')
    expect(fmtRatePct(0.182)).toBe('18%')
    expect(fmtRatePct(null)).toBe(DASH)
  })
})
