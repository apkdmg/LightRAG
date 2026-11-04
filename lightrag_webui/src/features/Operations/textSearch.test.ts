import { describe, expect, test } from 'bun:test'
import { DEFAULT_LOG_FILTER, effectiveWindow, exportFilters, logsQuery, normalizeFilter, textSearchWindow, trgmSearchable } from './logs'

const NOW = Date.parse('2026-11-04T10:00:30Z')
const filter = normalizeFilter({ ...DEFAULT_LOG_FILTER, window: '7d', q: 'budget' })
const fromHours = (from: unknown) => (Date.parse('2026-11-04T10:00:00Z') - Date.parse(String(from))) / 3600_000

describe('text search reach (V7, FR-20)', () => {
  test('24 h unless the server has a valid pg_trgm index', () => {
    expect(textSearchWindow(undefined)).toBe('24h')
    expect(textSearchWindow({})).toBe('24h')
    expect(textSearchWindow({ log_trgm: false })).toBe('24h')
    expect(textSearchWindow({ log_trgm: true })).toBe('7d')
  })

  test('without the index a 7-day text search is narrowed to 24 h; with it the whole window stays', () => {
    expect(effectiveWindow(filter)).toBe('24h')
    expect(effectiveWindow(filter, '7d')).toBe('7d')
    expect(fromHours(logsQuery(filter, NOW).from)).toBe(24)
    expect(fromHours(logsQuery(filter, NOW, 100, '7d').from)).toBe(168)
    expect(fromHours(exportFilters(filter, NOW, '7d').from)).toBe(168)
  })

  test('a filter without text is never narrowed', () => {
    const plain = normalizeFilter({ ...DEFAULT_LOG_FILTER, window: '7d' })
    expect(effectiveWindow(plain)).toBe('7d')
    expect(effectiveWindow({ ...filter, window: '6h' }, '24h')).toBe('6h')
  })

  test('a q too short for a trigram keeps the 24 h cap even with the index (server rule)', () => {
    for (const q of ['ab', 'db', 'a b', '1.2', '-->', 'a_b', '  ab  ']) {
      const short = normalizeFilter({ ...DEFAULT_LOG_FILTER, window: '7d', q })
      expect(trgmSearchable(q)).toBe(false)
      expect(effectiveWindow({ window: '7d', q }, '7d')).toBe('24h')
      expect(fromHours(logsQuery(short, NOW, 100, '7d').from)).toBe(24)
      expect(fromHours(exportFilters(short, NOW, '7d').from)).toBe(24)
    }
    for (const q of ['abc', 'x db1', 'error: abc', '实体名']) {
      expect(trgmSearchable(q)).toBe(true)
      expect(effectiveWindow({ window: '7d', q }, '7d')).toBe('7d')
    }
    expect(effectiveWindow({ window: '6h', q: 'ab' }, '7d')).toBe('6h')
  })
})
