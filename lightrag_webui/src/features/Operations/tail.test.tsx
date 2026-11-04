import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'
import type { LogTail, OpsStatus } from './api'

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

const { EMPTY_TAIL, TAIL_CAP, mergeTail, normalizeLogTail, normalizeTailLine, sortTail, tailFeature, tailKey, tailQuery, takesBatch } = await import('./logview/tail')
const { default: TailView, TailRows } = await import('./logview/TailView')
const { OpsPrefsContext } = await import('./prefs')
const { OpsShellContext } = await import('./context')

/** One line as GET /logs/tail returns it (ops/jsonl.py entry_view). */
const line = (sec: number, msg: string, extra: Record<string, unknown> = {}) => ({
  ts: new Date(Date.UTC(2026, 10, 4, 10, 0, sec)).toISOString(),
  level: 20,
  level_name: 'INFO',
  logger: 'lightrag',
  module: 'pipeline',
  lineno: 812,
  pid: 811,
  workspace: 'fsktm',
  request_id: null,
  run_id: null,
  msg,
  exc_type: null,
  exc: null,
  ...extra
})

const answer = (items: unknown[], extra: Record<string, unknown> = {}): LogTail =>
  normalizeLogTail({ generated_at: '2026-11-04T10:01:00Z', enabled: true, level: 'INFO', module: null, items, sources: [{ pid: 811, alive: true, truncated: false }], lines: 200, ...extra })

const KEY = tailKey('INFO', '')
const live = { baseline: false, atNewest: true }

describe('the tail answer (GET /logs/tail)', () => {
  test('reads the server shape: level by number or name, sources as files, a cut-short read', () => {
    const t = answer([line(2, 'b'), line(1, 'a', { level: 0, level_name: 'WARNING' })], { sources: [{ pid: 1 }, { pid: 2, truncated: true }] })
    expect(t.enabled).toBe(true)
    expect(t.files).toBe(2)
    expect(t.truncated).toBe(true)
    expect(t.items.map((l) => l.level)).toEqual([20, 30])
    expect(normalizeLogTail({ enabled: false, items: [], sources: [] })).toMatchObject({ enabled: false, items: [], files: 0, truncated: false })
    expect(normalizeLogTail({ items: [], sources_capped: true }).truncated).toBe(true)
  })

  test('a line without a readable time is dropped; epoch seconds are accepted', () => {
    expect(normalizeTailLine({ msg: 'x' })).toBeNull()
    expect(normalizeTailLine({ ts: 'soon', msg: 'x' })).toBeNull()
    expect(normalizeTailLine({ ts: 1793786400.5, msg: 'x', pid: 1 })?.ts).toBe(new Date(1793786400500).toISOString())
  })

  test('the same line read by two polls has the same key; different lines do not', () => {
    const a = normalizeTailLine(line(1, 'Processed 3 chunks'))!
    const b = normalizeTailLine(line(1, 'Processed 3 chunks'))!
    const c = normalizeTailLine(line(1, 'Processed 4 chunks'))!
    const d = normalizeTailLine(line(1, 'Processed 3 chunks', { pid: 812 }))!
    expect(a.key).toBe(b.key)
    expect(a.key).not.toBe(c.key)
    expect(a.key).not.toBe(d.key)
    // The key never carries the message itself.
    expect(a.key).not.toContain('Processed')
  })

  test('query: lines bounded, module only when chosen', () => {
    expect(tailQuery('INFO', '')).toEqual({ lines: 200, level: 'INFO', module: undefined })
    expect(tailQuery('ERROR', 'operate', 5000)).toEqual({ lines: 1000, level: 'ERROR', module: 'operate' })
  })

  test('offered unless the server says the files are off', () => {
    expect(tailFeature({ log_tail: true })).toBe(true)
    expect(tailFeature({ log_tail: false })).toBe(false)
    expect(tailFeature({})).toBeNull()
    expect(tailFeature(null)).toBeNull()
  })
})

describe('merging polls (newest first, deduplicated, capped)', () => {
  test('lines from several processes are ordered by time, newest first; equal times keep the server order', () => {
    const t = answer([line(3, 'c', { pid: 812 }), line(1, 'a'), line(2, 'b', { pid: 813 }), line(2, 'b2', { pid: 811 })])
    expect(sortTail(t.items).map((l) => l.msg)).toEqual(['c', 'b', 'b2', 'a'])
  })

  test('the first answer goes in unmarked; an overlapping poll adds only what is new, and marks it', () => {
    const first = mergeTail(EMPTY_TAIL, KEY, answer([line(2, 'b'), line(1, 'a')]), live)
    expect(first.items.map((l) => l.msg)).toEqual(['b', 'a'])
    expect(first.fresh.size).toBe(0)
    const next = mergeTail(first, KEY, answer([line(3, 'c'), line(2, 'b'), line(1, 'a')]), live)
    expect(next.items.map((l) => l.msg)).toEqual(['c', 'b', 'a'])
    expect([...next.fresh]).toEqual([next.items[0].key])
  })

  test('nothing new keeps the very same state (no re-render)', () => {
    const first = mergeTail(EMPTY_TAIL, KEY, answer([line(1, 'a')]), live)
    expect(mergeTail(first, KEY, answer([line(1, 'a')]), live)).toBe(first)
  })

  test('a late line from another process lands in its time order', () => {
    const first = mergeTail(EMPTY_TAIL, KEY, answer([line(5, 'e'), line(1, 'a')]), live)
    const next = mergeTail(first, KEY, answer([line(5, 'e'), line(3, 'late', { pid: 900 })]), live)
    expect(next.items.map((l) => l.msg)).toEqual(['e', 'late', 'a'])
  })

  test('the page keeps at most TAIL_CAP lines; the oldest fall off', () => {
    const many = Array.from({ length: TAIL_CAP + 40 }, (_, i) => line(0, `m${i}`, { ts: new Date(Date.UTC(2026, 10, 4, 10, 0, 0, i)).toISOString() }))
    const state = mergeTail(EMPTY_TAIL, KEY, answer(many), live)
    expect(state.items.length).toBe(TAIL_CAP)
    expect(state.items[0].msg).toBe(`m${TAIL_CAP + 39}`)
    expect(state.items.at(-1)!.msg).toBe('m40')
    const more = mergeTail(state, KEY, answer([line(59, 'newest')]), live)
    expect(more.items.length).toBe(TAIL_CAP)
    expect(more.items[0].msg).toBe('newest')
  })

  test('a baseline poll (resume, visible again) adds lines unmarked', () => {
    const first = mergeTail(EMPTY_TAIL, KEY, answer([line(1, 'a')]), live)
    expect(mergeTail(first, KEY, answer([line(2, 'b')]), { baseline: true, atNewest: true }).fresh.size).toBe(0)
  })

  test('scrolled away, new lines count into the "↑ N new" pill; at the top it stays 0', () => {
    const first = mergeTail(EMPTY_TAIL, KEY, answer([line(1, 'a')]), live)
    const away = mergeTail(first, KEY, answer([line(2, 'b'), line(3, 'c')]), { baseline: false, atNewest: false })
    expect(away.pending).toBe(2)
    expect(mergeTail(away, KEY, answer([line(4, 'd')]), live).pending).toBe(0)
  })

  test('another level or module starts over', () => {
    const first = mergeTail(EMPTY_TAIL, KEY, answer([line(1, 'a')]), live)
    const other = mergeTail(first, tailKey('ERROR', ''), answer([line(9, 'z', { level: 40 })]), live)
    expect(other.key).toBe(tailKey('ERROR', ''))
    expect(other.items.map((l) => l.msg)).toEqual(['z'])
    expect(other.fresh.size).toBe(0)
  })

  test('the server turning the files off is kept, so the view can say so', () => {
    const first = mergeTail(EMPTY_TAIL, KEY, answer([line(1, 'a')]), live)
    expect(mergeTail(first, KEY, normalizeLogTail({ enabled: false, items: [] }), live).enabled).toBe(false)
  })
})

describe('pause', () => {
  test('paused, no answer goes in; the first answer for a new level or module still does', () => {
    expect(takesBatch(true, KEY, KEY, KEY)).toBe(true)
    expect(takesBatch(false, KEY, KEY, KEY)).toBe(false)
    expect(takesBatch(false, KEY, tailKey('ERROR', ''), tailKey('ERROR', ''))).toBe(true)
    // An answer for a filter that is no longer current never goes in.
    expect(takesBatch(true, KEY, tailKey('ERROR', ''), KEY)).toBe(false)
  })
})

const prefs = (patch: { reduceMotion?: boolean; paused?: boolean } = {}) => ({ paused: false, setPaused: () => {}, reduceMotion: false, setReduceMotion: () => {}, ...patch })
const status = (features: Record<string, boolean>): OpsStatus => ({
  available: true,
  features,
  process: { mode: 'uvicorn', expected_workers: 1, workers_live: 1 },
  operator: true,
  version: null,
  generated_at: '2026-11-04T10:00:00Z'
})
const render = (node: React.ReactNode, patch: { reduceMotion?: boolean } = {}, s: OpsStatus = status({ log_tail: true })) =>
  renderToString(
    <OpsShellContext.Provider value={{ status: s, openInsights: () => {} }}>
      <OpsPrefsContext.Provider value={prefs(patch)}>{node}</OpsPrefsContext.Provider>
    </OpsShellContext.Provider>
  )

describe('Live tail rows (§2.12)', () => {
  const state = () => {
    const first = mergeTail(EMPTY_TAIL, KEY, answer([line(1, 'a', { request_id: 'req-12345678abcd' })]), live)
    return mergeTail(first, KEY, answer([line(2, 'user ⟨email⟩ signed in', { level: 30, level_name: 'WARNING' })]), live)
  }

  test('a new line carries the marker and fades in; reduced motion keeps only the marker', () => {
    const s = state()
    const html = render(<TailRows items={s.items} fresh={s.fresh} />)
    expect(html).toContain('data-new=""')
    expect(html).toContain('ops-log-enter')
    const reduced = render(<TailRows items={s.items} fresh={s.fresh} />, { reduceMotion: true })
    expect(reduced).toContain('data-new=""')
    expect(reduced).not.toContain('ops-log-enter')
    expect(reduced).not.toMatch(/transform[:=]/i)
  })

  test('level as text, redaction tokens as chips, the request ID as a pivot (no drawer: the line has no stored entry)', () => {
    const s = state()
    const html = render(<TailRows items={s.items} fresh={s.fresh} />)
    expect(html).toContain('WARNING')
    expect(html).toContain('INFO')
    expect(html).toContain('operations.logs.ctx.request')
    expect(html).toContain('olq=')
    expect(html).not.toContain('olog=')
  })

  test('the view says so when the server writes no JSONL files', () => {
    const html = render(<TailView module="" enabled={false} />)
    expect(html).toContain('operations.tail.off')
    // Nothing to pause when there is nothing to read.
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>.*operations\.tail\.pause/)
  })

  test('before the first answer it shows a skeleton and the controls', () => {
    const html = render(<TailView module="operate" enabled />)
    expect(html).toContain('aria-busy="true"')
    expect(html).toContain('operations.tail.pause')
    expect(html).toContain('operations.tail.module')
  })
})
