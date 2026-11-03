import { describe, expect, test } from 'bun:test'
import type { CorrelateMatch, LogEntry, LogPage } from './api'
import { DEFAULT_OPS_HASH, readOpsHash, writeOpsHash } from './hashState'
import {
  DEFAULT_LOG_FILTER,
  EMPTY_STREAM,
  LOG_CAP,
  activeFields,
  anchoredScrollTop,
  applyFollow,
  applyHead,
  applyOlder,
  beforeParam,
  canLoadOlder,
  cleanCorrelateInput,
  compareNewestFirst,
  correlateView,
  decodeLogFilter,
  effectiveWindow,
  encodeLogFilter,
  exportFilters,
  filterKey,
  filenameFromDisposition,
  followSince,
  levelName,
  levelNumber,
  logsQuery,
  logsView,
  newestTs,
  nextPending,
  normalizeFilter,
  onlyWorker,
  sourceLabel,
  splitRedacted
} from './logs'
import { normalizeCorrelate, normalizeCursor, normalizeLogEntry, normalizeLogPage } from './logsNormalize'

const T0 = Date.parse('2026-11-03T04:00:00.000Z')
const iso = (offsetMs: number) => new Date(T0 + offsetMs).toISOString()

const row = (id: number, offsetMs: number, patch: Partial<LogEntry> = {}): LogEntry => ({
  id,
  ts: iso(offsetMs),
  level: 30,
  logger: 'lightrag',
  module: 'pipeline',
  lineno: 100,
  pid: 811,
  worker_id: null,
  workspace: 'fsktm',
  request_id: null,
  client_request_id: null,
  run_id: null,
  track_id: null,
  doc_id: null,
  operation: null,
  source: null,
  msg: `line ${id}`,
  exc_type: null,
  has_exc: false,
  exc: null,
  fingerprint: 'abcd',
  repeat_count: 1,
  ...patch
})

const page = (items: LogEntry[], next: LogPage['next_before'] = null): LogPage => ({ items, next_before: next, newest_ts: null })
const ids = (items: LogEntry[]) => items.map((r) => r.id)

describe('the filter and its URL form', () => {
  test('the default filter has no olq at all', () => {
    expect(encodeLogFilter(DEFAULT_LOG_FILTER)).toBeNull()
    expect(decodeLogFilter(null)).toEqual(DEFAULT_LOG_FILTER)
  })

  test('round-trips through olq and the page hash', () => {
    const filter = normalizeFilter({ level: 'ERROR', window: '24h', workspace: 'fsktm', run_id: '7f3a9c2e', q: 'budget exhausted' })
    const olq = encodeLogFilter(filter)
    expect(olq).toBe('lv=ERROR&w=24h&ws=fsktm&q=budget+exhausted&run=7f3a9c2e')
    const hash = writeOpsHash({ ...DEFAULT_OPS_HASH, tab: 'logs', sub: 'stream', logQuery: olq })
    const back = readOpsHash(hash)
    expect(back?.tab).toBe('logs')
    expect(decodeLogFilter(back?.logQuery)).toEqual(filter)
  })

  test('links the server builds (correlate, entry pivots) decode to their filter', () => {
    // lightrag/api/ops/logs.py::_logs_link, e.g. '#/?ops=1&otab=logs&olq=req%3D3c9e41d0%26w%3D7d'
    const back = readOpsHash('#/?ops=1&otab=logs&olq=req%3D3c9e41d0%26w%3D7d')
    expect(back?.tab).toBe('logs')
    expect(decodeLogFilter(back?.logQuery)).toMatchObject({ request_id: '3c9e41d0', window: '7d' })
    expect(decodeLogFilter('fp=00000000000000a7&w=7d')).toMatchObject({ fingerprint: '00000000000000a7', window: '7d' })
    expect(decodeLogFilter('trk=upload_20261103_101010_aaaa0004&w=7d').track_id).toBe('upload_20261103_101010_aaaa0004')
  })

  test('bad values fall back to the default, control characters are dropped, lengths are capped', () => {
    const f = decodeLogFilter('lv=TRACE&w=30d&ws=a%0Ab&doc=' + 'x'.repeat(400) + '&zz=1')
    expect(f.level).toBe('WARNING')
    expect(f.window).toBe('1h')
    expect(f.workspace).toBe('a b')
    expect(f.doc_id.length).toBe(256)
  })

  test('active fields count what the user set beyond level and window', () => {
    expect(activeFields(normalizeFilter({ level: 'ERROR', workspace: 'x', request_id: 'r' }))).toEqual(['workspace', 'request_id'])
  })
})

describe('query building', () => {
  test('the window becomes `from`, empty fields are left out, the limit is capped at 500', () => {
    const now = T0 + 30_000
    const q = logsQuery(normalizeFilter({ workspace: 'fsktm', window: '1h' }), now, 900)
    expect(q.level).toBe('WARNING')
    // A minute bucket keeps polls comparable.
    expect(q.from).toBe(new Date(T0 - 3600_000).toISOString())
    expect(q.workspace).toBe('fsktm')
    expect(q.limit).toBe(500)
    expect(q.q).toBeUndefined()
    expect(q.run_id).toBeUndefined()
  })

  test('a text search narrows a wider window to 24 h', () => {
    const f = normalizeFilter({ window: '7d', q: 'timeout' })
    expect(effectiveWindow(f)).toBe('24h')
    expect(logsQuery(f, T0).from).toBe(new Date(T0 - 24 * 3600_000).toISOString())
    expect(effectiveWindow(normalizeFilter({ window: '7d' }))).toBe('7d')
  })

  test('export filters carry the query without a page size', () => {
    const f = exportFilters(normalizeFilter({ run_id: 'r1', level: 'ERROR' }), T0)
    expect(f).toEqual({ level: 'ERROR', from: new Date(T0 - 3600_000).toISOString(), run_id: 'r1' })
  })

  test('the keyset cursor is sent as "ts,id"', () => {
    expect(beforeParam({ ts: '2026-11-03T04:00:00Z', id: 42 })).toBe('2026-11-03T04:00:00Z,42')
    expect(beforeParam(null)).toBeUndefined()
  })
})

describe('keyset order and follow merge', () => {
  test('equal timestamps are ordered by id, newest first', () => {
    const rows = [row(1, 0), row(3, 0), row(2, 0), row(4, -1)]
    expect(ids([...rows].sort(compareNewestFirst))).toEqual([3, 2, 1, 4])
  })

  test('a head page is sorted, de-duplicated and unmarked', () => {
    const s = applyHead('k', page([row(1, 0), row(2, 1000), row(2, 1000)], { ts: iso(0), id: 1 }))
    expect(ids(s.items)).toEqual([2, 1])
    expect(s.fresh.size).toBe(0)
    expect(s.older).toEqual({ ts: iso(0), id: 1 })
  })

  test('follow starts 10 s before the newest row', () => {
    expect(newestTs([row(1, 0), row(2, 5000), row(3, 2000)])).toBe(iso(5000))
    expect(followSince(iso(5000))).toBe(iso(-5000))
    expect(followSince(null)).toBeNull()
  })

  test('the overlap never duplicates, and a late commit lands in its place', () => {
    const s0 = applyHead('k', page([row(10, 10_000), row(9, 9000)]))
    // The follow answer repeats row 10 (overlap) and brings a late row 8 with an older timestamp.
    const s1 = applyFollow(s0, page([row(11, 12_000), row(10, 10_000), row(8, 9500)]), { baseline: false, atNewest: true, limit: 500 })
    expect(ids(s1.items)).toEqual([11, 10, 8, 9])
    expect([...s1.fresh].sort()).toEqual([11, 8])
    expect(s1.pending).toBe(0)
  })

  test('an answer with nothing new keeps the same state object', () => {
    const s0 = applyHead('k', page([row(2, 1000), row(1, 0)]))
    expect(applyFollow(s0, page([row(2, 1000)]), { baseline: false, atNewest: true, limit: 500 })).toBe(s0)
  })

  test('the marker moves to the next batch and a baseline batch is unmarked', () => {
    const s0 = applyHead('k', page([row(1, 0)]))
    const s1 = applyFollow(s0, page([row(2, 1000)]), { baseline: false, atNewest: true, limit: 500 })
    const s2 = applyFollow(s1, page([row(3, 2000)]), { baseline: false, atNewest: true, limit: 500 })
    expect([...s2.fresh]).toEqual([3])
    const s3 = applyFollow(s2, page([row(4, 3000)]), { baseline: true, atNewest: true, limit: 500 })
    expect(s3.fresh.size).toBe(0)
  })

  test('newer rows push the oldest out past the cap and move the older cursor', () => {
    const s0 = applyHead('k', page([row(3, 3000), row(2, 2000), row(1, 1000)], null), 3)
    const s1 = applyFollow(s0, page([row(5, 5000), row(4, 4000)]), { baseline: false, atNewest: true, limit: 500, cap: 3 })
    expect(ids(s1.items)).toEqual([5, 4, 3])
    expect(s1.older).toEqual({ ts: iso(3000), id: 3 })
  })

  test('a full follow page restarts the list instead of leaving a gap', () => {
    const s0 = applyHead('k', page([row(1, 0)]))
    const full = page([row(9, 9000), row(8, 8000)], { ts: iso(8000), id: 8 })
    const s1 = applyFollow(s0, full, { baseline: false, atNewest: true, limit: 2 })
    expect(ids(s1.items)).toEqual([9, 8])
    expect(s1.fresh.size).toBe(0)
    expect(s1.older).toEqual({ ts: iso(8000), id: 8 })
  })

  test('the first rows into an empty list are a marked batch and bring their cursor', () => {
    const empty = applyHead('k', page([]))
    const s1 = applyFollow(empty, page([row(2, 1000), row(1, 0)], { ts: iso(0), id: 1 }), { baseline: false, atNewest: true, limit: 100 })
    expect(ids(s1.items)).toEqual([2, 1])
    expect([...s1.fresh].sort()).toEqual([1, 2])
    expect(s1.older).toEqual({ ts: iso(0), id: 1 })
  })

  test('the DOM cap is 1000 rows', () => {
    const many = Array.from({ length: 1200 }, (_, i) => row(i + 1, i * 10))
    const s = applyHead('k', page(many))
    expect(s.items.length).toBe(LOG_CAP)
    expect(s.items[0].id).toBe(1200)
    expect(s.older).toEqual({ ts: iso(200 * 10), id: 201 })
    expect(canLoadOlder(s)).toBe(false)
  })

  test('load older appends below, de-duplicated, and stops at the window end', () => {
    const s0 = applyHead('k', page([row(5, 5000), row(4, 4000)], { ts: iso(4000), id: 4 }))
    expect(canLoadOlder(s0)).toBe(true)
    const s1 = applyOlder(s0, page([row(4, 4000), row(3, 3000)], null))
    expect(ids(s1.items)).toEqual([5, 4, 3])
    expect(s1.older).toBeNull()
    expect(canLoadOlder(s1)).toBe(false)
  })

  test('the empty stream never passes for the default filter\'s rows', () => {
    expect(EMPTY_STREAM.key).not.toBe(filterKey(DEFAULT_LOG_FILTER))
  })

  test('an unavailable answer marks the stream unavailable', () => {
    expect(applyHead('k', { ...page([]), available: false }).unavailable).toBe(true)
    expect(applyFollow(EMPTY_STREAM, { ...page([]), available: false }, { baseline: false, atNewest: true, limit: 5 }).unavailable).toBe(true)
  })
})

describe('the "N new" pill and the scroll anchor', () => {
  test('rows that arrive while scrolled away count up; back at the newest end they clear', () => {
    expect(nextPending(0, 3, false)).toBe(3)
    expect(nextPending(3, 2, false)).toBe(5)
    expect(nextPending(5, 1, true)).toBe(0)
    expect(nextPending(2, -1, false)).toBe(2)
  })

  test('follow while scrolled away raises the pill count; at the newest end it stays 0', () => {
    const s0 = applyHead('k', page([row(1, 0)]))
    const away = applyFollow(s0, page([row(2, 1000), row(3, 2000)]), { baseline: false, atNewest: false, limit: 500 })
    expect(away.pending).toBe(2)
    const more = applyFollow(away, page([row(4, 3000)]), { baseline: false, atNewest: false, limit: 500 })
    expect(more.pending).toBe(3)
    expect(applyFollow(more, page([row(5, 4000)]), { baseline: false, atNewest: true, limit: 500 }).pending).toBe(0)
  })

  test('scrolled away, the scroll position moves by the height that went in above', () => {
    expect(anchoredScrollTop({ scrollTop: 300, scrollHeight: 2000 }, 2120)).toBe(420)
    // At the newest end nothing moves: the new rows are what the reader sees.
    expect(anchoredScrollTop({ scrollTop: 0, scrollHeight: 2000 }, 2120)).toBe(0)
  })
})

describe('levels, sources and redacted text', () => {
  test('level names and numbers', () => {
    expect(levelNumber('ERROR')).toBe(40)
    expect(levelNumber('warn')).toBe(30)
    expect(levelNumber(50)).toBe(50)
    expect(levelNumber('35')).toBe(35)
    expect(levelName(45)).toBe('ERROR')
    expect(levelName(30)).toBe('WARNING')
  })

  test('the source column shows the module for lightrag records and the logger otherwise', () => {
    expect(sourceLabel({ logger: 'lightrag', module: 'pipeline', lineno: 1 })).toBe('pipeline')
    expect(sourceLabel({ logger: 'lightrag.api.email', module: 'email_routes', lineno: 1 })).toBe('email_routes')
    expect(sourceLabel({ logger: 'uvicorn.error', module: 'httptools_impl', lineno: 1 })).toBe('uvicorn.error')
  })

  test('placeholders and masked file names become spans; the rest stays text', () => {
    const parts = splitRedacted('Email from <email> skipped: "Sen•••er.pdf" key=<redacted> ic <ic>')
    expect(parts).toEqual([
      { kind: 'text', text: 'Email from ' },
      { kind: 'token', token: 'email' },
      { kind: 'text', text: ' skipped: "' },
      { kind: 'file', text: 'Sen•••er.pdf' },
      { kind: 'text', text: '" key=' },
      { kind: 'token', token: 'redacted' },
      { kind: 'text', text: ' ic ' },
      { kind: 'token', token: 'ic' }
    ])
    expect(splitRedacted('<script>')).toEqual([{ kind: 'text', text: '<script>' }])
    expect(splitRedacted('')).toEqual([])
  })
})

describe('normalising server answers', () => {
  test('rows without an id or time are dropped; levels and ids accept both spellings', () => {
    const p = normalizeLogPage({
      items: [
        { id: '7', ts: '2026-11-03T04:00:00Z', level: 'ERROR', logger: 'lightrag', msg: 'x', repeat_count: 3 },
        { id: 8, ts: '2026-11-03T04:00:01Z', level: 40, message: 'y', exc: 'Traceback…' },
        { ts: '2026-11-03T04:00:02Z', level: 40 },
        { id: 9, level: 40 }
      ],
      next_before: '2026-11-03T04:00:00Z,7',
      newest_ts: '2026-11-03T04:00:01Z'
    })
    expect(p.items.map((r) => [r.id, r.level, r.msg, r.has_exc, r.repeat_count])).toEqual([
      [7, 40, 'x', false, 3],
      [8, 40, 'y', true, 1]
    ])
    expect(p.next_before).toEqual({ ts: '2026-11-03T04:00:00Z', id: 7 })
  })

  test('cursors in every shape', () => {
    expect(normalizeCursor({ ts: 'a', id: 3 })).toEqual({ ts: 'a', id: 3 })
    expect(normalizeCursor(['a', '3'])).toEqual({ ts: 'a', id: 3 })
    expect(normalizeCursor('2026-11-03T04:00:00+00:00,12')).toEqual({ ts: '2026-11-03T04:00:00+00:00', id: 12 })
    expect(normalizeCursor('nonsense')).toBeNull()
    expect(normalizeCursor(null)).toBeNull()
  })

  test('a log entry never throws on junk', () => {
    expect(normalizeLogEntry(null)).toBeNull()
    expect(normalizeLogEntry({ id: 1, ts: 'x' })?.msg).toBe('')
  })

  test('correlate keeps known types (and aliases), at most 10', () => {
    const c = normalizeCorrelate({
      matches: [
        { type: 'request_id', id: 'abc' },
        { type: 'run', id: 'r1', workspace: 'fsktm' },
        { type: 'mystery', id: 'x' },
        { type: 'owner', id: 'a•••@unimas.my', workspace: 'fsktm' },
        ...Array.from({ length: 20 }, (_, i) => ({ type: 'doc', id: `doc-${i}`, workspace: 'w' }))
      ]
    })
    expect(c.matches.length).toBe(10)
    expect(c.matches.slice(0, 3).map((m) => m.type)).toEqual(['request', 'run', 'owner'])
  })
})

describe('correlate drill-down', () => {
  const base = { ...DEFAULT_OPS_HASH, tab: 'processing' as const, sub: 'active' }
  const m = (patch: Partial<CorrelateMatch>): CorrelateMatch => ({ type: 'run', id: 'x', workspace: null, ts: null, label: null, count: null, ...patch })

  test('each id type lands on its inspector', () => {
    expect(correlateView(base, m({ type: 'run', id: 'r1' }))?.drawer).toEqual({ kind: 'run', id: 'r1' })
    expect(correlateView(base, m({ type: 'log', id: '42' }))?.drawer).toEqual({ kind: 'log', id: '42' })
    expect(correlateView(base, m({ type: 'doc', id: 'doc-1', workspace: 'fsktm' }))?.drawer).toEqual({ kind: 'document', workspace: 'fsktm', id: 'doc-1' })
    expect(correlateView(base, m({ type: 'workspace', id: 'fsktm' }))?.drawer).toEqual({ kind: 'workspace', id: 'fsktm' })
    expect(correlateView(base, m({ type: 'owner', id: 'a•••@x', workspace: 'fsktm' }))?.drawer).toEqual({ kind: 'workspace', id: 'fsktm' })
    expect(correlateView(base, m({ type: 'owner', id: 'a•••@x' }))).toBeNull()
  })

  test('drawers open over the current tab; request and track ids open the Logs tab filtered', () => {
    expect(correlateView(base, m({ type: 'run', id: 'r1' }))?.tab).toBe('processing')
    const req = correlateView(base, m({ type: 'request', id: '3c9e41d0' }))
    expect(req?.tab).toBe('logs')
    expect(req?.drawer).toBeNull()
    expect(decodeLogFilter(req?.logQuery)).toMatchObject({ request_id: '3c9e41d0', window: '7d' })
    expect(decodeLogFilter(correlateView(base, m({ type: 'track', id: 'email_1' }))?.logQuery).track_id).toBe('email_1')
    // A doc without a known workspace is looked up in the logs.
    expect(decodeLogFilter(correlateView(base, m({ type: 'doc', id: 'doc-9' }))?.logQuery).doc_id).toBe('doc-9')
  })

  test('the correlate input is one printable token', () => {
    expect(cleanCorrelateInput('  3c9e41d0 ')).toBe('3c9e41d0')
    expect(cleanCorrelateInput('two words')).toBeNull()
    expect(cleanCorrelateInput('')).toBeNull()
    expect(cleanCorrelateInput('x'.repeat(321))).toBeNull()
  })

  test('logsView closes the drawer and carries the filter', () => {
    const v = logsView({ run_id: 'r1' }, 'grouped')
    expect(v).toEqual({ tab: 'logs', sub: 'grouped', drawer: null, logQuery: 'run=r1' })
  })
})

describe('drawer sections and export helpers', () => {
  test('the Worker drawer keeps only its own rows', () => {
    const rows = [row(1, 0, { worker_id: 'w1' }), row(2, 0, { worker_id: 'w2' }), row(3, 0, { pid: 811 }), row(4, 0, { pid: 812 })]
    expect(ids(onlyWorker(rows, 'w1', 811))).toEqual([1, 3])
    expect(ids(onlyWorker(rows, 'w1', null))).toEqual([1])
  })

  test('export file names come from the server without a path', () => {
    expect(filenameFromDisposition('attachment; filename="ops-logs-2026-11-03.ndjson"', 'x')).toBe('ops-logs-2026-11-03.ndjson')
    expect(filenameFromDisposition('attachment; filename*=UTF-8\'\'..%2F..%2Fevil.csv', 'x')).toBe('evil.csv')
    expect(filenameFromDisposition(null, 'fallback.csv')).toBe('fallback.csv')
  })
})
