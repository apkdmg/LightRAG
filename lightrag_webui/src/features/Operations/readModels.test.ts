import { describe, expect, test } from 'bun:test'
import type { AttentionItem, OpsOverview, Stuck } from './api'
import { slotKindRows } from './api'
import { EMPTY_FEED, mergeFeed } from './drawers/feed'
import { DEFAULT_OPS_HASH, readOpsHash, viewForTarget, writeOpsHash } from './hashState'
import { attentionTarget, kpiState, markerOffset, overallState, overallWord, ringCandidates, sortAttention, stuckTotal } from './overview'
import { stuckRows } from './processing/stuck'

const item = (id: string, severity: string, extra: Partial<AttentionItem> = {}): AttentionItem => ({ id, rule: 'stale_docs', severity, ...extra })

const overview = (patch: Partial<OpsOverview> = {}): OpsOverview => ({ state: 'ok', kpis: {}, attention: [], ...patch })

describe('kpiState', () => {
  test('the server state wins', () => {
    expect(kpiState('backlog', { backlog: { inflight: 3, state: 'serious' } })).toBe('serious')
  })

  test('a tile with no data is unknown, never OK', () => {
    expect(kpiState('active', {})).toBe('unknown')
    expect(kpiState('deps', { deps: { ok: null, total: null } })).toBe('unknown')
    expect(kpiState('llm_15m', null)).toBe('unknown')
  })

  test('§1.7 defaults: stuck anything is a warning', () => {
    expect(kpiState('stuck', { stuck: { runs: 0, docs: 0, reservations: 0 } })).toBe('ok')
    expect(kpiState('stuck', { stuck: { docs: 4 } })).toBe('warning')
  })

  test('§1.7 defaults: LLM error rate needs 20 calls', () => {
    expect(kpiState('llm_15m', { llm_15m: { calls: 10, error_rate: 0.9 } })).toBe('ok')
    expect(kpiState('llm_15m', { llm_15m: { calls: 200, error_rate: 0.06 } })).toBe('warning')
    expect(kpiState('llm_15m', { llm_15m: { calls: 200, error_rate: 0.21 } })).toBe('serious')
  })

  test('§1.7 defaults: the oldest pending doc, unreadable queue files, workers', () => {
    expect(kpiState('backlog', { backlog: { inflight: 5, oldest_s: 3 * 3600 } })).toBe('warning')
    expect(kpiState('backlog', { backlog: { inflight: 5, oldest_s: 7 * 3600 } })).toBe('serious')
    expect(kpiState('email_queue', { email_queue: { records: 3, bad: 1 } })).toBe('warning')
    expect(kpiState('workers', { workers: { live: 0, expected: 3 } })).toBe('critical')
  })

  test('stuckTotal sums the known parts', () => {
    expect(stuckTotal({ stuck: { runs: 1, docs: 4 } })).toBe(5)
    expect(stuckTotal({})).toBeNull()
    expect(stuckTotal({ stuck: {} })).toBeNull()
  })
})

describe('header state', () => {
  test('words: Healthy / Degraded / Down / Unknown / Maintenance', () => {
    expect(overallWord('ok')).toBe('healthy')
    expect(overallWord('warning')).toBe('degraded')
    expect(overallWord('critical')).toBe('degraded')
    expect(overallWord('down')).toBe('down')
    expect(overallWord(null)).toBe('unknown')
    expect(overallWord('ok', { until: 'x' })).toBe('maintenance')
  })

  test('the chip shows the worst open severity; Down is critical', () => {
    expect(overallState(overview({ state: 'degraded', worst: 'serious' }))).toBe('serious')
    expect(overallState(overview({ state: 'down' }))).toBe('critical')
    expect(overallState(null)).toBe('unknown')
  })
})

describe('attention', () => {
  test('worst first, server order otherwise', () => {
    const sorted = sortAttention([item('a', 'warning'), item('b', 'critical'), item('c', 'warning'), item('d', 'serious')])
    expect(sorted.map((a) => a.id)).toEqual(['b', 'd', 'a', 'c'])
  })

  test('ring candidates: the chip on a rank rise, new critical items only', () => {
    const prev = overview({ state: 'warning', attention: [item('old', 'critical'), item('w', 'warning')] })
    const next = overview({ state: 'critical', attention: [item('old', 'critical'), item('new', 'critical'), item('w2', 'warning')] })
    expect(ringCandidates(prev, next)).toEqual([
      { key: 'chip', rank: 3, source: 'chip' },
      { key: 'att:new', rank: 3, source: 'attention' }
    ])
  })

  test('an improvement or an unchanged poll asks for nothing', () => {
    const bad = overview({ state: 'critical', attention: [item('x', 'critical')] })
    expect(ringCandidates(bad, overview({ state: 'warning' }))).toEqual([])
    expect(ringCandidates(bad, bad)).toEqual([])
    expect(ringCandidates(null, bad)).toEqual([])
  })
})

describe('attentionTarget', () => {
  test('the server target wins; otherwise the rule decides, with the workspace drawer', () => {
    expect(attentionTarget(item('a', 'warning', { target: { tab: 'health' } }))).toEqual({ tab: 'health' })
    expect(attentionTarget(item('a', 'warning'))).toEqual({ tab: 'processing', sub: 'stuck' })
    expect(attentionTarget(item('a', 'warning', { rule: 'email_queue_bad', workspace: 'space1' }))).toEqual({ tab: 'processing', sub: 'email', workspace: 'space1' })
  })
})

describe('viewForTarget', () => {
  test('a tab target switches tab and sub-view and closes the drawer', () => {
    const base = { ...DEFAULT_OPS_HASH, drawer: { kind: 'workspace' as const, id: 'law' } }
    expect(viewForTarget(base, { tab: 'processing', sub: 'stuck' })).toEqual({ tab: 'processing', sub: 'stuck', drawer: null, logQuery: null })
  })

  test('a workspace or document target opens its drawer on the current tab', () => {
    expect(viewForTarget(DEFAULT_OPS_HASH, { workspace: 'fsktm' }).drawer).toEqual({ kind: 'workspace', id: 'fsktm' })
    expect(viewForTarget(DEFAULT_OPS_HASH, { workspace: 'fsktm', doc_id: 'doc-9a2f' }).drawer).toEqual({ kind: 'document', workspace: 'fsktm', id: 'doc-9a2f' })
  })

  test('malformed targets are ignored, and the result round-trips through the URL', () => {
    const v = viewForTarget(DEFAULT_OPS_HASH, { tab: 'nope', sub: 'x', workspace: 'has space' })
    expect(v).toEqual({ ...DEFAULT_OPS_HASH, logQuery: null })
    const target = viewForTarget(DEFAULT_OPS_HASH, { tab: 'processing', sub: 'backlog', workspace: 'a_bin_unimas_my' })
    expect(readOpsHash(writeOpsHash(target))).toEqual(target)
  })
})

describe('markerOffset', () => {
  test('positions inside the window, null outside', () => {
    expect(markerOffset(150, 100, 200)).toBe(50)
    expect(markerOffset(99, 100, 200)).toBeNull()
    expect(markerOffset('not a date', 100, 200)).toBeNull()
  })
})

describe('stuckRows', () => {
  const data: Stuck = {
    orphaned: [{ workspace: 'fsktm', run_id: 'r1', age_s: 420 }],
    hung: [{ workspace: 'hr', run_id: 'r2', age_s: 1320 }],
    busy_without_run: [{ workspace: 'law' }],
    reservations: [{ workspace: 'space1', age_s: 840 }],
    stale_docs: { items: [{ workspace: 'library', doc_id: 'doc-77', status: 'processing', file_path_masked: 'R*****.pdf', age_s: 4320 }], total: 1 },
    thresholds: { hung_min: 15, doc_min: 60 }
  }

  test('single-process mode never shows orphaned runs or reservations', () => {
    expect(stuckRows(data, false).map((r) => r.cls)).toEqual(['busy_without_run', 'hung', 'stale_doc'])
  })

  test('multi-worker mode shows every class, worst first', () => {
    expect(stuckRows(data, true).map((r) => r.cls)).toEqual(['orphaned', 'busy_without_run', 'reservation', 'hung', 'stale_doc'])
  })

  test('a stale doc drills to its document, a run class to its workspace', () => {
    const rows = stuckRows(data, true)
    expect(rows.find((r) => r.cls === 'stale_doc')?.drawer).toEqual({ kind: 'document', workspace: 'library', id: 'doc-77' })
    expect(rows.find((r) => r.cls === 'hung')?.drawer).toEqual({ kind: 'workspace', id: 'hr' })
  })

  test('the PR-2 shape (stale docs only, empty run classes) renders', () => {
    const only = { ...data, orphaned: [], hung: [], busy_without_run: [], reservations: [] }
    expect(stuckRows(only, true)).toHaveLength(1)
    expect(stuckRows(null, true)).toEqual([])
  })
})

describe('mergeFeed (live feed with reset detection)', () => {
  test('the first answer is a snapshot of the newest lines', () => {
    const f = mergeFeed(EMPTY_FEED, { lines: ['a', 'b'], total: 40, reset: false }, null)
    expect(f).toEqual({ lines: ['a', 'b'], next: 40, reset: false })
  })

  test('later answers append from the offset', () => {
    const f1 = mergeFeed(EMPTY_FEED, { lines: ['a', 'b'], total: 2, reset: false }, null)
    const f2 = mergeFeed(f1, { lines: ['c'], total: 3, reset: false }, 2)
    expect(f2).toEqual({ lines: ['a', 'b', 'c'], next: 3, reset: false })
  })

  test('an unchanged feed keeps the same object (no re-render)', () => {
    const f1 = mergeFeed(EMPTY_FEED, { lines: ['a'], total: 1, reset: false }, null)
    expect(mergeFeed(f1, { lines: [], total: 1, reset: false }, 1)).toBe(f1)
  })

  test('a wipe replaces the lines instead of interleaving runs', () => {
    const f1 = mergeFeed(EMPTY_FEED, { lines: ['old 1', 'old 2', 'old 3'], total: 3, reset: false }, null)
    expect(mergeFeed(f1, { lines: ['new 1'], total: 1, reset: true }, 3)).toEqual({ lines: ['new 1'], next: 1, reset: true })
    // A total below our offset means the same, even without the flag.
    expect(mergeFeed(f1, { lines: ['new 1'], total: 1, reset: false }, 3).reset).toBe(true)
  })

  test('keeps at most 200 lines', () => {
    const many = Array.from({ length: 250 }, (_, i) => `line ${i}`)
    const f = mergeFeed(EMPTY_FEED, { lines: many, total: 250, reset: false }, null)
    expect(f.lines).toHaveLength(200)
    expect(f.lines[0]).toBe('line 50')
  })
})

describe('slotKindRows', () => {
  test('flat and nested shapes give the same rows, largest first', () => {
    const flat = slotKindRows({ 'extract:budget_exhausted': 14, 'query:invalid_key': 2 })
    const nested = slotKindRows({ extract: { budget_exhausted: 14 }, query: { invalid_key: 2 } })
    expect(flat).toEqual([
      { slot: 'extract', kind: 'budget_exhausted', count: 14 },
      { slot: 'query', kind: 'invalid_key', count: 2 }
    ])
    expect(nested).toEqual(flat)
    expect(slotKindRows(null)).toEqual([])
  })
})
