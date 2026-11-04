import { describe, expect, test } from 'bun:test'
import { mergeFeed, EMPTY_FEED } from './drawers/feed'
import {
  normalizeAttention,
  normalizeDocument,
  normalizeEmailQueue,
  normalizeEvents,
  normalizeFeed,
  normalizeKpis,
  normalizeOverview,
  normalizeStuck,
  normalizeWorkspace
} from './normalize'
import { kpiState } from './overview'

describe('overview from the /overview route', () => {
  const raw = {
    generated_at: '2026-11-03T04:00:00+00:00',
    state: 'degraded',
    worst: 'warning',
    kpis: {
      active: { value: 3, longest_s: 2520, approximate: true, scope: 'host', state: 'ok' },
      backlog: { value: 412, oldest_inflight_at: '2026-11-03T01:00:00+00:00', delta_1h: null, scope: 'host', state: 'ok' },
      failed_24h: { value: 17, cancelled: 5, dup: 812, state: 'ok' },
      stuck: { value: 27, runs: 0, docs: 27, reservations: 0, state: 'warning' },
      email_queue: { value: 38, workspaces: 7, longest_pending_s: 2460, bad: 0, state: 'ok' },
      llm: { available: false, error_rate: null, state: 'unknown' },
      deps: { available: false, ok: null, total: null, state: 'unknown' },
      workers: { live: 1, expected: 1, mode: 'uvicorn', state: 'ok' }
    },
    attention: [
      { id: 'stale_docs', rule: 'stale_docs', severity: 'warning', title: '27 document(s) in progress for over 60 min', params: { count: 27 }, link: '#/?ops=1&otab=processing&osub=stuck', runbook: 'Open Stuck' }
    ],
    checked: { workspaces_with_backlog: null, active_candidates: 4, email_queue: true, at: '2026-11-03T04:00:00+00:00' },
    timeline: { buckets: [], events: [{ ts: '2026-11-03T03:10:00+00:00', kind: 'deployment', detail: { version: '1.5.2' }, source: 'usage' }] },
    active_top: [],
    deps: []
  }

  test('tile values come from `value`, and the backlog age from its timestamp', () => {
    const o = normalizeOverview(raw)
    expect(o.kpis.active?.count).toBe(3)
    expect(o.kpis.backlog?.inflight).toBe(412)
    expect(o.kpis.backlog?.oldest_s).toBe(3 * 3600)
    expect(o.kpis.failed_24h?.real).toBe(17)
    expect(o.kpis.stuck?.docs).toBe(27)
    expect(o.kpis.email_queue?.records).toBe(38)
  })

  test('an unavailable LLM or dependency source is unknown, not zero', () => {
    const o = normalizeOverview(raw)
    expect(o.kpis.llm_15m).toBeUndefined()
    expect(kpiState('llm_15m', o.kpis)).toBe('unknown')
    expect(o.kpis.deps?.total).toBeNull()
  })

  test('attention links become drill targets; counts come from params', () => {
    const [a] = normalizeOverview(raw).attention
    expect(a.target).toEqual({ tab: 'processing', sub: 'stuck', workspace: null, doc_id: null, run_id: null, dep: null, worker_id: null })
    expect(a.count).toBe(27)
  })

  test('the "checked" sentence reads what the route reports', () => {
    expect(normalizeOverview(raw).checked).toEqual({ workspaces: null, active: 4, email_queue: true, at: '2026-11-03T04:00:00+00:00' })
  })

  test('timeline events are kept for the markers', () => {
    expect(normalizeOverview(raw).timeline?.events?.[0].kind).toBe('deployment')
  })

  test('a partial or empty payload never throws', () => {
    expect(normalizeOverview(null)).toMatchObject({ state: 'unknown', kpis: {}, attention: [] })
    expect(normalizeKpis({ stuck: { value: null } }, null).stuck).toEqual({ state: null, scope: null })
    // A failed stale-doc read: parts are null (or even 0 from an older server), the tile is unknown, not "0 stuck".
    expect(normalizeKpis({ stuck: { value: null, runs: 0, docs: 0, reservations: 0, state: 'unknown' } }, null).stuck).toEqual({ state: 'unknown', scope: null })
    expect(normalizeKpis({ stuck: { value: null, runs: null, docs: null, reservations: null } }, null).stuck).toEqual({ state: null, scope: null })
    expect(normalizeAttention([{ severity: 'warning' }, 'junk'])).toEqual([])
  })
})

describe('other read models', () => {
  test('stuck: threshold minutes and the stale doc `since`', () => {
    const s = normalizeStuck({
      stale_docs: { items: [{ workspace: 'w', doc_id: 'd', status: 'processing', since: '2026-11-03T03:00:00Z', file: 'R*****.pdf' }], total: 1, page: 1, page_size: 50 },
      thresholds: { doc_minutes: 60, hung_minutes: 15, orphan_seconds: 60 },
      checked_classes: ['stale_docs']
    })
    expect(s.thresholds).toEqual({ orphan_s: 60, hung_min: 15, doc_min: 60 })
    expect(s.stale_docs.items[0]).toMatchObject({ updated_at: '2026-11-03T03:00:00Z', file_path_masked: 'R*****.pdf' })
    expect(s.orphaned).toEqual([])
  })

  test('workspace: nested status counts, the job label, the live pause', () => {
    const w = normalizeWorkspace({
      workspace: 'fsktm',
      idle: false,
      manager: 'ok',
      live: {
        busy: true,
        job: 'Rep… [12 files]',
        latest_message_redacted: 'Extracting 4/12',
        provider_rejection: { role: 'extract', kind: 'budget_exhausted', status_code: 402, active: true, since: '2026-11-03T02:39:00Z' }
      },
      counts: { statuses: { pending: 310, processing: 12, failed: 4, processed: 9812 }, cancelled: 1, dup: 2 }
    })
    expect(w.live?.job_name).toBe('Rep… [12 files]')
    expect(w.live?.latest_message).toBe('Extracting 4/12')
    expect(w.live?.paused).toBe(true)
    expect(w.counts).toMatchObject({ pending: 310, processing: 12, failed: 4, processed: 9812, cancelled: 1, dup: 2 })
    expect(w.pause).toEqual([{ workspace: 'fsktm', slot: 'extract', kind: 'budget_exhausted', status_code: 402, since: '2026-11-03T02:39:00Z', retry_at: null, byo: null }])
    expect(w.live_status).toBe('ok')
  })

  test('workspace: idle and an unresponsive Manager', () => {
    expect(normalizeWorkspace({ workspace: 'x', idle: true, live: null }).live_status).toBe('idle')
    expect(normalizeWorkspace({ workspace: 'x', manager: 'unresponsive' }).live_status).toBe('unresponsive')
  })

  test('document: stages in seconds become milliseconds', () => {
    const d = normalizeDocument({ doc: { status: 'failed', stages: [{ id: 'parse', duration_s: 0.4, state: 'done' }], error: 'boom' }, calls_available: false })
    expect(d.doc.stages).toEqual([{ stage: 'parse', started_at: null, ended_at: null, duration_ms: 400, state: 'done' }])
    expect(d.doc.error_redacted).toBe('boom')
    expect(d.calls_available).toBe(false)
  })

  test('email queue pages count workspaces (total_items)', () => {
    expect(normalizeEmailQueue({ total: 38, workspaces: 7, items: [], total_items: 7 }).total_items).toBe(7)
    expect(normalizeEmailQueue({ total: 38, workspaces: 7 }).total_items).toBe(7)
  })

  test('events: a list or an {items} envelope', () => {
    const e = { ts: '2026-11-03T03:10:00Z', kind: 'ops_action', detail: { action: 'cancel' } }
    expect(normalizeEvents([e])).toHaveLength(1)
    expect(normalizeEvents({ items: [e, { kind: 'x' }] })).toHaveLength(1)
  })

  test('feed: the server next_offset wins when appending', () => {
    const f1 = mergeFeed(EMPTY_FEED, normalizeFeed({ lines: ['a', 'b'], start: 38, total: 40, next_offset: 40 }), null)
    expect(f1.next).toBe(40)
    const f2 = mergeFeed(f1, normalizeFeed({ lines: ['c'], start: 40, total: 41, next_offset: 41 }), 40)
    expect(f2).toEqual({ lines: ['a', 'b', 'c'], next: 41, reset: false })
  })
})
