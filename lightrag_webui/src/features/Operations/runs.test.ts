import { describe, expect, test } from 'bun:test'
import type { OpsRun, Stuck } from './api'
import { fmtStamp } from './format'
import { DEFAULT_OPS_HASH, readOpsHash, viewForTarget, writeOpsHash } from './hashState'
import {
  normalizeActive,
  normalizeAttention,
  normalizeDocument,
  normalizeEmailQueue,
  normalizeFailureDocs,
  normalizeFailures,
  normalizeRun,
  normalizeRunDetail,
  normalizeRuns,
  normalizeStuck,
  normalizeWorkspace,
  runDisplayState
} from './normalize'
import { attentionTarget } from './overview'
import {
  DEFAULT_RUNS_FILTER,
  failureKindCounts,
  mergeRunPages,
  runDrawerOf,
  runDurationSeconds,
  runGuidance,
  runsQuery
} from './processing/runs'
import { checkedClasses, needsRestart, stuckRows, stuckWhy } from './processing/stuck'
import { isLiveRunState, runLookOf } from './severity'

const run = (id: string, patch: Partial<OpsRun> = {}): OpsRun => ({ run_id: id, workspace: 'fsktm', status: 'succeeded', state: 'succeeded', ...patch })

describe('run state (what a run shows as)', () => {
  test('a finished run shows its outcome, whatever the live state says', () => {
    expect(runDisplayState('succeeded', 'running')).toBe('succeeded')
    expect(runDisplayState('orphaned', null)).toBe('orphaned')
    expect(runDisplayState('paused', undefined)).toBe('paused')
  })

  test('a running run shows the live classification, or running', () => {
    expect(runDisplayState('running', 'hung')).toBe('hung')
    expect(runDisplayState('running', 'unresponsive')).toBe('unresponsive')
    expect(runDisplayState('running', null)).toBe('running')
  })

  test('server spellings of outcomes are folded into the §3.5 vocabulary', () => {
    expect(runDisplayState('ok', null)).toBe('succeeded')
    expect(runDisplayState('canceled', null)).toBe('cancelled')
    expect(runDisplayState(null, null)).toBe('unknown')
  })

  test('every outcome has a look with icon and text; failed is not critical, cancelled is neutral', () => {
    expect(runLookOf('succeeded')).toMatchObject({ state: 'ok', label: 'operations.run.succeeded.label', icon: 'check' })
    expect(runLookOf('failed').state).toBe('serious')
    expect(runLookOf('cancelled').state).toBe('unknown')
    expect(isLiveRunState('hung')).toBe(true)
    expect(isLiveRunState('orphaned')).toBe(false)
    expect(isLiveRunState('succeeded')).toBe(false)
  })
})

describe('normalizeRun and friends', () => {
  test('a run row keeps its fields and derives its duration', () => {
    const r = normalizeRun({
      run_id: '7f3a9c2e',
      workspace: 'fsktm',
      kind: 'process',
      trigger: 'email',
      status: 'succeeded',
      started_at: '2026-11-03T01:00:00Z',
      finished_at: '2026-11-03T01:02:30Z',
      docs_total: 12,
      docs_done: 12,
      docs_failed: 0
    })
    expect(r).toMatchObject({ run_id: '7f3a9c2e', state: 'succeeded', duration_s: 150, was_orphaned: false, docs_total: 12 })
  })

  test('rows without an id are dropped, a missing workspace takes the drawer workspace', () => {
    expect(normalizeRun({ workspace: 'x' })).toBeNull()
    expect(normalizeRun({ run_id: 'r1' })).toBeNull()
    expect(normalizeRun({ run_id: 'r1' }, 'law')?.workspace).toBe('law')
    expect(normalizeRun({ id: 'r1', workspace: 'w' })?.run_id).toBe('r1')
  })

  test('a run without a status and without a finish time is running', () => {
    expect(normalizeRun({ run_id: 'r', workspace: 'w', state: 'hung' })).toMatchObject({ status: 'running', state: 'hung' })
  })

  test('/runs: items, a string cursor, or a {ts} cursor; junk is empty', () => {
    expect(normalizeRuns({ items: [{ run_id: 'a', workspace: 'w', status: 'failed' }], next_before: '2026-11-03T00:00:00Z' })).toMatchObject({
      items: [{ run_id: 'a', state: 'failed' }],
      next_before: '2026-11-03T00:00:00Z'
    })
    expect(normalizeRuns({ items: [], next_before: { ts: 't1', id: 'x' } }).next_before).toBe('t1')
    expect(normalizeRuns({ items: [], next_before: null }).next_before).toBeNull()
    expect(normalizeRuns('junk')).toMatchObject({ items: [], next_before: null })
  })

  test('/runs/{id}: run, failures (newest first as sent) and the log count', () => {
    const d = normalizeRunDetail({
      run: { run_id: 'r1', workspace: 'fsktm', status: 'running', state: 'hung', pid: 812 },
      failures: [
        { ts: '2026-11-03T02:44:00Z', doc_id: 'doc-1', stage: 'extract', error_kind: 'provider_budget', http_status: 402, attempt: 3, model: 'gpt-x' },
        { doc_id: 'no-ts' }
      ],
      log_count: null
    })
    expect(d.run).toMatchObject({ run_id: 'r1', state: 'hung', pid: 812 })
    expect(d.failures).toHaveLength(1)
    expect(d.failures[0]).toMatchObject({ error_kind: 'provider_budget', http_status: 402, attempt: 3 })
    expect(d.log_count).toBeNull()
    expect(normalizeRunDetail({}).run).toBeNull()
  })

  test('/processing/active: durable rows show their live state; interim rows keep theirs', () => {
    const a = normalizeActive({
      runs: [
        { run_id: 'r1', workspace: 'a', status: 'running', state: 'hung' },
        { run_id: null, workspace: 'b', state: 'stale_doc' },
        { workspace: '' }
      ],
      approximate: false
    })
    expect(a.runs.map((r) => [r.run_id, r.state])).toEqual([
      ['r1', 'hung'],
      [null, 'stale_doc']
    ])
    expect(a.approximate).toBe(false)
  })

  test('workspace drawer: recent runs as run rows, failures by kind from either spelling', () => {
    const w = normalizeWorkspace({
      workspace: 'fsktm',
      recent_runs: [{ run_id: 'r1', status: 'orphaned', kind: 'process' }, { kind: 'scan' }],
      failures_by_kind: [{ error_kind: 'provider_budget', count: 3 }, { key: 'parser_timeout', count: 1 }, { key: 'x', count: 0 }]
    })
    expect(w.recent_runs).toEqual([expect.objectContaining({ run_id: 'r1', workspace: 'fsktm', state: 'orphaned' })])
    expect(w.failures_by_kind?.map((f) => [f.key, f.count])).toEqual([
      ['provider_budget', 3],
      ['parser_timeout', 1]
    ])
  })

  test('document drawer: metadata.failure and the attempt history', () => {
    const d = normalizeDocument({
      doc: { status: 'failed', failure: { stage: 'extract', error_kind: 'provider_budget', http_status: 402, attempt: 3, last_run_id: 'r9', model: 'gpt-x' } },
      failures: [{ ts: '2026-11-03T02:44:00Z', run_id: 'r9', attempt: 3, error_kind: 'provider_budget' }]
    })
    expect(d.doc.failure).toMatchObject({ error_kind: 'provider_budget', attempt: 3, last_run_id: 'r9' })
    expect(d.failures[0].run_id).toBe('r9')
    expect(normalizeDocument({ doc: { status: 'processed', failure: {} } }).doc.failure).toBeNull()
    expect(normalizeDocument({ doc: { status: 'failed', failure: null } }).doc.failure).toBeNull()
  })

  test('failure groups: the key may be named after the group', () => {
    const f = normalizeFailures({ groups: [{ error_kind: 'llm_bad_output', count: 6 }, { model: 'gpt-x', count: 2 }], durable: true })
    expect(f.groups.map((g) => g.key)).toEqual(['llm_bad_output', 'gpt-x'])
    expect(f.durable).toBe(true)
  })

  test('failure docs: classified fields and the failure time', () => {
    const p = normalizeFailureDocs({ items: [{ workspace: 'w', doc_id: 'd', ts: '2026-11-03T00:00:00Z', error_kind: 'parser_timeout', attempt: 2, message: 'ReadTimeout' }] })
    expect(p.items[0]).toMatchObject({ updated_at: '2026-11-03T00:00:00Z', error_kind: 'parser_timeout', attempt: 2, error_redacted: 'ReadTimeout' })
  })

  test('email queue: the backoff deadline under either spelling, and the last attempt', () => {
    const q = normalizeEmailQueue({ items: [{ workspace: 'fsktm', pending: 2, attempts: 7, last_result_class: 'slot_busy', next_attempt_at: 't1', last_attempt_at: 't0' }] })
    expect(q.items[0]).toMatchObject({ backoff_until: 't1', last_attempt_at: 't0', attempts: 7, drainer: null })
    expect(normalizeWorkspace({ workspace: 'w', email_queue: { pending: 1, backoff_until: 't2' } }).email_queue?.backoff_until).toBe('t2')
  })

  test('stuck run classes: owner fields and reservation flags are read', () => {
    const s = normalizeStuck({
      orphaned: [{ workspace: 'fsktm', run_id: 'r1', owner_pid: 811, heartbeat_age_s: 540, confirmed_by: 'pid_gone' }],
      reservations: [{ workspace: 'space1', scanning: true, age_s: 840 }],
      checked_classes: ['orphaned', 'hung', 'stale_docs', 'busy_without_run', 'reservations', 7],
      checked: { workspaces: 3012, runs: 3 }
    })
    expect(s.orphaned[0]).toMatchObject({ pid: 811, heartbeat_age_s: 540, confirmed_by: 'pid_gone' })
    expect(s.reservations[0].scanning).toBe(true)
    expect(s.checked_classes).toEqual(['orphaned', 'hung', 'stale_docs', 'busy_without_run', 'reservations'])
    expect(s.checked?.workspaces).toBe(3012)
  })
})

describe('run history paging', () => {
  test('the query leaves out empty filters and carries the keyset cursor', () => {
    expect(runsQuery(DEFAULT_RUNS_FILTER)).toEqual({ workspace: undefined, kind: undefined, status: undefined, before: undefined })
    expect(runsQuery({ workspace: '  fsktm ', kind: 'email_batch', status: 'orphaned' }, 'c1')).toEqual({
      workspace: 'fsktm',
      kind: 'email_batch',
      status: 'orphaned',
      before: 'c1'
    })
    expect(String(runsQuery({ ...DEFAULT_RUNS_FILTER, workspace: 'x'.repeat(300) }).workspace).length).toBe(255)
  })

  test('older pages append after the polled first page; the first page wins a duplicate', () => {
    const first = [run('c', { state: 'running', status: 'running' }), run('b')]
    const older = [[run('b', { state: 'failed' }), run('a')]]
    const merged = mergeRunPages(first, older)
    expect(merged.map((r) => r.run_id)).toEqual(['c', 'b', 'a'])
    expect(merged[1].state).toBe('succeeded')
  })

  test('duration: the server value, the finish time, or elapsed for a running run', () => {
    expect(runDurationSeconds({ duration_s: 42, status: 'succeeded' }, null)).toBe(42)
    expect(runDurationSeconds({ started_at: '2026-11-03T00:00:00Z', finished_at: '2026-11-03T00:01:00Z', status: 'failed' }, null)).toBe(60)
    expect(runDurationSeconds({ started_at: '2026-11-03T00:00:00Z', status: 'running' }, Date.parse('2026-11-03T00:00:30Z'))).toBe(30)
    expect(runDurationSeconds({ started_at: '2026-11-03T00:00:00Z', status: 'orphaned' }, Date.parse('2026-11-03T00:00:30Z'))).toBeNull()
  })
})

describe('run drawer helpers', () => {
  test('failures counted by kind, most frequent first, unclassified as unknown', () => {
    expect(
      failureKindCounts([
        { ts: 't', error_kind: 'provider_budget' },
        { ts: 't', error_kind: 'parser_timeout' },
        { ts: 't', error_kind: 'provider_budget' },
        { ts: 't' }
      ])
    ).toEqual([
      { key: 'provider_budget', count: 2 },
      { key: 'parser_timeout', count: 1 },
      { key: 'unknown', count: 1 }
    ])
  })

  test('guidance for runs that need a person; orphaned only under Gunicorn', () => {
    expect(runGuidance('hung', false)).toBe('hung')
    expect(runGuidance('unresponsive', true)).toBe('unresponsive')
    expect(runGuidance('paused', false)).toBe('paused')
    expect(runGuidance('orphaned', true)).toBe('orphaned')
    expect(runGuidance('orphaned', false)).toBeNull()
    expect(runGuidance('running', true)).toBeNull()
    expect(runGuidance('succeeded', true)).toBeNull()
  })

  test('an active row opens its Run drawer, an interim row its workspace', () => {
    expect(runDrawerOf({ run_id: 'r1', workspace: 'w' })).toEqual({ kind: 'run', id: 'r1' })
    expect(runDrawerOf({ run_id: null, workspace: 'w' })).toEqual({ kind: 'workspace', id: 'w' })
  })
})

describe('Run drawer addressing (hashState)', () => {
  test('orun round-trips and opens the Run drawer', () => {
    const view = { ...DEFAULT_OPS_HASH, tab: 'processing' as const, sub: 'active', drawer: { kind: 'run' as const, id: '7f3a9c2e-1b2c' } }
    expect(writeOpsHash(view)).toBe('#/?ops=1&otab=processing&osub=active&orun=7f3a9c2e-1b2c')
    expect(readOpsHash(writeOpsHash(view))).toEqual(view)
  })

  test('the runs sub-view is addressable', () => {
    expect(readOpsHash('#/?ops=1&otab=processing&osub=runs')?.sub).toBe('runs')
  })

  test('a run target opens the Run drawer; a doc target still wins; a bad run id falls back to the workspace', () => {
    expect(viewForTarget(DEFAULT_OPS_HASH, { run_id: 'r1', workspace: 'fsktm' }).drawer).toEqual({ kind: 'run', id: 'r1' })
    expect(viewForTarget(DEFAULT_OPS_HASH, { run_id: 'r1', workspace: 'fsktm', doc_id: 'd' }).drawer).toEqual({ kind: 'document', workspace: 'fsktm', id: 'd' })
    expect(viewForTarget(DEFAULT_OPS_HASH, { run_id: 'bad id', workspace: 'fsktm' }).drawer).toEqual({ kind: 'workspace', id: 'fsktm' })
  })

  test('a hung-run attention item drills to its run', () => {
    const [item] = normalizeAttention([{ id: 'hung:r1', rule: 'run_hung', severity: 'serious', params: { workspace: 'fsktm', run_id: 'r1' } }])
    expect(item.run_id).toBe('r1')
    const view = viewForTarget(DEFAULT_OPS_HASH, attentionTarget(item))
    expect(view).toMatchObject({ tab: 'processing', sub: 'stuck', drawer: { kind: 'run', id: 'r1' } })
  })
})

describe('the five stuck classes', () => {
  const data: Stuck = {
    orphaned: [{ workspace: 'fsktm', run_id: 'r1', pid: 811, heartbeat_age_s: 540, confirmed_by: 'pid_gone' }],
    hung: [{ workspace: 'hr', run_id: 'r2', pid: 812, progress_age_s: 1320 }],
    busy_without_run: [{ workspace: 'law' }],
    reservations: [{ workspace: 'space1', scanning: true }, { workspace: 'lib', pending_enqueues: 2 }],
    stale_docs: { items: [], total: 0 },
    thresholds: {}
  }

  test('a reason is built from the row when the server sends none', () => {
    const rows = stuckRows(data, true)
    const by = (cls: string, ws?: string) => rows.find((r) => r.cls === cls && (!ws || r.workspace === ws))!
    expect(stuckWhy(by('orphaned'))).toEqual({ key: 'orphaned', values: { pid: 811, how: 'pid_gone' } })
    expect(stuckWhy(by('hung'))).toEqual({ key: 'hung', values: { pid: 812 } })
    expect(stuckWhy(by('reservation', 'space1')).key).toBe('reservationScan')
    expect(stuckWhy(by('reservation', 'lib'))).toEqual({ key: 'reservationEnqueue', values: { count: 2 } })
    expect(stuckWhy(by('busy_without_run')).key).toBe('busy_without_run')
    expect(stuckWhy({ ...by('hung'), pid: null }).key).toBe('hungNoPid')
  })

  test('restart guidance for the classes only a restart clears today', () => {
    expect(needsRestart('reservation')).toBe(true)
    expect(needsRestart('busy_without_run')).toBe(true)
    expect(needsRestart('orphaned')).toBe(true)
    expect(needsRestart('hung')).toBe(false)
    expect(needsRestart('stale_doc')).toBe(false)
  })

  test('checked classes: what the server says, limited to the process mode; an old server checked stale docs', () => {
    const all = { ...data, checked_classes: ['orphaned', 'hung', 'stale_docs', 'busy_without_run', 'reservations'] }
    expect(checkedClasses(all, true)).toEqual(['orphaned', 'hung', 'stale_docs', 'busy_without_run', 'reservations'])
    expect(checkedClasses(all, false)).toEqual(['hung', 'stale_docs', 'busy_without_run'])
    expect(checkedClasses({ ...data, checked_classes: undefined }, true)).toEqual(['stale_docs'])
  })
})

describe('fmtStamp', () => {
  const now = new Date(2026, 10, 3, 12, 0, 0).getTime()
  test('time only today, month-day on another day, the year before this one', () => {
    expect(fmtStamp(new Date(2026, 10, 3, 9, 14, 5).getTime(), now)).toBe('09:14:05')
    expect(fmtStamp(new Date(2026, 10, 1, 9, 14, 5).getTime(), now)).toBe('11-01 09:14')
    expect(fmtStamp(new Date(2025, 11, 30, 9, 14, 5).getTime(), now)).toBe('2025-12-30 09:14')
    expect(fmtStamp(null, now)).toBe('–')
    expect(fmtStamp('nope', now)).toBe('–')
  })
})

/**
 * The exact shapes the PR-4 backend sends (OpsService._running_state,
 * reconcile.check_shared_state, OpsService._compute_stuck_runs), so the
 * contract between the two sides is pinned in one place.
 */
describe('backend contract', () => {
  test('every live state the server classifies a run with has a look and an explanation', () => {
    for (const state of ['running', 'cancelling', 'paused', 'hung', 'unresponsive']) {
      const look = runLookOf(state)
      expect(look.label).toBe(`operations.run.${state}.label`)
      expect(look.explain).toBe(`operations.run.${state}.explain`)
    }
    expect(isLiveRunState('cancelling')).toBe(true)
    expect(isLiveRunState('unresponsive')).toBe(true)
    expect(runGuidance('unresponsive', false)).toBe('unresponsive')
    const active = normalizeActive({ runs: [{ run_id: 'r1', workspace: 'fsktm', status: 'running', state: 'unresponsive', worker_heartbeat_age_s: 420 }] })
    expect(active.runs[0]).toMatchObject({ state: 'unresponsive', worker_heartbeat_age_s: 420 })
  })

  test('reason codes, reservation flags and the unresponsive marker as the server sends them', () => {
    const s = normalizeStuck({
      orphaned: [{ workspace: 'fsktm', run_id: 'r1', pid: 811, age_s: 900, confirmed_by: 'host_gone', state: 'orphaned' }],
      hung: [
        { workspace: 'hr', run_id: 'r2', pid: 812, age_s: 1320, progress_age_s: 1320, state: 'hung' },
        { workspace: 'law', run_id: 'r3', pid: 813, age_s: 420, worker_heartbeat_age_s: 420, state: 'unresponsive', owner_unresponsive: true }
      ],
      busy_without_run: [
        { workspace: 'space1', reason_code: 'run_finished', run_id: 'r4', age_s: 180 },
        { workspace: 'lib', reason_code: 'run_unknown', run_id: 'r5', age_s: 180 },
        { workspace: 'eng', reason: 'no_run', age_s: 600 }
      ],
      reservations: [
        { workspace: 'hr_office', kind: 'scanning', scanning: true, reason_code: 'owner_dead', count: null, age_s: 400 },
        // A server one release behind: kind and count only, reason under `reason`.
        { workspace: 'fsktm', kind: 'pending_enqueues', reason: 'held', count: 2, age_s: 400 }
      ],
      stale_docs: { items: [], total: 0 },
      checked_classes: ['orphaned', 'hung', 'busy_without_run', 'reservations', 'stale_docs']
    })
    const rows = stuckRows(s, true)
    const by = (ws: string) => rows.find((r) => r.workspace === ws && r.cls !== 'stale_doc')!
    expect(by('law').cls).toBe('unresponsive')
    expect(stuckWhy(by('law'))).toEqual({ key: 'unresponsive', values: { pid: 813 } })
    expect(by('law').workerHeartbeatAge).toBe(420)
    expect(needsRestart('unresponsive')).toBe(true)
    expect(stuckWhy(by('space1')).key).toBe('busy_run_finished')
    expect(stuckWhy(by('eng')).key).toBe('busy_no_run')
    // A run the table does not hold opens the workspace, not an empty Run drawer.
    expect(by('lib').drawer).toEqual({ kind: 'workspace', id: 'lib' })
    expect(by('space1').drawer).toEqual({ kind: 'run', id: 'r4' })
    const scan = rows.find((r) => r.workspace === 'hr_office')!
    expect(scan.scanning).toBe(true)
    expect(stuckWhy(scan).key).toBe('reservationScan_owner_dead')
    const held = rows.find((r) => r.cls === 'reservation' && r.workspace === 'fsktm')!
    expect(held.scanning).toBe(false)
    expect(stuckWhy(held)).toEqual({ key: 'reservationEnqueue', values: { count: 2 } })
    expect(stuckWhy(by('fsktm'))).toEqual({ key: 'orphaned', values: { pid: 811, how: 'host_gone' } })
    // No row carries server text to print.
    expect(rows.some((r) => 'reason' in r)).toBe(false)
  })

  test('email queue: drains of the last hour', () => {
    const q = normalizeEmailQueue({ items: [{ workspace: 'w', pending: 3, drains_1h: { ok: 2, failed: 1, slot_busy: 5 } }, { workspace: 'x', pending: 1 }] })
    expect(q.items[0].drains_1h).toEqual({ ok: 2, failed: 1, slot_busy: 5 })
    expect(q.items[1].drains_1h).toBeNull()
  })
})
