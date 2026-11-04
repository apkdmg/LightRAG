import { describe, expect, test } from 'bun:test'
import {
  STRIP_CELLS,
  SLOT_MS,
  canRecheck,
  inRecheckAll,
  pgHostFacts,
  cellLabel,
  depCounts,
  depIsStale,
  depState,
  detailParts,
  drawerStatus,
  groupConfig,
  normalizeConfig,
  normalizeDep,
  normalizeDepDetail,
  normalizeHealth,
  normalizeProbeResult,
  normalizeSmtpTest,
  normalizeStrip,
  normalizeWorkerHistory,
  normalizeWorkers,
  poolText,
  slotRange,
  sortDeps,
  stripUptime,
  toLightragStatus,
  toProbeStatus,
  workerLook
} from './model'

// Untranslated labels: the key and its params, so assertions read the logic, not the copy.
const t = (key: string, opts?: Record<string, unknown>) => (opts ? `${key} ${JSON.stringify(opts)}` : key)

const AT = Date.parse('2026-11-03T04:04:20Z')

describe('probe status', () => {
  test('server spellings map onto ok / warn / down / unknown / unprobed', () => {
    expect(toProbeStatus('OK')).toBe('ok')
    expect(toProbeStatus('slow')).toBe('warn')
    expect(toProbeStatus('timeout')).toBe('down')
    expect(toProbeStatus('not_probed')).toBe('unprobed')
    expect(toProbeStatus('weird')).toBe('unknown')
    expect(toProbeStatus(null)).toBe('unknown')
  })

  test('down reads Critical, not probed reads Unknown (never OK)', () => {
    expect(depState('down')).toBe('critical')
    expect(depState('warn')).toBe('warning')
    expect(depState('unprobed')).toBe('unknown')
    expect(depState('ok')).toBe('ok')
  })
})

describe('availability strip', () => {
  test('always 48 cells, oldest first, padded with unknown at the start', () => {
    const cells = normalizeStrip(['ok', 'down', 'warn'], AT)
    expect(cells).toHaveLength(STRIP_CELLS)
    expect(cells.slice(0, 45).every((c) => c.status === 'unknown')).toBe(true)
    expect(cells.slice(45).map((c) => c.status)).toEqual(['ok', 'down', 'warn'])
  })

  test('cells without times count back from the half hour that holds the latest cycle', () => {
    const cells = normalizeStrip(['ok'], AT)
    const last = Date.parse('2026-11-03T04:00:00Z')
    expect(cells[47].start).toBe(last)
    expect(cells[0].start).toBe(last - 47 * SLOT_MS)
  })

  test('a longer history keeps the newest 48', () => {
    const cells = normalizeStrip([...Array(50).fill('down'), 'ok'], AT)
    expect(cells).toHaveLength(48)
    expect(cells[47].status).toBe('ok')
    expect(cells[0].status).toBe('down')
  })

  test('counted cells derive their state; their own times win', () => {
    const cells = normalizeStrip(
      [
        { t: '2026-11-03T03:00:00Z', ok: 60, failed: 0 },
        { t: '2026-11-03T03:30:00Z', total: 60, failed: 2 },
        { t: '2026-11-03T04:00:00Z', total: 4, failed: 4 },
        { total: 0 },
        0.5
      ],
      AT
    )
    const tail = cells.slice(-5)
    expect(tail.map((c) => c.status)).toEqual(['ok', 'warn', 'down', 'unknown', 'warn'])
    expect(tail[0].start).toBe(Date.parse('2026-11-03T03:00:00Z'))
    expect(tail[1].total).toBe(60)
    expect(tail[1].failed).toBe(2)
  })

  test('every cell has a text label, with the slot range and the counts', () => {
    const [cell] = normalizeStrip([{ t: '2026-11-03T03:00:00Z', total: 60, failed: 2 }], AT).slice(-1)
    const label = cellLabel(cell, t)
    expect(label).toContain('operations.strip.cell.failed')
    expect(label).toContain('"failed":2')
    expect(label).toContain('"total":60')
    expect(label).toContain(slotRange(cell.start) as string)
    expect(cellLabel({ start: null, status: 'unknown', total: null, failed: null, slow: null }, t)).toContain('operations.strip.cell.unknown')
  })

  test('uptime counts only cells with data', () => {
    expect(stripUptime(normalizeStrip(['ok', 'ok', 'down', null], AT))).toBeCloseTo(2 / 3)
    expect(stripUptime(normalizeStrip([], AT))).toBeNull()
  })
})

describe('/health', () => {
  const raw = {
    generated_at: '2026-11-03T04:04:31Z',
    available: false,
    cycle: { at: '2026-11-03T04:04:20Z', duration_ms: 1400, holder: 'a1b2c3d4-0000', pid: 812 },
    deps: [
      { id: 'milvus', status: 'down', latency_ms: null, error_class: 'TimeoutError', last_ok_at: '2026-11-03T04:00:02Z', consecutive_failures: 8, strip: ['ok', 'down'] },
      { id: 'postgres', status: 'ok', latency_ms: 3, detail: { connections: 61, max_connections: 200, blocked: 0 } },
      { id: 'etcd', status: 'not_probed' },
      { status: 'ok' }
    ]
  }

  test('reads the cycle, the deps, and drops entries without an id', () => {
    const h = normalizeHealth(raw)
    expect(h.cycle?.holder_pid).toBe(812)
    expect(h.cycle?.duration_ms).toBe(1400)
    expect(h.deps.map((d) => d.id)).toEqual(['milvus', 'postgres', 'etcd'])
    expect(h.deps[0].strip).toHaveLength(48)
    expect(h.deps[1].strip).toEqual([])
    expect(h.deps[2].status).toBe('unprobed')
  })

  test('probes run without PostgreSQL KV, so a payload with deps is never "unavailable"', () => {
    expect(normalizeHealth(raw).available).toBe(true)
    expect(normalizeHealth({ available: false, deps: [] }).available).toBe(false)
  })

  test('a map keyed by id is accepted too', () => {
    const h = normalizeHealth({ deps: { postgres: { status: 'ok' }, smtp: { status: 'warn' } } })
    expect(h.deps.map((d) => `${d.id}:${d.status}`)).toEqual(['postgres:ok', 'smtp:warn'])
  })

  test('grid order puts storage first; counts skip unprobed', () => {
    const deps = normalizeHealth(raw).deps
    expect(sortDeps(deps).map((d) => d.id)).toEqual(['postgres', 'milvus', 'etcd'])
    expect(depCounts(deps)).toEqual({ ok: 1, total: 2 })
  })

  test('passive and unprobed dependencies cannot be re-checked', () => {
    expect(canRecheck(normalizeDep({ id: 'postgres', status: 'ok' })!)).toBe(true)
    expect(canRecheck(normalizeDep({ id: 'llm', status: 'ok' })!)).toBe(false)
    expect(canRecheck(normalizeDep({ id: 'etcd', status: 'unprobed' })!)).toBe(false)
  })

  test('host Postgres facts come from the postgres card extras', () => {
    const pg = normalizeDep({
      id: 'postgres',
      status: 'ok',
      detail: { connections: 50, max_connections: 200, db_size_bytes: 1024, blocked: 0, db_delta: { window_s: 300, commits: 10, cache_hit_pct: 98.5 } }
    })!
    const facts = pgHostFacts([pg])!
    expect(facts.connectionsPct).toBe(25)
    expect(facts.dbSizeBytes).toBe(1024)
    expect(facts.delta).toEqual({ windowS: 300, commits: 10, rollbacks: null, deadlocks: null, tempBytes: null, cacheHitPct: 98.5 })
    expect(pgHostFacts([normalizeDep({ id: 'postgres', status: 'ok', detail: { connect_ms: 1 } })!])).toBeNull()
    expect(pgHostFacts([])).toBeNull()
  })

  test('"Re-check all" leaves out the slow-cadence probes (SMTP, TLS, IdP)', () => {
    expect(inRecheckAll(normalizeDep({ id: 'postgres', status: 'ok' })!)).toBe(true)
    expect(inRecheckAll(normalizeDep({ id: 'smtp', status: 'ok' })!)).toBe(false)
    expect(inRecheckAll(normalizeDep({ id: 'tls', status: 'ok', interval_s: 900 })!)).toBe(false)
    expect(inRecheckAll(normalizeDep({ id: 'idp', status: 'ok' })!)).toBe(false)
    expect(canRecheck(normalizeDep({ id: 'smtp', status: 'ok' })!)).toBe(true)
  })

  test('a card goes stale after twice its own cadence: 60 s for most, 30 min for SMTP', () => {
    const pg = normalizeDep({ id: 'postgres', status: 'ok', checked_at: '2026-11-03T04:00:00Z' })!
    const smtp = normalizeDep({ id: 'smtp', status: 'ok', checked_at: '2026-11-03T04:00:00Z' })!
    const fiveMin = Date.parse('2026-11-03T04:05:00Z')
    expect(depIsStale(pg, fiveMin, null)).toBe(true)
    expect(depIsStale(smtp, fiveMin, null)).toBe(false)
    expect(depIsStale(smtp, Date.parse('2026-11-03T04:31:00Z'), null)).toBe(true)
  })

  test('the detail line comes from whitelisted keys only', () => {
    const pg = normalizeHealth(raw).deps[1]
    const keys = detailParts(pg).map(([k]) => k)
    expect(keys).toEqual(['operations.health.detail.conns', 'operations.health.detail.blocked'])
    const vllm = normalizeDep({ id: 'vllm_rerank', status: 'ok', detail: { metrics_available: false } })!
    expect(detailParts(vllm).map(([k]) => k)).toEqual(['operations.health.detail.metricsUnavailable'])
    const tls = normalizeDep({ id: 'tls', status: 'warn', detail: { days_left: 12.7, self_signed: true } })!
    expect(detailParts(tls)).toEqual([['operations.health.detail.certSelfSigned', { days: 12 }]])
    const pgGraph = normalizeDep({ id: 'graph', kind: 'graph', status: 'unprobed', detail: { reason: 'covered_by_postgres' } })!
    expect(detailParts(pgGraph)).toEqual([['operations.health.detail.reason.covered_by_postgres', {}]])
    const odd = normalizeDep({ id: 'etcd', status: 'unprobed', detail: { reason: '<b>x</b>' } })!
    expect(detailParts(odd)).toEqual([])
  })

  test('the passive Sign-in card: 15-minute counts, passive, never who', () => {
    const signin = normalizeDep({ id: 'signin', name: 'Sign-in', kind: 'internal', passive: true, status: 'warn', detail: { window: '15m', sso_errors: 6, token_401: 3, lockouts: 0 } })!
    expect(signin.passive).toBe(true)
    expect(canRecheck(signin)).toBe(false)
    expect(detailParts(signin)).toEqual([['operations.health.detail.signin', { sso: 6, token401: 3, lockouts: 0 }]])
    expect(detailParts(normalizeDep({ id: 'signin', status: 'unknown', detail: { window: '15m' } })!)).toEqual([])
  })

  test('the server card shape: target, not-probed flag, stale without a result', () => {
    const etcd = normalizeDep({ id: 'etcd', name: 'etcd', kind: 'vector_backend', probed: false, status: 'unknown', strip: [] })!
    expect(etcd.status).toBe('unprobed')
    expect(depIsStale(etcd, AT, null)).toBe(false)
    const neo = normalizeDep({ id: 'neo4j', name: 'Neo4j', kind: 'graph', target: 'neo4j:7687', status: 'unknown', checked_at: null, stale: true })!
    expect(neo.target).toBe('neo4j:7687')
    expect(neo.name).toBe('Neo4j')
    expect(depIsStale(neo, AT, AT)).toBe(true)
  })

  test('vLLM, TLS, disk and recorder detail lines', () => {
    const embed = normalizeDep({ id: 'embed', status: 'ok', detail: { metrics: 'ok', running: 3, waiting: 0, kv_cache_pct: 22.4 } })!
    expect(detailParts(embed)).toEqual([
      ['operations.health.detail.queue', { running: 3, waiting: 0 }],
      ['operations.health.detail.kvCache', { pct: '22%' }]
    ])
    const tls = normalizeDep({ id: 'tls', status: 'warn', detail: { targets: [{ target: 'a:443', days_left: 40 }, { target: 'b:443', days_left: 9, self_signed: true }] } })!
    expect(detailParts(tls)).toEqual([['operations.health.detail.certSelfSigned', { days: 9 }]])
    const disk = normalizeDep({ id: 'disk', status: 'ok', detail: { free_pct: 38.2, paths: [] } })!
    expect(detailParts(disk)).toEqual([['operations.health.detail.diskFree', { pct: '38%' }]])
    const rec = normalizeDep({ id: 'recorder', status: 'ok', detail: { buffer: 0, lost_5m: 0, dropped_5m: 2, last_flush_age_s: 3 } })!
    expect(detailParts(rec)).toEqual([['operations.health.detail.recorder', { buffer: 0, lost: 0, dropped: 2 }]])
  })

  test('probe and SMTP test answers become cards and results', () => {
    expect(normalizeProbeResult('milvus', { status: 'down', error_class: 'TimeoutError' }).status).toBe('down')
    expect(normalizeProbeResult('milvus', { result: { status: 'ok', latency_ms: 9 } }).latency_ms).toBe(9)
    expect(normalizeSmtpTest({ ok: false, error_class: 'SMTPConnectError' })).toEqual({ ok: false, error_class: 'SMTPConnectError', latency_ms: null, detail: null })
    expect(normalizeSmtpTest({ status: 'ok', latency_ms: 120 }).ok).toBe(true)
  })
})

describe('/health/{dep}', () => {
  test('recent newest first, series oldest first, error classes counted', () => {
    const d = normalizeDepDetail('milvus', {
      recent: [
        { ts: '2026-11-03T04:03:50Z', status: 'down', error_class: 'TimeoutError' },
        { ts: '2026-11-03T04:04:20Z', status: 'down', error_class: 'TimeoutError' },
        { ts: '2026-11-03T04:00:02Z', status: 'ok', latency_ms: 14 }
      ],
      latency_series: [
        { t: '2026-11-03T04:00:00Z', latency_ms: 14 },
        { t: '2026-11-03T03:30:00Z', latency_ms: 12 }
      ]
    })
    expect(d.recent[0].ts).toBe('2026-11-03T04:04:20Z')
    expect(d.latency_series[0].t).toBe(Date.parse('2026-11-03T03:30:00Z'))
    expect(d.error_classes).toEqual([{ key: 'TimeoutError', count: 2 }])
    expect(drawerStatus(d)).toBe('down')
  })

  test('Postgres blocking detail keeps durations, statement type and fingerprint, never text', () => {
    const d = normalizeDepDetail('postgres', {
      detail: {
        blocking: [{ pid: 4411, blocking_pids: [4410], wait_s: 42, statement_type: 'UPDATE', query_fingerprint: '9f3a', query_len: 412, query: 'UPDATE secret' }]
      }
    })
    expect(d.blocking).toEqual([
      { pid: 4411, blocked_by: [4410], duration_s: 42, state: null, statement_type: 'UPDATE', fingerprint: '9f3a', query_length: 412, wait_event: null }
    ])
    expect(JSON.stringify(d.blocking)).not.toContain('secret')
  })
})

describe('/workers', () => {
  const raw = {
    generated_at: '2026-11-03T04:04:31Z',
    items: [
      { worker_id: 'old', pid: 1, started_at: '2026-11-01T00:00:00Z', heartbeat_age_s: 3600, superseded: true, snapshot: {} },
      {
        worker_id: 'w2',
        pid: 812,
        mode: 'uvicorn',
        started_at: '2026-11-01T04:00:00Z',
        heartbeat_at: '2026-11-03T04:04:25Z',
        leases_held: ['ops:probes'],
        snapshot: {
          rss_mb: 2048,
          cpu: 12.5,
          loop_lag_ms: 3,
          pool: { size: 12, idle: 4, max_size: 50, acquire_wait_ms: 3 },
          queues: { llm: { running: 14, queued: 40 }, embed: { running: 1, waiting: 0 } },
          recorder: { buffer_len: 0, last_flush_ok_at: '2026-11-03T04:04:28Z', dropped_events: 0 },
          drainers: ['fsktm', { workspace: 'space1', state: 'draining' }]
        }
      },
      { worker_id: 'w3', pid: 811, started_at: '2026-11-02T00:00:00Z', heartbeat_age_s: 540, snapshot: {} }
    ],
    host: { memory: { current: 7.1e9, max: 16e9 }, loadavg: [3.2, 2.9, 2.7, 9], disks: { '/app/data': { used: 62, total: 100 } } }
  }

  test('alive from the heartbeat age (45 s rule); live first, superseded last', () => {
    const w = normalizeWorkers(raw)
    expect(w.items.map((x) => x.worker_id)).toEqual(['w2', 'w3', 'old'])
    expect(w.items[0].heartbeat_age_s).toBe(6)
    expect(w.items[0].alive).toBe(true)
    expect(w.items[1].alive).toBe(false)
  })

  test('snapshot spellings: rss in MB, cpu, pool, queues, recorder, drainers', () => {
    const s = normalizeWorkers(raw).items[0].snapshot
    expect(s.rss).toBe(2048 * 1024 * 1024)
    expect(s.cpu_pct).toBe(12.5)
    expect(s.pool).toEqual({ size: 12, idle: 4, max: 50, wait_ms: 3 })
    expect(s.queues.llm).toEqual({ running: 14, queued: 40, in_flight: null })
    expect(s.queues.embed).toEqual({ running: 1, queued: 0, in_flight: null })
    expect(s.queues.rerank).toBeNull()
    expect(s.recorder?.buffer).toBe(0)
    expect(s.drainers.map((d) => d.workspace)).toEqual(['fsktm', 'space1'])
    expect(poolText(s.pool)).toBe('12/50 (3 ms)')
  })

  test('host numbers: cgroup memory, three load averages, disks', () => {
    const h = normalizeWorkers(raw).host!
    expect(h.mem_used).toBe(7.1e9)
    expect(h.mem_limit).toBe(16e9)
    expect(h.load).toEqual([3.2, 2.9, 2.7])
    expect(h.disks).toEqual([{ path: '/app/data', used: 62, total: 100 }])
  })

  test('row states: alive, no heartbeat, superseded, exited', () => {
    const [alive, stale, old] = normalizeWorkers(raw).items
    expect(workerLook(alive).state).toBe('ok')
    expect(workerLook(stale)).toEqual({ state: 'critical', label: 'operations.worker.state.stale' })
    expect(workerLook(old)).toEqual({ state: 'unknown', label: 'operations.worker.state.superseded' })
    expect(workerLook({ ...alive, exited_at: '2026-11-03T04:00:00Z' }).label).toBe('operations.worker.state.exited')
  })

  test('history times in epoch seconds or ms are both read as ms', () => {
    const at = Date.parse('2026-11-03T04:00:00Z')
    expect(normalizeWorkerHistory({ items: [{ t: at / 1000, rss: 1 }, { t: at + 60_000, rss: 2 }] }).map((s) => s.t)).toEqual([at, at + 60_000])
  })

  test('history samples sorted, both spellings', () => {
    const h = normalizeWorkerHistory({
      samples: [
        { minute: '2026-11-03T04:01:00Z', rss: 2, cpu_pct: 5, loop_lag_ms: 3, pool_wait_ms: 1 },
        { t: '2026-11-03T04:00:00Z', rss: 1, cpu: 4, loop_lag: 2, pool_wait: null },
        { rss: 9 }
      ]
    })
    expect(h.map((s) => s.rss)).toEqual([1, 2])
    expect(h[1]).toEqual({ t: Date.parse('2026-11-03T04:01:00Z'), rss: 2, cpu: 5, loop_lag: 3, pool_wait: 1 })
  })
})

describe('/config', () => {
  const raw = {
    configuration: {
      llm_binding: 'openai',
      llm_binding_host: 'https://api.example.com/v1',
      embedding_model: 'bge-m3',
      kv_storage: 'PGKVStorage',
      summary_language: 'English',
      force_llm_summary_on_merge: 8,
      cosine_threshold: 0.2,
      webhook_x: 'y'
    },
    core_version: '1.5.1'
  }

  test('groups keys for the viewer', () => {
    const groups = groupConfig(normalizeConfig(raw).configuration)
    const byId = Object.fromEntries(groups.map((g) => [g.id, g.entries.map(([k]) => k)]))
    expect(byId.llm).toEqual(['llm_binding', 'llm_binding_host'])
    expect(byId.storage).toEqual(['kv_storage'])
    expect(byId.other).toEqual(['webhook_x'])
  })

  test('the StatusCard shape is built only from a complete configuration', () => {
    const status = toLightragStatus(normalizeConfig(raw), null)
    expect(status?.configuration.kv_storage).toBe('PGKVStorage')
    expect(toLightragStatus(normalizeConfig({ configuration: { llm_binding: 'openai' } }))).toBeNull()
    expect(toLightragStatus(null)).toBeNull()
  })
})
