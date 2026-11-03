import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'
import type { DepHealth, OpsHealth, OpsStatus, OpsWorkers, ProbeStatus } from './api'

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
const { default: TopologyMap } = await import('./components/TopologyMap')
const { default: AvailabilityStrip } = await import('./components/AvailabilityStrip')
const { default: Sparkline } = await import('./components/Sparkline')
const { default: WorkersTable } = await import('./components/WorkersTable')
const { default: WorkersCard } = await import('./components/WorkersCard')
const { default: DepsView } = await import('./health/DepsView')
const { HostView, WorkersView } = await import('./health/WorkersView')
const { default: ConfigView } = await import('./health/ConfigView')
const { summarizeRechecks } = await import('./health/actions')
const { normalizeStrip, normalizeWorkers, normalizeConfig } = await import('./health/model')
const { OpsPrefsContext } = await import('./prefs')
const { OpsShellContext } = await import('./context')
const { normalizeOverview } = await import('./normalize')
const { attentionTarget } = await import('./overview')
const { DEFAULT_OPS_HASH, viewForTarget, writeOpsHash } = await import('./hashState')

const prefs = (patch: { reduceMotion?: boolean; paused?: boolean } = {}) => ({
  paused: false,
  setPaused: () => {},
  reduceMotion: false,
  setReduceMotion: () => {},
  ...patch
})

const withPrefs = (node: React.ReactNode, patch: { reduceMotion?: boolean; paused?: boolean } = {}) =>
  renderToString(<OpsPrefsContext.Provider value={prefs(patch)}>{node}</OpsPrefsContext.Provider>)

/** Every looping or attention-grabbing animation class (§2.12 rules 1–3). */
const ANIMATION_CLASSES = ['ops-breathe', 'ops-flow', 'ops-ring-a', 'ops-ring-b', 'ops-pulse', 'ops-cell-new', 'ops-fade-in']
const animated = (html: string) => ANIMATION_CLASSES.filter((c) => html.includes(c))
/** Inline transforms (style or attribute) would mean layout-affecting motion in the markup. */
const hasTransform = (html: string) => /transform[:=]/i.test(html)

// Fresh data (panels and cards measure staleness against the real clock).
const AT = Date.now() - 5000
const dep = (id: string, status: ProbeStatus = 'ok', extra: Partial<DepHealth> = {}): DepHealth => ({
  id,
  status,
  detail: {},
  strip: normalizeStrip(Array(48).fill(status === 'down' ? 'down' : 'ok'), AT),
  checked_at: new Date(AT).toISOString(),
  latency_ms: status === 'down' ? null : 3,
  ...extra
})

const PROD: DepHealth[] = [
  dep('postgres', 'ok', { detail: { connections: 61, max_connections: 200, blocked: 0 } }),
  dep('neo4j'),
  dep('milvus', 'down', { error_class: 'TimeoutError', last_ok_at: '2026-11-03T04:00:02Z' }),
  dep('vllm_embed'),
  dep('vllm_rerank', 'warn'),
  dep('docling'),
  dep('llm', 'ok', { passive: true }),
  dep('smtp', 'ok', { interval_s: 900 }),
  dep('idp'),
  dep('tls', 'ok', { detail: { days_left: 115 } })
]

describe('TopologyMap (§2.13, static in this release)', () => {
  const render = (patch: { reduceMotion?: boolean; paused?: boolean } = {}) =>
    withPrefs(<TopologyMap deps={PROD} processLabel="uvicorn · 1 process" />, patch)

  test('reduced motion: final state on first paint, no transforms, no animation classes', () => {
    const html = render({ reduceMotion: true })
    expect(hasTransform(html)).toBe(false)
    expect(animated(html)).toEqual([])
    // Non-colour marks: the down edge is dashed critical with ✖, unprobed nodes are dashed.
    expect(html).toContain('is-down')
    expect(html).toContain('✖')
    expect(html).toContain('is-unprobed')
    expect(html).toContain('operations.topology.notProbed')
  })

  test('normal motion renders the same markup: a repeated render adds no animation', () => {
    const a = render()
    const b = render()
    expect(a).toBe(b)
    expect(animated(a)).toEqual([])
    expect(hasTransform(a)).toBe(false)
    // Static until the rate series exist (PR-6): no edge flows.
    expect(a).not.toContain('ops-flow')
  })

  test('"⏸ auto" adds no loop either', () => {
    expect(animated(render({ paused: true }))).toEqual([])
  })

  test('nodes are focusable buttons whose names carry their state in text', () => {
    const html = render()
    expect(html).toContain('role="button"')
    expect(html).toContain('tabindex="0"')
    // Untranslated in tests, a dependency's name falls back to its node's generic label.
    expect(html).toContain('aria-label="operations.topology.node.vector.label, operations.state.critical"')
    expect(html).toContain('aria-label="operations.topology.node.etcd.label, operations.topology.notProbed"')
  })

  test('the info panel opens on the worst node and links to its drawer', () => {
    const html = render()
    expect(html).toContain('aria-pressed="true"')
    expect(html).toContain('data-node="vector"')
    expect(html).toContain('href="#/?ops=1&amp;otab=overview&amp;odep=milvus"')
  })

  test('both layers are in the markup; the Monitoring layer is marked for hiding', () => {
    const html = render()
    expect(html).toContain('operations.topology.traffic')
    expect(html).toContain('operations.topology.monitoring')
    expect(html).toContain('ops-te-probe is-mon')
  })

  test('without dependencies it still draws the server, and no dependency links', () => {
    const html = withPrefs(<TopologyMap deps={[]} />)
    expect(html).toContain('operations.topology.node.http.label')
    expect(html).not.toContain('odep=')
    expect(html).not.toMatch(/<polyline[^>]*is-down/)
  })
})

describe('AvailabilityStrip (§2.6)', () => {
  const cells = normalizeStrip([...Array(44).fill('ok'), { total: 60, failed: 2 }, 'down', 'warn', null], AT)

  test('48 cells, each with a text label; non-colour marks by state', () => {
    const html = withPrefs(<AvailabilityStrip cells={cells} />)
    expect((html.match(/role="img"/g) ?? []).length).toBe(48)
    expect(html).toContain('data-s="down"')
    expect(html).toContain('data-s="warn"')
    expect(html).toContain('data-s="unknown"')
    expect(html).toContain('operations.strip.cell.failed')
    expect(html).toContain('operations.strip.cell.unknown')
  })

  test('the first paint is final: no fade, also under reduced motion', () => {
    expect(withPrefs(<AvailabilityStrip cells={cells} />)).not.toContain('ops-cell-new')
    const reduced = withPrefs(<AvailabilityStrip cells={cells} />, { reduceMotion: true })
    expect(animated(reduced)).toEqual([])
    expect(hasTransform(reduced)).toBe(false)
  })

  test('renders nothing without cells', () => {
    expect(withPrefs(<AvailabilityStrip cells={[]} />)).toBe('')
  })
})

describe('Sparkline', () => {
  test('a static line with gaps split, aria-hidden', () => {
    const html = renderToString(<Sparkline values={[1, 2, null, 4, 5]} />)
    expect(html).toContain('aria-hidden="true"')
    expect((html.match(/<polyline/g) ?? []).length).toBe(2)
    expect(hasTransform(html)).toBe(false)
  })

  test('no data draws only the baseline', () => {
    expect(renderToString(<Sparkline values={[null, null]} />)).not.toContain('<polyline')
  })
})

const status: OpsStatus = {
  available: true,
  features: { durable: true, runs: false, logs: false, probes: true, alerts: false, insights: true, release_busy: false, gpu: false },
  process: { mode: 'uvicorn', expected_workers: 1, workers_live: 1 },
  operator: true,
  version: '1.5.1',
  generated_at: '2026-11-03T04:04:31Z'
}

const pollOf = <T,>(data: T | null, patch: Partial<{ error: any; updatedAt: number | null }> = {}) => ({
  data,
  error: null,
  loading: false,
  slow: false,
  updatedAt: AT,
  previousAt: null,
  baseline: true,
  reload: () => {},
  ...patch
})

const withShell = (node: React.ReactNode, patch: { reduceMotion?: boolean; paused?: boolean } = {}) =>
  withPrefs(<OpsShellContext.Provider value={{ status, openInsights: () => {} }}>{node}</OpsShellContext.Provider>, patch)

describe('DepsView', () => {
  const health: OpsHealth = { cycle: { at: new Date(AT).toISOString(), duration_ms: 1400, holder: 'w1', holder_pid: 812, interval_s: 30 }, deps: PROD }

  test('one card per dependency, each a link to its drawer, with state in text and the strip', () => {
    const html = withShell(<DepsView poll={pollOf(health)} processLabel="uvicorn · 1 process" probesOff={false} />)
    expect((html.match(/class="ops-dep /g) ?? []).length).toBe(PROD.length)
    expect(html).toContain('odep=postgres')
    expect(html).toContain('operations.health.detail.conns')
    expect(html).toContain('TimeoutError')
    expect(html).toContain('operations.health.probedBy')
    expect(html).toContain('operations.health.recheckAll')
    expect(html).toContain('operations.health.smtpTest')
    expect(html).toContain('operations.health.passive')
  })

  test('a server with probes off says so instead of showing green', () => {
    const html = withShell(<DepsView poll={pollOf({ cycle: null, deps: [] })} processLabel={null} probesOff />)
    expect(html).toContain('operations.health.probesOff')
    expect(html).toContain('operations.health.probesOffEmpty')
    expect(html).not.toContain('ops-tone-ok')
  })

  test('reduced motion: no animation classes and no inline transforms anywhere on the view', () => {
    const html = withShell(<DepsView poll={pollOf(health)} processLabel="uvicorn" probesOff={false} />, { reduceMotion: true })
    expect(animated(html)).toEqual([])
    expect(hasTransform(html)).toBe(false)
  })
})

const workersRaw = {
  generated_at: '2026-11-03T04:04:31Z',
  items: [
    { worker_id: 'w-812', pid: 812, mode: 'uvicorn', started_at: '2026-11-01T04:00:00Z', heartbeat_age_s: 6, snapshot: { rss: 2.1e9, cpu_pct: 12, loop_lag_ms: 3, pool: { size: 12, max: 50, wait_ms: 3 } } },
    { worker_id: 'w-old', pid: 1, started_at: '2026-10-30T00:00:00Z', heartbeat_age_s: 7200, superseded: true, snapshot: {} }
  ],
  host: { mem_used: 7.1e9, mem_limit: 16e9, load: [3.2, 2.9, 2.7], disks: [{ path: '/app/data', used: 62e9, total: 100e9 }] }
}

describe('Workers', () => {
  const workers: OpsWorkers = normalizeWorkers(workersRaw)

  test('single process: one "server" row, no Manager column, rows open the Worker drawer', () => {
    const html = withShell(<WorkersTable items={workers.items} now={AT} multiWorker={false} />)
    expect(html).toContain('operations.worker.server')
    expect(html).toContain('operations.worker.state.superseded')
    expect(html).not.toContain('operations.worker.col.manager')
    expect(html).toContain('owk=w-812')
    expect(html).toContain('12/50 (3 ms)')
  })

  test('multi-worker shows the Manager round trip', () => {
    expect(withShell(<WorkersTable items={workers.items} now={AT} multiWorker />)).toContain('operations.worker.col.manager')
  })

  test('a worker with no heartbeat is Critical, with ✖ and text', () => {
    const dead = normalizeWorkers({ items: [{ worker_id: 'w-811', pid: 811, heartbeat_age_s: 540, snapshot: {} }] }).items
    const html = withShell(<WorkersTable items={dead} now={AT} multiWorker />)
    expect(html).toContain('operations.worker.state.stale')
    expect(html).toContain('✖')
  })

  test('the view counts live of expected; host numbers are labelled', () => {
    const view = withShell(<WorkersView poll={pollOf(workers)} multiWorker={false} expected={1} />)
    expect(view).toContain('operations.worker.liveSingle')
    expect(view).toContain('ops-tone-ok')
    const host = withShell(<HostView poll={pollOf(workers)} />)
    expect(host).toContain('operations.host.memOf')
    expect(host).toContain('3.20 / 2.90 / 2.70')
    expect(host).toContain('operations.host.gpuNote')
    // FR-33: the Postgres numbers come from the postgres card's extras.
    const pg = {
      id: 'postgres',
      status: 'ok',
      detail: {
        connections: 150,
        max_connections: 200,
        connections_pct: 75,
        db_size_bytes: 44_000_000_000,
        blocked: 1,
        blocked_max_s: 95,
        db_delta: { window_s: 300, commits: 600, rollbacks: 3, deadlocks: 0, temp_bytes: 4096, cache_hit_pct: 99 },
        extras_at: '2026-11-03T04:00:00Z'
      }
    } as unknown as DepHealth
    const withPg = withShell(<HostView poll={pollOf(workers)} deps={[pg]} />)
    expect(withPg).toContain('operations.host.pg.connOf')
    expect(withPg).toContain('operations.host.pg.blockedFor')
    expect(withPg).toContain('operations.host.pg.delta')
    expect(withShell(<HostView poll={pollOf(workers)} deps={[]} />)).toContain('operations.host.pg.none')
  })

  test('the Overview card lists workers and links to Health › Workers', () => {
    const html = withShell(<WorkersCard workers={workers.items} now={AT} multiWorker={false} />)
    expect(html).toContain('operations.worker.line')
    expect(html).toContain('osub=workers')
    expect(withShell(<WorkersCard workers={[]} now={AT} multiWorker={false} />)).toContain('operations.system.workersPending')
  })
})

describe('ConfigView', () => {
  test('grouped, read-only, nested maps one row per entry', () => {
    const config = normalizeConfig({
      configuration: {
        llm_binding: 'openai',
        llm_binding_host: 'https://api.example.com/v1',
        role_llm_config: { extract: { model: 'gpt-x', host: 'https://api.example.com/v1' } },
        kv_storage: 'PGKVStorage'
      }
    })
    const html = withShell(<ConfigView poll={pollOf(config)} />)
    expect(html).toContain('operations.config.group.llm')
    expect(html).toContain('role_llm_config.extract')
    expect(html).toContain('model=gpt-x')
    expect(html).toContain('PGKVStorage')
  })
})

describe('overview read model and drill targets', () => {
  test('deps and workers are normalised; the Deps tile is counted when the server sends none', () => {
    const o = normalizeOverview({
      generated_at: '2026-11-03T04:04:31Z',
      state: 'ok',
      kpis: {},
      attention: [],
      deps: [{ id: 'postgres', status: 'ok' }, { id: 'milvus', status: 'down' }, { id: 'etcd', status: 'unprobed' }],
      workers: [{ worker_id: 'w1', pid: 812, heartbeat_at: '2026-11-03T04:04:20Z', snapshot: {} }]
    })
    expect(o.deps?.map((d) => d.status)).toEqual(['ok', 'down', 'unprobed'])
    expect(o.kpis.deps).toEqual({ ok: 1, total: 2 })
    expect(o.workers?.[0].alive).toBe(true)
    expect(o.workers?.[0].heartbeat_age_s).toBe(11)
  })

  test('dependency and worker targets open their drawers', () => {
    expect(writeOpsHash(viewForTarget(DEFAULT_OPS_HASH, { tab: 'health', sub: 'deps', dep: 'milvus' }))).toBe('#/?ops=1&otab=health&osub=deps&odep=milvus')
    expect(writeOpsHash(viewForTarget(DEFAULT_OPS_HASH, { tab: 'health', worker_id: 'w-812' }))).toBe('#/?ops=1&otab=health&owk=w-812')
    expect(attentionTarget({ id: 'a', rule: 'dependency_down', severity: 'critical', dep: 'milvus' })).toEqual({ tab: 'health', sub: 'deps', dep: 'milvus' })
  })

  test('a server link to a dependency drawer becomes a target', () => {
    const o = normalizeOverview({ state: 'warning', kpis: {}, attention: [{ id: 'x', rule: 'dependency_down', severity: 'critical', link: '#/?ops=1&otab=health&odep=milvus' }] })
    expect(o.attention[0].target?.dep).toBe('milvus')
  })
})

describe('Re-check all summary', () => {
  const err = (status: number) => Object.assign(new Error(String(status)), { status })

  test('rate-limited answers are told apart from failures', () => {
    const s = summarizeRechecks([
      { status: 'fulfilled', value: 1 },
      { status: 'rejected', reason: err(429) },
      { status: 'rejected', reason: err(500) }
    ])
    expect(s).toEqual({ done: 1, total: 3, refused: 1, failed: 1 })
  })
})
