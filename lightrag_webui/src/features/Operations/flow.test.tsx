import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'
import type { DepHealth, ProbeStatus } from './api'
import type { FlowRates } from './performance/model'

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
const { OpsPrefsContext } = await import('./prefs')
const { FLOW_EDGES, edgeFlow, flowAllowed, flowClasses, flowStillReason } = await import('./flow')
const { buildTopology } = await import('./topology')

const prefs = (patch: { reduceMotion?: boolean; paused?: boolean } = {}) => ({
  paused: false,
  setPaused: () => {},
  reduceMotion: false,
  setReduceMotion: () => {},
  ...patch
})

const dep = (id: string, status: ProbeStatus = 'ok'): DepHealth => ({ id, status, detail: {}, strip: [] })
const PROD = ['postgres', 'neo4j', 'milvus', 'vllm_embed', 'vllm_rerank', 'docling', 'llm', 'smtp', 'idp', 'tls'].map((id) => dep(id))

/** HTTP busy, embedding usual, rerank below usual, LLM idle. */
const RATES: FlowRates = {
  http: { perMin: 75, p50: 20, p90: 60 },
  embed: { perMin: 12, p50: 10, p90: 40 },
  rerank: { perMin: 2, p50: 5, p90: 20 },
  llm: { perMin: 0, p50: 100, p90: 300 }
}

const render = (opts: { reduceMotion?: boolean; paused?: boolean; stale?: boolean; deps?: DepHealth[]; flows?: FlowRates | null } = {}) =>
  renderToString(
    <OpsPrefsContext.Provider value={prefs(opts)}>
      <TopologyMap deps={opts.deps ?? PROD} processLabel="uvicorn · 1 process" flows={opts.flows === undefined ? RATES : opts.flows} stale={opts.stale} />
    </OpsPrefsContext.Provider>
  )

const LOOP = /ops-flow(?!-)|ops-te-flow|ops-flow-(low|medium|high)/
const polyline = (html: string, cls: string) => (html.match(new RegExp(`<polyline[^>]*class="[^"]*${cls}[^"]*"[^>]*>`, 'g')) ?? [])
/** Inline transforms (style or attribute) would mean layout-affecting motion. */
const hasTransform = (html: string) => /transform[:=]/i.test(html)

describe('flow gate (§2.12 rule 3)', () => {
  const ok = { reduced: false, paused: false, hidden: false, stale: false, inView: true }

  test('flows only when nothing stops it', () => {
    expect(flowAllowed(ok)).toBe(true)
    for (const k of ['reduced', 'paused', 'hidden', 'stale'] as const) expect(flowAllowed({ ...ok, [k]: true })).toBe(false)
    expect(flowAllowed({ ...ok, inView: false })).toBe(false)
  })

  test('the still reason names the first control that stopped it', () => {
    expect(flowStillReason(ok)).toBeNull()
    expect(flowStillReason({ ...ok, reduced: true, paused: true })).toBe('reduced')
    expect(flowStillReason({ ...ok, paused: true, stale: true })).toBe('paused')
    expect(flowStillReason({ ...ok, stale: true })).toBe('stale')
    expect(flowStillReason({ ...ok, hidden: true })).toBe('hidden')
    expect(flowStillReason({ ...ok, inView: false })).toBe('offscreen')
  })

  test('at most five edges ever flow, and only data edges with a rate series', () => {
    const topo = buildTopology(PROD)
    const flowing = topo.edges.filter((e) => edgeFlow(e.id, e.down, RATES, true) !== null)
    expect(flowing.map((e) => e.id).sort()).toEqual(['api-llm', 'api-vembed', 'api-vrerank', 'browser-proxy', 'proxy-http'])
    expect(new Set(Object.values(FLOW_EDGES))).toEqual(new Set(['http', 'embed', 'rerank', 'llm']))
  })

  test('buckets per edge from its own p50/p90; a down target never flows', () => {
    expect(edgeFlow('proxy-http', false, RATES, true)).toEqual({ source: 'http', perMin: 75, bucket: 'high', moving: true })
    expect(edgeFlow('api-vembed', false, RATES, true)?.bucket).toBe('medium')
    expect(edgeFlow('api-vrerank', false, RATES, true)?.bucket).toBe('low')
    expect(edgeFlow('api-llm', false, RATES, true)).toEqual({ source: 'llm', perMin: 0, bucket: 'idle', moving: false })
    expect(edgeFlow('api-vembed', true, RATES, true)).toEqual({ source: 'embed', perMin: 12, bucket: 'idle', moving: false })
    expect(edgeFlow('api-postgres', false, RATES, true)).toBeNull()
    expect(edgeFlow('proxy-http', false, null, true)).toBeNull()
  })

  test('classes: thickness always, the dash loop only while moving', () => {
    const still = edgeFlow('proxy-http', false, RATES, false)
    expect(flowClasses(still)).toEqual(['ops-te-rate-high'])
    expect(flowClasses(edgeFlow('proxy-http', false, RATES, true))).toEqual(['ops-te-rate-high', 'ops-te-flow', 'ops-flow', 'ops-loop', 'ops-flow-high'])
    expect(flowClasses(null)).toEqual([])
  })
})

describe('TopologyMap edge flow (§2.12, §2.13)', () => {
  test('normal motion: busy edges flow at their bucket speed; idle edges stay still', () => {
    const html = render()
    expect(polyline(html, 'ops-flow-high').length).toBe(2) // browser → proxy → API share the HTTP series
    expect(polyline(html, 'ops-flow-medium').length).toBe(1)
    expect(polyline(html, 'ops-flow-low').length).toBe(1)
    expect(polyline(html, 'ops-te-rate-idle')[0]).not.toMatch(LOOP)
    expect(html).toContain('operations.topology.flow.moving')
    // Only stroke-dashoffset moves (CSS); nothing in the markup transforms.
    expect(hasTransform(html)).toBe(false)
  })

  test('reduced motion: static solid lines; the rate label and thickness carry the same information', () => {
    const html = render({ reduceMotion: true })
    expect(html).not.toMatch(LOOP)
    expect(hasTransform(html)).toBe(false)
    expect(polyline(html, 'ops-te-rate-high').length).toBe(2)
    expect(html).toContain('75/min')
    expect(html).toContain('12/min')
    expect(html).toContain('operations.topology.flow.still.reduced')
    expect(html).toContain('data-flow-source="http"')
  })

  test('"⏸ auto" stops every loop', () => {
    const html = render({ paused: true })
    expect(html).not.toMatch(LOOP)
    expect(html).toContain('operations.topology.flow.still.paused')
  })

  test('a stale panel stops every loop', () => {
    const html = render({ stale: true })
    expect(html).not.toMatch(LOOP)
    expect(html).toContain('operations.topology.flow.still.stale')
  })

  test('a down target: its edge is the dashed critical line with ✖, never flowing', () => {
    const html = render({ deps: PROD.map((d) => (d.id === 'vllm_embed' ? dep(d.id, 'down') : d)) })
    const embed = polyline(html, 'is-down')
    expect(embed.length).toBe(1)
    expect(embed[0]).not.toMatch(LOOP)
    // The edge's rate label is left out (the ✖ marks it); the list under the map still names the rate.
    expect(html).not.toMatch(/<text class="ops-te-lab ops-te-rate"[^>]*>12\/min/)
    expect(html).toMatch(/<text class="ops-te-lab ops-te-rate"[^>]*>75\/min/)
  })

  test('without rates every edge is still and the header says so', () => {
    const html = render({ flows: null })
    expect(html).not.toMatch(LOOP)
    expect(html).not.toContain('ops-te-rate-')
    expect(html).toContain('operations.topology.staticNote')
    expect(html).not.toContain('operations.topology.flow.title')
  })

  test('rule 1: the same rates render the same markup', () => {
    expect(render()).toBe(render())
  })
})
