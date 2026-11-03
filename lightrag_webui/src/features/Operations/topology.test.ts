import { describe, expect, test } from 'bun:test'
import type { DepHealth, ProbeStatus } from './api'
import { AttentionSlot } from './motion'
import { buildTopology, defaultSelection, nodeOfDep, nodeRingCandidates, polylineLength, polylineMid } from './topology'

const dep = (id: string, status: ProbeStatus = 'ok', extra: Partial<DepHealth> = {}): DepHealth => ({ id, status, detail: {}, strip: [], ...extra })

const PROD = [
  dep('postgres'),
  dep('neo4j'),
  dep('milvus'),
  dep('vllm_embed'),
  dep('vllm_rerank'),
  dep('docling'),
  dep('llm'),
  dep('smtp'),
  dep('idp'),
  dep('tls')
]

describe('polyline geometry', () => {
  test('length and midpoint by length along orthogonal segments', () => {
    expect(polylineLength([[0, 0], [10, 0], [10, 10]])).toBe(20)
    expect(polylineMid([[0, 0], [10, 0], [10, 10]])).toEqual([10, 0])
    expect(polylineMid([[0, 0], [30, 0], [30, 10]])).toEqual([20, 0])
    expect(polylineMid([[5, 5]])).toEqual([5, 5])
  })
})

describe('buildTopology', () => {
  test('dependency ids, store names and kinds map onto nodes', () => {
    expect(nodeOfDep({ id: 'neo4j' })).toBe('graph')
    expect(nodeOfDep({ id: 'qdrant' })).toBe('vector')
    expect(nodeOfDep({ id: 'custom', kind: 'vector' })).toBe('vector')
    expect(nodeOfDep({ id: 'disk' })).toBeNull()
  })

  test('the prod layout: every reported dependency, etcd and MinIO behind Milvus as not probed', () => {
    const topo = buildTopology(PROD)
    const ids: string[] = topo.nodes.map((n) => n.id)
    for (const id of ['postgres', 'graph', 'vector', 'etcd', 'minio', 'vembed', 'vrerank', 'docling', 'llm', 'smtp', 'idp', 'proxy', 'http', 'collectors']) {
      expect(ids).toContain(id)
    }
    expect(ids).not.toContain('redis')
    expect(topo.byId.etcd?.unprobed).toBe(true)
    expect(topo.byId.etcd?.state).toBe('unknown')
    expect(topo.edges.find((e) => e.id === 'vector-etcd')?.cls).toBe('unprobed')
  })

  test('nothing is claimed that the server does not report', () => {
    const topo = buildTopology([dep('postgres'), dep('qdrant')])
    const ids: string[] = topo.nodes.map((n) => n.id)
    expect(ids).toContain('vector')
    expect(ids).not.toContain('etcd')
    expect(ids).not.toContain('smtp')
    expect(ids).not.toContain('proxy')
    // Without a proxy, callers reach the HTTP stack directly.
    expect(topo.edges.some((e) => e.id === 'browser-http')).toBe(true)
    expect(topo.edges.some((e) => e.id.includes('proxy'))).toBe(false)
  })

  test('a down target makes its data edges dashed critical with ✖ at the midpoint; probe edges stay as they are', () => {
    const topo = buildTopology(PROD.map((d) => (d.id === 'milvus' ? { ...d, status: 'down' } : d)))
    const edge = topo.edges.find((e) => e.id === 'api-vector')!
    expect(edge.down).toBe(true)
    expect(edge.mid).toEqual([770, 308])
    expect(topo.edges.find((e) => e.id === 'probe-vector')?.down).toBe(false)
    expect(topo.edges.filter((e) => e.down).map((e) => e.id)).toEqual(['api-vector'])
    expect(topo.byId.vector?.state).toBe('critical')
  })

  test('two ids on one node: the worse state wins', () => {
    expect(buildTopology([dep('graph', 'ok'), dep('neo4j', 'warn')]).byId.graph?.state).toBe('warning')
  })

  test('the info panel opens on the worst node, else Postgres', () => {
    expect(defaultSelection(buildTopology(PROD))).toBe('postgres')
    expect(defaultSelection(buildTopology(PROD.map((d) => (d.id === 'docling' ? { ...d, status: 'down' } : d))))).toBe('docling')
    expect(defaultSelection(buildTopology([]))).toBe('http')
  })
})

describe('node rings (§2.12 rule 2)', () => {
  const down = PROD.map((d) => (d.id === 'milvus' ? { ...d, status: 'down' as const } : d))

  test('only a node that has just entered Critical asks for a ring', () => {
    expect(nodeRingCandidates(PROD, down)).toEqual([{ key: 'node:vector', rank: 3, source: 'node' }])
    expect(nodeRingCandidates(down, down)).toEqual([])
    expect(nodeRingCandidates(down, PROD)).toEqual([])
    expect(nodeRingCandidates(null, down)).toEqual([])
  })

  test('the state chip wins the slot over a node in the same poll; the cooldown blocks a flapping node', () => {
    const slot = new AttentionSlot()
    const node = nodeRingCandidates(PROD, down)
    expect(slot.offer([...node, { key: 'chip', rank: 3, source: 'chip' }], 0)?.source).toBe('chip')
    expect(slot.offer(node, 5_000)?.key).toBe('node:vector')
    expect(slot.offer(node, 60_000)).toBeNull()
  })
})
