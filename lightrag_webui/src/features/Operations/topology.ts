import type { DepHealth, ProbeStatus } from './api'
import { depState } from './health/model'
import { RingCandidate, isWorsening } from './motion'
import { OpsState, rankOf } from './severity'

/**
 * The system topology (spec §2.13) as data: a fixed layout for the
 * LightRAG container and its clients, plus one node per dependency the probe
 * cycle reports. Nodes for dependencies the server does not report are left
 * out, so the map never claims a service the deployment does not have.
 * Edges are orthogonal polylines, so their midpoint (where a down target gets
 * its ✖) is computed without measuring the DOM. Static in PR-3: no edge flows
 * until the rate series exist (PR-6).
 */

export const TOPO_WIDTH = 1100
export const TOPO_HEIGHT = 640

export type NodeId =
  | 'idp'
  | 'llm'
  | 'smtp'
  | 'browser'
  | 'connector'
  | 'clients'
  | 'proxy'
  | 'http'
  | 'opsapi'
  | 'pipeline'
  | 'drainer'
  | 'insights'
  | 'collectors'
  | 'queue'
  | 'postgres'
  | 'graph'
  | 'vector'
  | 'etcd'
  | 'minio'
  | 'vembed'
  | 'vrerank'
  | 'docling'
  | 'redis'

/** ext: outside the host; client: callers; int: inside the LightRAG process; svc: a service on the host. */
export type NodeKind = 'ext' | 'client' | 'int' | 'svc'

export type Box = { x: number; y: number; w: number; h: number }

type NodeDef = Box & { kind: NodeKind; tall?: boolean }

export const NODES: Record<NodeId, NodeDef> = {
  idp: { x: 220, y: 10, w: 150, h: 48, kind: 'ext' },
  llm: { x: 800, y: 10, w: 160, h: 48, kind: 'ext' },
  smtp: { x: 990, y: 10, w: 100, h: 48, kind: 'ext' },
  browser: { x: 20, y: 110, w: 150, h: 56, kind: 'client' },
  connector: { x: 20, y: 332, w: 150, h: 56, kind: 'client' },
  clients: { x: 20, y: 540, w: 150, h: 56, kind: 'client' },
  proxy: { x: 220, y: 316, w: 140, h: 84, kind: 'svc' },
  http: { x: 430, y: 128, w: 290, h: 46, kind: 'int' },
  opsapi: { x: 430, y: 184, w: 290, h: 40, kind: 'int' },
  pipeline: { x: 430, y: 234, w: 290, h: 46, kind: 'int' },
  drainer: { x: 430, y: 290, w: 290, h: 46, kind: 'int' },
  insights: { x: 430, y: 346, w: 290, h: 46, kind: 'int' },
  collectors: { x: 430, y: 402, w: 290, h: 62, kind: 'int', tall: true },
  queue: { x: 430, y: 476, w: 290, h: 44, kind: 'int' },
  postgres: { x: 800, y: 110, w: 160, h: 60, kind: 'svc' },
  graph: { x: 800, y: 196, w: 160, h: 48, kind: 'svc' },
  vector: { x: 800, y: 282, w: 160, h: 52, kind: 'svc' },
  etcd: { x: 990, y: 272, w: 100, h: 32, kind: 'svc' },
  minio: { x: 990, y: 314, w: 100, h: 32, kind: 'svc' },
  vembed: { x: 800, y: 372, w: 160, h: 50, kind: 'svc' },
  vrerank: { x: 800, y: 436, w: 160, h: 50, kind: 'svc' },
  docling: { x: 800, y: 500, w: 160, h: 48, kind: 'svc' },
  redis: { x: 800, y: 574, w: 160, h: 44, kind: 'svc' }
}

/** The LightRAG process box and the dashed box around the model and vector services. */
export const CONTAINER: Box = { x: 410, y: 96, w: 330, h: 438 }
export const SERVICES: Box = { x: 786, y: 270, w: 190, h: 292 }

/** Internal nodes are always drawn; client nodes too (they are the callers, not probed). */
const ALWAYS: NodeId[] = ['browser', 'connector', 'clients', 'http', 'opsapi', 'pipeline', 'drainer', 'insights', 'collectors', 'queue']

/** Which node a dependency id from the probe cycle belongs to. */
const DEP_NODE: Record<string, NodeId> = {
  postgres: 'postgres',
  pg: 'postgres',
  graph: 'graph',
  neo4j: 'graph',
  memgraph: 'graph',
  mongo_graph: 'graph',
  vector: 'vector',
  milvus: 'vector',
  qdrant: 'vector',
  faiss: 'vector',
  vllm_embed: 'vembed',
  embed: 'vembed',
  embedding: 'vembed',
  vllm_rerank: 'vrerank',
  rerank: 'vrerank',
  docling: 'docling',
  llm: 'llm',
  llm_provider: 'llm',
  smtp: 'smtp',
  idp: 'idp',
  keycloak: 'idp',
  oidc: 'idp',
  sso: 'idp',
  tls: 'proxy',
  tls_proxy: 'proxy',
  proxy: 'proxy',
  redis: 'redis',
  etcd: 'etcd',
  minio: 'minio',
  recorder: 'insights',
  insights: 'insights',
  log_sink: 'collectors'
}

export const nodeOfDep = (d: Pick<DepHealth, 'id' | 'kind'>): NodeId | null =>
  DEP_NODE[d.id] ?? (d.kind ? (DEP_NODE[d.kind] ?? null) : null)

/** Milvus keeps its metadata in etcd and its segments in MinIO; neither is probed (§2.13). */
const isMilvus = (d: DepHealth | undefined) =>
  !!d && (d.id === 'milvus' || (d.kind ?? '').toLowerCase() === 'milvus' || /milvus/i.test(d.name ?? '') || /milvus/i.test(String(d.detail?.backend ?? d.detail?.storage ?? '')))

export type Pt = [number, number]

export type EdgeClass = 'data' | 'unprobed' | 'bypass' | 'monitor' | 'probe'

type EdgeDef = {
  id: string
  points: Pt[]
  cls: EdgeClass
  /** The node whose state colours the edge: a down target turns it into a dashed critical line with ✖. */
  to?: NodeId
  /** Drawn only when every one of these nodes is on the map. */
  needs: NodeId[]
  /** i18n key under operations.topology.edge.* and its position. */
  label?: { key: string; at: Pt; anchor?: 'start' | 'middle' | 'end' }
  /** Monitoring-layer edges are hidden on the Traffic layer, and the reverse for nothing: data edges dim. */
  monitor?: boolean
}

const viaProxy = (from: Pt, bend: number, y: number): Pt[] => [from, [bend, from[1]], [bend, y], [220, y]]

/** Callers reach the HTTP stack through the proxy when one is probed, else directly. */
function clientEdges(proxy: boolean): EdgeDef[] {
  if (proxy) {
    return [
      { id: 'browser-proxy', points: viaProxy([170, 138], 195, 340), cls: 'data', needs: ['browser', 'proxy'] },
      { id: 'connector-proxy', points: [[170, 360], [220, 360]], cls: 'data', needs: ['connector', 'proxy'] },
      { id: 'clients-proxy', points: viaProxy([170, 568], 195, 380), cls: 'data', needs: ['clients', 'proxy'] },
      { id: 'proxy-http', points: [[360, 350], [395, 350], [395, 151], [430, 151]], cls: 'data', needs: ['proxy', 'http'] }
    ]
  }
  return [
    { id: 'browser-http', points: [[170, 138], [395, 138], [395, 151], [430, 151]], cls: 'data', needs: ['browser', 'http'] },
    { id: 'connector-http', points: [[170, 360], [395, 360], [395, 151], [430, 151]], cls: 'data', needs: ['connector', 'http'] },
    { id: 'clients-http', points: [[170, 568], [395, 568], [395, 151], [430, 151]], cls: 'data', needs: ['clients', 'http'] }
  ]
}

const PROBE_FROM: Pt[] = [[720, 456], [778, 456]]

const EDGES: EdgeDef[] = [
  { id: 'browser-idp', points: [[95, 110], [95, 34], [220, 34]], cls: 'data', to: 'idp', needs: ['browser', 'idp'], label: { key: 'sso', at: [100, 26] } },
  { id: 'api-idp', points: [[450, 96], [450, 40], [370, 40]], cls: 'data', to: 'idp', needs: ['idp'] },
  { id: 'api-llm', points: [[640, 96], [640, 34], [800, 34]], cls: 'data', to: 'llm', needs: ['llm'], label: { key: 'llm', at: [648, 26] } },
  { id: 'api-smtp', points: [[700, 96], [700, 74], [1040, 74], [1040, 58]], cls: 'data', to: 'smtp', needs: ['smtp'], label: { key: 'alertEmail', at: [880, 68] } },
  { id: 'api-postgres', points: [[740, 140], [800, 140]], cls: 'data', to: 'postgres', needs: ['postgres'] },
  { id: 'api-graph', points: [[740, 220], [800, 220]], cls: 'data', to: 'graph', needs: ['graph'] },
  { id: 'api-vector', points: [[740, 308], [800, 308]], cls: 'data', to: 'vector', needs: ['vector'] },
  { id: 'api-vembed', points: [[740, 397], [800, 397]], cls: 'data', to: 'vembed', needs: ['vembed'] },
  { id: 'api-vrerank', points: [[740, 461], [800, 461]], cls: 'data', to: 'vrerank', needs: ['vrerank'] },
  { id: 'api-docling', points: [[740, 524], [800, 524]], cls: 'data', to: 'docling', needs: ['docling'] },
  { id: 'api-redis', points: [[700, 534], [700, 596], [800, 596]], cls: 'data', to: 'redis', needs: ['redis'] },
  { id: 'vector-etcd', points: [[960, 300], [990, 288]], cls: 'unprobed', needs: ['vector', 'etcd'] },
  { id: 'vector-minio', points: [[960, 316], [990, 330]], cls: 'unprobed', needs: ['vector', 'minio'] },
  { id: 'queue-in', points: [[720, 151], [732, 151], [732, 498], [720, 498]], cls: 'data', needs: ['http', 'queue'] },
  { id: 'queue-out', points: [[430, 498], [418, 498], [418, 313], [430, 313]], cls: 'data', needs: ['queue', 'drainer'] },
  {
    id: 'bypass',
    points: [[290, 400], [290, 628], [772, 628], [772, 476], [800, 476]],
    cls: 'bypass',
    needs: ['proxy', 'vrerank'],
    label: { key: 'bypass', at: [430, 622], anchor: 'start' }
  },
  { id: 'bypass-embed', points: [[772, 476], [772, 410], [800, 410]], cls: 'bypass', needs: ['proxy', 'vembed', 'vrerank'] },
  // Monitoring layer: what the collectors write, and what they probe.
  {
    id: 'collectors-postgres',
    points: [[720, 430], [760, 430], [760, 160], [800, 160]],
    cls: 'monitor',
    needs: ['collectors', 'postgres'],
    monitor: true,
    label: { key: 'writes', at: [764, 356], anchor: 'start' }
  },
  { id: 'opsapi-postgres', points: [[720, 204], [750, 204], [750, 124], [800, 124]], cls: 'monitor', needs: ['opsapi', 'postgres'], monitor: true, label: { key: 'reads', at: [764, 118], anchor: 'start' } },
  { id: 'probe-graph', points: [...PROBE_FROM, [778, 220], [800, 220]], cls: 'probe', needs: ['graph'], monitor: true },
  { id: 'probe-vector', points: [...PROBE_FROM, [778, 308], [800, 308]], cls: 'probe', needs: ['vector'], monitor: true },
  { id: 'probe-vembed', points: [...PROBE_FROM, [778, 397], [800, 397]], cls: 'probe', needs: ['vembed'], monitor: true },
  { id: 'probe-vrerank', points: [...PROBE_FROM, [778, 461], [800, 461]], cls: 'probe', needs: ['vrerank'], monitor: true },
  { id: 'probe-docling', points: [...PROBE_FROM, [778, 524], [800, 524]], cls: 'probe', needs: ['docling'], monitor: true },
  { id: 'probe-redis', points: [...PROBE_FROM, [778, 586], [800, 586]], cls: 'probe', needs: ['redis'], monitor: true },
  { id: 'probe-smtp', points: [...PROBE_FROM, [778, 84], [1060, 84], [1060, 58]], cls: 'probe', needs: ['smtp'], monitor: true },
  { id: 'probe-idp', points: [...PROBE_FROM, [778, 84], [290, 84], [290, 58]], cls: 'probe', needs: ['idp'], monitor: true, label: { key: 'probes', at: [780, 98], anchor: 'start' } },
  { id: 'probe-proxy', points: [[430, 452], [380, 452], [380, 392], [360, 392]], cls: 'probe', needs: ['proxy', 'collectors'], monitor: true }
]

export type TopoNode = Box & {
  id: NodeId
  kind: NodeKind
  /** The dependency behind the node (for its name, latency and drawer). */
  dep: DepHealth | null
  state: OpsState
  probe: ProbeStatus | null
  /** Shown dashed: nothing probes it (etcd, MinIO), or its probe reports "not probed". */
  unprobed: boolean
  tall?: boolean
}

export type TopoEdge = {
  id: string
  points: Pt[]
  cls: EdgeClass
  monitor: boolean
  /** The target is down: dashed critical line with ✖ at the midpoint. */
  down: boolean
  mid: Pt
  label?: EdgeDef['label']
}

export type Topology = { nodes: TopoNode[]; edges: TopoEdge[]; byId: Partial<Record<NodeId, TopoNode>> }

/** Total length of a polyline. */
export const polylineLength = (pts: Pt[]) => pts.slice(1).reduce((a, p, i) => a + Math.hypot(p[0] - pts[i][0], p[1] - pts[i][1]), 0)

/** The point halfway along a polyline (by length), where a down edge carries its ✖. */
export function polylineMid(pts: Pt[]): Pt {
  if (pts.length === 0) return [0, 0]
  if (pts.length === 1) return pts[0]
  let rest = polylineLength(pts) / 2
  for (let i = 1; i < pts.length; i += 1) {
    const [x0, y0] = pts[i - 1]
    const [x1, y1] = pts[i]
    const seg = Math.hypot(x1 - x0, y1 - y0)
    if (rest <= seg && seg > 0) {
      const f = rest / seg
      return [Math.round(x0 + (x1 - x0) * f), Math.round(y0 + (y1 - y0) * f)]
    }
    rest -= seg
  }
  return pts[pts.length - 1]
}

/** SVG `points` attribute for a polyline. */
export const pointsAttr = (pts: Pt[]) => pts.map(([x, y]) => `${x},${y}`).join(' ')

/** When two dependencies map to one node (Neo4j as "graph" and "neo4j"), the worse state wins. */
const worse = (a: DepHealth | undefined, b: DepHealth) => (!a || rankOf(depState(b.status)) > rankOf(depState(a.status)) ? b : a)

/**
 * Nodes and edges for the reported dependencies. Etcd and MinIO appear only
 * behind Milvus, always as "not probed"; the proxy only when its TLS probe
 * exists (otherwise callers reach the HTTP stack directly).
 */
export function buildTopology(deps: DepHealth[]): Topology {
  const byNode = new Map<NodeId, DepHealth>()
  for (const d of deps) {
    const n = nodeOfDep(d)
    if (n) byNode.set(n, worse(byNode.get(n), d))
  }
  const present = new Set<NodeId>(ALWAYS)
  for (const n of byNode.keys()) present.add(n)
  if (isMilvus(byNode.get('vector'))) {
    present.add('etcd')
    present.add('minio')
  }
  const nodes: TopoNode[] = (Object.keys(NODES) as NodeId[])
    .filter((id) => present.has(id))
    .map((id) => {
      const def = NODES[id]
      const dep = byNode.get(id) ?? null
      const probe = dep ? dep.status : id === 'etcd' || id === 'minio' ? 'unprobed' : null
      return {
        id,
        x: def.x,
        y: def.y,
        w: def.w,
        h: def.h,
        kind: def.kind,
        tall: def.tall,
        dep,
        probe,
        state: probe ? depState(probe) : 'ok',
        unprobed: probe === 'unprobed'
      }
    })
  const byId: Partial<Record<NodeId, TopoNode>> = Object.fromEntries(nodes.map((n) => [n.id, n]))
  const edges: TopoEdge[] = [...clientEdges(present.has('proxy')), ...EDGES]
    .filter((e) => e.needs.every((n) => present.has(n)))
    .map((e) => {
      const target = e.to ? byId[e.to] : undefined
      return {
        id: e.id,
        points: e.points,
        cls: e.cls,
        monitor: !!e.monitor,
        down: !e.monitor && !!target && target.probe === 'down',
        mid: polylineMid(e.points),
        label: e.label
      }
    })
  return { nodes, edges, byId }
}

/** The node the info panel opens on: the worst one, else Postgres, else the HTTP stack. */
export function defaultSelection(topo: Topology): NodeId {
  let pick: TopoNode | null = null
  for (const n of topo.nodes) if (rankOf(n.state) > 0 && (!pick || rankOf(n.state) > rankOf(pick.state))) pick = n
  if (pick) return pick.id
  return topo.byId.postgres ? 'postgres' : 'http'
}

/**
 * Rings a new (non-baseline) health poll may ask for: one per node that has
 * just entered Critical/Down (§2.12 "Topology: failing node"). The attention
 * slot grants at most one, after the state chip and attention items.
 */
export function nodeRingCandidates(prev: DepHealth[] | null | undefined, next: DepHealth[] | null | undefined): RingCandidate[] {
  if (!prev || !next) return []
  const before = buildTopology(prev).byId
  const out: RingCandidate[] = []
  for (const n of buildTopology(next).nodes) {
    if (n.state !== 'critical' || !n.dep) continue
    const was = before[n.id]?.state ?? 'unknown'
    if (isWorsening(was, n.state)) out.push({ key: `node:${n.id}`, rank: rankOf(n.state), source: 'node' })
  }
  return out
}

/** Clip a label to fit a node box (SVG text does not wrap). */
export const clip = (text: string, max: number) => (text.length > max ? `${text.slice(0, Math.max(1, max - 1))}…` : text)
