import { FlowBucket, flowBucket } from './motion'
import type { FlowRates, FlowSource } from './performance/model'
import type { Pt } from './topology'

/**
 * Topology edge flow (spec §2.12 "Topology: edges", §2.13). At most five
 * edges carry a 60 s rate series: browser → proxy → API (HTTP requests from
 * RequestIdMiddleware), API → vLLM embed and API → vLLM rerank (the
 * _record_dep_latency counts) and API → the LLM provider (calls per minute
 * from the usage table). Dashes flow along an edge at one of three speeds,
 * chosen against that edge's own 24 h p50/p90; the speed changes only when
 * the bucket changes. Flow stops under reduced motion, while "⏸ auto" is on,
 * while the tab is hidden, while the panel is stale and while the map is
 * less than half in view. The rate label and the line thickness carry the
 * same information without motion, so nothing depends on the dashes.
 */

/** Which rate series drives each edge. Edges not listed never flow. */
export const FLOW_EDGES: Record<string, FlowSource> = {
  'browser-proxy': 'http',
  'proxy-http': 'http',
  'browser-http': 'http',
  'api-vembed': 'embed',
  'api-vrerank': 'rerank',
  'api-llm': 'llm'
}

/** Where an edge's rate label sits; one label per source (browser → proxy shares the proxy's). */
export const RATE_LABEL_AT: Record<string, { at: Pt; anchor: 'start' | 'middle' | 'end' }> = {
  'proxy-http': { at: [388, 300], anchor: 'end' },
  'browser-http': { at: [282, 130], anchor: 'middle' },
  'api-vembed': { at: [770, 390], anchor: 'middle' },
  'api-vrerank': { at: [770, 454], anchor: 'middle' },
  'api-llm': { at: [796, 26], anchor: 'end' }
}

/** Seconds per 12 px dash period: 4, 8 and 16 px/s (§2.12). Idle edges are a still solid line. */
export const FLOW_PERIOD_S: Record<Exclude<FlowBucket, 'idle'>, number> = { low: 3, medium: 1.5, high: 0.75 }

export type FlowCtx = {
  reduced: boolean
  paused: boolean
  hidden: boolean
  stale: boolean
  /** At least half of the map is inside the viewport. */
  inView: boolean
}

/** Whether any edge may move right now (rule 3: loops stop when nobody can vouch for them). */
export const flowAllowed = (ctx: FlowCtx) => !ctx.reduced && !ctx.paused && !ctx.hidden && !ctx.stale && ctx.inView

/** Why the dashes are still, for the text under the map; null while they may move. */
export function flowStillReason(ctx: FlowCtx): 'reduced' | 'paused' | 'stale' | 'hidden' | 'offscreen' | null {
  if (ctx.reduced) return 'reduced'
  if (ctx.paused) return 'paused'
  if (ctx.stale) return 'stale'
  if (ctx.hidden) return 'hidden'
  if (!ctx.inView) return 'offscreen'
  return null
}

export type EdgeFlow = {
  source: FlowSource
  /** Per minute, latest complete minute. */
  perMin: number
  bucket: FlowBucket
  /** The dashes move (bucket above idle, flow allowed, target not down). */
  moving: boolean
}

/**
 * The flow of one edge, or null for an edge without a rate series or
 * without data yet. A down target never flows: its edge is the dashed
 * critical line with ✖ instead.
 */
export function edgeFlow(edgeId: string, down: boolean, rates: FlowRates | null | undefined, allowed: boolean): EdgeFlow | null {
  const source = FLOW_EDGES[edgeId]
  const rate = source ? rates?.[source] : undefined
  if (!source || !rate) return null
  const bucket = down ? 'idle' : flowBucket(rate.perMin, rate)
  return { source, perMin: rate.perMin, bucket, moving: allowed && bucket !== 'idle' }
}

/** The classes a flowing edge carries: a thickness bucket always, the dash loop only while moving. */
export function flowClasses(flow: EdgeFlow | null): string[] {
  if (!flow) return []
  const out = [`ops-te-rate-${flow.bucket}`]
  if (flow.moving) out.push('ops-te-flow', 'ops-flow', 'ops-loop', `ops-flow-${flow.bucket}`)
  return out
}
