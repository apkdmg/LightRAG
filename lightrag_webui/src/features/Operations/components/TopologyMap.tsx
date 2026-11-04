import { CSSProperties, KeyboardEvent, memo, useId, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Seg } from '@/features/TokenInsights/ui'
import { cn } from '@/lib/utils'
import type { DepHealth } from '../api'
import { useOpsNav } from '../context'
import { EdgeFlow, RATE_LABEL_AT, edgeFlow, flowAllowed, flowClasses, flowStillReason } from '../flow'
import { fmtClock, fmtCount, fmtLatency, utcTitle } from '../format'
import { useInView } from '../hooks'
import { useOpsMotion, useRing, useRingToken } from '../motion'
import { FlowRates, FlowSource, fmtPerMin } from '../performance/model'
import { lookOf } from '../severity'
import {
  CONTAINER,
  NodeId,
  SERVICES,
  TOPO_HEIGHT,
  TOPO_WIDTH,
  TopoEdge,
  TopoNode,
  buildTopology,
  clip,
  defaultSelection,
  pointsAttr
} from '../topology'
import OpsLink from './OpsLink'
import { StateIcon, StatusChip } from './StatusChip'

type Layer = 'traffic' | 'monitor'

/** Characters that fit a node box at the label (12.5 px) and subtitle (10.5 px mono) sizes. */
const labelChars = (w: number) => Math.floor((w - 34) / 7)
const subChars = (w: number) => Math.floor((w - 16) / 6.4)

/** Nodes whose dependency is a check on them, not the node itself: the proxy is watched through its TLS certificate. */
const CHECKED_THROUGH = new Set(['proxy'])

/** The node's display name: the server's name for the dependency, else the generic one. */
function useNodeText(n: TopoNode) {
  const { t } = useTranslation()
  const generic = t(`operations.topology.node.${n.id}.label`)
  const label = n.dep && !CHECKED_THROUGH.has(n.id) ? (n.dep.name ?? t(`operations.dep.${n.dep.id}`, { defaultValue: generic })) : generic
  const sub = n.dep?.target ?? t(`operations.topology.node.${n.id}.sub`)
  const sub2 = n.tall ? t(`operations.topology.node.${n.id}.sub2`) : null
  return { label, sub, sub2 }
}

function NodeBox({ n, selected, onSelect }: { n: TopoNode; selected: boolean; onSelect: (id: NodeId) => void }) {
  const { t } = useTranslation()
  const { label, sub, sub2 } = useNodeText(n)
  const ring = useRing<SVGRectElement>(useRingToken(`node:${n.id}`), { measure: false })
  const look = lookOf(n.state)
  const probed = !!n.probe && !n.unprobed
  const stateText = n.unprobed ? t('operations.topology.notProbed') : probed ? t(look.label) : null
  const onKeyDown = (e: KeyboardEvent<SVGGElement>) => {
    if (e.key !== 'Enter' && e.key !== ' ') return
    e.preventDefault()
    onSelect(n.id)
  }
  // The ring grows by at most 8 px per side, whatever the box size (§2.12).
  const ringStyle = { '--ring-sx': (1 + 16 / n.w).toFixed(3), '--ring-sy': (1 + 16 / n.h).toFixed(3) } as CSSProperties
  return (
    <g
      className={cn('ops-tn', `ops-tn-${n.kind}`, n.unprobed && 'is-unprobed', selected && 'is-sel')}
      data-node={n.id}
      data-tone={probed ? look.tone : undefined}
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      aria-label={stateText ? `${label}, ${stateText}` : label}
      onClick={() => onSelect(n.id)}
      onKeyDown={onKeyDown}
    >
      <rect className="ops-tn-box" x={n.x} y={n.y} width={n.w} height={n.h} rx={6} />
      <text className="ops-tn-l" x={n.x + 10} y={n.y + 19}>
        {clip(label, labelChars(n.w))}
      </text>
      <text className="ops-tn-s" x={n.x + 10} y={n.y + 34}>
        {clip(sub, subChars(n.w))}
      </text>
      {sub2 ? (
        <text className="ops-tn-s" x={n.x + 10} y={n.y + 48}>
          {clip(sub2, subChars(n.w))}
        </text>
      ) : null}
      {probed && n.h >= 40 ? (
        <StateIcon icon={look.icon} x={n.x + n.w - 21} y={n.y + 8} width={14} height={14} className="ops-tn-ico size-[14px]" />
      ) : null}
      <rect ref={ring} className="ops-topo-ring" x={n.x} y={n.y} width={n.w} height={n.h} rx={6} style={ringStyle} aria-hidden="true" />
    </g>
  )
}

function Edge({ e, markers, flow }: { e: TopoEdge; markers: string; flow: EdgeFlow | null }) {
  const marker = e.down ? 'crit' : e.cls === 'monitor' || e.cls === 'probe' ? 'mon' : flow && flow.bucket !== 'idle' ? 'flow' : 'base'
  return (
    <>
      <polyline
        className={cn('ops-te', `ops-te-${e.cls}`, e.down && 'is-down', e.monitor && 'is-mon', ...flowClasses(flow))}
        data-flow={flow ? flow.bucket : undefined}
        points={pointsAttr(e.points)}
        markerEnd={e.cls === 'unprobed' || e.cls === 'probe' ? undefined : `url(#${markers}-${marker})`}
      />
      {e.down ? (
        <text className="ops-te-x" x={e.mid[0]} y={e.mid[1] + 4} textAnchor="middle" aria-hidden="true">
          ✖
        </text>
      ) : null}
    </>
  )
}

/** The edge's rate as text ("75/min"): the same information as the dash speed, without motion. */
function RateLabel({ e, flow }: { e: TopoEdge; flow: EdgeFlow | null }) {
  const place = RATE_LABEL_AT[e.id]
  if (!place || !flow || e.down) return null
  return (
    <text className="ops-te-lab ops-te-rate" x={place.at[0]} y={place.at[1]} textAnchor={place.anchor} aria-hidden="true">
      {fmtPerMin(flow.perMin)}
    </text>
  )
}

function EdgeLabel({ e }: { e: TopoEdge }) {
  const { t } = useTranslation()
  if (!e.label) return null
  return (
    <text className={cn('ops-te-lab', e.monitor && 'is-mon')} x={e.label.at[0]} y={e.label.at[1]} textAnchor={e.label.anchor ?? 'start'}>
      {t(`operations.topology.edge.${e.label.key}`)}
    </text>
  )
}

function Marker({ id, color }: { id: string; color: string }) {
  return (
    <marker id={id} viewBox="0 0 8 8" refX={7} refY={4} markerWidth={7} markerHeight={7} orient="auto-start-reverse">
      <path d="M0,0 L8,4 L0,8 z" fill={color} />
    </marker>
  )
}

/** What the selected node is, how Operations watches it, and its latest probe. */
function InfoPanel({ node }: { node: TopoNode }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const { label } = useNodeText(node)
  const dep: DepHealth | null = node.dep
  return (
    <div className="flex min-w-0 flex-col gap-2 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="m-0 text-[15px] font-semibold">{label}</h3>
        {node.unprobed ? (
          <StatusChip state="unknown">{t('operations.topology.notProbed')}</StatusChip>
        ) : dep ? (
          <StatusChip state={node.state} />
        ) : null}
      </div>
      <dl className="m-0 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 text-[13px]">
        <dt className="text-muted-foreground">{t('operations.topology.info.role')}</dt>
        <dd className="m-0">{t(`operations.topology.node.${node.id}.role`)}</dd>
        <dt className="text-muted-foreground">{t('operations.topology.info.watch')}</dt>
        <dd className="m-0">{t(`operations.topology.node.${node.id}.watch`)}</dd>
        {dep && !node.unprobed ? (
          <>
            <dt className="text-muted-foreground">{t('operations.topology.info.latest')}</dt>
            <dd className="m-0" title={utcTitle(dep.checked_at)}>
              {[
                typeof dep.latency_ms === 'number' ? fmtLatency(dep.latency_ms) : null,
                dep.error_class,
                dep.checked_at ? t('operations.health.checkedAt', { time: fmtClock(dep.checked_at) }) : null
              ]
                .filter(Boolean)
                .join(' · ') || '–'}
            </dd>
            {dep.status !== 'ok' && dep.last_ok_at ? (
              <>
                <dt className="text-muted-foreground">{t('operations.health.lastOk')}</dt>
                <dd className="m-0" title={utcTitle(dep.last_ok_at)}>
                  {fmtClock(dep.last_ok_at)}
                </dd>
              </>
            ) : null}
            {typeof dep.consecutive_failures === 'number' && dep.consecutive_failures > 0 ? (
              <>
                <dt className="text-muted-foreground">{t('operations.health.failures')}</dt>
                <dd className="m-0">{fmtCount(dep.consecutive_failures)}</dd>
              </>
            ) : null}
          </>
        ) : null}
      </dl>
      {dep ? (
        <div>
          <OpsLink
            to={{ ...nav.view, drawer: { kind: 'dependency', id: dep.id } }}
            className="hover:bg-muted inline-flex min-h-8 items-center rounded-lg border px-2.5 text-[13px] no-underline"
          >
            {t('operations.topology.info.open')}
          </OpsLink>
        </div>
      ) : null}
    </div>
  )
}

/** Sources in the order the flow list names them. */
const FLOW_ORDER: FlowSource[] = ['http', 'embed', 'rerank', 'llm']

/**
 * "Live traffic, last minute" under the map: every source with its rate and
 * speed bucket in text, and why the dashes are still when they are. This is
 * the text alternative to the moving dashes (§2.12 rule 6).
 */
function FlowList({ flows, reason }: { flows: Partial<Record<FlowSource, EdgeFlow>>; reason: string | null }) {
  const { t } = useTranslation()
  const rows = FLOW_ORDER.filter((s) => flows[s])
  if (!rows.length) return null
  return (
    <div className="flex flex-col gap-1 border-t px-4 py-3 text-xs">
      <span className="font-medium">{t('operations.topology.flow.title')}</span>
      <ul className="m-0 flex list-none flex-wrap gap-x-5 gap-y-1 p-0">
        {rows.map((s) => {
          const f = flows[s]!
          return (
            <li key={s} data-flow-source={s}>
              {t(`operations.topology.flow.source.${s}`)}: <span className="font-mono">{fmtPerMin(f.perMin)}</span>
              <span className="text-muted-foreground"> · {t(`operations.topology.flow.bucket.${f.bucket}`)}</span>
            </li>
          )
        })}
      </ul>
      <span className="text-muted-foreground">{reason ? t(`operations.topology.flow.still.${reason}`) : t('operations.topology.flow.moving')}</span>
    </div>
  )
}

function LegendLine({ cls, children }: { cls: string; children: React.ReactNode }) {
  return (
    <li className="flex items-center gap-2">
      <svg width="40" height="10" aria-hidden="true" focusable="false" className="flex-none">
        <line x1="0" y1="5" x2="40" y2="5" className={cn('ops-te', cls)} />
      </svg>
      {children}
    </li>
  )
}

/**
 * The system topology at the top of the Health tab (§2.13). Node status
 * follows the latest probe cycle (icon and outline, never colour alone); a
 * target that is down turns its edges into dashed critical lines with ✖ at
 * the midpoint; services nothing probes are dashed grey. A node that has
 * just failed gets one ring when the page's attention slot grants it.
 *
 * Up to five edges carry live traffic (`flows`, from the per-minute rate
 * series): dashes flow at a speed bucketed against the edge's own 24 h
 * p50/p90, with the rate as a label and the bucket as line thickness. The
 * dashes stop under reduced motion, "⏸ auto", a hidden tab, a stale panel
 * or while the map is less than half on screen; only stroke-dashoffset
 * moves (paint only, no layout).
 *
 * Nodes are keyboard-focusable buttons that open the info panel; the
 * Monitoring layer shows what the ops collectors write and probe.
 */
function TopologyMap({
  deps,
  processLabel,
  flows,
  stale = false
}: {
  deps: DepHealth[]
  processLabel?: string | null
  /** Per-source rates for the flowing edges; null or missing keeps every edge still. */
  flows?: FlowRates | null
  /** The health or rate data is stale: nothing can vouch the traffic is still flowing. */
  stale?: boolean
}) {
  const { t } = useTranslation()
  const { reduced, paused, hidden } = useOpsMotion()
  const box = useRef<HTMLDivElement | null>(null)
  const inView = useInView(box, 0.5)
  const ctx = { reduced, paused, hidden, stale, inView }
  const allowed = flowAllowed(ctx)
  const markers = `ops-topo-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`
  const topo = useMemo(() => buildTopology(deps), [deps])
  const [layer, setLayer] = useState<Layer>('traffic')
  const [picked, setPicked] = useState<NodeId | null>(null)
  const selected = picked && topo.byId[picked] ? picked : defaultSelection(topo)
  const node = topo.byId[selected] ?? topo.nodes[0]
  const servicesShown = topo.nodes.some((n) => n.id === 'vector' || n.id === 'vembed' || n.id === 'vrerank' || n.id === 'docling')
  const names = topo.nodes.filter((n) => n.dep).map((n) => n.dep!.name ?? t(`operations.dep.${n.dep!.id}`, { defaultValue: n.dep!.id }))
  const data = topo.edges.filter((e) => !e.monitor)
  const mon = topo.edges.filter((e) => e.monitor)
  const edgeFlows = new Map<string, EdgeFlow | null>(data.map((e) => [e.id, edgeFlow(e.id, e.down, flows, allowed)]))
  const bySource: Partial<Record<FlowSource, EdgeFlow>> = {}
  for (const f of edgeFlows.values()) if (f && !bySource[f.source]) bySource[f.source] = f
  const hasFlows = Object.keys(bySource).length > 0
  const anyMoving = [...edgeFlows.values()].some((f) => f?.moving)
  return (
    <div className="bg-card min-w-0 overflow-hidden rounded-[10px] border">
      <div className="bg-muted/40 flex flex-wrap items-center gap-x-4 gap-y-2 border-b px-4 py-2.5">
        <h2 className="m-0 text-[15px] font-semibold">{t('operations.topology.title')}</h2>
        <Seg
          value={layer}
          onChange={setLayer}
          label={t('operations.topology.layer')}
          options={[
            { value: 'traffic', label: t('operations.topology.traffic') },
            { value: 'monitor', label: t('operations.topology.monitoring') }
          ]}
        />
        <span className="text-muted-foreground text-xs">{hasFlows ? t('operations.topology.flowNote') : t('operations.topology.staticNote')}</span>
      </div>
      <div className="overflow-x-auto" ref={box}>
        <div className={cn('ops-topo', layer === 'monitor' && 'is-mon')} data-flowing={anyMoving ? '' : undefined}>
          <svg
            className="ops-dim"
            viewBox={`0 0 ${TOPO_WIDTH} ${TOPO_HEIGHT}`}
            role="group"
            aria-label={t('operations.topology.svgLabel', { deps: names.length ? names.join(', ') : t('operations.topology.noDeps') })}
          >
            <defs>
              <Marker id={`${markers}-base`} color="var(--muted-foreground)" />
              <Marker id={`${markers}-mon`} color="var(--ops-info)" />
              <Marker id={`${markers}-crit`} color="var(--ops-crit)" />
              <Marker id={`${markers}-flow`} color="var(--ops-s1)" />
            </defs>
            <rect className="ops-grp" x={CONTAINER.x} y={CONTAINER.y} width={CONTAINER.w} height={CONTAINER.h} rx={10} />
            <text className="ops-grp-lab" x={CONTAINER.x + 14} y={CONTAINER.y + 20}>
              {clip(t('operations.topology.group.container', { mode: processLabel ?? t('operations.process.unknown') }), 48)}
            </text>
            {servicesShown ? (
              <>
                <rect className="ops-grp-svc" x={SERVICES.x} y={SERVICES.y} width={SERVICES.w} height={SERVICES.h} rx={10} />
                <text className="ops-grp-lab" x={SERVICES.x + 6} y={SERVICES.y - 6}>
                  {t('operations.topology.group.services')}
                </text>
              </>
            ) : null}
            <text className="ops-grp-lab" x={476} y={20}>
              {t('operations.topology.group.external')}
            </text>
            {data.map((e) => (
              <Edge key={e.id} e={e} markers={markers} flow={edgeFlows.get(e.id) ?? null} />
            ))}
            {mon.map((e) => (
              <Edge key={e.id} e={e} markers={markers} flow={null} />
            ))}
            {topo.edges.map((e) => (
              <EdgeLabel key={e.id} e={e} />
            ))}
            {data.map((e) => (
              <RateLabel key={`rate-${e.id}`} e={e} flow={edgeFlows.get(e.id) ?? null} />
            ))}
            {topo.nodes.map((n) => (
              <NodeBox key={n.id} n={n} selected={n.id === selected} onSelect={setPicked} />
            ))}
          </svg>
        </div>
      </div>
      <div className="grid border-t md:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        {node ? <InfoPanel node={node} /> : <div />}
        <ul className="text-muted-foreground m-0 flex list-none flex-col gap-1.5 border-t p-4 text-xs md:border-t-0 md:border-l">
          <LegendLine cls="ops-te-data">{t('operations.topology.legend.connection')}</LegendLine>
          {hasFlows ? <LegendLine cls="ops-te-data ops-te-rate-medium ops-te-legend-flow">{t('operations.topology.legend.traffic')}</LegendLine> : null}
          <LegendLine cls="ops-te-unprobed">{t('operations.topology.legend.unprobed')}</LegendLine>
          <LegendLine cls="ops-te-data is-down">{t('operations.topology.legend.down')}</LegendLine>
          <LegendLine cls="ops-te-bypass">{t('operations.topology.legend.bypass')}</LegendLine>
          <LegendLine cls="ops-te-probe">{t('operations.topology.legend.probe')}</LegendLine>
        </ul>
      </div>
      <FlowList flows={bySource} reason={flowStillReason(ctx)} />
    </div>
  )
}

/** Re-renders only when the dependencies, the rates or the process label change, not on the page's clock ticks. */
export default memo(TopologyMap)
