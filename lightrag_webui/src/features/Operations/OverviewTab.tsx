import { useTranslation } from 'react-i18next'
import { Card, Kv, Title } from '@/features/TokenInsights/ui'
import { OpsFeatures, OpsOverview } from './api'
import ActiveRunsTable from './components/ActiveRunsTable'
import AttentionList from './components/AttentionList'
import KpiTile from './components/KpiTile'
import OpsLink from './components/OpsLink'
import { Panel } from './components/Panel'
import ProcessChip from './components/ProcessChip'
import { StatusChip, ToneChip } from './components/StatusChip'
import TimelinePanel from './components/TimelinePanel'
import { useOpsNav, useOpsOverview, useOpsShell } from './context'
import { fmtAge, fmtClock, fmtCount, fmtLatency, fmtPercent } from './format'
import { viewForTarget } from './hashState'
import { useStale } from './hooks'
import { KPI_TARGETS, KpiId, OVERVIEW_REFRESH_SECONDS, kpiState, stuckTotal } from './overview'
import { OpsPoll } from './poll'
import { isMultiWorker } from './severity'

const FEATURE_ORDER: (keyof OpsFeatures)[] = ['durable', 'runs', 'logs', 'probes', 'alerts', 'release_busy', 'gpu', 'insights']
const TOP_ACTIVE = 5

/** Why a feature is off: Insights can be switched off, release busy needs Gunicorn, the rest ships later. */
const offLabel = (key: keyof OpsFeatures, singleProcess: boolean) => {
  if (key === 'insights') return 'operations.feature.off'
  if (key === 'release_busy' && singleProcess) return 'operations.feature.multiWorkerOnly'
  return 'operations.feature.later'
}

/** What /status reports about this server (the workers strip arrives with heartbeats). */
export function ServerCard() {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  if (!status) return null
  const features = status.features ?? {}
  const singleProcess = !isMultiWorker(status.process.mode)
  return (
    <Card className="flex min-w-0 flex-col gap-3">
      <Title sub={t('operations.system.checked', { time: fmtClock(status.generated_at) })}>{t('operations.system.title')}</Title>
      <div className="flex flex-col gap-1.5">
        <Kv k={t('operations.system.mode')}>
          <ProcessChip process={status.process} />
        </Kv>
        <Kv k={t('operations.system.workers')}>
          {typeof status.process.workers_live === 'number'
            ? t('operations.system.workersLive', { live: status.process.workers_live, expected: status.process.expected_workers ?? '–' })
            : t('operations.system.workersPending')}
        </Kv>
        <Kv k={t('operations.system.storage')}>
          {status.available ? (
            <ToneChip tone="ok" icon="check">
              {t('operations.system.storageOk')}
            </ToneChip>
          ) : (
            <ToneChip tone="unknown" icon="dashed-circle">
              {t('operations.system.storageMissing')}
            </ToneChip>
          )}
        </Kv>
        <Kv k={t('operations.system.access')}>{status.operator ? t('operations.system.operator') : t('operations.system.adminOnly')}</Kv>
        {status.version ? <Kv k={t('operations.system.version')}>{status.version}</Kv> : null}
      </div>
      {singleProcess ? <p className="text-muted-foreground m-0 text-xs">{t('operations.system.singleProcessNote')}</p> : null}
      <div>
        <h3 className="m-0 mb-1.5 text-[13px] font-semibold">{t('operations.system.features')}</h3>
        <ul className="m-0 grid list-none grid-cols-1 gap-1 p-0 sm:grid-cols-2">
          {FEATURE_ORDER.map((key) => (
            <li key={key} className="flex items-center justify-between gap-3 text-xs">
              <span className="text-muted-foreground">{t(`operations.feature.${key}`)}</span>
              {features[key] ? (
                <ToneChip tone="ok" icon="check">
                  {t('operations.feature.on')}
                </ToneChip>
              ) : (
                <ToneChip tone="unknown" icon="dashed-circle">
                  {t(offLabel(key, singleProcess))}
                </ToneChip>
              )}
            </li>
          ))}
        </ul>
      </div>
    </Card>
  )
}

/** The eight KPI tiles (FR-3), each a link to its drill-down. */
export function KpiRow({ data, poll }: { data: OpsOverview; poll: Pick<OpsPoll<OpsOverview>, 'baseline' | 'previousAt' | 'updatedAt' | 'error'> }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const { status } = useOpsShell()
  const { stale } = useStale(poll.updatedAt, OVERVIEW_REFRESH_SECONDS)
  const k = data.kpis ?? {}
  const common = (id: KpiId) => ({
    state: kpiState(id, k),
    scope: (k[id] as { scope?: string | null } | undefined)?.scope ?? 'host',
    to: viewForTarget(nav.view, KPI_TARGETS[id]),
    baseline: poll.baseline,
    since: poll.previousAt,
    stale: stale || !!poll.error
  })
  const stuck = k.stuck
  const stuckParts = [
    stuck?.runs ? t('operations.kpi.stuck.runs', { count: stuck.runs }) : null,
    stuck?.docs ? t('operations.kpi.stuck.docs', { count: stuck.docs }) : null,
    stuck?.reservations ? t('operations.kpi.stuck.reservations', { count: stuck.reservations }) : null
  ].filter(Boolean)
  const failedParts = [
    typeof k.failed_24h?.cancelled === 'number' ? t('operations.kpi.failed.cancelled', { count: fmtCount(k.failed_24h.cancelled) }) : null,
    typeof k.failed_24h?.dup === 'number' ? t('operations.kpi.failed.dup', { count: fmtCount(k.failed_24h.dup) }) : null
  ].filter(Boolean)
  const workersLive = k.workers?.live ?? status?.process.workers_live ?? null
  const workersExpected = k.workers?.expected ?? status?.process.expected_workers ?? null
  const singleProcess = status ? !isMultiWorker(status.process.mode) : false
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4 2xl:grid-cols-8">
      <KpiTile
        {...common('active')}
        label={t('operations.kpi.active.label')}
        value={k.active?.count}
        sub={
          typeof k.active?.longest_s === 'number'
            ? t('operations.kpi.active.longest', { age: fmtAge(k.active.longest_s) })
            : k.active?.approximate
              ? t('operations.approximate.label')
              : t('operations.kpi.active.none')
        }
      />
      <KpiTile
        {...common('backlog')}
        label={t('operations.kpi.backlog.label')}
        value={k.backlog?.inflight}
        sub={
          typeof k.backlog?.delta_1h === 'number'
            ? t('operations.kpi.backlog.delta', { sign: k.backlog.delta_1h >= 0 ? '+' : '−', count: fmtCount(Math.abs(k.backlog.delta_1h)) })
            : typeof k.backlog?.oldest_s === 'number'
              ? t('operations.kpi.backlog.oldest', { age: fmtAge(k.backlog.oldest_s) })
              : t('operations.kpi.backlog.sub')
        }
      />
      <KpiTile
        {...common('failed_24h')}
        label={t('operations.kpi.failed.label')}
        value={k.failed_24h?.real}
        sub={failedParts.length ? failedParts.join(' · ') : t('operations.kpi.failed.sub')}
      />
      <KpiTile
        {...common('stuck')}
        label={t('operations.kpi.stuck.label')}
        value={stuckTotal(k)}
        sub={
          stuckParts.length
            ? stuckParts.join(' · ')
            : stuckTotal(k) === null
              ? t('operations.kpi.stuck.unknown')
              : t('operations.kpi.stuck.none')
        }
      />
      <KpiTile
        {...common('email_queue')}
        label={t('operations.kpi.email.label')}
        value={k.email_queue?.records}
        sub={
          typeof k.email_queue?.bad === 'number' && k.email_queue.bad > 0
            ? t('operations.kpi.email.bad', { count: k.email_queue.bad })
            : typeof k.email_queue?.longest_pending_s === 'number' && k.email_queue.longest_pending_s > 0
              ? t('operations.kpi.email.longest', { age: fmtAge(k.email_queue.longest_pending_s) })
              : t('operations.kpi.email.sub')
        }
      />
      <KpiTile
        {...common('llm_15m')}
        label={t('operations.kpi.llm.label')}
        value={k.llm_15m?.error_rate}
        format={(n) => t('operations.kpi.llm.errorRate', { rate: fmtPercent(n) })}
        sub={
          k.llm_15m
            ? t('operations.kpi.llm.sub', { p95: fmtLatency(k.llm_15m.p95_ms ?? null), calls: fmtCount(k.llm_15m.calls ?? null) })
            : t('operations.kpi.llm.unavailable')
        }
      />
      <KpiTile
        {...common('deps')}
        label={t('operations.kpi.deps.label')}
        display={typeof k.deps?.ok === 'number' && typeof k.deps?.total === 'number' ? `${k.deps.ok}/${k.deps.total}` : '–'}
        sub={k.deps && typeof k.deps.total === 'number' ? t('operations.kpi.deps.sub') : t('operations.kpi.deps.later')}
      />
      <KpiTile
        {...common('workers')}
        label={t('operations.kpi.workers.label')}
        display={
          singleProcess
            ? t('operations.kpi.workers.single')
            : typeof workersLive === 'number' && typeof workersExpected === 'number'
              ? `${workersLive}/${workersExpected}`
              : '–'
        }
        sub={singleProcess ? t('operations.process.uvicorn') : typeof workersLive === 'number' ? t('operations.kpi.workers.sub') : t('operations.system.workersPending')}
      />
    </div>
  )
}

function AttentionPanel({ poll }: { poll: OpsPoll<OpsOverview> }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  return (
    <Panel
      poll={poll}
      freshness={OVERVIEW_REFRESH_SECONDS}
      title={t('operations.attention.title')}
      sub={poll.updatedAt ? t('operations.attention.checked', { time: fmtClock(poll.updatedAt) }) : undefined}
    >
      {(data) => {
        const c = data.checked
        const time = fmtClock(c?.at ?? poll.updatedAt)
        const empty =
          c && typeof c.workspaces === 'number'
            ? t('operations.attention.none', { workspaces: fmtCount(c.workspaces), active: fmtCount(c.active ?? 0), time })
            : t('operations.attention.noneSimple', { time })
        return <AttentionList items={data.attention ?? []} base={nav.view} baseline={poll.baseline} now={poll.updatedAt} empty={empty} />
      }}
    </Panel>
  )
}

function ProcessingNow({ poll }: { poll: OpsPoll<OpsOverview> }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const { status } = useOpsShell()
  const { stale } = useStale(poll.updatedAt, OVERVIEW_REFRESH_SECONDS)
  return (
    <Panel
      poll={poll}
      freshness={OVERVIEW_REFRESH_SECONDS}
      title={t('operations.overview.processingNow')}
      approximate={!!poll.data?.kpis?.active?.approximate}
      actions={
        <OpsLink to={viewForTarget(nav.view, { tab: 'processing', sub: 'active' })} className="hover:bg-muted rounded-lg border px-2.5 py-1 text-[13px] no-underline">
          {t('operations.overview.allActive')}
        </OpsLink>
      }
    >
      {(data) => {
        const runs = (data.active_top ?? []).slice(0, TOP_ACTIVE)
        return runs.length ? (
          <ActiveRunsTable runs={runs} now={poll.updatedAt} stale={stale} baseline={poll.baseline} compact multiWorker={isMultiWorker(status?.process.mode)} />
        ) : (
          <p className="text-muted-foreground m-0 text-sm">{t('operations.active.none')}</p>
        )
      }}
    </Panel>
  )
}

function DepsSummary({ data }: { data: OpsOverview | null }) {
  const { t } = useTranslation()
  const deps = data?.deps ?? []
  return (
    <Card className="flex min-w-0 flex-col gap-2">
      <Title>{t('operations.overview.deps')}</Title>
      {deps.length ? (
        <ul className="m-0 flex list-none flex-wrap gap-1.5 p-0">
          {deps.map((d) => (
            <li key={d.id}>
              <StatusChip state={d.status}>{t(`operations.dep.${d.id}`, { defaultValue: d.id })}</StatusChip>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-muted-foreground m-0 text-sm">{t('operations.overview.depsLater')}</p>
      )}
    </Card>
  )
}

/**
 * The Overview (§2.3): KPI tiles, Needs attention, the server card, the last
 * 6 hours, processing now and the dependency summary. The /overview poll is
 * shared with the header chip (15 s); timeline markers refresh every 60 s.
 */
export default function OverviewTab() {
  const poll = useOpsOverview()
  if (!poll) return <ServerCard />
  return <OverviewContent poll={poll} />
}

function OverviewContent({ poll }: { poll: OpsPoll<OpsOverview> }) {
  const data = poll.data
  return (
    <div className="flex flex-col gap-4">
      {data && data.available !== false ? <KpiRow data={data} poll={poll} /> : null}
      <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <AttentionPanel poll={poll} />
        <ServerCard />
      </div>
      <TimelinePanel buckets={data?.timeline?.buckets} events={data?.timeline?.events} />
      <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <ProcessingNow poll={poll} />
        <DepsSummary data={data} />
      </div>
    </div>
  )
}
