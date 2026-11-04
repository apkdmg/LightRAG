import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { DownloadIcon } from 'lucide-react'
import { Kv } from '@/components/admin'
import { AlertEmailState, OpsAlert, opsApi } from '../api'
import { alertActionsEnabled, alertTitle, ruleTitle } from '../activity/alertActions'
import { alertChannelOf, alertOpenSeconds, alertTarget, emailFromChannel } from '../activity/alerts'
import { NotifiedCell, RunbookCell } from '../activity/AlertsView'
import { AckButton, MuteMenu } from '../components/AlertActions'
import ErrorPanel from '../components/ErrorPanel'
import IncidentExportDialog from '../components/IncidentExportDialog'
import OpsLink from '../components/OpsLink'
import { StaleChip } from '../components/Panel'
import Skeleton from '../components/Skeleton'
import { StatusChip, ToneChip } from '../components/StatusChip'
import { useOpsNav, useOpsOverview, useOpsShell } from '../context'
import { fmtAge, fmtStamp, utcTitle } from '../format'
import { viewForTarget } from '../hashState'
import { useStale } from '../hooks'
import { useOpsPoll } from '../poll'
import Section from './Section'

/** Drawers refresh at 15 s (§3.10). */
const DRAWER_SECONDS = 15

const linkButton = 'text-left underline underline-offset-2 break-all'

/** Firing or resolved, as an icon and text (an ack does not change it, FR-61). */
function AlertStateChip({ alert }: { alert: OpsAlert }) {
  const { t } = useTranslation()
  return alert.state === 'resolved' ? (
    <ToneChip tone="ok" icon="check">
      {t('operations.alerts.state.resolved')}
    </ToneChip>
  ) : (
    <StatusChip state={alert.severity}>{t('operations.alerts.state.firing')}</StatusChip>
  )
}

/** The alert's facts: severity, state, rule, scope, times, email and ack. */
export function AlertSummary({ alert, now, email: own = null }: { alert: OpsAlert; now: number; email?: AlertEmailState | null }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const overview = useOpsOverview()
  const email = own ?? emailFromChannel(alertChannelOf(overview?.data))
  const at = now
  const s = alert.scope
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col items-start gap-1.5">
        <span className="flex flex-wrap items-center gap-2">
          <StatusChip state={alert.severity} />
          <AlertStateChip alert={alert} />
        </span>
        <p className="m-0 text-[15px] font-medium">{alertTitle(t, alert)}</p>
      </div>
      <div className="flex flex-col gap-1">
        <Kv k={t('operations.alertDrawer.rule')}>
          <span className="flex flex-col items-end">
            <span>{ruleTitle(t, alert.rule, null, alert.label)}</span>
            <span className="text-muted-foreground font-mono text-xs">{alert.rule}</span>
          </span>
        </Kv>
        {s.workspace ? (
          <Kv k={t('operations.alertDrawer.workspace')}>
            <button type="button" className={linkButton} onClick={() => nav.openDrawer({ kind: 'workspace', id: s.workspace as string }, { push: true })}>
              {s.workspace}
            </button>
          </Kv>
        ) : null}
        {s.dep ? (
          <Kv k={t('operations.alertDrawer.dependency')}>
            <button type="button" className={linkButton} onClick={() => nav.openDrawer({ kind: 'dependency', id: s.dep as string }, { push: true })}>
              {t(`operations.dep.${s.dep}`, { defaultValue: s.dep })}
            </button>
          </Kv>
        ) : null}
        {s.worker_id ? (
          <Kv k={t('operations.alertDrawer.worker')}>
            <button type="button" className={`${linkButton} font-mono text-xs`} onClick={() => nav.openDrawer({ kind: 'worker', id: s.worker_id as string }, { push: true })}>
              {s.worker_id}
            </button>
          </Kv>
        ) : null}
        {s.run_id ? (
          <Kv k={t('operations.alertDrawer.run')}>
            <button type="button" className={`${linkButton} font-mono text-xs`} onClick={() => nav.openDrawer({ kind: 'run', id: s.run_id as string }, { push: true })}>
              {s.run_id}
            </button>
          </Kv>
        ) : null}
        <Kv k={t('operations.alertDrawer.firstSeen')}>
          <span title={utcTitle(alert.first_seen)}>{fmtStamp(alert.first_seen, at)}</span>
        </Kv>
        <Kv k={t('operations.alertDrawer.lastSeen')}>
          <span title={utcTitle(alert.last_seen)}>{fmtStamp(alert.last_seen, at)}</span>
        </Kv>
        {alert.resolved_at ? (
          <Kv k={t('operations.alertDrawer.resolved')}>
            <span title={utcTitle(alert.resolved_at)}>{fmtStamp(alert.resolved_at, at)}</span>
          </Kv>
        ) : null}
        <Kv k={alert.resolved_at ? t('operations.alertDrawer.wasOpen') : t('operations.alertDrawer.openFor')}>{fmtAge(alertOpenSeconds(alert, at))}</Kv>
        <Kv k={t('operations.alertDrawer.notified')}>
          <NotifiedCell alert={alert} email={email} now={now} />
        </Kv>
        <Kv k={t('operations.alertDrawer.acked')}>
          {alert.acked_by ? (
            <span className="flex flex-col items-end">
              <span>
                {alert.acked_by}
                {alert.acked_at ? <span className="text-muted-foreground"> · {fmtStamp(alert.acked_at, at)}</span> : null}
              </span>
              {alert.ack_note ? <span className="text-muted-foreground text-xs break-words">{alert.ack_note}</span> : null}
            </span>
          ) : (
            '–'
          )}
        </Kv>
      </div>
    </div>
  )
}

/** Earlier occurrences of the same fingerprint: a re-fire after resolving is a new row (FR-61). */
function AlertHistory({ items, now }: { items: OpsAlert[]; now: number }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  if (!items.length) return null
  return (
    <Section title={t('operations.alertDrawer.history', { count: items.length })}>
      <ul className="m-0 flex list-none flex-col p-0">
        {items.map((a) => (
          <li key={a.id}>
            <button
              type="button"
              className="hover:bg-muted grid w-full grid-cols-[auto_minmax(0,1fr)_auto] items-baseline gap-x-3 rounded-lg px-2 py-1.5 text-left text-[13px]"
              onClick={() => nav.openDrawer({ kind: 'alert', id: a.id }, { push: true })}
            >
              <StatusChip state={a.severity} />
              <span className="font-mono text-xs" title={utcTitle(a.first_seen)}>
                {fmtStamp(a.first_seen, now)}
              </span>
              <span className="text-muted-foreground text-xs whitespace-nowrap">
                {a.resolved_at ? t('operations.alerts.openFor', { age: fmtAge(alertOpenSeconds(a, now)) }) : t('operations.alerts.state.firing')}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </Section>
  )
}

/**
 * The Alert drawer (FR-51, FR-52, §2.9 flow 2): the alert's state, what it
 * is about (each a drill-down on the breadcrumb), when it opened and was
 * last seen, whether it was emailed and who acked it, the runbook step, Ack
 * and Mute ▾, the view it points at, and an incident export around it.
 * URL-addressable as `oal=<id>`, the only link an alert email carries.
 */
export default function AlertDrawer({ alertId }: { alertId: string }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const { status } = useOpsShell()
  const [exporting, setExporting] = useState(false)
  const poll = useOpsPoll((signal) => opsApi.alert(alertId, signal), [alertId], DRAWER_SECONDS)
  const { stale, ageSeconds: age } = useStale(poll.updatedAt, DRAWER_SECONDS)
  const alert = poll.data?.alert ?? null
  const unavailable = poll.data?.available === false
  const missing = (poll.error?.kind === 'not_found' && !poll.data) || (!!poll.data && !alert && !unavailable)
  const canAct = alertActionsEnabled(status)
  return (
    <>
      {stale && alert ? <StaleChip ageSeconds={age} /> : null}
      {missing ? <p className="text-muted-foreground m-0 text-sm">{t('operations.alertDrawer.notFound')}</p> : null}
      {unavailable ? <p className="text-muted-foreground m-0 text-sm">{t('operations.unavailableShort')}</p> : null}
      {poll.error && !poll.data && !missing ? <ErrorPanel error={poll.error} onRetry={poll.reload} /> : null}
      {poll.error && poll.data ? <ErrorPanel error={poll.error} onRetry={poll.reload} compact /> : null}
      {!poll.data && !poll.error ? <Skeleton className="h-40 w-full" /> : null}
      {alert ? (
        <div className={stale ? 'ops-stale flex flex-col gap-4' : 'flex flex-col gap-4'}>
          <AlertSummary alert={alert} now={poll.updatedAt ?? 0} email={poll.data?.email ?? null} />
          <div className="flex flex-col gap-0.5 rounded-lg border border-dashed px-3 py-2 text-[13px]" role="note">
            <strong>{t('operations.alertDrawer.whatToDo')}</strong>
            <RunbookCell rule={alert.rule} text={alert.runbook ?? poll.data?.rule?.runbook} url={alert.runbook_url ?? poll.data?.rule?.runbook_url} />
          </div>
          <div className="flex flex-wrap gap-2">
            <OpsLink to={viewForTarget(nav.view, alertTarget(alert))} className="hover:bg-muted inline-flex min-h-8 items-center rounded-lg border px-2.5 text-[13px] no-underline">
              {t('operations.alertDrawer.open')}
            </OpsLink>
            {canAct && alert.state !== 'resolved' && !alert.acked_by ? (
              <AckButton alertId={alert.id} title={alertTitle(t, alert)} onOutcome={poll.reload} />
            ) : null}
            {canAct ? <MuteMenu rule={alert.rule} ruleTitle={ruleTitle(t, alert.rule, null, alert.label)} onOutcome={poll.reload} /> : null}
            {status?.operator ? (
              <button type="button" className="hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-2.5 text-[13px]" onClick={() => setExporting(true)}>
                <DownloadIcon className="size-3.5" aria-hidden="true" />
                {t('operations.incident.button')}
              </button>
            ) : null}
          </div>
          <AlertHistory items={poll.data?.history ?? []} now={poll.updatedAt ?? 0} />
          <p className="text-muted-foreground m-0 text-xs">{t('operations.alertDrawer.retention')}</p>
          {status?.operator ? <IncidentExportDialog open={exporting} onOpenChange={setExporting} alert={alert} /> : null}
        </div>
      ) : null}
    </>
  )
}
