import { Fragment, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ArrowRightIcon, ExternalLinkIcon, WrenchIcon } from 'lucide-react'
import Input from '@/components/ui/Input'
import { Seg } from '@/components/admin'
import { cn } from '@/lib/utils'
import { AlertEmailState, AlertMute, AlertRule, AlertRules, AlertsPage, OpsAlert, opsApi } from '../api'
import { ActionOutcome } from '../actions'
import { AckButton, EndMaintenanceDialog, MaintenanceDialog, MuteMenu, UnmuteButton } from '../components/AlertActions'
import ConfirmActionDialog from '../components/ConfirmActionDialog'
import KeysetPager from '../components/KeysetPager'
import OpsLink from '../components/OpsLink'
import { Panel } from '../components/Panel'
import { StatusChip, ToneChip } from '../components/StatusChip'
import { useOpsNav, useOpsOverview, useOpsShell } from '../context'
import { ageSeconds, fmtAge, fmtClock, fmtStamp, utcTitle } from '../format'
import { viewForTarget } from '../hashState'
import { useNow } from '../hooks'
import { Cursor, FIRST_PAGE } from '../keyset'
import { useOpsPoll } from '../poll'
import { alertActionsEnabled, alertTitle, ruleTitle, runbookText, safeHttpUrl, sendAlertAction } from './alertActions'
import {
  RulesDraft,
  activeMutes,
  alertOpenSeconds,
  alertTarget,
  changeLines,
  draftErrors,
  draftOf,
  emailStatus,
  maintenanceActive,
  muteOf,
  rulesChange,
  sortOpenAlerts
} from './alerts'

/** §3.10: Activity refreshes every 60 s. */
const ACTIVITY_SECONDS = 60

export type AlertsSub = 'open' | 'history' | 'mutes' | 'rules' | 'maintenance'

const linkButton = 'text-left underline-offset-2 hover:underline break-all'

/** What alert email does on this server: record-only, on with N recipients, or on with nobody to send to. */
export function EmailModeNote({ email }: { email: AlertEmailState | null }) {
  const { t } = useTranslation()
  if (!email || email.enabled === null) return null
  let text: string
  if (!email.enabled) text = t('operations.alerts.emailMode.recordOnly')
  else if (email.recipients === 0) text = t('operations.alerts.emailMode.noRecipients')
  else if (typeof email.recipients === 'number') text = t('operations.alerts.emailMode.on', { count: email.recipients })
  else text = t('operations.alerts.emailMode.onUnknown')
  return (
    <p className={cn('m-0 rounded-lg border border-dashed px-3 py-2 text-[13px]', email.enabled && email.recipients === 0 && 'text-[color:var(--ops-warn)]')} role="note">
      {text}
      {email.enabled && typeof email.failures === 'number' && email.failures > 0 ? (
        <span className="text-[color:var(--ops-warn)]"> {t('operations.alerts.emailMode.failures', { count: email.failures })}</span>
      ) : null}
    </p>
  )
}

/** Workspace, dependency, worker or run: each opens its drawer on the breadcrumb. */
function ScopeCell({ alert }: { alert: OpsAlert }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const s = alert.scope
  const parts: React.ReactNode[] = []
  if (s.workspace) {
    parts.push(
      <button key="ws" type="button" className={linkButton} onClick={() => nav.openDrawer({ kind: 'workspace', id: s.workspace as string })}>
        {s.workspace}
      </button>
    )
  }
  if (s.dep) {
    parts.push(
      <button key="dep" type="button" className={linkButton} onClick={() => nav.openDrawer({ kind: 'dependency', id: s.dep as string })}>
        {t(`operations.dep.${s.dep}`, { defaultValue: s.dep })}
      </button>
    )
  }
  if (s.worker_id) {
    parts.push(
      <button key="wk" type="button" className={`${linkButton} font-mono text-xs`} onClick={() => nav.openDrawer({ kind: 'worker', id: s.worker_id as string })}>
        {t('operations.alerts.scope.worker', { id: s.worker_id })}
      </button>
    )
  }
  if (s.run_id) {
    parts.push(
      <button key="run" type="button" className={`${linkButton} font-mono text-xs`} onClick={() => nav.openDrawer({ kind: 'run', id: s.run_id as string })}>
        {t('operations.alerts.scope.run', { id: (s.run_id as string).slice(0, 8) })}
      </button>
    )
  }
  if (!parts.length) return <span className="text-muted-foreground">{t('operations.alerts.scope.host')}</span>
  return <span className="flex flex-col items-end gap-0.5 md:items-start">{parts}</span>
}

/** The "Notified" cell (FR-62): sent with its time, or why not. */
export function NotifiedCell({ alert, email, now }: { alert: OpsAlert; email: AlertEmailState | null; now: number | null }) {
  const { t } = useTranslation()
  const status = emailStatus(alert, email)
  return (
    <span className="flex flex-col items-end md:items-start">
      <span className={cn(status === 'failed' && 'text-[color:var(--ops-crit)]', status === 'pending' && 'text-[color:var(--ops-warn)]')}>
        {t(`operations.alerts.email.${status}`)}
      </span>
      {alert.last_emailed_at ? (
        <span className="text-muted-foreground font-mono text-xs" title={utcTitle(alert.last_emailed_at)}>
          {fmtStamp(alert.last_emailed_at, now ?? undefined)}
        </span>
      ) : null}
    </span>
  )
}

function AckedCell({ alert, now }: { alert: OpsAlert; now: number | null }) {
  if (!alert.acked_by) return <span className="text-muted-foreground">–</span>
  return (
    <span className="flex flex-col items-end md:items-start" title={alert.ack_note ?? undefined}>
      <span className="break-all">{alert.acked_by}</span>
      {alert.acked_at ? (
        <span className="text-muted-foreground font-mono text-xs" title={utcTitle(alert.acked_at)}>
          {fmtStamp(alert.acked_at, now ?? undefined)}
        </span>
      ) : null}
      {alert.ack_note ? <span className="text-muted-foreground max-w-[220px] truncate text-xs">{alert.ack_note}</span> : null}
    </span>
  )
}

/** The runbook step, with its page when OPS_RUNBOOK_BASE_URL is set (FR-60). */
export function RunbookCell({ rule, text, url }: { rule: string; text?: string | null; url?: string | null }) {
  const { t } = useTranslation()
  const step = runbookText(t, rule, text)
  const href = safeHttpUrl(url)
  if (!step && !href) return <span className="text-muted-foreground">–</span>
  return (
    <span className="flex max-w-[320px] flex-col items-end gap-0.5 md:items-start">
      {step ? <span className="text-left">{step}</span> : null}
      {href ? (
        <a href={href} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-xs underline underline-offset-2">
          {t('operations.alerts.runbookLink')}
          <ExternalLinkIcon className="size-3" aria-hidden="true" />
        </a>
      ) : null}
    </span>
  )
}

/**
 * The alerts table (§2.8): rule, severity, scope, first and last seen,
 * notified, acked by, runbook and Mute ▾ / Ack. Open alerts list worst
 * first; history shows when each resolved and how long it was open. The
 * title opens the Alert drawer; the arrow goes where the alert points.
 */
export function AlertTable({
  alerts,
  mode,
  now,
  email,
  mutes = [],
  canAct,
  onChanged
}: {
  alerts: OpsAlert[]
  mode: 'open' | 'history'
  /** When the rows were confirmed (epoch ms), for ages and mutes. */
  now: number
  email: AlertEmailState | null
  mutes?: AlertMute[]
  canAct: boolean
  onChanged?: (o: ActionOutcome) => void
}) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const col = (key: string) => t(`operations.alerts.col.${key}`)
  const rows = mode === 'open' ? sortOpenAlerts(alerts) : alerts
  const at = now
  return (
    <div className="overflow-x-auto">
      <table className="ops-stack w-full min-w-[1080px] text-[13px]">
        <thead>
          <tr className="text-muted-foreground border-b text-xs">
            <th className="p-2 text-left font-medium">{col('severity')}</th>
            <th className="p-2 text-left font-medium">{col('alert')}</th>
            <th className="p-2 text-left font-medium">{col('scope')}</th>
            <th className="p-2 text-left font-medium">{mode === 'open' ? col('seen') : col('opened')}</th>
            {mode === 'history' ? <th className="p-2 text-left font-medium">{col('resolved')}</th> : null}
            <th className="p-2 text-left font-medium">{col('notified')}</th>
            <th className="p-2 text-left font-medium">{col('acked')}</th>
            {mode === 'open' ? <th className="p-2 text-left font-medium">{col('runbook')}</th> : null}
            {mode === 'open' ? <th className="p-2 text-right font-medium">{col('actions')}</th> : null}
          </tr>
        </thead>
        <tbody>
          {rows.map((a) => {
            const title = alertTitle(t, a)
            const mute = muteOf(a.rule, mutes, at)
            const open = alertOpenSeconds(a, at)
            return (
              <tr key={a.id} className="border-b align-top last:border-0">
                <td className="p-2" data-label={col('severity')}>
                  <StatusChip state={a.severity} />
                </td>
                <td className="p-2" data-label={col('alert')}>
                  <span className="flex flex-col items-end gap-0.5 md:items-start">
                    <span className="inline-flex items-start gap-1.5">
                      <button type="button" className="text-left font-medium underline-offset-2 hover:underline" onClick={() => nav.openDrawer({ kind: 'alert', id: a.id })}>
                        {title}
                      </button>
                      <OpsLink
                        to={viewForTarget(nav.view, alertTarget(a))}
                        className="text-muted-foreground hover:text-foreground mt-0.5"
                        aria-label={t('operations.alerts.goTo', { title })}
                      >
                        <ArrowRightIcon className="size-3.5" aria-hidden="true" />
                      </OpsLink>
                    </span>
                    <span className="text-muted-foreground font-mono text-xs">{a.rule}</span>
                    {mute ? (
                      <ToneChip tone="unknown" icon="pause">
                        {t('operations.alerts.mutedUntil', { time: fmtClock(mute.until).slice(0, 5) })}
                      </ToneChip>
                    ) : null}
                  </span>
                </td>
                <td className="p-2" data-label={col('scope')}>
                  <ScopeCell alert={a} />
                </td>
                <td className="p-2 font-mono text-xs whitespace-nowrap" data-label={mode === 'open' ? col('seen') : col('opened')}>
                  <span className="flex flex-col items-end md:items-start">
                    <span title={utcTitle(a.first_seen)}>{fmtStamp(a.first_seen, at)}</span>
                    {mode === 'open' ? (
                      <span className="text-muted-foreground" title={utcTitle(a.last_seen)}>
                        {t('operations.alerts.lastSeen', { age: fmtAge(ageSeconds(a.last_seen, at)) })}
                      </span>
                    ) : null}
                  </span>
                </td>
                {mode === 'history' ? (
                  <td className="p-2 font-mono text-xs whitespace-nowrap" data-label={col('resolved')}>
                    <span className="flex flex-col items-end md:items-start">
                      <span title={utcTitle(a.resolved_at)}>{fmtStamp(a.resolved_at, at)}</span>
                      <span className="text-muted-foreground">{t('operations.alerts.openFor', { age: fmtAge(open) })}</span>
                    </span>
                  </td>
                ) : null}
                <td className="p-2" data-label={col('notified')}>
                  <NotifiedCell alert={a} email={email} now={now} />
                </td>
                <td className="p-2" data-label={col('acked')}>
                  <AckedCell alert={a} now={now} />
                </td>
                {mode === 'open' ? (
                  <td className="p-2 text-xs" data-label={col('runbook')}>
                    <RunbookCell rule={a.rule} text={a.runbook} url={a.runbook_url} />
                  </td>
                ) : null}
                {mode === 'open' ? (
                  <td className="p-2" data-label={col('actions')}>
                    {canAct ? (
                      <span className="flex flex-wrap justify-end gap-1.5">
                        {!a.acked_by && a.state !== 'resolved' ? <AckButton alertId={a.id} title={title} compact onOutcome={onChanged} /> : null}
                        <MuteMenu rule={a.rule} ruleTitle={ruleTitle(t, a.rule, null, a.label)} compact onOutcome={onChanged} />
                      </span>
                    ) : (
                      <span className="text-muted-foreground">–</span>
                    )}
                  </td>
                ) : null}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function AlertList({
  mode,
  email,
  mutes,
  canAct,
  onChanged
}: {
  mode: 'open' | 'history'
  email: AlertEmailState | null
  mutes: AlertMute[]
  canAct: boolean
  /** An ack or mute went through: what else shows it refreshes too (mutes, Needs attention). */
  onChanged?: () => void
}) {
  const { t } = useTranslation()
  const [cursor, setCursor] = useState<Cursor>(FIRST_PAGE)
  const state = mode === 'open' ? 'open' : 'resolved'
  const poll = useOpsPoll((signal) => opsApi.alerts({ state, before: cursor.before }, signal), [state, cursor.before], ACTIVITY_SECONDS)
  const changed = (o: ActionOutcome) => {
    if (o.tone === 'refused' && o.error) return
    poll.reload()
    onChanged?.()
  }
  return (
    <Panel
      poll={poll}
      freshness={ACTIVITY_SECONDS}
      title={mode === 'open' ? t('operations.alerts.openTitle') : t('operations.alerts.historyTitle')}
      sub={mode === 'open' ? t('operations.alerts.openSub') : t('operations.alerts.historySub')}
    >
      {(data) => (
        <div className={cn('flex flex-col gap-3', poll.loading && 'opacity-60')}>
          <AlertListBody data={data} mode={mode} now={poll.updatedAt ?? 0} email={email} mutes={mutes} canAct={canAct} onChanged={changed} />
          {data.enabled !== false ? <KeysetPager next={data.next_before} cursor={cursor} onCursor={setCursor} /> : null}
        </div>
      )}
    </Panel>
  )
}

/** One page of alerts: the table, an honest empty state, or "turned off" from a server whose evaluator is off. */
export function AlertListBody({
  data,
  mode,
  now,
  email,
  mutes,
  canAct,
  onChanged
}: {
  data: AlertsPage
  mode: 'open' | 'history'
  now: number
  email: AlertEmailState | null
  mutes: AlertMute[]
  canAct: boolean
  onChanged?: (o: ActionOutcome) => void
}) {
  const { t } = useTranslation()
  if (data.enabled === false) return <p className="text-muted-foreground m-0 text-sm">{t('operations.alerts.off')}</p>
  if (!data.items.length) {
    return <p className="text-muted-foreground m-0 text-sm">{mode === 'open' ? t('operations.alerts.noneOpen') : t('operations.alerts.noneHistory')}</p>
  }
  return <AlertTable alerts={data.items} mode={mode} now={now} email={data.email ?? email} mutes={mutes} canAct={canAct} onChanged={onChanged} />
}

/** Rules muted now, with who, why and until when; Unmute… ends one. */
export function MutesList({
  mutes,
  rules = [],
  now,
  canAct,
  onChanged
}: {
  mutes: AlertMute[]
  /** The catalogue, for the rules' names. */
  rules?: AlertRule[]
  now: number
  canAct: boolean
  onChanged?: () => void
}) {
  const { t } = useTranslation()
  const active = activeMutes(mutes, now)
  const titleOf = (rule: string) => ruleTitle(t, rule, null, rules.find((r) => r.rule === rule)?.label)
  if (!active.length) return <p className="text-muted-foreground m-0 text-sm">{t('operations.alerts.mutes.none')}</p>
  return (
    <ul className="m-0 flex list-none flex-col gap-2 p-0">
      {active.map((m) => (
        <li key={m.rule} className="flex flex-wrap items-start justify-between gap-3 rounded-lg border px-3 py-2">
          <span className="flex min-w-0 flex-col gap-0.5 text-[13px]">
            <span className="font-medium">{titleOf(m.rule)}</span>
            <span className="text-muted-foreground font-mono text-xs">{m.rule}</span>
            <span className="text-xs" title={utcTitle(m.until)}>
              {t('operations.alerts.mutes.until', { time: fmtStamp(m.until, now), left: fmtAge(-(ageSeconds(m.until, now) ?? 0)) })}
            </span>
            {m.by || m.reason ? (
              <span className="text-muted-foreground text-xs break-words">{t('operations.alerts.mutes.by', { by: m.by ?? '–', reason: m.reason ?? '–' })}</span>
            ) : null}
          </span>
          {canAct ? <UnmuteButton rule={m.rule} ruleTitle={titleOf(m.rule)} onOutcome={onChanged} /> : null}
        </li>
      ))}
    </ul>
  )
}

const thresholdLabel = (t: ReturnType<typeof useTranslation>['t'], key: string) => t(`operations.alerts.threshold.${key}`, { defaultValue: key.replace(/_/g, ' ') })

/**
 * The rule catalogue (§1.7) with each rule's severities, whether it is
 * emailed, its condition, current thresholds and runbook. Operators edit the
 * thresholds in place: each value is checked as it is typed (a number above
 * zero, whole where it must be, inside the server's bounds, the warning
 * level before the serious one), and Save… sends only what changed, with a
 * reason, to the audited PUT. Other admins read the same table.
 */
export function RulesEditor({
  data,
  operator,
  canAct,
  now,
  onSaved
}: {
  data: AlertRules
  operator: boolean
  canAct: boolean
  now: number
  onSaved?: () => void
}) {
  const { t } = useTranslation()
  const [edits, setEdits] = useState<RulesDraft>({})
  const [saving, setSaving] = useState(false)
  const base = draftOf(data.rules)
  const draft: RulesDraft = Object.fromEntries(Object.entries(base).map(([rule, values]) => [rule, { ...values, ...edits[rule] }]))
  const errors = draftErrors(data.rules, draft)
  const change = rulesChange(data.rules, draft)
  const errorCount = Object.values(errors).reduce((n, e) => n + Object.keys(e).length, 0)
  const editable = operator && canAct
  const setValue = (rule: string, key: string, value: string) => setEdits((e) => ({ ...e, [rule]: { ...e[rule], [key]: value } }))
  const col = (key: string) => t(`operations.alerts.rules.col.${key}`)
  const lines = change ? changeLines(data.rules, change) : []

  return (
    <div className="flex flex-col gap-3">
      {!operator ? <p className="text-muted-foreground m-0 text-[13px]">{t('operations.alerts.rules.operatorOnly')}</p> : null}
      <div className="overflow-x-auto">
        <table className="ops-stack w-full min-w-[1080px] text-[13px]">
          <thead>
            <tr className="text-muted-foreground border-b text-xs">
              <th className="p-2 text-left font-medium">{col('rule')}</th>
              <th className="p-2 text-left font-medium">{col('severity')}</th>
              <th className="p-2 text-left font-medium">{col('email')}</th>
              <th className="p-2 text-left font-medium">{col('thresholds')}</th>
              <th className="p-2 text-left font-medium">{col('runbook')}</th>
              <th className="p-2 text-right font-medium">{col('mute')}</th>
            </tr>
          </thead>
          <tbody>
            {data.rules.map((r) => (
              <RuleRow
                key={r.rule}
                rule={r}
                values={draft[r.rule] ?? {}}
                errors={errors[r.rule] ?? {}}
                editable={editable}
                mute={muteOf(r.rule, data.mutes, now)}
                canAct={canAct}
                onValue={(key, value) => setValue(r.rule, key, value)}
                onChanged={onSaved}
              />
            ))}
          </tbody>
        </table>
      </div>
      {editable ? (
        <div className="flex flex-wrap items-center justify-end gap-2">
          {errorCount ? (
            <span className="me-auto text-xs text-[color:var(--ops-warn)]" role="alert">
              {t('operations.alerts.rules.errors', { count: errorCount })}
            </span>
          ) : change ? (
            <span className="text-muted-foreground me-auto text-xs">{t('operations.alerts.rules.changed', { count: lines.length })}</span>
          ) : null}
          <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px] disabled:opacity-50" disabled={!Object.keys(edits).length} onClick={() => setEdits({})}>
            {t('operations.alerts.rules.reset')}
          </button>
          <button
            type="button"
            className="bg-primary text-primary-foreground min-h-9 rounded-lg px-3 text-[13px] hover:opacity-90 disabled:opacity-50"
            disabled={!change || errorCount > 0}
            onClick={() => setSaving(true)}
          >
            {t('operations.alerts.rules.save')}
          </button>
        </div>
      ) : null}
      {editable ? (
        <ConfirmActionDialog
          open={saving}
          onOpenChange={setSaving}
          action="rules"
          title={t('operations.alerts.rules.saveTitle')}
          body={<p className="m-0">{t('operations.alerts.rules.saveBody')}</p>}
          extra={
            <ul className="bg-muted/50 m-0 flex list-none flex-col gap-1 rounded-md border px-3 py-2 text-[13px]">
              {lines.map((l) => (
                <li key={`${l.rule}:${l.key}`}>
                  <span className="font-mono text-xs">{l.rule}</span> · {thresholdLabel(t, l.key)}: <span className="font-mono">{l.from}</span> →{' '}
                  <span className="font-mono font-medium">{l.to}</span>
                </li>
              ))}
            </ul>
          }
          ready={!!change && errorCount === 0}
          confirmLabel={t('operations.alerts.rules.saveConfirm')}
          onConfirm={({ reason }) => sendAlertAction('rules', () => opsApi.saveRules(change ?? {}, reason))}
          onOutcome={(o) => {
            if (o.tone === 'success') {
              setEdits({})
              onSaved?.()
            }
          }}
        />
      ) : null}
    </div>
  )
}

function RuleRow({
  rule,
  values,
  errors,
  editable,
  mute,
  canAct,
  onValue,
  onChanged
}: {
  rule: AlertRule
  values: Record<string, string>
  errors: Record<string, string>
  editable: boolean
  mute: AlertMute | null
  canAct: boolean
  onValue: (key: string, value: string) => void
  onChanged?: () => void
}) {
  const { t } = useTranslation()
  const col = (key: string) => t(`operations.alerts.rules.col.${key}`)
  const title = ruleTitle(t, rule.rule, null, rule.label)
  return (
    <tr className="border-b align-top last:border-0">
      <td className="p-2" data-label={col('rule')}>
        <span className="flex flex-col items-end gap-0.5 md:items-start">
          <span className="font-medium">{title}</span>
          <span className="text-muted-foreground font-mono text-xs">{rule.rule}</span>
          {rule.condition ? <span className="text-muted-foreground text-xs">{rule.condition}</span> : null}
        </span>
      </td>
      <td className="p-2" data-label={col('severity')}>
        <span className="flex flex-wrap justify-end gap-1 md:justify-start">
          {rule.severities.length ? rule.severities.map((s) => <StatusChip key={s} state={s} />) : <span className="text-muted-foreground">–</span>}
        </span>
      </td>
      <td className="p-2 text-xs" data-label={col('email')}>
        {rule.email ? t('operations.alerts.rules.emailed') : t('operations.alerts.rules.inAppOnly')}
      </td>
      <td className="p-2" data-label={col('thresholds')}>
        {rule.thresholds.length ? (
          <span className="flex flex-col items-end gap-2 md:items-start">
            {rule.thresholds.map((th) => {
              const id = `ops-th-${rule.rule}-${th.key}`
              const err = errors[th.key]
              return (
                <span key={th.key} className="flex flex-col items-end gap-0.5 md:items-start">
                  <label htmlFor={editable ? id : undefined} className="text-muted-foreground text-xs">
                    {thresholdLabel(t, th.key)}
                    {th.unit ? ` (${th.unit})` : ''}
                  </label>
                  {editable ? (
                    <Input
                      id={id}
                      inputMode="decimal"
                      value={values[th.key] ?? ''}
                      onChange={(e) => onValue(th.key, e.target.value)}
                      aria-invalid={err ? true : undefined}
                      aria-describedby={err ? `${id}-err` : undefined}
                      className="h-8 w-28 font-mono text-[13px]"
                    />
                  ) : (
                    <span className="font-mono">{values[th.key] ?? '–'}</span>
                  )}
                  {err ? (
                    <span id={`${id}-err`} className="text-xs text-[color:var(--ops-warn)]">
                      {t(`operations.alerts.thresholdError.${err}`, { min: th.min ?? '', max: th.max ?? '' })}
                    </span>
                  ) : th.default !== null && String(th.default) !== values[th.key] ? (
                    <span className="text-muted-foreground text-xs">{t('operations.alerts.rules.default', { value: th.default })}</span>
                  ) : null}
                </span>
              )
            })}
          </span>
        ) : (
          <span className="text-muted-foreground text-xs">{t('operations.alerts.rules.fixed')}</span>
        )}
      </td>
      <td className="p-2 text-xs" data-label={col('runbook')}>
        <RunbookCell rule={rule.rule} text={rule.runbook} url={rule.runbook_url} />
      </td>
      <td className="p-2" data-label={col('mute')}>
        <span className="flex flex-col items-end gap-1">
          {mute ? (
            <ToneChip tone="unknown" icon="pause">
              {t('operations.alerts.mutedUntil', { time: fmtClock(mute.until).slice(0, 5) })}
            </ToneChip>
          ) : null}
          {canAct ? <MuteMenu rule={rule.rule} ruleTitle={title} compact onOutcome={onChanged} /> : null}
        </span>
      </td>
    </tr>
  )
}

/** Maintenance mode (FR-64): its state, and Start… or End…. */
export function MaintenanceView({ data, now, canAct, onChanged }: { data: AlertRules; now: number; canAct: boolean; onChanged?: () => void }) {
  const { t } = useTranslation()
  const overview = useOpsOverview()
  const [starting, setStarting] = useState(false)
  const [ending, setEnding] = useState(false)
  const m = data.maintenance ?? overview?.data?.maintenance ?? null
  const on = maintenanceActive(m, now)
  const changed = () => {
    onChanged?.()
    overview?.reload()
  }
  return (
    <div className="flex flex-col gap-3">
      <p className="text-muted-foreground m-0 text-[13px]">{t('operations.maintenance.explain')}</p>
      {on && m ? (
        <div className="flex flex-wrap items-start justify-between gap-3 rounded-lg border px-3 py-2">
          <span className="flex items-start gap-2 text-[13px]">
            <WrenchIcon className="mt-0.5 size-4 flex-none" aria-hidden="true" />
            <span className="flex flex-col">
              <span className="font-medium">{t('operations.maintenance.onUntil', { time: fmtStamp(m.until, now) })}</span>
              {m.by || m.reason ? <span className="text-muted-foreground text-xs break-words">{t('operations.maintenance.setBy', { by: m.by ?? '–', reason: m.reason ?? '–' })}</span> : null}
            </span>
          </span>
          {canAct ? (
            <button type="button" className="hover:bg-muted min-h-8 rounded-lg border px-2.5 text-[13px]" onClick={() => setEnding(true)}>
              {t('operations.maintenance.endButton')}
            </button>
          ) : null}
        </div>
      ) : (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-dashed px-3 py-2">
          <span className="text-[13px]">{t('operations.maintenance.off')}</span>
          {canAct ? (
            <button type="button" className="hover:bg-muted min-h-8 rounded-lg border px-2.5 text-[13px]" onClick={() => setStarting(true)}>
              {t('operations.maintenance.startButton')}
            </button>
          ) : null}
        </div>
      )}
      {canAct ? (
        <>
          <MaintenanceDialog open={starting} onOpenChange={setStarting} onOutcome={(o) => o.tone === 'success' && changed()} />
          <EndMaintenanceDialog open={ending} onOpenChange={setEnding} onOutcome={(o) => o.tone === 'success' && changed()} />
        </>
      ) : null}
    </div>
  )
}

/**
 * Activity › Alerts (§2.8, FR-60–FR-64): open alerts, history, mutes, the
 * rule catalogue and maintenance mode. The rules answer (60 s) carries the
 * email state, mutes and maintenance that every sub-view uses; the open and
 * history lists poll on their own while shown. Alerts are recorded even when
 * email is off (record-only) or held (mute, maintenance).
 */
export default function AlertsView() {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const overview = useOpsOverview()
  const [sub, setSub] = useState<AlertsSub>('open')
  const [fadeSub, setFadeSub] = useState<AlertsSub | null>(null)
  const enabled = status?.features?.alerts === true
  const rules = useOpsPoll((signal) => opsApi.alertRules(signal), [], ACTIVITY_SECONDS, { enabled: enabled && !!status?.available })
  const now = useNow(60_000)
  const canAct = alertActionsEnabled(status)
  const operator = !!status?.operator

  if (status && !status.available) return <p className="text-muted-foreground m-0 text-sm">{t('operations.unavailableShort')}</p>
  if (status && !enabled) {
    return (
      <section className="bg-card flex flex-col gap-1 rounded-[10px] border p-4">
        <h2 className="m-0 text-[15px] font-semibold">{t('operations.alerts.title')}</h2>
        <p className="text-muted-foreground m-0 text-sm">{t('operations.alerts.off')}</p>
      </section>
    )
  }

  const data = rules.data
  const mutes = data?.mutes ?? []
  const open = overview?.data?.alerts?.open
  const muted = activeMutes(mutes, now).length
  const maintenance = maintenanceActive(data?.maintenance ?? null, now)
  const select = (v: AlertsSub) => {
    setFadeSub(v)
    setSub(v)
  }
  const reloadAround = () => {
    rules.reload()
    overview?.reload()
  }
  const options: { value: AlertsSub; label: string }[] = [
    { value: 'open', label: typeof open === 'number' && open > 0 ? `${t('operations.alerts.sub.open')} ${open}` : t('operations.alerts.sub.open') },
    { value: 'history', label: t('operations.alerts.sub.history') },
    { value: 'mutes', label: muted ? `${t('operations.alerts.sub.mutes')} ${muted}` : t('operations.alerts.sub.mutes') },
    { value: 'rules', label: t('operations.alerts.sub.rules') },
    { value: 'maintenance', label: maintenance ? `${t('operations.alerts.sub.maintenance')} · ${t('operations.alerts.sub.on')}` : t('operations.alerts.sub.maintenance') }
  ]

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="m-0 text-[15px] font-semibold">{t('operations.alerts.title')}</h2>
        <div className="overflow-x-auto">
          <Seg value={sub} onChange={select} options={options} label={t('operations.alerts.sub.label')} />
        </div>
      </div>
      <EmailModeNote email={data?.email ?? null} />
      <div key={sub} className={fadeSub === sub ? 'ops-fade-in' : undefined}>
        {sub === 'open' || sub === 'history' ? (
          <AlertList mode={sub} email={data?.email ?? null} mutes={mutes} canAct={canAct} onChanged={reloadAround} />
        ) : (
          <Panel
            poll={rules}
            freshness={ACTIVITY_SECONDS}
            title={t(`operations.alerts.${sub}Title`)}
            sub={sub === 'rules' ? t('operations.alerts.rules.sub') : undefined}
          >
            {(d) => (
              <Fragment>
                {sub === 'mutes' ? <MutesList mutes={d.mutes} rules={d.rules} now={now} canAct={canAct} onChanged={rules.reload} /> : null}
                {sub === 'rules' ? (
                  <RulesEditor data={d} operator={typeof d.operator === 'boolean' ? d.operator : operator} canAct={canAct} now={now} onSaved={rules.reload} />
                ) : null}
                {sub === 'maintenance' ? <MaintenanceView data={d} now={now} canAct={canAct} onChanged={rules.reload} /> : null}
              </Fragment>
            )}
          </Panel>
        )}
      </div>
    </div>
  )
}
