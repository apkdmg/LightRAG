import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import { actionsEnabled } from '../actions'
import OpsLink from '../components/OpsLink'
import { Pager, Panel } from '../components/Panel'
import { StatusChip } from '../components/StatusChip'
import { WorkspaceActionBar } from '../components/WorkspaceActions'
import { useOpsNav, useOpsShell } from '../context'
import { EmailQueueItem, opsApi } from '../api'
import { ageSeconds, fmtAge, fmtClock, fmtCount, shortId, utcTitle } from '../format'
import { useOpsPoll } from '../poll'

/** The queue is scanned once a minute by one leased process (§3.4). */
const EMAIL_SECONDS = 60

/** ok / slot busy / failed as a state: a busy slot is a wait, a failure is a warning. */
const resultState = (cls: string | null | undefined) => (cls === 'ok' ? 'ok' : cls === 'slot_busy' ? 'paused' : cls ? 'warning' : 'unknown')

/**
 * The drainer column (§2.4 "worker 813 · draining", "worker 812 · backoff
 * 40s"): the live drainer from heartbeats, else the backoff deadline the
 * drainer recorded (KV resync:<ws>). The time left is measured against when
 * the data was confirmed, so it does not drift between polls.
 */
export function DrainerText({ item, now }: { item: Omit<EmailQueueItem, 'workspace'>; now: number | null }) {
  const { t } = useTranslation()
  const d = item.drainer
  const until = d?.backoff_until ?? item.backoff_until ?? null
  const left = until && now !== null ? -(ageSeconds(until, now) ?? 0) : null
  const parts: string[] = []
  if (d) {
    const who = typeof d.pid === 'number' ? t('operations.email.worker', { id: d.pid }) : d.worker_id ? t('operations.email.worker', { id: shortId(d.worker_id) }) : null
    if (who) parts.push(who)
    if (d.state && !(d.state === 'backoff' && left !== null && left > 0)) parts.push(t(`operations.workerDrawer.drainer.${d.state}`, { defaultValue: d.state }))
  }
  if (left !== null && left > 0) parts.push(t('operations.email.backoffLeft', { age: fmtAge(left) }))
  if (!parts.length) return <>–</>
  return <span title={utcTitle(until)}>{parts.join(' · ')}</span>
}

/** Drains of the last hour (FR-15): is the drainer making progress, or failing again and again? */
function DrainsText({ drains }: { drains: EmailQueueItem['drains_1h'] }) {
  const { t } = useTranslation()
  if (!drains) return <>–</>
  if (!drains.ok && !drains.failed && !drains.slot_busy) return <span className="text-muted-foreground">{t('operations.email.drainsNone')}</span>
  return (
    <span className="flex flex-wrap gap-x-1.5">
      {drains.ok ? <span>{t('operations.email.drainsOk', { count: drains.ok })}</span> : null}
      {drains.failed ? <span style={{ color: 'var(--ops-warn)' }}>{t('operations.email.drainsFailed', { count: drains.failed })}</span> : null}
      {drains.slot_busy ? <span className="text-muted-foreground">{t('operations.email.drainsBusy', { count: drains.slot_busy })}</span> : null}
    </span>
  )
}

/**
 * The email re-sync queue (§2.4, FR-15): records per workspace, how long the
 * oldest has waited, `.bad` files, the drainer (live, or its backoff
 * deadline), attempts, the drains of the last hour (ok / failed / slot busy)
 * and the last result, including `slot_busy` when the pipeline held the
 * workspace. Contents are never returned; the scanner
 * counts files without opening them. When actions are on, each row offers
 * "Drain email…" (at most once a minute per workspace).
 */
export default function EmailQueueView() {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const { status } = useOpsShell()
  const showActions = actionsEnabled(status)
  const [page, setPage] = useState(1)
  const poll = useOpsPoll((signal) => opsApi.emailQueue({ page }, signal), [page], EMAIL_SECONDS)
  const d = poll.data
  const col = (key: string) => t(`operations.email.col.${key}`)
  return (
    <Panel
      poll={poll}
      freshness={EMAIL_SECONDS}
      title={t('operations.email.title')}
      sub={
        d
          ? t('operations.email.summary', {
            records: fmtCount(d.total),
            workspaces: fmtCount(d.workspaces),
            longest: fmtAge(d.longest_pending_s ?? null),
            bad: fmtCount(d.bad ?? 0),
            scanned: d.scanned_at ? fmtAge(ageSeconds(d.scanned_at, poll.updatedAt ?? undefined)) : '–'
          })
          : undefined
      }
    >
      {(data) => (
        <div className={cn('flex flex-col gap-3', poll.loading && 'opacity-60')}>
          {!data.scanned_at ? <p className="text-muted-foreground m-0 text-sm">{t('operations.email.notScanned')}</p> : null}
          {data.items.length ? (
            <div className="overflow-x-auto">
              <table className="ops-stack w-full min-w-[1020px] text-[13px]">
                <thead>
                  <tr className="text-muted-foreground border-b text-xs">
                    <th className="p-2 text-left font-medium">{col('workspace')}</th>
                    <th className="p-2 text-right font-medium">{col('pending')}</th>
                    <th className="p-2 text-left font-medium">{col('since')}</th>
                    <th className="p-2 text-right font-medium">{col('bad')}</th>
                    <th className="p-2 text-left font-medium">{col('drainer')}</th>
                    <th className="p-2 text-right font-medium">{col('attempts')}</th>
                    <th className="p-2 text-left font-medium">{col('drains')}</th>
                    <th className="p-2 text-left font-medium">{col('last')}</th>
                    <th className="p-2 text-left font-medium">{col('lastOk')}</th>
                    {showActions ? (
                      <th className="p-2 text-right font-medium">
                        <span className="sr-only">{col('actions')}</span>
                      </th>
                    ) : null}
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((r) => {
                    const since = r.pending_since ? ageSeconds(r.pending_since, poll.updatedAt ?? undefined) : null
                    return (
                      <tr key={r.workspace} className="border-b align-top last:border-0">
                        <td className="p-2" data-label={col('workspace')}>
                          <OpsLink
                            to={{ ...nav.view, drawer: { kind: 'workspace', id: r.workspace } }}
                            onNavigate={() => nav.openDrawer({ kind: 'workspace', id: r.workspace })}
                            className="break-all underline-offset-2 hover:underline"
                          >
                            {r.workspace}
                          </OpsLink>
                        </td>
                        <td className="p-2 text-right tabular-nums" data-label={col('pending')}>
                          {fmtCount(r.pending)}
                        </td>
                        <td className="p-2 font-mono text-xs whitespace-nowrap" data-label={col('since')} title={utcTitle(r.pending_since)}>
                          {fmtAge(since)}
                        </td>
                        <td className="p-2 text-right tabular-nums" data-label={col('bad')}>
                          {typeof r.bad === 'number' && r.bad > 0 ? <StatusChip state="warning">{fmtCount(r.bad)}</StatusChip> : fmtCount(r.bad ?? 0)}
                        </td>
                        <td className="p-2 text-xs" data-label={col('drainer')}>
                          <DrainerText item={r} now={poll.updatedAt} />
                        </td>
                        <td className="p-2 text-right tabular-nums" data-label={col('attempts')} title={utcTitle(r.last_attempt_at)}>
                          {fmtCount(r.attempts ?? null)}
                        </td>
                        <td className="p-2 text-xs whitespace-nowrap" data-label={col('drains')}>
                          <DrainsText drains={r.drains_1h} />
                        </td>
                        <td className="p-2" data-label={col('last')}>
                          {r.last_result_class ? (
                            <span className="flex flex-col items-end gap-0.5 md:items-start">
                              <StatusChip state={resultState(r.last_result_class)}>
                                {t(`operations.email.result.${r.last_result_class}`, { defaultValue: r.last_result_class })}
                              </StatusChip>
                              {r.reason ? <span className="text-muted-foreground text-xs">{r.reason}</span> : null}
                            </span>
                          ) : (
                            '–'
                          )}
                        </td>
                        <td className="p-2 font-mono text-xs whitespace-nowrap" data-label={col('lastOk')} title={utcTitle(r.last_ok_at)}>
                          {fmtClock(r.last_ok_at ?? null)}
                        </td>
                        {showActions ? (
                          <td className="p-2 text-right" data-label={col('actions')}>
                            <WorkspaceActionBar
                              workspace={r.workspace}
                              kinds={['drain_email']}
                              ctx={{ busy: null, emailPending: r.pending }}
                              facts={{ emailPending: r.pending }}
                              onChanged={poll.reload}
                            />
                          </td>
                        ) : null}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          ) : data.scanned_at ? (
            <p className="text-muted-foreground m-0 text-sm">{t('operations.email.empty', { time: fmtClock(data.scanned_at) })}</p>
          ) : null}
          <Pager page={data.page ?? page} pageSize={data.page_size ?? Math.max(data.items.length, 50)} total={data.total_items ?? data.workspaces} onPage={setPage} />
          <p className="text-muted-foreground m-0 text-xs">{t('operations.email.privacy')}</p>
        </div>
      )}
    </Panel>
  )
}
