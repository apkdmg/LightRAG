import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Seg } from '@/components/admin'
import { cn } from '@/lib/utils'
import { OpsEvent, opsApi } from '../api'
import { Panel } from '../components/Panel'
import { EventLabel } from '../components/TimelinePanel'
import { useOpsNav } from '../context'
import { fmtStamp, utcTitle } from '../format'
import { useOpsPoll } from '../poll'

/** §3.10: Activity refreshes every 60 s. */
const ACTIVITY_SECONDS = 60

const WINDOWS = { '6h': 6, '24h': 24, '7d': 168 } as const
type EventWindow = keyof typeof WINDOWS

/** Markers that mean something went wrong. */
const BAD_KINDS = new Set(['provider_rejection', 'run_orphaned'])

export function EventsTable({ events, now }: { events: OpsEvent[]; now: number | null }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const col = (key: string) => t(`operations.events.col.${key}`)
  const rows = [...events].sort((a, b) => Date.parse(b.ts) - Date.parse(a.ts))
  return (
    <div className="overflow-x-auto">
      <table className="ops-stack w-full min-w-[640px] text-[13px]">
        <thead>
          <tr className="text-muted-foreground border-b text-xs">
            <th className="p-2 text-left font-medium">{col('time')}</th>
            <th className="p-2 text-left font-medium">{col('event')}</th>
            <th className="p-2 text-left font-medium">{col('workspace')}</th>
            <th className="p-2 text-left font-medium">{col('source')}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((e, i) => (
            <tr key={`${e.ts}-${e.kind}-${i}`} className="border-b align-top last:border-0">
              <td className="p-2 font-mono text-xs whitespace-nowrap" data-label={col('time')} title={utcTitle(e.ts)}>
                {fmtStamp(e.ts, now ?? undefined)}
              </td>
              <td className={cn('p-2', BAD_KINDS.has(e.kind) && 'text-[color:var(--ops-crit)]')} data-label={col('event')}>
                <EventLabel e={{ ...e, workspace: null }} />
              </td>
              <td className="p-2 break-all" data-label={col('workspace')}>
                {e.workspace ? (
                  <button type="button" className="text-left underline-offset-2 hover:underline" onClick={() => nav.openDrawer({ kind: 'workspace', id: e.workspace as string })}>
                    {e.workspace}
                  </button>
                ) : (
                  '–'
                )}
              </td>
              <td className="text-muted-foreground p-2 text-xs" data-label={col('source')}>
                {e.source ? t(`operations.events.source.${e.source}`, { defaultValue: e.source }) : '–'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/**
 * Activity › System events (§2.8 section 3, flow 6): deployments, model
 * changes, provider rejections, ingestion, orphaned runs and admin actions,
 * merged at read time from the usage events, the audit log and the run
 * table. "Emails stopped arriving" is answered here: no ingest since 09:40.
 */
export default function EventsView() {
  const { t } = useTranslation()
  const [win, setWin] = useState<EventWindow>('24h')
  const poll = useOpsPoll(
    (signal) => {
      const to = Date.now()
      return opsApi.events({ from: new Date(to - WINDOWS[win] * 3600_000).toISOString(), to: new Date(to).toISOString() }, signal)
    },
    [win],
    ACTIVITY_SECONDS
  )
  return (
    <Panel
      poll={poll}
      freshness={ACTIVITY_SECONDS}
      title={t('operations.events.title')}
      sub={t('operations.events.sub')}
      actions={
        <Seg
          value={win}
          onChange={setWin}
          label={t('operations.events.windowLabel')}
          options={(Object.keys(WINDOWS) as EventWindow[]).map((w) => ({ value: w, label: t(`operations.events.window.${w}`) }))}
        />
      }
    >
      {(events) =>
        events.length ? (
          <EventsTable events={events} now={poll.updatedAt} />
        ) : (
          <p className="text-muted-foreground m-0 text-sm">{t('operations.events.none', { window: t(`operations.events.window.${win}`) })}</p>
        )
      }
    </Panel>
  )
}
