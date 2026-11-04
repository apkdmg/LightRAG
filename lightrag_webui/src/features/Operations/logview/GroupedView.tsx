import { useTranslation } from 'react-i18next'
import { opsApi } from '../api'
import { LevelChip } from '../components/LogRow'
import OpsLink from '../components/OpsLink'
import { Panel } from '../components/Panel'
import { RedactedText } from '../components/RedactedSpan'
import { fmtCount, fmtStamp, utcTitle } from '../format'
import { LogFilter, WINDOW_SECONDS, callSite, effectiveWindow, logsView } from '../logs'
import { useOpsPoll } from '../poll'

/** Groups change slowly; a minute keeps the counts honest without load. */
const GROUPS_SECONDS = 60

/**
 * Rows grouped by call site (FR-23): count, level, where it was logged, the
 * exception type, a redacted sample, first and last seen, and how many
 * workspaces hit it. "Show" opens the stream filtered on that fingerprint.
 * The server groups over the window and workspace; the other filters
 * belong to the stream.
 */
export default function GroupedView({ filter }: { filter: LogFilter }) {
  const { t } = useTranslation()
  const win = effectiveWindow({ ...filter, q: '' })
  const hours = Math.max(1, Math.ceil(WINDOW_SECONDS[win] / 3600))
  const params = { hours, workspace: filter.workspace || undefined, level: filter.level }
  const poll = useOpsPoll((signal) => opsApi.logGroups(params, signal), [params], GROUPS_SECONDS)
  // A server without the grouped route answers 404 (or 422 when /logs/{id} catches the path).
  const missing = !poll.data && (poll.error?.status === 404 || poll.error?.status === 422)
  if (missing) return <p className="text-muted-foreground m-0 text-sm">{t('operations.logs.groupsUnavailable')}</p>
  const col = (k: string) => t(`operations.logs.groupCol.${k}`)
  return (
    <Panel poll={poll} freshness={GROUPS_SECONDS} title={t('operations.logs.groupsTitle')} sub={t('operations.logs.groupsSub', { hours })}>
      {(data) =>
        data.groups.length ? (
          <div className="overflow-x-auto">
            <table className="ops-stack w-full min-w-[860px] text-[13px]">
              <thead>
                <tr className="text-muted-foreground border-b text-xs">
                  <th className="p-2 text-right font-medium">{col('count')}</th>
                  <th className="p-2 text-left font-medium">{col('level')}</th>
                  <th className="p-2 text-left font-medium">{col('site')}</th>
                  <th className="p-2 text-left font-medium">{col('message')}</th>
                  <th className="p-2 text-left font-medium">{col('first')}</th>
                  <th className="p-2 text-left font-medium">{col('last')}</th>
                  <th className="p-2 text-right font-medium">{col('workspaces')}</th>
                  <th className="p-2 text-left font-medium">
                    <span className="sr-only">{col('show')}</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.groups.map((g) => (
                  <tr key={g.fingerprint} className="border-b align-top last:border-0">
                    <td className="p-2 text-right font-mono whitespace-nowrap" data-label={col('count')}>
                      ×{fmtCount(g.count)}
                    </td>
                    <td className="p-2" data-label={col('level')}>
                      <LevelChip level={g.level} />
                    </td>
                    <td className="p-2 font-mono text-xs break-all" data-label={col('site')} title={g.logger || undefined}>
                      {callSite(g)}
                    </td>
                    <td className="p-2" data-label={col('message')}>
                      <span className="flex flex-col gap-0.5">
                        {g.exc_type ? <span className="font-mono text-xs font-medium">{g.exc_type}</span> : null}
                        <RedactedText text={g.sample} className="line-clamp-2 break-words" />
                      </span>
                    </td>
                    <td className="p-2 font-mono text-xs whitespace-nowrap" data-label={col('first')} title={utcTitle(g.first_seen)}>
                      {fmtStamp(g.first_seen, poll.updatedAt ?? undefined)}
                    </td>
                    <td className="p-2 font-mono text-xs whitespace-nowrap" data-label={col('last')} title={utcTitle(g.last_seen)}>
                      {fmtStamp(g.last_seen, poll.updatedAt ?? undefined)}
                    </td>
                    <td className="p-2 text-right tabular-nums" data-label={col('workspaces')}>
                      {fmtCount(g.workspaces)}
                    </td>
                    <td className="p-2" data-label="">
                      <OpsLink to={logsView({ ...filter, fingerprint: g.fingerprint })} className="hover:bg-muted inline-flex min-h-8 items-center rounded-lg border px-2.5 text-[13px]">
                        {col('show')}
                      </OpsLink>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-muted-foreground m-0 text-sm">{t('operations.logs.groupsEmpty', { hours })}</p>
        )
      }
    </Panel>
  )
}
