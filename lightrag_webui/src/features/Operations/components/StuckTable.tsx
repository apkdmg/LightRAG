import { useTranslation } from 'react-i18next'
import { useOpsNav } from '../context'
import { fmtAge, fmtClock, shortId, utcTitle } from '../format'
import { StuckRow, needsRestart, stuckWhy } from '../processing/stuck'
import { runLookOf } from '../severity'
import OpsLink from './OpsLink'
import { StateIcon } from './StatusChip'

/** The age a row's reason talks about: the owner's last beat, or the last progress. */
const reasonAge = (row: StuckRow) => {
  if (row.cls === 'orphaned') return row.heartbeatAge ?? row.ageSeconds
  if (row.cls === 'unresponsive') return row.workerHeartbeatAge ?? row.ageSeconds
  if (row.cls === 'hung') return row.progressAge ?? row.ageSeconds
  return row.ageSeconds
}

/**
 * Why a row is stuck, in plain language (US-3), built from its class, the
 * server's reason code and its fields. Server codes are never shown as-is.
 */
function Why({ row }: { row: StuckRow }) {
  const { t } = useTranslation()
  const { key, values } = stuckWhy(row)
  if (row.cls === 'stale_doc') {
    return <>{t('operations.stuck.why.stale_doc', { status: t(`operations.docStatus.${row.docStatus}`, { defaultValue: row.docStatus ?? '' }), time: fmtClock(row.since ?? null) })}</>
  }
  const how = typeof values.how === 'string' ? t(`operations.stuck.confirmedBy.${values.how}`, { defaultValue: t('operations.stuck.confirmedBy.pid_gone') }) : ''
  return <>{t(`operations.stuck.why.${key}`, { ...values, how, age: fmtAge(reasonAge(row)) })}</>
}

/**
 * The Stuck view's table (§2.4): class (icon and text), workspace, what,
 * age and why. Rows link to the run, document or workspace drawer. Classes
 * that only a container restart clears today carry that guidance under the
 * reason. In single-process mode the multi-worker classes never appear.
 */
export default function StuckTable({ rows }: { rows: StuckRow[] }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const col = (key: string) => t(`operations.stuck.col.${key}`)
  return (
    <div className="overflow-x-auto">
      <table className="ops-stack w-full min-w-[760px] text-[13px]">
        <thead>
          <tr className="text-muted-foreground border-b text-xs">
            <th className="p-2 text-left font-medium">{col('class')}</th>
            <th className="p-2 text-left font-medium">{col('workspace')}</th>
            <th className="p-2 text-left font-medium">{col('what')}</th>
            <th className="p-2 text-right font-medium">{col('age')}</th>
            <th className="p-2 text-left font-medium">{col('why')}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const look = runLookOf(row.cls)
            const to = { ...nav.view, drawer: row.drawer }
            const open = () => nav.openDrawer(row.drawer)
            return (
              <tr
                key={row.key}
                className="ops-row-link border-b align-top last:border-0"
                onClick={(e) => {
                  if ((e.target as HTMLElement).closest('a')) return
                  open()
                }}
              >
                <td className="p-2" data-label={col('class')}>
                  <span className="inline-flex items-center gap-1.5 text-xs font-semibold whitespace-nowrap" style={{ color: `var(--ops-${look.tone})` }}>
                    <StateIcon icon={look.icon} />
                    {t(look.label)}
                  </span>
                </td>
                <td className="p-2 break-all" data-label={col('workspace')}>
                  {row.workspace}
                </td>
                <td className="p-2" data-label={col('what')}>
                  <OpsLink to={to} onNavigate={open} className="font-mono text-xs break-all underline-offset-2 hover:underline">
                    {row.docId
                      ? row.docName || shortId(row.docId, 12)
                      : row.cls === 'busy_without_run'
                        ? t('operations.stuck.flagBusy')
                        : row.runId
                          ? t('operations.stuck.run', { id: shortId(row.runId) })
                          : row.cls === 'reservation'
                            ? row.scanning
                              ? t('operations.stuck.flagScanning')
                              : t('operations.stuck.flagEnqueues', { count: row.pendingEnqueues ?? 0 })
                            : t('operations.stuck.openWorkspace')}
                  </OpsLink>
                </td>
                <td className="p-2 text-right font-mono text-xs whitespace-nowrap" data-label={col('age')} title={utcTitle(row.since)}>
                  {fmtAge(row.ageSeconds)}
                </td>
                <td className="text-muted-foreground p-2" data-label={col('why')}>
                  <Why row={row} />
                  {needsRestart(row.cls) ? <div className="text-foreground mt-0.5 text-xs">{t('operations.stuck.restart')}</div> : null}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
