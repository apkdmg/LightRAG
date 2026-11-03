import { useTranslation } from 'react-i18next'
import { useOpsNav } from '../context'
import { fmtAge, fmtClock, shortId, utcTitle } from '../format'
import { StuckRow } from '../processing/stuck'
import { runLookOf } from '../severity'
import OpsLink from './OpsLink'
import { StateIcon } from './StatusChip'

/** Why a row is stuck, in plain language (FR-53). */
function Why({ row }: { row: StuckRow }) {
  const { t } = useTranslation()
  if (row.reason) return <>{row.reason}</>
  if (row.cls === 'stale_doc') {
    return <>{t('operations.stuck.why.stale_doc', { status: t(`operations.docStatus.${row.docStatus}`, { defaultValue: row.docStatus ?? '' }), time: fmtClock(row.since ?? null) })}</>
  }
  return <>{t(runLookOf(row.cls).explain)}</>
}

/**
 * The Stuck view's table (§2.4): class (icon and text), workspace, what,
 * age and why. Rows link to the document or workspace drawer. In
 * single-process mode the multi-worker classes never appear.
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
                    {row.docId ? row.docName || shortId(row.docId, 12) : row.runId ? t('operations.stuck.run', { id: shortId(row.runId) }) : t('operations.stuck.openWorkspace')}
                  </OpsLink>
                </td>
                <td className="p-2 text-right font-mono text-xs whitespace-nowrap" data-label={col('age')} title={utcTitle(row.since)}>
                  {fmtAge(row.ageSeconds)}
                </td>
                <td className="text-muted-foreground p-2" data-label={col('why')}>
                  <Why row={row} />
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
