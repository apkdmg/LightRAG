import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { opsApi } from '../api'
import { Pager, Panel } from '../components/Panel'
import StuckTable from '../components/StuckTable'
import { useOpsShell } from '../context'
import { fmtAge, fmtClock, fmtCount } from '../format'
import { useOpsPoll } from '../poll'
import { isMultiWorker } from '../severity'
import { STUCK_CLASSES, checkedClasses, stuckRows } from './stuck'

/** §3.6: stuck docs are served from a 15 s cache; the run classes come from the 60 s reconcile pass. */
const STUCK_SECONDS = 15

/**
 * What is stuck (§2.4, US-3): orphaned runs (Gunicorn only, once the OS has
 * confirmed the owner is gone), runs that are alive but hung, workspaces busy
 * without a live run, reservations without an owner (Gunicorn only, with
 * restart guidance) and stale documents. The view says which classes the
 * server actually checked, so a class that was not checked never reads as
 * "nothing stuck".
 */
export default function StuckView() {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const multiWorker = isMultiWorker(status?.process.mode)
  const [page, setPage] = useState(1)
  const poll = useOpsPoll((signal) => opsApi.stuck({ page }, signal), [page], STUCK_SECONDS)
  const th = poll.data?.thresholds
  const thresholds = [
    multiWorker && typeof th?.orphan_s === 'number' ? t('operations.stuck.th.orphan', { age: fmtAge(th.orphan_s) }) : null,
    typeof th?.hung_min === 'number' ? t('operations.stuck.th.hung', { age: fmtAge(th.hung_min * 60) }) : null,
    typeof th?.doc_min === 'number' ? t('operations.stuck.th.doc', { age: fmtAge(th.doc_min * 60) }) : null
  ].filter(Boolean)
  return (
    <Panel
      poll={poll}
      freshness={STUCK_SECONDS}
      title={t('operations.stuck.title')}
      sub={thresholds.length ? t('operations.stuck.thresholds', { list: thresholds.join(' · ') }) : undefined}
    >
      {(data) => {
        const rows = stuckRows(data, multiWorker)
        const docs = data.stale_docs ?? { items: [], total: 0 }
        const checked = checkedClasses(data, multiWorker)
        const possible = STUCK_CLASSES.filter((c) => multiWorker || (c !== 'orphaned' && c !== 'reservations'))
        const unchecked = possible.filter((c) => !checked.includes(c))
        const list = checked.map((c) => t(`operations.stuck.class.${c}`)).join(', ')
        const time = fmtClock(data.checked?.at ?? poll.updatedAt)
        return (
          <div className="flex flex-col gap-3">
            {rows.length ? (
              <StuckTable rows={rows} />
            ) : (
              <p className="text-muted-foreground m-0 text-sm">
                {typeof data.checked?.workspaces === 'number' && typeof data.checked?.runs === 'number'
                  ? t('operations.stuck.noneCounted', { workspaces: fmtCount(data.checked.workspaces), runs: fmtCount(data.checked.runs), list, time })
                  : t('operations.stuck.noneClasses', { list, time })}
              </p>
            )}
            <Pager page={docs.page ?? page} pageSize={docs.page_size ?? Math.max(docs.items.length, 50)} total={docs.total} onPage={setPage} />
            {unchecked.length ? (
              <p className="text-muted-foreground m-0 text-xs">
                {status?.features?.runs === false
                  ? t('operations.stuck.trackingOff')
                  : t('operations.stuck.notChecked', { list: unchecked.map((c) => t(`operations.stuck.class.${c}`)).join(', ') })}
              </p>
            ) : null}
            {!multiWorker ? <p className="text-muted-foreground m-0 text-xs">{t('operations.stuck.singleProcess')}</p> : null}
          </div>
        )
      }}
    </Panel>
  )
}
