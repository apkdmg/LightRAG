import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { opsApi } from '../api'
import { Pager, Panel } from '../components/Panel'
import StuckTable from '../components/StuckTable'
import { useOpsShell } from '../context'
import { fmtAge, fmtClock } from '../format'
import { useOpsPoll } from '../poll'
import { isMultiWorker } from '../severity'
import { stuckRows } from './stuck'

/** §3.6: stuck docs are served from a 15 s cache. */
const STUCK_SECONDS = 15

/**
 * What is stuck (§2.4, US-3). In this release the server detects stale
 * documents (in parsing, analyzing or processing longer than the threshold);
 * hung, orphaned and busy-without-run checks arrive with run tracking, and
 * the view says so instead of implying they were checked.
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
        return (
          <div className="flex flex-col gap-3">
            {rows.length ? (
              <StuckTable rows={rows} />
            ) : (
              <p className="text-muted-foreground m-0 text-sm">{t('operations.stuck.none', { time: fmtClock(poll.updatedAt) })}</p>
            )}
            <Pager page={docs.page ?? page} pageSize={docs.page_size ?? Math.max(docs.items.length, 50)} total={docs.total} onPage={setPage} />
            {!status?.features?.runs || (data.checked_classes && !data.checked_classes.includes('hung')) ? (
              <p className="text-muted-foreground m-0 text-xs">{t('operations.stuck.runsLater')}</p>
            ) : null}
          </div>
        )
      }}
    </Panel>
  )
}
