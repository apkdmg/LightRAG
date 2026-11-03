import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import { opsApi, slotKindRows } from '../api'
import OpsLink from '../components/OpsLink'
import { Pager, Panel } from '../components/Panel'
import { StatusChip } from '../components/StatusChip'
import { useOpsNav } from '../context'
import { fmtClock, fmtCount, utcTitle } from '../format'
import { useOpsPoll } from '../poll'

/** The pause reader refreshes once a minute (§3.6). */
const PAUSES_SECONDS = 60

/**
 * Provider pauses (§2.4, FR-16): counts by slot and kind, then the paused
 * workspaces, with bring-your-own-key pauses flagged. Paused workspaces
 * resume on their own once the provider accepts calls again; keys are
 * managed on the Providers page.
 */
export default function PausesView() {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const [page, setPage] = useState(1)
  const poll = useOpsPoll((signal) => opsApi.pauses({ page }, signal), [page], PAUSES_SECONDS)
  const col = (key: string) => t(`operations.pauses.col.${key}`)
  return (
    <Panel
      poll={poll}
      freshness={PAUSES_SECONDS}
      title={t('operations.pauses.title')}
      approximate={!!poll.data?.approximate}
      actions={
        <button type="button" className="hover:bg-muted min-h-8 rounded-lg border px-2.5 text-[13px]" onClick={() => useSettingsStore.getState().setCurrentTab('providers')}>
          {t('operations.pauses.openProviders')}
        </button>
      }
    >
      {(data) => {
        const groups = slotKindRows(data.by_slot_kind)
        const total = typeof data.total === 'number' ? data.total : data.items.length
        return (
          <div className={cn('flex flex-col gap-3', poll.loading && 'opacity-60')}>
            {groups.length ? (
              <ul className="m-0 flex list-none flex-wrap gap-1.5 p-0">
                {groups.map((g) => (
                  <li key={`${g.slot}:${g.kind}`}>
                    <StatusChip state="paused">
                      {t('operations.pauses.group', {
                        slot: g.slot,
                        kind: g.kind ? t(`operations.pauses.kind.${g.kind}`, { defaultValue: g.kind }) : '–',
                        count: g.count,
                        countText: fmtCount(g.count)
                      })}
                    </StatusChip>
                  </li>
                ))}
              </ul>
            ) : null}
            {data.items.length ? (
              <div className="overflow-x-auto">
                <table className="ops-stack w-full min-w-[720px] text-[13px]">
                  <thead>
                    <tr className="text-muted-foreground border-b text-xs">
                      <th className="p-2 text-left font-medium">{col('workspace')}</th>
                      <th className="p-2 text-left font-medium">{col('slot')}</th>
                      <th className="p-2 text-left font-medium">{col('kind')}</th>
                      <th className="p-2 text-right font-medium">{col('status')}</th>
                      <th className="p-2 text-left font-medium">{col('key')}</th>
                      <th className="p-2 text-left font-medium">{col('since')}</th>
                      <th className="p-2 text-left font-medium">{col('retry')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.items.map((p, i) => (
                      <tr key={`${p.workspace}:${p.slot}:${i}`} className="border-b align-top last:border-0">
                        <td className="p-2" data-label={col('workspace')}>
                          <OpsLink
                            to={{ ...nav.view, drawer: { kind: 'workspace', id: p.workspace } }}
                            onNavigate={() => nav.openDrawer({ kind: 'workspace', id: p.workspace })}
                            className="break-all underline-offset-2 hover:underline"
                          >
                            {p.workspace}
                          </OpsLink>
                        </td>
                        <td className="p-2" data-label={col('slot')}>
                          {p.slot ?? '–'}
                        </td>
                        <td className="p-2" data-label={col('kind')}>
                          {p.kind ? t(`operations.pauses.kind.${p.kind}`, { defaultValue: p.kind }) : '–'}
                        </td>
                        <td className="p-2 text-right font-mono text-xs" data-label={col('status')}>
                          {typeof p.status_code === 'number' ? p.status_code : '–'}
                        </td>
                        <td className="p-2" data-label={col('key')}>
                          {p.byo ? t('operations.pauses.byo') : t('operations.pauses.systemKey')}
                        </td>
                        <td className="p-2 font-mono text-xs whitespace-nowrap" data-label={col('since')} title={utcTitle(p.since)}>
                          {fmtClock(p.since ?? null)}
                        </td>
                        <td className="p-2 font-mono text-xs whitespace-nowrap" data-label={col('retry')} title={utcTitle(p.retry_at)}>
                          {fmtClock(p.retry_at ?? null)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-muted-foreground m-0 text-sm">{t('operations.pauses.none')}</p>
            )}
            <Pager page={data.page ?? page} pageSize={data.page_size ?? Math.max(data.items.length, 50)} total={total} onPage={setPage} />
            <p className="text-muted-foreground m-0 text-xs">{t('operations.pauses.note')}</p>
          </div>
        )
      }}
    </Panel>
  )
}
