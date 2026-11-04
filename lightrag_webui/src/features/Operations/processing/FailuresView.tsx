import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import { Seg } from '@/components/admin'
import { FailureGroup, opsApi } from '../api'
import ErrorPanel from '../components/ErrorPanel'
import { Pager, Panel } from '../components/Panel'
import Skeleton from '../components/Skeleton'
import { useOpsNav, useOpsShell } from '../context'
import { ErrorKindText } from '../drawers/RunDrawer'
import { fmtClock, fmtCount, shortId, utcTitle } from '../format'
import { useOpsPoll } from '../poll'

/** Failure history changes slowly; one minute is plenty (the summary bar refreshes faster). */
const FAILURES_SECONDS = 60
const WINDOWS = ['1h', '24h', '7d'] as const
const GROUPS = ['error_kind', 'stage', 'workspace', 'model'] as const
/** Legacy rows have no error kind; their docs are fetched by message signature. */
const LEGACY_GROUP = 'signature'

type Period = (typeof WINDOWS)[number]
type Group = (typeof GROUPS)[number]
/** A group's documents to list; `source` says which side a name used by both (workspace) comes from. */
type Picked = { key: string; group: string; label: string; source: 'legacy' | 'durable' }

/** What a legacy row's key is: a message signature, a workspace, or the cancellations bucket. */
const legacyKind = (row: FailureGroup, fallback: string | null | undefined) => row.kind || fallback || LEGACY_GROUP

function GroupLabel({ group, row, legacy, legacyGroup }: { group: string; row: FailureGroup; legacy: boolean; legacyGroup?: string | null }) {
  const { t } = useTranslation()
  if (legacy) {
    const kind = legacyKind(row, legacyGroup)
    if (kind === 'cancelled') return <>{t('operations.failures.cancelledLabel')}</>
    if (kind === 'workspace') return <span className="break-all">{row.key || '–'}</span>
    return <>{t('operations.failures.legacyLabel')}</>
  }
  if (group === 'error_kind') return <ErrorKindText kind={row.key} status={row.http_status} />
  if (group === 'stage') return <>{t(`operations.stage.${row.key}`, { defaultValue: row.key })}</>
  return <span className="break-all">{row.key || '–'}</span>
}

/** The documents behind one group, masked, each opening the Document drawer. */
function GroupDocs({ picked, period, onClose }: { picked: Picked; period: Period; onClose: () => void }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const [page, setPage] = useState(1)
  const poll = useOpsPoll(
    (signal) => opsApi.failureDocs({ key: picked.key, group: picked.group, window: period, page, source: picked.source }, signal),
    [picked.key, picked.group, picked.source, period, page],
    FAILURES_SECONDS
  )
  const data = poll.data
  return (
    <section className="flex flex-col gap-2 rounded-lg border p-3" aria-label={t('operations.failures.docsFor', { label: picked.label })}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="m-0 text-[13px] font-semibold">{t('operations.failures.docsFor', { label: picked.label })}</h3>
        <button type="button" className="hover:bg-muted min-h-7 rounded-lg border px-2.5 text-xs" onClick={onClose}>
          {t('operations.failures.closeDocs')}
        </button>
      </div>
      {poll.error && !data ? <ErrorPanel error={poll.error} onRetry={poll.reload} compact /> : null}
      {!data && !poll.error ? <Skeleton className="h-16 w-full" /> : null}
      {data ? (
        data.items.length ? (
          <ul className={cn('m-0 flex list-none flex-col p-0', poll.loading && 'opacity-60')}>
            {data.items.map((d) => (
              <li key={`${d.workspace}/${d.doc_id}`}>
                <button
                  type="button"
                  className="hover:bg-muted grid w-full grid-cols-[minmax(0,1fr)_auto] gap-x-3 rounded-lg px-2 py-1.5 text-left text-[13px]"
                  onClick={() => nav.openDrawer({ kind: 'document', workspace: d.workspace, id: d.doc_id })}
                >
                  <span className="truncate font-mono text-xs">{d.file_path_masked || shortId(d.doc_id, 16)}</span>
                  <span className="text-muted-foreground font-mono text-xs" title={utcTitle(d.updated_at)}>
                    {fmtClock(d.updated_at ?? null)}
                  </span>
                  {d.error_kind || d.stage ? (
                    <span className="col-span-2 text-xs">
                      {d.stage ? `${t(`operations.stage.${d.stage}`, { defaultValue: d.stage })} · ` : ''}
                      <span className="font-medium">
                        <ErrorKindText kind={d.error_kind} status={d.http_status} />
                      </span>
                      {d.model ? <span className="text-muted-foreground font-mono"> · {d.model}</span> : null}
                      {typeof d.attempt === 'number' ? <span className="text-muted-foreground"> · {t('operations.document.attempt', { n: d.attempt })}</span> : null}
                    </span>
                  ) : null}
                  <span className="text-muted-foreground col-span-2 truncate text-xs">
                    {d.workspace}
                    {d.error_redacted ? ` · ${d.error_redacted}` : ''}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-muted-foreground m-0 text-sm">{t('operations.failures.noDocs')}</p>
        )
      ) : null}
      {data ? <Pager page={data.page} pageSize={data.page_size} total={data.total} onPage={setPage} /> : null}
    </section>
  )
}

function GroupTable({
  rows,
  group,
  legacy,
  legacyGroup,
  onPick
}: {
  rows: FailureGroup[]
  group: string
  legacy: boolean
  legacyGroup?: string | null
  onPick: (p: Picked) => void
}) {
  const { t } = useTranslation()
  const col = (key: string) => t(`operations.failures.col.${key}`)
  return (
    <div className="overflow-x-auto">
      <table className="ops-stack w-full min-w-[760px] text-[13px]">
        <thead>
          <tr className="text-muted-foreground border-b text-xs">
            <th className="p-2 text-left font-medium">{legacy && legacyGroup !== 'workspace' ? col('signature') : t(`operations.failures.group.${legacy ? 'workspace' : group}`)}</th>
            <th className="p-2 text-right font-medium">{col('count')}</th>
            <th className="p-2 text-right font-medium">{col('workspaces')}</th>
            <th className="p-2 text-left font-medium">{col('first')}</th>
            <th className="p-2 text-left font-medium">{col('last')}</th>
            <th className="p-2 text-left font-medium">{col('example')}</th>
            <th className="p-2 text-left font-medium">
              <span className="sr-only">{col('docs')}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const kind = legacy ? legacyKind(r, legacyGroup) : group
            const label = legacy
              ? kind === 'workspace'
                ? r.key
                : kind === 'cancelled'
                  ? t('operations.failures.cancelledLabel')
                  : t('operations.failures.legacyLabel')
              : group === 'error_kind'
                ? t(`operations.errorKind.${r.key}`, { defaultValue: r.key })
                : group === 'stage'
                  ? t(`operations.stage.${r.key}`, { defaultValue: r.key })
                  : r.key
            return (
              <tr key={`${legacy ? 'legacy' : group}:${r.key}`} className="border-b align-top last:border-0">
                <td className="p-2 font-medium" data-label={legacy ? col('signature') : t(`operations.failures.group.${group}`)}>
                  <GroupLabel group={group} row={r} legacy={legacy} legacyGroup={legacyGroup} />
                </td>
                <td className="p-2 text-right tabular-nums" data-label={col('count')}>
                  {fmtCount(r.count)}
                </td>
                <td className="p-2 text-right tabular-nums" data-label={col('workspaces')}>
                  {fmtCount(r.workspaces ?? null)}
                </td>
                <td className="p-2 font-mono text-xs whitespace-nowrap" data-label={col('first')} title={utcTitle(r.first_seen)}>
                  {fmtClock(r.first_seen ?? null)}
                </td>
                <td className="p-2 font-mono text-xs whitespace-nowrap" data-label={col('last')} title={utcTitle(r.last_seen)}>
                  {fmtClock(r.last_seen ?? null)}
                </td>
                <td className="text-muted-foreground max-w-[360px] p-2 font-mono text-xs break-words" data-label={col('example')}>
                  {r.example ? `“${r.example}”` : '–'}
                </td>
                <td className="p-2" data-label="">
                  <button
                    type="button"
                    className="hover:bg-muted min-h-7 rounded-lg border px-2.5 text-xs whitespace-nowrap"
                    onClick={() => onPick({ key: r.key, group: kind, label, source: legacy ? 'legacy' : 'durable' })}
                  >
                    {col('docs')}
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/**
 * Failure analysis (§2.4, FR-14): classified failures from the failure
 * history, grouped by kind, stage, workspace or model, with first and last
 * seen, a masked example and a drill to the documents. Real failures only:
 * duplicates are excluded and cancellations are counted separately (or
 * included on request). Failures recorded before classification existed
 * (or while run tracking is off) are grouped by a normalised message
 * signature under "Unclassified (legacy)".
 */
export default function FailuresView() {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const [period, setPeriod] = useState<Period>('24h')
  const [group, setGroup] = useState<Group>('error_kind')
  const [includeCancelled, setIncludeCancelled] = useState(false)
  const [picked, setPicked] = useState<Picked | null>(null)
  const poll = useOpsPoll(
    (signal) => opsApi.failures({ window: period, group, include_cancelled: includeCancelled }, signal),
    [period, group, includeCancelled],
    FAILURES_SECONDS
  )
  return (
    <Panel
      poll={poll}
      freshness={FAILURES_SECONDS}
      title={t('operations.failures.title')}
      sub={
        poll.data
          ? t('operations.failures.excluded', {
            dup: fmtCount(poll.data.excluded?.dup ?? 0),
            cancelled: fmtCount(poll.data.excluded?.cancelled ?? 0)
          })
          : undefined
      }
    >
      {(data) => (
        <div className={cn('flex flex-col gap-3', poll.loading && 'opacity-60')}>
          <div className="flex flex-wrap items-end gap-3">
            <div className="text-muted-foreground flex flex-col gap-0.5 text-xs">
              {t('operations.failures.window')}
              <Seg value={period} onChange={(w) => { setPeriod(w); setPicked(null) }} options={WINDOWS.map((w) => ({ value: w, label: t(`operations.failures.windows.${w}`) }))} />
            </div>
            <div className="text-muted-foreground flex flex-col gap-0.5 text-xs">
              {t('operations.failures.groupBy')}
              <Seg value={group} onChange={(g) => { setGroup(g); setPicked(null) }} options={GROUPS.map((g) => ({ value: g, label: t(`operations.failures.group.${g}`) }))} />
            </div>
            <label className="flex min-h-8 items-center gap-2 text-[13px]">
              <input type="checkbox" checked={includeCancelled} onChange={(e) => setIncludeCancelled(e.target.checked)} className="size-4" />
              {t('operations.failures.includeCancelled')}
            </label>
          </div>
          {data.groups.length ? <GroupTable rows={data.groups} group={group} legacy={false} onPick={setPicked} /> : null}
          {data.durable === false ? (
            <p className="text-muted-foreground m-0 text-xs">
              {status?.features?.runs === false ? t('operations.failures.trackingOff') : t('operations.failures.notDurable')}
            </p>
          ) : null}
          {data.legacy?.length ? (
            <div className="flex flex-col gap-1.5">
              <h3 className="m-0 text-[13px] font-semibold">{t('operations.failures.legacyTitle')}</h3>
              <p className="text-muted-foreground m-0 text-xs">{t('operations.failures.legacyNote')}</p>
              <GroupTable rows={data.legacy} group={group} legacy legacyGroup={data.legacy_group} onPick={setPicked} />
            </div>
          ) : null}
          {!data.groups.length && !data.legacy?.length ? (
            <p className="text-muted-foreground m-0 text-sm">{t('operations.failures.none', { window: t(`operations.failures.windows.${period}`) })}</p>
          ) : null}
          {picked ? <GroupDocs key={`${picked.source}:${picked.group}:${picked.key}:${period}`} picked={picked} period={period} onClose={() => setPicked(null)} /> : null}
          <p className="text-muted-foreground m-0 text-xs">{t('operations.maskNote')}</p>
        </div>
      )}
    </Panel>
  )
}
