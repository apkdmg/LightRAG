import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import Input from '@/components/ui/Input'
import { cn } from '@/lib/utils'
import { Seg } from '@/components/admin'
import { useDebounce } from '@/hooks/useDebounce'
import { OpsRun, opsApi } from '../api'
import ErrorPanel from '../components/ErrorPanel'
import OpsLink from '../components/OpsLink'
import { Panel } from '../components/Panel'
import { RunStateLabel } from '../components/RunState'
import { useOpsNav, useOpsShell } from '../context'
import { ErrorKindText } from '../drawers/RunDrawer'
import { OpsError, isAbort, toOpsError } from '../errors'
import { fmtAge, fmtCount, fmtStamp, utcTitle } from '../format'
import { useOpsPoll } from '../poll'
import { DEFAULT_RUNS_FILTER, MAX_WORKSPACE_LENGTH, RUN_KINDS, RUN_STATUSES, RunsFilter, mergeRunPages, runDurationSeconds, runsQuery } from './runs'

const SEARCH_DEBOUNCE_MS = 250
/** Run rows are written within 2 s; the newest page refreshes at this pace (older pages are fixed history). */
const RUNS_SECONDS = 30

type Older = { view: string; pages: OpsRun[][]; next: string | null; loading: boolean; error: OpsError | null }

export function RunsTable({ runs, now }: { runs: OpsRun[]; now: number | null }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const col = (key: string) => t(`operations.runs.col.${key}`)
  return (
    <div className="overflow-x-auto">
      <table className="ops-stack w-full min-w-[900px] text-[13px]">
        <thead>
          <tr className="text-muted-foreground border-b text-xs">
            <th className="p-2 text-left font-medium">{col('started')}</th>
            <th className="p-2 text-left font-medium">{col('workspace')}</th>
            <th className="p-2 text-left font-medium">{col('kind')}</th>
            <th className="p-2 text-left font-medium">{col('trigger')}</th>
            <th className="p-2 text-left font-medium">{col('outcome')}</th>
            <th className="p-2 text-right font-medium">{col('duration')}</th>
            <th className="p-2 text-right font-medium">{col('docs')}</th>
            <th className="p-2 text-right font-medium">{col('failed')}</th>
            <th className="p-2 text-left font-medium">{col('error')}</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const open = () => nav.openDrawer({ kind: 'run', id: run.run_id })
            return (
              <tr
                key={run.run_id}
                className="ops-row-link border-b align-top last:border-0"
                onClick={(e) => {
                  if ((e.target as HTMLElement).closest('a')) return
                  open()
                }}
              >
                <td className="p-2 font-mono text-xs whitespace-nowrap" data-label={col('started')} title={utcTitle(run.started_at)}>
                  <OpsLink to={{ ...nav.view, drawer: { kind: 'run', id: run.run_id } }} onNavigate={open} className="underline-offset-2 hover:underline">
                    {fmtStamp(run.started_at ?? null, now ?? undefined)}
                  </OpsLink>
                </td>
                <td className="p-2 break-all" data-label={col('workspace')}>
                  <OpsLink
                    to={{ ...nav.view, drawer: { kind: 'workspace', id: run.workspace } }}
                    onNavigate={() => nav.openDrawer({ kind: 'workspace', id: run.workspace })}
                    className="underline-offset-2 hover:underline"
                  >
                    {run.workspace}
                  </OpsLink>
                </td>
                <td className="p-2 font-mono text-xs" data-label={col('kind')}>
                  {run.kind ? t(`operations.kind.${run.kind}`, { defaultValue: run.kind }) : '–'}
                </td>
                <td className="p-2" data-label={col('trigger')}>
                  {run.trigger ? t(`operations.trigger.${run.trigger}`, { defaultValue: run.trigger }) : '–'}
                </td>
                <td className="p-2" data-label={col('outcome')}>
                  <span className="flex flex-col items-end gap-0.5 md:items-start">
                    {/* History rows never breathe: the dot is static here (§2.12, running dot only in Active and the Run drawer). */}
                    <RunStateLabel state={run.state} stale />
                    {run.was_orphaned ? <span className="text-muted-foreground text-xs">{t('operations.runDrawer.wasOrphaned')}</span> : null}
                  </span>
                </td>
                <td className="p-2 text-right whitespace-nowrap tabular-nums" data-label={col('duration')}>
                  {fmtAge(runDurationSeconds(run, now))}
                </td>
                <td className="p-2 text-right font-mono text-xs whitespace-nowrap" data-label={col('docs')}>
                  {typeof run.docs_total === 'number' ? `${fmtCount(run.docs_done ?? 0)}/${fmtCount(run.docs_total)}` : '–'}
                </td>
                <td className="p-2 text-right tabular-nums" data-label={col('failed')}>
                  {fmtCount(run.docs_failed ?? null)}
                </td>
                <td className="p-2 text-xs" data-label={col('error')}>
                  {run.error_kind ? <ErrorKindText kind={run.error_kind} /> : '–'}
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
 * Run history for 30 days (FR-17): kind, trigger, duration, docs done and
 * failed, and the outcome (succeeded, failed, cancelled, paused, orphaned;
 * "was orphaned" for a run whose owner came back and finished it). Filtered
 * on the server by exact workspace, kind and outcome, newest first, with a
 * keyset "Load older" so a long history never ships at once. Rows open the
 * Run drawer.
 */
export default function RunsView() {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const [filter, setFilter] = useState<RunsFilter>(DEFAULT_RUNS_FILTER)
  const [wsInput, setWsInput] = useState('')
  const workspace = useDebounce(wsInput, SEARCH_DEBOUNCE_MS).trim()
  const effective: RunsFilter = { ...filter, workspace }
  const params = runsQuery(effective)
  const view = JSON.stringify(params)
  const poll = useOpsPoll((signal) => opsApi.runs(params, signal), [params], RUNS_SECONDS)
  // Older pages belong to one filter; a new filter starts from the newest page again.
  const [olderState, setOlder] = useState<Older>({ view, pages: [], next: null, loading: false, error: null })
  const older = olderState.view === view ? olderState : { view, pages: [], next: null, loading: false, error: null }

  const firstPage = poll.data?.items ?? []
  const runs = mergeRunPages(firstPage, older.pages)
  const cursor = older.pages.length ? older.next : (poll.data?.next_before ?? null)

  const loadOlder = async () => {
    if (!cursor || older.loading) return
    setOlder({ ...older, loading: true, error: null })
    try {
      const page = await opsApi.runs(runsQuery(effective, cursor))
      setOlder((o) => (o.view === view ? { view, pages: [...o.pages, page.items], next: page.next_before, loading: false, error: null } : o))
    } catch (err) {
      if (isAbort(err)) return
      setOlder((o) => (o.view === view ? { ...o, loading: false, error: toOpsError(err) } : o))
    }
  }

  const trackingOff = status?.features?.runs === false || poll.data?.enabled === false
  return (
    <Panel poll={poll} freshness={RUNS_SECONDS} title={t('operations.runs.title')} sub={t('operations.runs.sub')}>
      {() => (
        <div className={cn('flex flex-col gap-3', poll.loading && 'opacity-60')}>
          <div className="flex flex-wrap items-end gap-3">
            <label className="text-muted-foreground flex flex-col gap-0.5 text-xs">
              {t('operations.runs.workspace')}
              <Input
                value={wsInput}
                onChange={(e) => setWsInput(e.target.value)}
                placeholder={t('operations.runs.workspacePlaceholder')}
                className="h-9 w-56"
                maxLength={MAX_WORKSPACE_LENGTH}
              />
            </label>
            <label className="text-muted-foreground flex flex-col gap-0.5 text-xs">
              {t('operations.runs.kind')}
              <select
                value={filter.kind}
                onChange={(e) => setFilter((f) => ({ ...f, kind: e.target.value as RunsFilter['kind'] }))}
                className="border-input bg-background text-foreground h-9 rounded-md border px-2 text-[13px]"
              >
                <option value="all">{t('operations.runs.allKinds')}</option>
                {RUN_KINDS.map((k) => (
                  <option key={k} value={k}>
                    {t(`operations.kind.${k}`)}
                  </option>
                ))}
              </select>
            </label>
            <div className="text-muted-foreground flex max-w-full flex-col gap-0.5 overflow-x-auto text-xs">
              {t('operations.runs.outcome')}
              <Seg
                value={filter.status}
                onChange={(v) => setFilter((f) => ({ ...f, status: v }))}
                label={t('operations.runs.outcome')}
                options={[
                  { value: 'all' as const, label: t('operations.runs.allOutcomes') },
                  ...RUN_STATUSES.map((s) => ({ value: s, label: t(`operations.run.${s}.label`) }))
                ]}
              />
            </div>
          </div>
          {runs.length ? (
            <RunsTable runs={runs} now={poll.updatedAt} />
          ) : (
            <p className="text-muted-foreground m-0 text-sm">
              {trackingOff ? t('operations.runs.trackingOff') : params.workspace || params.kind || params.status ? t('operations.runs.noneMatching') : t('operations.runs.none')}
            </p>
          )}
          {older.error ? <ErrorPanel error={older.error} onRetry={loadOlder} compact /> : null}
          {cursor ? (
            <div className="flex justify-end">
              <button
                type="button"
                className="hover:bg-muted min-h-8 rounded-lg border px-3 text-[13px] disabled:opacity-50"
                disabled={older.loading}
                aria-busy={older.loading || undefined}
                onClick={loadOlder}
              >
                {older.loading ? t('operations.runs.loading') : t('operations.runs.loadOlder')}
              </button>
            </div>
          ) : runs.length ? (
            <p className="text-muted-foreground m-0 text-right text-xs">{t('operations.runs.end')}</p>
          ) : null}
        </div>
      )}
    </Panel>
  )
}
