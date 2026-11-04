import { useEffect, useId, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { EllipsisIcon } from 'lucide-react'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import { cn } from '@/lib/utils'
import { openHostedRun, useActionHost } from '../actionHost'
import { ActionResponse, opsApi } from '../api'
import {
  ActionContext,
  ActionKind,
  ActionOffer,
  ActionOutcome,
  DryRunCounts,
  actionOffers,
  dryRunCounts,
  outcomeOf,
  outcomeOfError,
  releaseFacts
} from '../actions'
import { useOpsShell } from '../context'
import { fmtAge, fmtCount } from '../format'
import { useOpsPoll } from '../poll'
import ConfirmActionDialog from './ConfirmActionDialog'

/** Facts the dialogs quote: how many emails wait, whether a run was asked for again. */
export type ActionFacts = {
  /** Records waiting in the email queue, for the drain copy. */
  emailPending?: number | null
  /** A provider pause is active: the run dialog says so next to "override pause". */
  paused?: boolean | null
}

const send = async (action: ActionKind, call: () => Promise<ActionResponse>, hints?: { requestPending?: boolean }): Promise<ActionOutcome> => {
  try {
    return outcomeOf(action, await call(), hints)
  } catch (err) {
    return outcomeOfError(action, err)
  }
}

function CancelDialog({ workspace, open, onOpenChange, onOutcome }: DialogProps) {
  const { t } = useTranslation()
  return (
    <ConfirmActionDialog
      open={open}
      onOpenChange={onOpenChange}
      action="cancel"
      title={t('operations.actions.cancel.title', { workspace })}
      body={<p className="m-0">{t('operations.actions.cancel.body')}</p>}
      confirmLabel={t('operations.actions.cancel.confirm')}
      cancelLabel={t('operations.actions.cancel.keep')}
      onConfirm={({ reason }) => send('cancel', () => opsApi.cancel(workspace, reason))}
      onOutcome={onOutcome}
    />
  )
}

/** "Will process 310 pending + 4 failed (including 2 user-cancelled) docs in fsktm." plus what a run now would meet. */
export function DryRunPreview({
  counts,
  workspace,
  live
}: {
  counts: DryRunCounts
  workspace: string
  live?: ActionResponse['live']
}) {
  const { t } = useTranslation()
  const meets = live?.destructive_busy ? 'destructive' : live?.busy ? 'busy' : live?.paused ? 'paused' : null
  return (
    <div className="bg-muted/50 flex flex-col gap-1 rounded-md border px-3 py-2 text-[13px]" aria-live="polite">
      <span className="font-medium">{t('operations.actions.run.previewTitle')}</span>
      {counts.total ? (
        <span>
          {t('operations.actions.run.preview', {
            pending: fmtCount(counts.pending),
            failed: fmtCount(counts.failed),
            cancelled: fmtCount(counts.cancelled),
            workspace
          })}
          {counts.inFlight ? ` ${t('operations.actions.run.previewInterrupted', { count: counts.inFlight })}` : ''}
        </span>
      ) : (
        <span>{t('operations.actions.run.previewNothing', { workspace })}</span>
      )}
      {meets ? <span className="text-[color:var(--ops-warn)]">{t(`operations.actions.run.meets.${meets}`)}</span> : null}
    </div>
  )
}

/** Each opening of a dialog counts once, so what it reads on open is read afresh. */
function useOpenSession(open: boolean) {
  const [session, setSession] = useState({ open, n: 0 })
  if (session.open !== open) setSession({ open, n: open ? session.n + 1 : session.n })
  return session.n
}

/**
 * Run pipeline now (§2.10): the dry run loads when the dialog opens (the
 * server counts what would run and starts nothing), then Run now starts it.
 * It retries every pending and failed document, user-cancelled ones
 * included, and is refused during a provider pause unless "override pause"
 * is ticked. With nothing to process the confirm stays disabled.
 */
function RunDialog({ workspace, open, onOpenChange, onOutcome, facts }: DialogProps) {
  const { t } = useTranslation()
  const [override, setOverride] = useState(false)
  const session = useOpenSession(open)
  const dry = useOpsPoll((signal) => opsApi.runPreview(workspace, signal), [workspace, session], undefined, { enabled: open })
  const fresh = !!dry.data && !dry.loading
  const counts = fresh ? dryRunCounts(dry.data?.would_process) : null
  const paused = !!dry.data?.live?.paused || !!facts?.paused
  const change = (next: boolean) => {
    if (!next) setOverride(false)
    onOpenChange(next)
  }
  return (
    <ConfirmActionDialog
      open={open}
      onOpenChange={change}
      action="run"
      title={t('operations.actions.run.title', { workspace })}
      body={<p className="m-0">{t('operations.actions.run.body', { workspace })}</p>}
      extra={
        <>
          {counts ? (
            <DryRunPreview counts={counts} workspace={workspace} live={dry.data?.live} />
          ) : dry.error && !dry.loading ? (
            <div className="flex flex-wrap items-center gap-2 text-[13px]" role="alert">
              <span className="text-[color:var(--ops-crit)]">
                {t('operations.actions.run.previewFailed')} {t(dry.error.messageKey)}
              </span>
              <button type="button" className="hover:bg-muted min-h-7 rounded-lg border px-2 text-xs" onClick={dry.reload}>
                {t('operations.error.retry')}
              </button>
            </div>
          ) : (
            <p className="text-muted-foreground m-0 text-[13px]">{t('operations.actions.run.previewLoading')}</p>
          )}
          <label className="inline-flex items-start gap-2 text-[13px]">
            <input type="checkbox" className="mt-0.5" checked={override} onChange={(e) => setOverride(e.target.checked)} />
            <span className="flex flex-col">
              {t('operations.actions.run.override')}
              <span className={cn('text-xs', paused ? 'text-[color:var(--ops-warn)]' : 'text-muted-foreground')}>
                {paused ? t('operations.actions.run.overridePaused') : t('operations.actions.run.overrideHint')}
              </span>
            </span>
          </label>
        </>
      }
      ready={!!counts && counts.total > 0}
      confirmLabel={t('operations.actions.run.confirm')}
      cancelLabel={t('operations.actions.run.notNow')}
      onConfirm={({ reason }) => send('run', () => opsApi.runNow(workspace, { reason, override_pause: override }))}
      onOutcome={onOutcome}
    />
  )
}

function DrainDialog({ workspace, open, onOpenChange, onOutcome, facts }: DialogProps) {
  const { t } = useTranslation()
  const count = facts?.emailPending
  return (
    <ConfirmActionDialog
      open={open}
      onOpenChange={onOpenChange}
      action="drain_email"
      title={t('operations.actions.drain.title', { workspace })}
      body={
        <>
          <p className="m-0">
            {typeof count === 'number' && count > 0 ? t('operations.actions.drain.body', { count }) : t('operations.actions.drain.bodyUnknown')}
          </p>
          <p className="m-0">{t('operations.actions.drain.limits')}</p>
        </>
      }
      confirmLabel={t('operations.actions.drain.confirm')}
      onConfirm={({ reason }) => send('drain_email', () => opsApi.drainEmail(workspace, reason))}
      onOutcome={onOutcome}
    />
  )
}

/**
 * Release busy (§2.10, Gunicorn only, operator only): for a worker that is
 * gone. The dialog reads the workspace's live state when it opens, quotes
 * who held `busy` and since when, and sends the run id and job start it saw
 * so the server can refuse if anything changed (CAS). The workspace name
 * must be typed.
 */
function ReleaseDialog({ workspace, open, onOpenChange, onOutcome, onRunNow }: DialogProps) {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  // Every opening reads the live state afresh: the copy and the CAS values must be current.
  const session = useOpenSession(open)
  const live = useOpsPoll((signal) => opsApi.workspace(workspace, signal), [workspace, session], undefined, { enabled: open })
  // Ages are measured against when the live state was read (it is only shown once read).
  const facts = releaseFacts(live.data, live.updatedAt ?? 0)
  const loaded = !!live.data && !live.loading
  const owner =
    facts.worker || facts.pid !== null
      ? facts.beatAge !== null
        ? t('operations.actions.release.ownerStopped', { worker: facts.worker ?? '–', pid: facts.pid ?? '–', age: fmtAge(facts.beatAge) })
        : t('operations.actions.release.ownerNoBeat', { worker: facts.worker ?? '–', pid: facts.pid ?? '–' })
      : t('operations.actions.release.noOwner')
  return (
    <ConfirmActionDialog
      open={open}
      onOpenChange={onOpenChange}
      action="release_busy"
      destructive
      title={t('operations.actions.release.title', { workspace })}
      body={
        <>
          <p className="m-0 font-medium">{t('operations.actions.release.onlyGone')}</p>
          {loaded && facts.busy ? (
            <p className="m-0">
              {owner}
              {facts.processGone ? ` ${t('operations.actions.release.processGone')}` : ''}
            </p>
          ) : null}
          <p className="m-0">{t('operations.actions.release.body')}</p>
          <p className="m-0">{t('operations.actions.release.frozen')}</p>
        </>
      }
      extra={
        live.error && !live.data ? (
          <p className="m-0 text-[13px] text-[color:var(--ops-crit)]">{t('operations.actions.release.liveFailed')}</p>
        ) : !loaded ? (
          <p className="text-muted-foreground m-0 text-[13px]">{t('operations.actions.release.loading')}</p>
        ) : !facts.busy ? (
          <p className="m-0 text-[13px]">{t('operations.actions.release.notBusy')}</p>
        ) : null
      }
      ready={loaded && facts.busy}
      allowed={!!status?.operator}
      notAllowedText={t('operations.actions.hint.operatorOnly')}
      typedExpected={workspace}
      confirmLabel={t('operations.actions.release.confirm')}
      onConfirm={({ reason, typed }) =>
        send(
          'release_busy',
          () =>
            opsApi.releaseBusy(workspace, {
              reason,
              expected_run_id: facts.runId,
              expected_job_start: facts.jobStart,
              confirm: typed.trim()
            }),
          { requestPending: facts.requestPending }
        )
      }
      onOutcome={(o) => {
        // The run changed since the dialog read it: read it again, so a retry compares the new values.
        if (o.code === 'cas_mismatch') live.reload()
        onOutcome?.(o)
      }}
      onRunNow={onRunNow}
    />
  )
}

type DialogProps = {
  workspace: string
  open: boolean
  onOpenChange: (open: boolean) => void
  onOutcome?: (outcome: ActionOutcome) => void
  onRunNow?: () => void
  facts?: ActionFacts
}

/**
 * The four action dialogs of one workspace. Closed dialogs render nothing;
 * they stay mounted once used so a closing dialog can play its exit.
 */
export function ActionDialogs({
  workspace,
  dialog,
  setDialog,
  facts,
  onChanged
}: {
  workspace: string
  dialog: ActionKind | null
  setDialog: (kind: ActionKind | null) => void
  facts?: ActionFacts
  /** A final outcome arrived: refresh what the view shows. */
  onChanged?: () => void
}) {
  const props = (kind: ActionKind): DialogProps => ({
    workspace,
    facts,
    open: dialog === kind,
    onOpenChange: (open) => setDialog(open ? kind : null),
    onOutcome: (o) => {
      if (o.tone !== 'refused' || o.code === 'cas_mismatch') onChanged?.()
    },
    // Released with a queued "process again" request: offer Run now (§3.9).
    // It opens in the shell's host: the released row may already be gone.
    onRunNow: () => openHostedRun(workspace)
  })
  return (
    <>
      <CancelDialog {...props('cancel')} />
      <RunDialog {...props('run')} />
      <DrainDialog {...props('drain_email')} />
      <ReleaseDialog {...props('release_busy')} />
    </>
  )
}

/**
 * The console's one action host (see actionHost.ts), rendered by the shell:
 * dialogs opened from a follow-up that outlived its row. Forgotten when the
 * console unmounts.
 */
export function HostedActionDialogs() {
  const hosted = useActionHost((s) => s.hosted)
  const setKind = useActionHost((s) => s.setKind)
  const reset = useActionHost((s) => s.reset)
  useEffect(() => reset, [reset])
  if (!hosted) return null
  return <ActionDialogs key={hosted.workspace} workspace={hosted.workspace} dialog={hosted.kind} setDialog={setKind} />
}

function OfferButton({ offer, onClick, compact }: { offer: ActionOffer; onClick: () => void; compact?: boolean }) {
  const { t } = useTranslation()
  const hintId = useId()
  return (
    <span title={offer.hint ? t(`operations.actions.hint.${offer.hint}`) : undefined} className="inline-flex">
      <button
        type="button"
        disabled={offer.disabled}
        aria-describedby={offer.hint ? hintId : undefined}
        onClick={onClick}
        className={cn(
          'hover:bg-muted rounded-lg border px-3 text-[13px] disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent',
          compact ? 'min-h-7 px-2 text-xs' : 'min-h-8',
          offer.kind === 'release_busy' && !offer.disabled && 'border-[color:var(--ops-crit)] text-[color:var(--ops-crit)]'
        )}
      >
        {t(`operations.actions.button.${offer.kind}`)}
      </button>
      {offer.hint ? (
        <span id={hintId} className="sr-only">
          {t(`operations.actions.hint.${offer.hint}`)}
        </span>
      ) : null}
    </span>
  )
}

/**
 * The Workspace drawer's action row (§2.9): Cancel…, Release busy…
 * (multi-worker only; disabled with a hint for non-operators), Run pipeline
 * now…, Drain email…. Nothing renders while actions are off on the server
 * (OPS_ACTIONS_ENABLED). Each opens its confirm dialog; every result is a
 * toast with the audit id.
 */
export function WorkspaceActionBar({
  workspace,
  ctx,
  facts,
  onChanged,
  kinds
}: {
  workspace: string
  ctx: Omit<ActionContext, 'status'>
  facts?: ActionFacts
  onChanged?: () => void
  kinds?: ActionKind[]
}) {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const [dialog, setDialog] = useState<ActionKind | null>(null)
  const [used, setUsed] = useState(false)
  if (dialog && !used) setUsed(true)
  const offers = actionOffers({ ...ctx, status }, kinds)
  if (!offers.length) return null
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap gap-2" role="group" aria-label={t('operations.actions.groupLabel', { workspace })}>
        {offers.map((o) => (
          <OfferButton key={o.kind} offer={o} onClick={() => setDialog(o.kind)} />
        ))}
      </div>
      {used ? <ActionDialogs workspace={workspace} dialog={dialog} setDialog={setDialog} facts={facts} onChanged={onChanged} /> : null}
    </div>
  )
}

/**
 * A row's "⋯" menu (§2.4 Active and Stuck): the actions that apply to the
 * row's workspace plus "Open workspace". The menu and its dialogs stop
 * clicks from reaching the row (which opens a drawer).
 */
export function RowActions({
  workspace,
  ctx,
  facts,
  onChanged,
  kinds,
  onOpenWorkspace
}: {
  workspace: string
  ctx: Omit<ActionContext, 'status'>
  facts?: ActionFacts
  onChanged?: () => void
  kinds?: ActionKind[]
  onOpenWorkspace?: () => void
}) {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const [menu, setMenu] = useState(false)
  const [dialog, setDialog] = useState<ActionKind | null>(null)
  const [used, setUsed] = useState(false)
  if (dialog && !used) setUsed(true)
  const offers = actionOffers({ ...ctx, status }, kinds)
  if (!offers.length) return null
  const pick = (kind: ActionKind) => {
    setMenu(false)
    setDialog(kind)
  }
  return (
    // React events bubble through portals: keep menu and dialog clicks off the row.
    <div className="inline-flex" onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
      <Popover open={menu} onOpenChange={setMenu}>
        <PopoverTrigger asChild>
          <button
            type="button"
            className="hover:bg-muted inline-flex size-8 items-center justify-center rounded-lg border"
            aria-label={t('operations.actions.menuLabel', { workspace })}
          >
            <EllipsisIcon className="size-4" aria-hidden="true" />
          </button>
        </PopoverTrigger>
        <PopoverContent align="end" className="ops-portal flex w-60 flex-col gap-0.5 p-1.5">
          {offers.map((o) => (
            <button
              key={o.kind}
              type="button"
              disabled={o.disabled}
              onClick={() => pick(o.kind)}
              className="hover:bg-muted flex flex-col items-start rounded-md px-2.5 py-1.5 text-left text-[13px] disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:bg-transparent"
            >
              <span>{t(`operations.actions.button.${o.kind}`)}</span>
              {o.hint ? <span className="text-muted-foreground text-xs">{t(`operations.actions.hint.${o.hint}`)}</span> : null}
            </button>
          ))}
          {onOpenWorkspace ? (
            <button
              type="button"
              onClick={() => {
                setMenu(false)
                onOpenWorkspace()
              }}
              className="hover:bg-muted mt-0.5 rounded-md border-t px-2.5 py-1.5 text-left text-[13px]"
            >
              {t('operations.actions.openWorkspace')}
            </button>
          ) : null}
        </PopoverContent>
      </Popover>
      {used ? <ActionDialogs workspace={workspace} dialog={dialog} setDialog={setDialog} facts={facts} onChanged={onChanged} /> : null}
    </div>
  )
}
