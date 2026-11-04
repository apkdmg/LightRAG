import { ReactNode, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { EyeIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { RevealKind, opsApi } from '../api'
import { ActionOutcome, actionsEnabled, outcomeOf, outcomeOfError } from '../actions'
import { useOpsShell } from '../context'
import { HIDDEN, REVEAL_SECONDS, RevealState, RevealTimer } from '../reveal'
import ConfirmActionDialog from './ConfirmActionDialog'

/**
 * A revealed value with its countdown (§2.9 "Revealed values show a 60 s
 * countdown with Keep visible"). The countdown text is plain text, so it
 * reads without the timer; "Keep visible" stops it and "Hide" drops the
 * value at once.
 */
export function RevealedValue({
  value,
  state,
  block,
  onKeep,
  onHide
}: {
  value: string
  state: RevealState
  block?: boolean
  onKeep: () => void
  onHide: () => void
}) {
  const { t } = useTranslation()
  const Box = block ? 'div' : 'span'
  return (
    <Box className={cn('flex flex-col gap-1', !block && 'items-start')}>
      <span
        className={cn(
          'rounded-md border border-dashed border-[color:var(--ops-warn)] px-2 py-1 font-mono text-xs break-all',
          block && 'bg-muted/40 whitespace-pre-wrap'
        )}
      >
        {value}
      </span>
      <span className="text-muted-foreground flex flex-wrap items-center gap-2 text-xs">
        <span>{state.kept ? t('operations.reveal.kept') : t('operations.reveal.hidesIn', { seconds: state.remaining })}</span>
        {!state.kept ? (
          <button type="button" className="hover:bg-muted min-h-7 rounded-lg border px-2" onClick={onKeep}>
            {t('operations.reveal.keep')}
          </button>
        ) : null}
        <button type="button" className="hover:bg-muted min-h-7 rounded-lg border px-2" onClick={onHide}>
          {t('operations.reveal.hide')}
        </button>
      </span>
    </Box>
  )
}

/**
 * A masked field with "Reveal…" (§2.10, §3.9, D3): an operator gives a
 * reason, the server reads the one field, redacts secrets and emails, audits
 * the reveal and answers the value. It is held in this component's state
 * only, never stored or put in the URL, and hides itself after 60 s unless
 * "Keep visible" is chosen. Non-operators, and servers with actions off,
 * see only the masked value. Log text is never revealable.
 */
export default function MaskedValue({
  kind,
  workspace,
  id,
  children,
  block
}: {
  kind: RevealKind
  workspace: string
  /** The document id (doc fields), or the run the job name belongs to (null when not known). */
  id: string | null
  /** The masked value as shown before a reveal. */
  children: ReactNode
  block?: boolean
}) {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const [open, setOpen] = useState(false)
  const [value, setValue] = useState<string | null>(null)
  const [reveal, setReveal] = useState<RevealState>(HIDDEN)
  const timerRef = useRef<RevealTimer | null>(null)
  const timer = () => {
    if (!timerRef.current)
      timerRef.current = new RevealTimer((s) => {
        setReveal(s)
        if (!s.visible) setValue(null)
      })
    return timerRef.current
  }
  // A different field, or leaving the drawer, drops the value.
  useEffect(
    () => () => {
      timerRef.current?.dispose()
      timerRef.current = null
    },
    [kind, workspace, id]
  )
  const [shownFor, setShownFor] = useState(`${kind}:${workspace}:${id}`)
  if (shownFor !== `${kind}:${workspace}:${id}`) {
    setShownFor(`${kind}:${workspace}:${id}`)
    setValue(null)
    setReveal(HIDDEN)
  }

  const allowed = actionsEnabled(status) && !!status?.operator
  const confirm = async ({ reason }: { reason: string }): Promise<ActionOutcome> => {
    try {
      const resp = await opsApi.reveal({ kind, workspace, id: id ?? undefined, reason })
      if (resp.value === null) return { ...outcomeOf('reveal', resp), tone: 'info', code: 'nothing_to_reveal' }
      setValue(resp.value)
      timer().start()
      return outcomeOf('reveal', resp)
    } catch (err) {
      return outcomeOfError('reveal', err)
    }
  }

  const Box = block ? 'div' : 'span'
  return (
    <Box className={cn('flex min-w-0 flex-col gap-1', !block && 'items-start')}>
      {value !== null && reveal.visible ? (
        <RevealedValue value={value} state={reveal} block={block} onKeep={() => timer().keep()} onHide={() => timer().hide()} />
      ) : (
        <Box className={cn('flex min-w-0 gap-2', block ? 'flex-col items-start' : 'flex-wrap items-center')}>
          {children}
          {allowed ? (
            <button
              type="button"
              className="hover:bg-muted inline-flex min-h-7 items-center gap-1 rounded-lg border px-2 text-xs"
              onClick={() => setOpen(true)}
            >
              <EyeIcon className="size-3.5" aria-hidden="true" />
              {t('operations.reveal.button')}
            </button>
          ) : null}
        </Box>
      )}
      {allowed ? (
        <ConfirmActionDialog
          open={open}
          onOpenChange={setOpen}
          action="reveal"
          title={t(`operations.reveal.title.${kind}`)}
          body={
            <>
              <p className="m-0">{t(`operations.reveal.body.${kind}`)}</p>
              <p className="m-0">{t('operations.reveal.hideNote', { seconds: REVEAL_SECONDS })}</p>
            </>
          }
          confirmLabel={t('operations.reveal.confirm')}
          onConfirm={confirm}
        />
      ) : null}
    </Box>
  )
}
