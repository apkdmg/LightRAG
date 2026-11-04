import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { TimerIcon } from 'lucide-react'
import Input from '@/components/ui/Input'
import { LogCapture, LogCaptureState, opsApi } from '../api'
import { ActionOutcome } from '../actions'
import { durationText, sendAlertAction } from '../activity/alertActions'
import { captureRemaining, capturePresets, fmtCountdown, validMinutes, validModuleName } from '../activity/alerts'
import { useNow } from '../hooks'
import { LogCaptureInfo } from '../logCapture'
import ConfirmActionDialog from './ConfirmActionDialog'

const selectClass = 'border-input bg-background text-foreground h-9 rounded-md border px-2 text-[13px]'

/**
 * "Capture INFO…" (FR-26, §2.10): one module, at most 60 minutes, a reason.
 * The server stores the capture in KV, every worker's log sink reads it, it
 * expires by itself and its volume is capped; the start is audited.
 */
export function CaptureDialog({
  open,
  onOpenChange,
  state,
  initialModule,
  onOutcome
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  state: LogCaptureState | null
  /** The module the log filter names, preselected when it may be captured. */
  initialModule?: string
  onOutcome?: (o: ActionOutcome, module: string) => void
}) {
  const { t } = useTranslation()
  const modules = state?.modules ?? []
  const presets = capturePresets(state?.max_minutes ?? 60)
  const pickDefault = () => (initialModule && (!modules.length || modules.includes(initialModule)) ? initialModule : (modules[0] ?? ''))
  const [module, setModule] = useState(pickDefault)
  const [minutes, setMinutes] = useState<number>(presets[presets.length - 1])
  const [wasOpen, setWasOpen] = useState(open)
  if (open !== wasOpen) {
    setWasOpen(open)
    if (open) {
      setModule(pickDefault())
      setMinutes(presets[presets.length - 1])
    }
  }
  const moduleOk = modules.length ? modules.includes(module) : validModuleName(module)
  const ok = moduleOk && validMinutes(minutes, presets[presets.length - 1])
  return (
    <ConfirmActionDialog
      open={open}
      onOpenChange={onOpenChange}
      action="capture"
      title={t('operations.capture.title')}
      body={<p className="m-0">{t('operations.capture.body')}</p>}
      extra={
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="flex flex-col gap-1 text-[13px]">
            <span className="font-medium">{t('operations.capture.module')}</span>
            {modules.length ? (
              <select className={selectClass} value={module} onChange={(e) => setModule(e.target.value)}>
                {modules.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            ) : (
              <Input
                value={module}
                onChange={(e) => setModule(e.target.value)}
                placeholder={t('operations.capture.modulePlaceholder')}
                spellCheck={false}
                autoComplete="off"
                className="font-mono"
                aria-invalid={module.trim() !== '' && !moduleOk ? true : undefined}
              />
            )}
            {!modules.length ? <span className="text-muted-foreground text-xs">{t('operations.capture.moduleHint')}</span> : null}
          </label>
          <label className="flex flex-col gap-1 text-[13px]">
            <span className="font-medium">{t('operations.capture.duration')}</span>
            <select className={selectClass} value={minutes} onChange={(e) => setMinutes(Number(e.target.value))}>
              {presets.map((m) => (
                <option key={m} value={m}>
                  {durationText(t, m)}
                </option>
              ))}
            </select>
          </label>
        </div>
      }
      ready={ok}
      confirmLabel={t('operations.capture.confirm')}
      onConfirm={({ reason }) => sendAlertAction('capture', () => opsApi.startCapture({ module: module.trim(), minutes, reason }))}
      onOutcome={(o) => onOutcome?.(o, module.trim())}
    />
  )
}

/** Stop the capture now (0 minutes); lines already stored stay until retention. */
function StopCaptureDialog({ open, onOpenChange, capture, onOutcome }: { open: boolean; onOpenChange: (open: boolean) => void; capture: LogCapture; onOutcome?: () => void }) {
  const { t } = useTranslation()
  return (
    <ConfirmActionDialog
      open={open}
      onOpenChange={onOpenChange}
      action="capture_stop"
      title={t('operations.capture.stopTitle', { module: capture.module })}
      body={<p className="m-0">{t('operations.capture.stopBody')}</p>}
      confirmLabel={t('operations.capture.stopConfirm')}
      onConfirm={({ reason }) => sendAlertAction('capture_stop', () => opsApi.startCapture({ module: capture.module, minutes: 0, reason }))}
      onOutcome={onOutcome}
    />
  )
}

/**
 * The countdown of a running capture: "INFO capture · pipeline · 42:13 left".
 * Plain text that updates once a second (no motion, so nothing changes under
 * reduced motion); at zero it asks for a fresh state once.
 */
export function CaptureCountdown({
  state,
  fetchedAt,
  onExpired
}: {
  state: LogCaptureState
  fetchedAt: number | null
  onExpired?: () => void
}) {
  const { t } = useTranslation()
  const now = useNow(1000, !!state.active)
  const remaining = captureRemaining(state, fetchedAt, now)
  const expired = remaining === 0
  useEffect(() => {
    if (expired) onExpired?.()
  }, [expired, onExpired])
  const a = state.active
  if (!a || remaining === null) return null
  return (
    <span className="inline-flex flex-wrap items-center gap-x-1.5 text-[13px]">
      <TimerIcon className="size-3.5 flex-none" aria-hidden="true" />
      <strong>{t('operations.capture.active', { module: a.module })}</strong>
      {expired ? (
        <span className="text-muted-foreground">{t('operations.capture.ended')}</span>
      ) : (
        <span className="font-mono" aria-label={t('operations.capture.leftLabel', { time: fmtCountdown(remaining) })}>
          {t('operations.capture.left', { time: fmtCountdown(remaining) })}
        </span>
      )}
      {a.by ? <span className="text-muted-foreground">· {t('operations.capture.by', { by: a.by })}</span> : null}
    </span>
  )
}

/**
 * The INFO capture control of the Logs tab (FR-26): the running capture with
 * its countdown, "Show INFO lines" and Stop…, or Capture INFO… for
 * operators (disabled with a hint for other admins). Nothing renders when
 * the server does not offer captures.
 */
export default function LogCaptureControl({
  info,
  open,
  onOpenChange,
  initialModule,
  onShow
}: {
  info: LogCaptureInfo
  open: boolean
  onOpenChange: (open: boolean) => void
  initialModule?: string
  /** Switch the log filter to INFO lines of this module. */
  onShow: (module: string) => void
}) {
  const { t } = useTranslation()
  const [stopping, setStopping] = useState(false)
  const { poll, supported, canCapture } = info
  if (!supported) return null
  const state = poll.data
  const active = state?.active ?? null
  // An expired capture asks for a fresh state at zero (CaptureCountdown), which clears it.
  const live = !!active
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-lg border border-dashed px-3 py-1.5" data-capture={live ? 'on' : 'off'}>
      {state && active ? (
        <>
          <CaptureCountdown state={state} fetchedAt={poll.updatedAt} onExpired={poll.reload} />
          <button type="button" className="hover:bg-muted min-h-7 rounded-lg border px-2 text-xs" onClick={() => onShow(active.module)}>
            {t('operations.capture.show')}
          </button>
          {canCapture && live ? (
            <button type="button" className="hover:bg-muted min-h-7 rounded-lg border px-2 text-xs" onClick={() => setStopping(true)}>
              {t('operations.capture.stop')}
            </button>
          ) : null}
        </>
      ) : (
        <>
          <span className="text-muted-foreground text-[13px]">{t('operations.capture.none')}</span>
          <button
            type="button"
            disabled={!canCapture}
            title={canCapture ? undefined : info.sinkOff ? t('operations.capture.sinkOff') : t('operations.capture.operatorOnly')}
            className="hover:bg-muted min-h-7 rounded-lg border px-2 text-xs disabled:cursor-not-allowed disabled:opacity-60"
            onClick={() => onOpenChange(true)}
          >
            {t('operations.capture.start')}
          </button>
        </>
      )}
      {canCapture ? (
        <CaptureDialog
          open={open}
          onOpenChange={onOpenChange}
          state={state ?? null}
          initialModule={initialModule}
          onOutcome={(o, module) => {
            if (o.tone !== 'success') return
            poll.reload()
            onShow(module)
          }}
        />
      ) : null}
      {canCapture && active ? <StopCaptureDialog open={stopping} onOpenChange={setStopping} capture={active} onOutcome={poll.reload} /> : null}
    </div>
  )
}
