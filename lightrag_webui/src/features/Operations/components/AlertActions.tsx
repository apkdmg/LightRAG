import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ChevronDownIcon } from 'lucide-react'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import { cn } from '@/lib/utils'
import { opsApi } from '../api'
import { ActionOutcome } from '../actions'
import { durationText, sendAlertAction } from '../activity/alertActions'
import { MAINTENANCE_MAX_MINUTES, MAINTENANCE_PRESETS, MUTE_PRESETS, validMinutes } from '../activity/alerts'
import { fmtClock } from '../format'
import { useNow } from '../hooks'
import ConfirmActionDialog from './ConfirmActionDialog'

/**
 * Alert actions (§2.8, §2.10, FR-61): Ack, Mute ▾ and maintenance mode. Each
 * needs a reason, is audited on the server and answers `{ok, audit_id}`;
 * every result is a toast with the audit id and a refusal stays in its
 * dialog. Any admin may use them (they are not operator-only); the server
 * still checks that the admin is current.
 */

const smallButton = 'hover:bg-muted inline-flex items-center gap-1 rounded-lg border text-[13px] disabled:cursor-not-allowed disabled:opacity-50'

/** Ack… (§2.8): records who is on it; the alert stays firing until its rule clears (FR-61). */
export function AckButton({ alertId, title, compact, onOutcome }: { alertId: string; title: string; compact?: boolean; onOutcome?: (o: ActionOutcome) => void }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  return (
    <>
      <button type="button" className={cn(smallButton, compact ? 'min-h-7 px-2 text-xs' : 'min-h-8 px-2.5')} onClick={() => setOpen(true)}>
        {t('operations.alerts.ack.button')}
      </button>
      <ConfirmActionDialog
        open={open}
        onOpenChange={setOpen}
        action="ack"
        title={t('operations.alerts.ack.title', { title })}
        body={<p className="m-0">{t('operations.alerts.ack.body')}</p>}
        confirmLabel={t('operations.alerts.ack.confirm')}
        onConfirm={({ reason }) => sendAlertAction('ack', () => opsApi.ackAlert(alertId, reason))}
        onOutcome={onOutcome}
      />
    </>
  )
}

/**
 * Mute ▾ (FR-61): 1 h, 4 h, 24 h or 7 days, then the reason. A muted rule
 * is still evaluated and recorded; no email is sent for it until the mute
 * ends.
 */
export function MuteMenu({ rule, ruleTitle, compact, onOutcome }: { rule: string; ruleTitle: string; compact?: boolean; onOutcome?: (o: ActionOutcome) => void }) {
  const { t } = useTranslation()
  const [menu, setMenu] = useState(false)
  const [minutes, setMinutes] = useState<number | null>(null)
  const pick = (m: number) => {
    setMenu(false)
    setMinutes(m)
  }
  return (
    <>
      <Popover open={menu} onOpenChange={setMenu}>
        <PopoverTrigger asChild>
          <button
            type="button"
            className={cn(smallButton, compact ? 'min-h-7 px-2 text-xs' : 'min-h-8 px-2.5')}
            aria-label={t('operations.alerts.mute.menuLabel', { rule: ruleTitle })}
          >
            {t('operations.alerts.mute.button')}
            <ChevronDownIcon className="size-3.5" aria-hidden="true" />
          </button>
        </PopoverTrigger>
        <PopoverContent align="end" className="ops-portal flex w-44 flex-col gap-0.5 p-1.5">
          {MUTE_PRESETS.map((m) => (
            <button key={m} type="button" onClick={() => pick(m)} className="hover:bg-muted rounded-md px-2.5 py-1.5 text-left text-[13px]">
              {t('operations.alerts.mute.for', { duration: durationText(t, m) })}
            </button>
          ))}
        </PopoverContent>
      </Popover>
      <ConfirmActionDialog
        open={minutes !== null}
        onOpenChange={(open) => !open && setMinutes(null)}
        action="mute"
        title={t('operations.alerts.mute.title', { rule: ruleTitle, duration: durationText(t, minutes ?? MUTE_PRESETS[0]) })}
        body={<p className="m-0">{t('operations.alerts.mute.body')}</p>}
        confirmLabel={t('operations.alerts.mute.confirm')}
        ready={minutes !== null}
        onConfirm={({ reason }) => sendAlertAction('mute', () => opsApi.muteRule(rule, minutes ?? MUTE_PRESETS[0], reason))}
        onOutcome={onOutcome}
      />
    </>
  )
}

/** Unmute…: ends a rule's mute now (minutes 0). */
export function UnmuteButton({ rule, ruleTitle, onOutcome }: { rule: string; ruleTitle: string; onOutcome?: (o: ActionOutcome) => void }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  return (
    <>
      <button type="button" className={cn(smallButton, 'min-h-8 px-2.5')} onClick={() => setOpen(true)}>
        {t('operations.alerts.unmute.button')}
      </button>
      <ConfirmActionDialog
        open={open}
        onOpenChange={setOpen}
        action="unmute"
        title={t('operations.alerts.unmute.title', { rule: ruleTitle })}
        body={<p className="m-0">{t('operations.alerts.unmute.body')}</p>}
        confirmLabel={t('operations.alerts.unmute.confirm')}
        onConfirm={({ reason }) => sendAlertAction('unmute', () => opsApi.muteRule(rule, 0, reason))}
        onOutcome={onOutcome}
      />
    </>
  )
}

const selectClass = 'border-input bg-background text-foreground h-9 rounded-md border px-2 text-[13px]'

/**
 * Start maintenance mode (FR-64): a duration and a reason. Alerts are still
 * evaluated and recorded; no alert email is sent and the header reads
 * Maintenance until it ends, by itself or with End.
 */
export function MaintenanceDialog({ open, onOpenChange, onOutcome }: { open: boolean; onOpenChange: (open: boolean) => void; onOutcome?: (o: ActionOutcome) => void }) {
  const { t } = useTranslation()
  const [minutes, setMinutes] = useState<number>(60)
  const now = useNow(30_000, open)
  const ok = validMinutes(minutes, MAINTENANCE_MAX_MINUTES)
  return (
    <ConfirmActionDialog
      open={open}
      onOpenChange={onOpenChange}
      action="maintenance"
      title={t('operations.maintenance.start.title')}
      body={<p className="m-0">{t('operations.maintenance.start.body')}</p>}
      extra={
        <label className="flex flex-col gap-1 text-[13px]">
          <span className="font-medium">{t('operations.maintenance.duration')}</span>
          <select className={selectClass} value={minutes} onChange={(e) => setMinutes(Number(e.target.value))}>
            {MAINTENANCE_PRESETS.map((m) => (
              <option key={m} value={m}>
                {durationText(t, m)}
              </option>
            ))}
          </select>
          <span className="text-muted-foreground text-xs">{t('operations.maintenance.endsAt', { time: fmtClock(now + minutes * 60_000).slice(0, 5) })}</span>
        </label>
      }
      ready={ok}
      confirmLabel={t('operations.maintenance.start.confirm')}
      onConfirm={({ reason }) => sendAlertAction('maintenance', () => opsApi.setMaintenance(minutes, reason))}
      onOutcome={onOutcome}
    />
  )
}

/** End maintenance now (minutes 0); alert email resumes for what is still open. */
export function EndMaintenanceDialog({ open, onOpenChange, onOutcome }: { open: boolean; onOpenChange: (open: boolean) => void; onOutcome?: (o: ActionOutcome) => void }) {
  const { t } = useTranslation()
  return (
    <ConfirmActionDialog
      open={open}
      onOpenChange={onOpenChange}
      action="maintenance_end"
      title={t('operations.maintenance.end.title')}
      body={<p className="m-0">{t('operations.maintenance.end.body')}</p>}
      confirmLabel={t('operations.maintenance.end.confirm')}
      onConfirm={({ reason }) => sendAlertAction('maintenance_end', () => opsApi.setMaintenance(0, reason))}
      onOutcome={onOutcome}
    />
  )
}
