import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { MailWarningIcon, WrenchIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { AlertChannel, Maintenance } from '../api'
import { alertActionsEnabled } from '../activity/alertActions'
import { alertChannelOf, bannerEnters, maintenanceActive } from '../activity/alerts'
import { useOpsNav, useOpsOverview, useOpsShell } from '../context'
import { fmtClock, fmtCount, fmtStamp } from '../format'
import { useNow } from '../hooks'
import { EndMaintenanceDialog } from './AlertActions'

const KNOWN_REASONS = new Set(['smtp_probe_failing', 'smtp_not_configured', 'send_failed', 'no_recipients'])

/**
 * "Alert email is NOT being delivered" (FR-63): the SMTP probe fails, SMTP
 * is not configured, or recent alert sends failed. Not a live region (only
 * the state chip and the "N new" pill are, NFR-10); it says why and since
 * when, and opens the SMTP dependency.
 */
export function AlertEmailBanner({ channel, enter, onOpenSmtp }: { channel: AlertChannel; enter?: boolean; onOpenSmtp?: () => void }) {
  const { t } = useTranslation()
  const reason = channel.reason && KNOWN_REASONS.has(channel.reason) ? channel.reason : 'unknown'
  return (
    <div className={cn('ops-banner', enter && 'ops-fade-in')} data-tone="crit" data-banner="alert-email">
      <MailWarningIcon className="ops-banner-icon size-4 flex-none" aria-hidden="true" />
      <span className="min-w-0 flex-1">
        <strong>{t('operations.banner.email.title')}</strong> {t(`operations.banner.email.reason.${reason}`)}
        {channel.since ? ` ${t('operations.banner.email.since', { time: fmtStamp(channel.since) })}` : ''}.
        {typeof channel.failures === 'number' && channel.failures > 0 ? ` ${t('operations.banner.email.failures', { count: channel.failures, countText: fmtCount(channel.failures) })}` : ''}
      </span>
      {onOpenSmtp ? (
        <button type="button" className="hover:bg-background/60 min-h-8 rounded-lg border px-2.5 text-[13px]" onClick={onOpenSmtp}>
          {t('operations.banner.email.open')}
        </button>
      ) : null}
    </div>
  )
}

/** "Maintenance mode until 13:00 (set by opso: "deploy")" with End… (FR-64). */
export function MaintenanceBanner({ maintenance, enter, onEnd }: { maintenance: Maintenance; enter?: boolean; onEnd?: () => void }) {
  const { t } = useTranslation()
  return (
    <div className={cn('ops-banner', enter && 'ops-fade-in')} data-tone="unknown" data-banner="maintenance">
      <WrenchIcon className="ops-banner-icon size-4 flex-none" aria-hidden="true" />
      <span className="min-w-0 flex-1">
        <strong>{t('operations.banner.maintenance.title', { time: fmtClock(maintenance.until).slice(0, 5) })}</strong>
        {maintenance.by || maintenance.reason ? ` ${t('operations.banner.maintenance.by', { by: maintenance.by ?? '–', reason: maintenance.reason ?? '–' })}` : ''}{' '}
        <span className="text-muted-foreground">{t('operations.banner.maintenance.effect')}</span>
      </span>
      {onEnd ? (
        <button type="button" className="hover:bg-background/60 min-h-8 rounded-lg border px-2.5 text-[13px]" onClick={onEnd}>
          {t('operations.maintenance.endButton')}
        </button>
      ) : null}
    </div>
  )
}

/** Whether a banner shown now fades in: only when it appeared after the first poll (§2.12 rule 7). */
function useEnters(shown: boolean, baseline: boolean) {
  const [seen, setSeen] = useState<{ shown: boolean | null; enter: boolean }>({ shown: null, enter: false })
  if (seen.shown !== shown) {
    const enter = bannerEnters(seen.shown, shown, baseline)
    setSeen({ shown, enter })
    return enter
  }
  return seen.enter
}

/**
 * The shell's banners, under the tab row (§2.2): alert email not delivered
 * (FR-63) and maintenance mode (FR-64). Both come from the /overview poll the
 * header already makes; nothing extra is fetched.
 */
export default function OpsBanners() {
  const nav = useOpsNav()
  const overview = useOpsOverview()
  const { status } = useOpsShell()
  const now = useNow(30_000)
  const [ending, setEnding] = useState(false)
  const data = overview?.data && overview.data.available !== false ? overview.data : null
  const channel = alertChannelOf(data)
  const emailBroken = channel?.ok === false
  const maintenance = data?.maintenance ?? null
  const inMaintenance = maintenanceActive(maintenance, now)
  const baseline = overview?.baseline ?? true
  const emailEnters = useEnters(emailBroken, baseline)
  const maintenanceEnters = useEnters(inMaintenance, baseline)
  const canAct = alertActionsEnabled(status)
  if (!emailBroken && !inMaintenance) return null
  return (
    <div className="flex flex-col gap-2">
      {emailBroken && channel ? (
        <AlertEmailBanner channel={channel} enter={emailEnters} onOpenSmtp={() => nav.openDrawer({ kind: 'dependency', id: 'smtp' })} />
      ) : null}
      {inMaintenance && maintenance ? (
        <MaintenanceBanner maintenance={maintenance} enter={maintenanceEnters} onEnd={canAct ? () => setEnding(true) : undefined} />
      ) : null}
      {canAct ? <EndMaintenanceDialog open={ending} onOpenChange={setEnding} onOutcome={(o) => o.tone === 'success' && overview?.reload()} /> : null}
    </div>
  )
}
