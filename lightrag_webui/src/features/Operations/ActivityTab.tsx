import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { DownloadIcon } from 'lucide-react'
import { Seg } from '@/features/TokenInsights/ui'
import AlertsView from './activity/AlertsView'
import AuditView from './activity/AuditView'
import EventsView from './activity/EventsView'
import IncidentExportDialog from './components/IncidentExportDialog'
import { useOpsNav, useOpsShell } from './context'
import { OPS_SUBS } from './hashState'

type Sub = 'alerts' | 'audit' | 'events' | 'export'

/** Incident export (§2.8 section 4): what the timeline holds, and the dialog that downloads it. */
function IncidentExportSection() {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const [open, setOpen] = useState(false)
  const operator = !!status?.operator
  return (
    <section className="bg-card flex max-w-[760px] flex-col gap-2 rounded-[10px] border p-4">
      <h2 className="m-0 text-[15px] font-semibold">{t('operations.incident.title')}</h2>
      <p className="text-muted-foreground m-0 text-sm">{t('operations.incident.body')}</p>
      <div className="flex flex-wrap items-center gap-2 pt-1">
        <button
          type="button"
          disabled={!operator}
          title={operator ? undefined : t('operations.incident.operatorOnly')}
          onClick={() => setOpen(true)}
          className="hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-2.5 text-[13px] disabled:cursor-not-allowed disabled:opacity-60"
        >
          <DownloadIcon className="size-3.5" aria-hidden="true" />
          {t('operations.incident.button')}
        </button>
        {!operator ? <span className="text-muted-foreground text-xs">{t('operations.incident.operatorOnly')}</span> : null}
      </div>
      {operator ? <IncidentExportDialog open={open} onOpenChange={setOpen} /> : null}
    </section>
  )
}

/**
 * Activity (§2.8): Alerts (open, history, mutes, rules, maintenance), the
 * audit log, system events and the incident export. The section lives in
 * the URL (`osub`), so a link opens it directly; only the visible section
 * polls.
 */
export default function ActivityTab() {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const sub = (nav.view.sub && OPS_SUBS.activity.includes(nav.view.sub) ? nav.view.sub : 'alerts') as Sub
  // Only a section the user switched to fades in; the first paint and deep links do not.
  const [fadeSub, setFadeSub] = useState<Sub | null>(null)
  const select = (v: Sub) => {
    setFadeSub(v)
    nav.setSub(v)
  }
  const options: { value: Sub; label: string }[] = [
    { value: 'alerts', label: t('operations.activity.sub.alerts') },
    { value: 'audit', label: t('operations.activity.sub.audit') },
    { value: 'events', label: t('operations.activity.sub.events') },
    { value: 'export', label: t('operations.activity.sub.export') }
  ]
  return (
    <div className="flex flex-col gap-4">
      <div className="overflow-x-auto">
        <Seg value={sub} onChange={select} options={options} label={t('operations.activity.sub.label')} />
      </div>
      <div key={sub} className={fadeSub === sub ? 'ops-fade-in' : undefined}>
        {sub === 'alerts' ? <AlertsView /> : null}
        {sub === 'audit' ? <AuditView /> : null}
        {sub === 'events' ? <EventsView /> : null}
        {sub === 'export' ? <IncidentExportSection /> : null}
      </div>
    </div>
  )
}
