import { useTranslation } from 'react-i18next'

/**
 * FR-4: every value carries its scope. "host" values come from Postgres rows
 * written by every process (or a leased singleton); "worker <pid>" values
 * describe one process only and never appear on the Overview.
 */
export default function ScopeBadge({ scope }: { scope?: string | null }) {
  const { t } = useTranslation()
  const raw = (scope ?? 'host').trim()
  const worker = /^worker[\s:]+(\S+)$/i.exec(raw)
  const text = worker ? t('operations.scope.worker', { pid: worker[1] }) : t('operations.scope.host')
  return (
    <span className="text-muted-foreground font-mono text-[11px]" title={t('operations.scope.hint')}>
      {text}
    </span>
  )
}
