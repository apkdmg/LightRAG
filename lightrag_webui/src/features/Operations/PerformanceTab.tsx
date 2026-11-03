import { useTranslation } from 'react-i18next'
import ComingLater from './components/ComingLater'
import { useOpsShell } from './context'

export default function PerformanceTab() {
  const { t } = useTranslation()
  const { status, openInsights } = useOpsShell()
  return (
    <ComingLater title={t('operations.later.performance.title')} body={t('operations.later.performance.body')}>
      {/* Cost and token questions belong to Token Insights (§2.1 coexistence rules). */}
      {status?.features?.insights ? (
        <button type="button" className="hover:bg-muted min-h-8 rounded-lg border px-3 text-[13px]" onClick={openInsights}>
          {t('operations.openInsights')}
        </button>
      ) : null}
    </ComingLater>
  )
}
