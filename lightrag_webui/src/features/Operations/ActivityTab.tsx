import { useTranslation } from 'react-i18next'
import AuditView from './activity/AuditView'
import ComingLater from './components/ComingLater'

/**
 * Activity (§2.8). The audit log ships with admin actions; alerts, system
 * events and the incident export arrive with alerting and say so.
 */
export default function ActivityTab() {
  const { t } = useTranslation()
  return (
    <div className="flex flex-col gap-4">
      <AuditView />
      <ComingLater title={t('operations.later.activity.title')} body={t('operations.later.activity.body')} />
    </div>
  )
}
