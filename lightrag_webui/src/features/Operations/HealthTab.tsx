import { useTranslation } from 'react-i18next'
import ComingLater from './components/ComingLater'

export default function HealthTab() {
  const { t } = useTranslation()
  return <ComingLater title={t('operations.later.health.title')} body={t('operations.later.health.body')} />
}
