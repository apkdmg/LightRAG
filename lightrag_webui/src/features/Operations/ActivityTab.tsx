import { useTranslation } from 'react-i18next'
import ComingLater from './components/ComingLater'

export default function ActivityTab() {
  const { t } = useTranslation()
  return <ComingLater title={t('operations.later.activity.title')} body={t('operations.later.activity.body')} />
}
