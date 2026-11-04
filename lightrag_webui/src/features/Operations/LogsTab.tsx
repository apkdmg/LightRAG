import { useTranslation } from 'react-i18next'
import ComingLater from './components/ComingLater'

export default function LogsTab() {
  const { t } = useTranslation()
  return <ComingLater title={t('operations.later.logs.title')} body={t('operations.later.logs.body')} />
}
