import { useTranslation } from 'react-i18next'
import ComingLater from './components/ComingLater'

export default function ProcessingTab() {
  const { t } = useTranslation()
  return <ComingLater title={t('operations.later.processing.title')} body={t('operations.later.processing.body')} />
}
