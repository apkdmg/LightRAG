import { useTranslation } from 'react-i18next'
import { CircleDashedIcon } from 'lucide-react'
import { DrawerKind } from '../hashState'

/**
 * A drawer link for an entity whose drawer ships later (run, log entry,
 * alert). Old links still open the page; they say so
 * instead of failing.
 */
export default function UnsupportedDrawer({ kind }: { kind: DrawerKind }) {
  const { t } = useTranslation()
  return (
    <div className="flex items-start gap-2">
      <CircleDashedIcon className="text-muted-foreground mt-0.5 size-4 flex-none" aria-hidden="true" />
      <p className="text-muted-foreground m-0 text-sm">{t('operations.drawer.later', { kind: t(`operations.drawer.kind.${kind}`) })}</p>
    </div>
  )
}
