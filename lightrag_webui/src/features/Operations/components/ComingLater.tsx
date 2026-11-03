import { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { CircleDashedIcon } from 'lucide-react'
import { Card } from '@/features/TokenInsights/ui'

/**
 * Honest empty state for a section that has not shipped yet: it says what
 * will appear here and, where one exists, where to look today.
 */
export default function ComingLater({ title, body, children }: { title: string; body: string; children?: ReactNode }) {
  const { t } = useTranslation()
  return (
    <Card className="flex max-w-[760px] flex-col gap-2">
      <div className="flex items-center gap-2">
        <CircleDashedIcon className="text-muted-foreground size-4 flex-none" aria-hidden="true" />
        <h2 className="m-0 text-[15px] font-semibold">{title}</h2>
      </div>
      <p className="text-muted-foreground m-0 text-sm">{body}</p>
      <p className="text-muted-foreground m-0 text-xs">{t('operations.later.note')}</p>
      {children ? <div className="flex flex-wrap gap-2 pt-1">{children}</div> : null}
    </Card>
  )
}
