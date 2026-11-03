import { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import {
  AlertTriangleIcon,
  CheckCircle2Icon,
  CircleDashedIcon,
  CircleDotIcon,
  OctagonAlertIcon,
  PauseCircleIcon,
  WrenchIcon,
  XCircleIcon
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { IconId, Tone, lookOf } from '../severity'

const ICONS: Record<IconId, typeof CheckCircle2Icon> = {
  check: CheckCircle2Icon,
  triangle: AlertTriangleIcon,
  octagon: OctagonAlertIcon,
  'x-circle': XCircleIcon,
  'dashed-circle': CircleDashedIcon,
  dot: CircleDotIcon,
  pause: PauseCircleIcon,
  wrench: WrenchIcon
}

export function StateIcon({ icon, className }: { icon: IconId; className?: string }) {
  const Icon = ICONS[icon]
  return <Icon className={cn('size-3.5 flex-none', className)} aria-hidden="true" />
}

/** A tone-coloured chip: icon plus text, never colour alone (§2.2). */
export function ToneChip({ tone, icon, children, className }: { tone: Tone; icon: IconId; children: ReactNode; className?: string }) {
  return (
    <span className={cn('ops-chip inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs font-semibold', `ops-tone-${tone}`, className)}>
      <StateIcon icon={icon} />
      {children}
    </span>
  )
}

/** A state from the status vocabulary, with its default label unless `children` overrides it. */
export function StatusChip({ state, children, className }: { state: string | null | undefined; children?: ReactNode; className?: string }) {
  const { t } = useTranslation()
  const look = lookOf(state)
  return (
    <ToneChip tone={look.tone} icon={look.icon} className={className}>
      {children ?? t(look.label)}
    </ToneChip>
  )
}
