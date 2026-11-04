import { cn } from '@/lib/utils'

/**
 * First-load placeholder at the final size, so nothing jumps when data lands.
 * It pulses twice (4 s) and then holds still; it is never shown on refresh.
 */
export default function Skeleton({ className }: { className?: string }) {
  return <span aria-hidden="true" className={cn('ops-skeleton bg-muted block rounded-md', className)} />
}
