import { AnchorHTMLAttributes, MouseEvent, ReactNode, Ref } from 'react'
import { OpsHash, writeOpsHash } from '../hashState'
import { useOpsNav } from '../context'

const isPlainClick = (e: MouseEvent) => e.button === 0 && !e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey

/**
 * A real link (`<a href="#…">`) to an Operations view, so drill targets can be
 * opened in a new tab or copied into chat (US-1, NFR-10). A plain click stays
 * in the page: `onNavigate` decides how (for example a drawer push), or the
 * view is applied as it is.
 */
export default function OpsLink({
  to,
  onNavigate,
  children,
  ref,
  ...rest
}: { to: OpsHash; onNavigate?: () => void; children: ReactNode; ref?: Ref<HTMLAnchorElement> } & Omit<AnchorHTMLAttributes<HTMLAnchorElement>, 'href'>) {
  const nav = useOpsNav()
  return (
    <a
      {...rest}
      ref={ref}
      href={writeOpsHash(to)}
      onClick={(e) => {
        rest.onClick?.(e)
        if (e.defaultPrevented || !isPlainClick(e)) return
        e.preventDefault()
        if (onNavigate) onNavigate()
        else nav.setView(to)
      }}
    >
      {children}
    </a>
  )
}
