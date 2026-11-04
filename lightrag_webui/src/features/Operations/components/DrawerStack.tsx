import { KeyboardEvent, ReactNode, useEffect, useRef } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { useTranslation } from 'react-i18next'
import { ChevronLeftIcon, XIcon } from 'lucide-react'
import { useOpsNav } from '../context'
import { shortId } from '../format'
import { Drawer } from '../hashState'
import { TWEEN, useOpsMotion } from '../motion'

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

const drawerKey = (d: Drawer) => (d.kind === 'document' ? `document:${d.workspace}/${d.id}` : `${d.kind}:${d.id}`)

function DrawerLabel({ drawer }: { drawer: Drawer }) {
  const { t } = useTranslation()
  if (drawer.kind === 'document') return <>{t('operations.drawer.documentTitle', { id: shortId(drawer.id, 10) })}</>
  if (drawer.kind === 'workspace') return <>{t('operations.drawer.workspaceTitle', { id: drawer.id })}</>
  if (drawer.kind === 'run') return <>{t('operations.drawer.runTitle', { id: shortId(drawer.id, 8) })}</>
  if (drawer.kind === 'dependency') return <>{t('operations.drawer.dependencyTitle', { name: t(`operations.dep.${drawer.id}`, { defaultValue: drawer.id }) })}</>
  if (drawer.kind === 'worker') return <>{t('operations.drawer.workerTitle', { id: shortId(drawer.id, 8) })}</>
  if (drawer.kind === 'log') return <>{t('operations.drawer.logTitle', { id: shortId(drawer.id, 12) })}</>
  return <>{t(`operations.drawer.kind.${drawer.kind}`)}</>
}

/**
 * The right-hand drawer (§2.9, §2.12): one drawer at a time, URL-addressable,
 * with an in-drawer breadcrumb. Opening slides in from x 40 with a fade
 * (240 ms, ease-out), closing runs the reverse with ease-in; a drawer opened
 * by a deep link on page load does not animate. Drilling deeper slides the
 * body left, back slides it right, and the header stays still. Under reduced
 * motion everything is a 150 ms fade. Esc closes, focus moves to the heading
 * on every mount, is trapped while open, and returns to the opener on close.
 */
export default function DrawerStack({ animateOpen, render }: { animateOpen: boolean; render: (drawer: Drawer) => ReactNode }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const { reduced } = useOpsMotion()
  const drawer = nav.view.drawer
  const open = drawer !== null
  const asideRef = useRef<HTMLElement | null>(null)
  const headingRef = useRef<HTMLHeadingElement | null>(null)
  const openerRef = useRef<Element | null>(null)
  const key = drawer ? drawerKey(drawer) : null

  // Remember who opened the drawer; give focus back on close.
  useEffect(() => {
    if (!open) return
    const active = document.activeElement
    openerRef.current = active && !asideRef.current?.contains(active) ? active : openerRef.current
    return () => {
      const opener = openerRef.current as HTMLElement | null
      openerRef.current = null
      if (opener && opener.isConnected && typeof opener.focus === 'function') opener.focus({ preventScroll: true })
    }
  }, [open])

  // Focus the heading when a drawer (or a drilled entity) mounts, not when the animation ends.
  useEffect(() => {
    if (key) headingRef.current?.focus({ preventScroll: true })
  }, [key])

  useEffect(() => {
    if (!open) return
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape' && !e.defaultPrevented) nav.closeDrawer()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, nav])

  const trap = (e: KeyboardEvent<HTMLElement>) => {
    if (e.key !== 'Tab' || !asideRef.current) return
    const items = Array.from(asideRef.current.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((el) => el.offsetParent !== null || el === document.activeElement)
    if (!items.length) return
    const first = items[0]
    const last = items[items.length - 1]
    if (e.shiftKey && (document.activeElement === first || document.activeElement === headingRef.current)) {
      e.preventDefault()
      last.focus()
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault()
      first.focus()
    }
  }

  const panel = reduced ? TWEEN.reducedFade : TWEEN.panel
  const panelExit = reduced ? TWEEN.reducedFade : TWEEN.panelExit
  const back = nav.move === 'back'
  return (
    <AnimatePresence>
      {drawer ? (
        <motion.div
          key="ops-scrim"
          aria-hidden="true"
          className="absolute inset-0 z-20 bg-black/20"
          initial={animateOpen ? { opacity: 0 } : false}
          animate={{ opacity: 1, transition: panel }}
          exit={{ opacity: 0, transition: panelExit }}
          onClick={nav.closeDrawer}
        />
      ) : null}
      {drawer ? (
        <motion.aside
          key="ops-drawer"
          ref={asideRef}
          role="dialog"
          aria-modal="true"
          aria-labelledby="ops-drawer-title"
          onKeyDown={trap}
          initial={animateOpen ? { x: 40, opacity: 0 } : false}
          animate={{ x: 0, opacity: 1, transition: panel }}
          exit={{ x: 40, opacity: 0, transition: panelExit }}
          className="bg-background absolute top-0 right-0 bottom-0 z-30 flex w-full max-w-[560px] flex-col border-l shadow-xl"
        >
          <div className="flex flex-col gap-1.5 border-b px-5 pt-4 pb-3">
            {nav.stack.length ? (
              <nav aria-label={t('operations.drawer.breadcrumb')} className="flex flex-wrap items-center gap-1.5 text-xs">
                {nav.stack.map((d, i) => (
                  <span key={`${drawerKey(d)}-${i}`} className="inline-flex items-center gap-1.5">
                    <button type="button" className="hover:bg-muted max-w-[220px] truncate rounded-full border px-2.5 py-0.5" onClick={() => nav.backDrawer(i)}>
                      <DrawerLabel drawer={d} />
                    </button>
                    <span className="text-muted-foreground" aria-hidden="true">
                      ›
                    </span>
                  </span>
                ))}
              </nav>
            ) : null}
            <div className="flex items-start gap-2">
              {nav.stack.length ? (
                <button type="button" className="hover:bg-muted rounded-lg border p-1.5" aria-label={t('operations.drawer.back')} onClick={() => nav.backDrawer()}>
                  <ChevronLeftIcon className="size-4" aria-hidden="true" />
                </button>
              ) : null}
              <h2 id="ops-drawer-title" ref={headingRef} tabIndex={-1} className="m-0 min-w-0 flex-1 text-lg font-semibold break-all outline-none">
                <DrawerLabel drawer={drawer} />
              </h2>
              <button type="button" className="hover:bg-muted rounded-lg border p-1.5" aria-label={t('operations.drawer.close')} onClick={nav.closeDrawer}>
                <XIcon className="size-4" aria-hidden="true" />
              </button>
            </div>
          </div>
          <div className="relative min-h-0 flex-1 overflow-x-hidden overflow-y-auto">
            <AnimatePresence mode="popLayout" initial={false} custom={back}>
              <motion.div
                key={key}
                custom={back}
                variants={{
                  enter: (isBack: boolean) => ({ opacity: 0, x: isBack ? -24 : 24 }),
                  center: { opacity: 1, x: 0, transition: panel },
                  leave: (isBack: boolean) => ({ opacity: 0, x: isBack ? 24 : -24, transition: panel })
                }}
                initial="enter"
                animate="center"
                exit="leave"
                className="flex flex-col gap-4 px-5 py-4"
              >
                {render(drawer)}
              </motion.div>
            </AnimatePresence>
          </div>
        </motion.aside>
      ) : null}
    </AnimatePresence>
  )
}
