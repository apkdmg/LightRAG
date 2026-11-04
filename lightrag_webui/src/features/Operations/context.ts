import { createContext, useContext } from 'react'
import type { InsightsTarget } from '@/features/TokenInsights/hash'
import { OpsOverview, OpsStatus } from './api'
import { DEFAULT_OPS_HASH, Drawer, OpsHash, OpsTab } from './hashState'
import { OpsPoll } from './poll'

export type OpsShell = {
  /** Last /admin/ops/status answer (null before the first one). */
  status: OpsStatus | null
  /**
   * Switches the app to Token Insights (the owner of cost and token
   * questions), optionally on a workspace, a purpose or one document's calls.
   */
  openInsights: (target?: InsightsTarget) => void
}

export const OpsShellContext = createContext<OpsShell>({ status: null, openInsights: () => {} })

export const useOpsShell = () => useContext(OpsShellContext)

/**
 * Navigation inside the console. The URL hash holds the view (tab, sub-view,
 * top drawer); the in-drawer breadcrumb is the stack of drawers drilled
 * through, kept in memory.
 */
export type DrawerMove = 'none' | 'open' | 'push' | 'back' | 'close'

export type OpsNav = {
  view: OpsHash
  /** Drawers below the current one (oldest first), for the breadcrumb. */
  stack: Drawer[]
  /** How the drawer last changed, so the body slides the right way. */
  move: DrawerMove
  go: (tab: OpsTab, sub?: string | null) => void
  setSub: (sub: string) => void
  /** push: drill from the open drawer (breadcrumb); otherwise replace the stack. */
  openDrawer: (drawer: Drawer, opts?: { push?: boolean }) => void
  /** Back to the drawer at `index` in the stack (the breadcrumb), or one level when omitted. */
  backDrawer: (index?: number) => void
  closeDrawer: () => void
  /** Applies a view from a link (keeps the breadcrumb only when the drawer stays). */
  setView: (view: OpsHash) => void
}

export const OpsNavContext = createContext<OpsNav>({
  view: DEFAULT_OPS_HASH,
  stack: [],
  move: 'none',
  go: () => {},
  setSub: () => {},
  openDrawer: () => {},
  backDrawer: () => {},
  closeDrawer: () => {},
  setView: () => {}
})

export const useOpsNav = () => useContext(OpsNavContext)

/**
 * The /overview poll is owned by the shell: the header state chip and the
 * Overview tab read the same answer, so it is fetched once per 15 s.
 */
export const OpsOverviewContext = createContext<OpsPoll<OpsOverview> | null>(null)

export const useOpsOverview = () => useContext(OpsOverviewContext)
