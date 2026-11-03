import { createContext, useContext } from 'react'
import { OpsStatus } from './api'

export type OpsShell = {
  /** Last /admin/ops/status answer (null before the first one). */
  status: OpsStatus | null
  /** Switches the app to the Token Insights tab (the owner of cost and token questions). */
  openInsights: () => void
}

export const OpsShellContext = createContext<OpsShell>({ status: null, openInsights: () => {} })

export const useOpsShell = () => useContext(OpsShellContext)
