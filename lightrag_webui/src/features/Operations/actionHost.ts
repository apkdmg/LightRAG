import { create } from 'zustand'
import type { ActionKind } from './actions'

/**
 * The console-level action dialog host. A follow-up that outlives the
 * component it came from (a release toast's "Run now…", clicked after the
 * released Stuck or Active row has left the table) opens its dialog here,
 * rendered once by the shell, instead of on the row that unmounted.
 */
export type HostedAction = { workspace: string; kind: ActionKind | null }

type ActionHostState = {
  hosted: HostedAction | null
  open: (workspace: string, kind: ActionKind) => void
  setKind: (kind: ActionKind | null) => void
  reset: () => void
}

export const useActionHost = create<ActionHostState>()((set) => ({
  hosted: null,
  open: (workspace, kind) => set({ hosted: { workspace, kind } }),
  // Closing keeps the workspace so the dialog can play its exit.
  setKind: (kind) => set((s) => (s.hosted ? { hosted: { ...s.hosted, kind } } : s)),
  reset: () => set({ hosted: null })
}))

/** Opens "Run pipeline now" for a workspace in the shell's host (§3.9 release follow-up). */
export const openHostedRun = (workspace: string) => useActionHost.getState().open(workspace, 'run')
