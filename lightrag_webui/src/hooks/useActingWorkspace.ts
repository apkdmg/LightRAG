import { useCallback, useEffect } from 'react'
import { useAuthStore } from '@/stores/state'
import { mayAct, shownActingWorkspace, useActingStore } from '@/stores/actingWorkspace'
import { useAdminCapability } from './useAdminCapability'

// A logout, or another user signing in on this browser, ends the mode: the
// next session never inherits someone else's target workspace.
useAuthStore.subscribe((state, previous) => {
  if (previous.isAuthenticated && (!state.isAuthenticated || state.username !== previous.username)) {
    useActingStore.getState().exit()
  }
})

/**
 * The admin "act on workspace" mode for Documents (OQ-13): the workspace the
 * Documents tab acts on (null for the user's own), `start` (admins only;
 * returns false when refused) and `exit`. When the server answers that the
 * user is not an admin, the mode ends at once.
 */
export function useActingWorkspace() {
  const isAdmin = useAdminCapability()
  const acting = useActingStore((s) => s.acting)
  const auth = useAuthStore((s) => s)
  const exit = useActingStore((s) => s.exit)
  const startRaw = useActingStore((s) => s.start)
  const username = auth.username
  const start = useCallback((workspace: string) => startRaw(workspace, { admin: isAdmin, user: username }), [startRaw, isAdmin, username])

  // A definite "not an admin" (or a mode another user entered) ends it.
  const allowed = mayAct(acting, auth)
  useEffect(() => {
    if (acting && (isAdmin === false || !allowed)) exit()
  }, [acting, isAdmin, allowed, exit])

  return { workspace: shownActingWorkspace(acting, auth, isAdmin), since: acting?.since ?? null, start, exit }
}

export default useActingWorkspace
