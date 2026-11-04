import { useCallback, useEffect, useSyncExternalStore } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { axiosInstance } from '@/api/lightrag'
import { useAuthStore } from '@/stores/state'
import { AuthSnapshot, Capability, createCapabilityStore } from './adminCapability'

const store = createCapabilityStore({
  probe: async () => (await axiosInstance.get('/admin/ops/status')).status,
  storage: () => {
    try {
      return sessionStorage
    } catch {
      return null
    }
  }
})

// A logout (or an expired session) forgets the answer, so the next user on
// this browser is asked again instead of inheriting it.
useAuthStore.subscribe((state, previous) => {
  if (previous.isAuthenticated && (!state.isAuthenticated || state.username !== previous.username)) {
    store.reset()
  }
})

/**
 * Whether the signed-in user may see the admin tabs: true, false, or null
 * while the server is being asked (SSO sessions without a known role).
 * Callers hide admin UI unless this is true, and only treat false as a
 * reason to move the user away from an admin tab.
 */
export function useAdminCapability(): Capability {
  const auth: AuthSnapshot = useAuthStore(
    useShallow((s) => ({
      isAuthenticated: s.isAuthenticated,
      isGuestMode: s.isGuestMode,
      isSSOMode: s.isSSOMode,
      role: s.role,
      username: s.username
    }))
  )
  const getSnapshot = useCallback(() => store.peek(auth), [auth])
  const capability = useSyncExternalStore(store.subscribe, getSnapshot, getSnapshot)
  useEffect(() => {
    store.ensure(auth)
  }, [auth, capability])
  return capability
}

export default useAdminCapability
