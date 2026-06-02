import { create } from 'zustand'
import { createSelectors } from '@/lib/utils'
import { ProviderConfigMasked } from '@/api/lightrag'

/**
 * Holds the masked per-workspace provider configuration for the settings panel.
 * Not persisted — loaded fresh from the API on mount (secrets never live here).
 */
interface ProvidersState {
  config: ProviderConfigMasked | null
  setConfig: (config: ProviderConfigMasked | null) => void
}

const useProvidersStoreBase = create<ProvidersState>((set) => ({
  config: null,
  setConfig: (config) => set({ config })
}))

const useProvidersStore = createSelectors(useProvidersStoreBase)

export { useProvidersStore }
