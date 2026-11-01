import type { ProviderSlotKey, ServerProviderView } from '@/api/lightrag'

const SLOT_KEYS: ProviderSlotKey[] = ['extraction', 'query', 'vision']

export const sameBaseUrl = (a: string | null | undefined, b: string | null | undefined): boolean =>
  (a ?? '').replace(/\/+$/, '') === (b ?? '').replace(/\/+$/, '')

/**
 * The preview of the key the server would reuse if the key field stays blank, or
 * null if this base URL needs its own key. Follows the same order as the server:
 * this slot's saved key, the other slots' saved keys, then the .env keys.
 */
export function knownKeyPreview(
  view: ServerProviderView,
  slotKey: ProviderSlotKey,
  baseUrl: string
): string | null {
  const order = [slotKey, ...SLOT_KEYS.filter((k) => k !== slotKey)]
  for (const key of order) {
    const slot = view[key]
    if (slot.active && sameBaseUrl(slot.base_url, baseUrl)) return slot.api_key_preview
  }
  for (const key of order) {
    const { env } = view[key]
    if (env.api_key_preview && sameBaseUrl(env.host, baseUrl)) return env.api_key_preview
  }
  return null
}
