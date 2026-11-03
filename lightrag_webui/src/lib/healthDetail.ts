/**
 * `/health` answers callers without a valid session with liveness only
 * (`{status, version}`, HEALTH_DETAIL=authenticated, the server default).
 * Signed-in and guest sessions get the full detail. This tells the two
 * apart, so a liveness-only answer (an expired token on the way to the
 * login page, for example) keeps the server marked healthy without replacing
 * the last detailed status that StatusCard, the graph limits and the
 * pipeline flags read.
 */
export function hasHealthDetail(health: unknown): boolean {
  if (!health || typeof health !== 'object') return false
  const configuration = (health as { configuration?: unknown }).configuration
  return !!configuration && typeof configuration === 'object' && !Array.isArray(configuration)
}
