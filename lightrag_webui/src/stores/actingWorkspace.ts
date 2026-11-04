import { create } from 'zustand'

/**
 * Admin "act on workspace" for the Documents tab (spec OQ-13).
 *
 * An admin picks a workspace in the Operations workspace drawer ("Open in
 * Documents"). From then on every Documents API call (`/documents…`) carries
 * `X-Target-Workspace: <ws>`, and the app shows a persistent "Acting on <ws>"
 * banner with Exit until the admin leaves the mode. The server stays the
 * authority: it accepts the header from admins only (dependencies.py) and
 * audits every admin write it applies to another workspace.
 *
 * The client guards on its own as well: the mode can only be entered by a
 * caller that knows the user is an admin, it belongs to the user who entered
 * it (another login on the same browser never inherits it), and a session
 * whose role is known and not admin never sends the header. The choice is
 * kept in sessionStorage so a reload keeps both the header and the banner;
 * it never outlives the browser tab.
 *
 * Pure parts are exported for tests; this module imports nothing from the
 * API client (which reads it from its request interceptor).
 */

export const ACTING_SESSION_KEY = 'LIGHTRAG-ACTING-WORKSPACE'
export const TARGET_WORKSPACE_HEADER = 'X-Target-Workspace'

/** Workspace ids the server would accept as a header value: printable, no whitespace, bounded. */
const MAX_WORKSPACE = 256

export function cleanWorkspace(raw: unknown): string | null {
  if (typeof raw !== 'string') return null
  const v = raw.trim()
  if (!v || v.length > MAX_WORKSPACE) return null
  for (let i = 0; i < v.length; i += 1) {
    const c = v.charCodeAt(i)
    // Control characters and anything outside Latin-1 cannot travel in an HTTP header.
    if (c <= 0x20 || c === 0x7f || c > 0xff) return null
  }
  return v
}

export type Acting = {
  workspace: string
  /** Who entered the mode; the header is only sent for this user. */
  user: string | null
  /** Epoch ms. */
  since: number
}

/** What the request interceptor knows about the session. */
export type ActingAuth = { isAuthenticated: boolean; username: string | null; role: string | null; isGuestMode?: boolean }

/** Paths of the Documents API: `/documents`, `/documents/…` and `/documents?…`, nothing else. */
export function isDocumentsCall(url: string | null | undefined): boolean {
  if (!url) return false
  let path = url
  // An absolute URL (a caller that passed the base URL along) is matched on its path.
  const scheme = path.indexOf('://')
  if (scheme >= 0) {
    const slash = path.indexOf('/', scheme + 3)
    path = slash >= 0 ? path.slice(slash) : '/'
  }
  return /^\/documents(?:[/?#]|$)/.test(path)
}

/**
 * Whether this session may send the header now: signed in, not a guest, the
 * same user who entered the mode, and a role that is admin or not known yet
 * (SSO sessions learn it from the server; the mode was only entered after
 * the server said admin).
 */
export function mayAct(acting: Acting | null, auth: ActingAuth): acting is Acting {
  if (!acting || !auth.isAuthenticated || auth.isGuestMode) return false
  if (auth.role !== null && auth.role !== 'admin') return false
  return acting.user === (auth.username ?? null)
}

/**
 * The workspace the banner names and Documents acts on: the mode's target
 * while this session may act, null otherwise (and null as soon as the server
 * has said the user is not an admin).
 */
export function shownActingWorkspace(acting: Acting | null, auth: ActingAuth, isAdmin: boolean | null): string | null {
  if (isAdmin === false || !mayAct(acting, auth)) return null
  return acting.workspace
}

/** The header for one request, or null when the request goes to the caller's own workspace. */
export function actingHeaderFor(url: string | null | undefined, acting: Acting | null, auth: ActingAuth): Record<string, string> | null {
  if (!isDocumentsCall(url) || !mayAct(acting, auth)) return null
  return { [TARGET_WORKSPACE_HEADER]: acting.workspace }
}

type StorageLike = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>

const sessionStore = (): StorageLike | null => {
  try {
    return globalThis.sessionStorage ?? null
  } catch {
    return null
  }
}

export function readActing(storage: StorageLike | null = sessionStore()): Acting | null {
  try {
    const raw = storage?.getItem(ACTING_SESSION_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    const workspace = cleanWorkspace(parsed?.workspace)
    if (!workspace) return null
    const user = typeof parsed.user === 'string' ? parsed.user : null
    const since = typeof parsed.since === 'number' && Number.isFinite(parsed.since) ? parsed.since : Date.now()
    return { workspace, user, since }
  } catch {
    return null
  }
}

export function writeActing(acting: Acting | null, storage: StorageLike | null = sessionStore()) {
  try {
    if (acting) storage?.setItem(ACTING_SESSION_KEY, JSON.stringify(acting))
    else storage?.removeItem(ACTING_SESSION_KEY)
  } catch {
    // Storage blocked: the mode still holds for this page, and ends with it.
  }
}

type ActingState = {
  acting: Acting | null
  /**
   * Enter the mode for `workspace`. Refused (false) unless `admin` is true
   * and the id is one the server could accept.
   */
  start: (workspace: string, who: { admin: boolean | null; user: string | null }) => boolean
  exit: () => void
}

export const useActingStore = create<ActingState>((set) => ({
  acting: readActing(),
  start: (workspace, { admin, user }) => {
    const ws = cleanWorkspace(workspace)
    if (admin !== true || !ws) return false
    const acting = { workspace: ws, user, since: Date.now() }
    writeActing(acting)
    set({ acting })
    return true
  },
  exit: () => {
    writeActing(null)
    set({ acting: null })
  }
}))
