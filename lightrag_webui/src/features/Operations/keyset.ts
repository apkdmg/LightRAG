/** A keyset position: `before` is the server's `next_before` of the page above; `trail` walks back up. */
export type Cursor = { before: string | null; trail: string[] }

export const FIRST_PAGE: Cursor = { before: null, trail: [] }

/** One page older: the server's `next_before` becomes the cursor. */
export const olderCursor = (cursor: Cursor, next: string): Cursor => ({ before: next, trail: [...cursor.trail, next] })

/** One page newer: back to the cursor of the page above. */
export function newerCursor(cursor: Cursor): Cursor {
  const trail = cursor.trail.slice(0, -1)
  return { before: trail.length ? trail[trail.length - 1] : null, trail }
}
