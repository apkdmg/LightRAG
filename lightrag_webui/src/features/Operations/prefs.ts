import { createContext, useContext } from 'react'

/**
 * Per-viewer page preferences: "⏸ auto" (pause polling and every loop) and
 * "Reduce motion". Both live in localStorage as conveniences only, wrapped in
 * try/catch so a blocked or private-mode storage just means the defaults.
 */

export const PAUSE_KEY = 'LIGHTRAG-OPS-PAUSED'
export const REDUCE_MOTION_KEY = 'LIGHTRAG-OPS-REDUCE-MOTION'

export const readFlag = (key: string): boolean => {
  try {
    return localStorage.getItem(key) === '1'
  } catch {
    return false
  }
}

export const writeFlag = (key: string, on: boolean) => {
  try {
    if (on) localStorage.setItem(key, '1')
    else localStorage.removeItem(key)
  } catch {
    // Storage blocked: the setting holds for this page only.
  }
}

export type OpsPrefs = {
  /** "⏸ auto": no polling, no loops. Resuming starts with a baseline poll. */
  paused: boolean
  setPaused: (paused: boolean) => void
  /** The page's own "Reduce motion" toggle (the OS setting is combined in useOpsMotion). */
  reduceMotion: boolean
  setReduceMotion: (reduce: boolean) => void
}

export const OpsPrefsContext = createContext<OpsPrefs>({
  paused: false,
  setPaused: () => {},
  reduceMotion: false,
  setReduceMotion: () => {}
})

export const useOpsPrefs = () => useContext(OpsPrefsContext)
