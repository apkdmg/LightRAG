import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { Params } from './api'
import { Purpose } from './meta'

export type Scope = {
  tab: string
  purpose: Purpose | ''
  workspace: string
  range: number | 'custom'
  start: string
  end: string
  unit: 'tokens' | 'cost'
  day: string | null
}

export type Drill = { dim: string; key: string; label: string }

export type InsightsContext = {
  scope: Scope
  set: (patch: Partial<Scope>) => void
  params: Params
  explore: { group: string; drill: Drill[] }
  setExplore: (patch: { group?: string; drill?: Drill[] }) => void
  goExplore: (group: string, drill: Drill[], patch?: Partial<Scope>) => void
  refreshHeader: () => void
}

export const Ctx = createContext<InsightsContext>(null as unknown as InsightsContext)
export const useInsights = () => useContext(Ctx)

/** Fetch on change, keeping the old view on screen (dimmed) until the new one lands. */
export function useData<T>(fetcher: () => Promise<T>, deps: unknown[], refreshSeconds?: number) {
  const depsKey = JSON.stringify(deps)
  const [data, setData] = useState<T | null>(null)
  const [loadedKey, setLoadedKey] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const latest = useRef(0)
  const fetcherRef = useRef(fetcher)
  useEffect(() => {
    fetcherRef.current = fetcher
  })

  const load = useCallback(async (key: string) => {
    const id = ++latest.current
    try {
      const result = await fetcherRef.current()
      if (id === latest.current) {
        setData(result)
        setError(null)
      }
    } catch (e: any) {
      if (id === latest.current) setError(e?.response?.data?.detail || e?.message || 'Error')
    } finally {
      if (id === latest.current) setLoadedKey(key)
    }
  }, [])

  useEffect(() => {
    load(depsKey)
    if (!refreshSeconds) return
    const timer = setInterval(() => load(depsKey), refreshSeconds * 1000)
    return () => clearInterval(timer)
  }, [depsKey, load, refreshSeconds])

  const reload = useCallback(() => load(depsKey), [depsKey, load])
  return { data, loading: loadedKey !== depsKey, error, reload }
}
