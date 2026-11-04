import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { DepHealth, SmtpTestResult, opsApi } from '../api'
import { OpsError, toOpsError } from '../errors'
import { fmtLatency } from '../format'
import { inRecheckAll } from './model'

/**
 * Re-check and SMTP test (§3.7 PR-3). They are read-mostly diagnostics, not
 * admin actions: the server rate-limits them (one re-check per dependency
 * every 10 s, one SMTP test a minute, host-wide) and the SMTP test never
 * sends mail. A finished re-check tells every open health view to refresh,
 * since the grid and the drawer poll separately.
 */

const HEALTH_CHANGED = 'ops:health-changed'
const bus: EventTarget | null = typeof EventTarget === 'undefined' ? null : new EventTarget()

export const announceHealthChanged = () => bus?.dispatchEvent(new Event(HEALTH_CHANGED))

/** Calls `reload` whenever a re-check finishes anywhere on the page. */
export function useHealthChanged(reload: () => void) {
  const ref = useRef(reload)
  useEffect(() => {
    ref.current = reload
  })
  useEffect(() => {
    if (!bus) return
    const fn = () => ref.current()
    bus.addEventListener(HEALTH_CHANGED, fn)
    return () => bus.removeEventListener(HEALTH_CHANGED, fn)
  }, [])
}

export type RecheckSummary = { done: number; total: number; refused: number; failed: number }

/** Counts for the "Re-check all" toast: refusals (rate limit) are told apart from errors. */
export function summarizeRechecks(results: PromiseSettledResult<unknown>[]): RecheckSummary {
  let done = 0
  let refused = 0
  let failed = 0
  for (const r of results) {
    if (r.status === 'fulfilled') done += 1
    else {
      const e = toOpsError(r.reason)
      if (e.kind === 'rate_limited' || e.refusal === 'rate_limited') refused += 1
      else failed += 1
    }
  }
  return { done, total: results.length, refused, failed }
}

export function useProbeActions() {
  const { t } = useTranslation()
  const [busy, setBusy] = useState<Set<string>>(() => new Set())
  const [smtp, setSmtp] = useState<{ pending: boolean; result: SmtpTestResult | null; error: OpsError | null }>({ pending: false, result: null, error: null })

  const mark = useCallback(
    (key: string, on: boolean) =>
      setBusy((s) => {
        const next = new Set(s)
        if (on) next.add(key)
        else next.delete(key)
        return next
      }),
    []
  )

  const errorText = useCallback(
    // A refusal's message already says what the code means; other errors add the server's detail.
    (e: OpsError) => [t(e.messageKey), e.refusal ? null : e.detail, e.requestId ? `${t('operations.error.ref')} ${e.requestId}` : null].filter(Boolean).join(' · '),
    [t]
  )

  const recheck = useCallback(
    async (dep: DepHealth) => {
      mark(dep.id, true)
      try {
        const result = await opsApi.recheck(dep.id)
        toast.success(t('operations.health.recheckDone', { name: t(`operations.dep.${dep.id}`, { defaultValue: dep.name ?? dep.id }), status: t(`operations.probe.${result.status}`) }))
        announceHealthChanged()
        return result
      } catch (err) {
        toast.error(errorText(toOpsError(err)))
        return null
      } finally {
        mark(dep.id, false)
      }
    },
    [t, errorText, mark]
  )

  const recheckAll = useCallback(
    async (deps: DepHealth[]) => {
      const targets = deps.filter(inRecheckAll)
      if (!targets.length) return
      mark('*', true)
      try {
        const s = summarizeRechecks(await Promise.allSettled(targets.map((d) => opsApi.recheck(d.id))))
        const text = t('operations.health.recheckAllDone', { done: s.done, total: s.total })
        const notes = [s.refused ? t('operations.health.recheckRefused', { count: s.refused }) : null, s.failed ? t('operations.health.recheckFailed', { count: s.failed }) : null]
          .filter(Boolean)
          .join(' · ')
        if (s.failed) toast.error(notes ? `${text} · ${notes}` : text)
        else toast.success(notes ? `${text} · ${notes}` : text)
        announceHealthChanged()
      } finally {
        mark('*', false)
      }
    },
    [t, mark]
  )

  const smtpTest = useCallback(async () => {
    setSmtp({ pending: true, result: null, error: null })
    try {
      const result = await opsApi.smtpTest()
      setSmtp({ pending: false, result, error: null })
      if (result.ok) toast.success(t('operations.health.smtpOk', { latency: fmtLatency(result.latency_ms ?? null) }))
      else toast.error(t('operations.health.smtpFailed', { error: result.error_class ?? t('operations.state.unknown') }))
    } catch (err) {
      const e = toOpsError(err)
      setSmtp({ pending: false, result: null, error: e })
      toast.error(errorText(e))
    }
  }, [t, errorText])

  return { busy, recheck, recheckAll, smtp, smtpTest, errorText }
}
