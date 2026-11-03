import { FormEvent, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Loader2Icon, SearchIcon, XIcon } from 'lucide-react'
import Input from '@/components/ui/Input'
import { Correlate, CorrelateMatch, opsApi } from '../api'
import { useOpsNav } from '../context'
import { OpsError, isAbort, toOpsError } from '../errors'
import { fmtStamp, shortId, utcTitle } from '../format'
import { MAX_CORRELATE, cleanCorrelateInput, correlateView } from '../logs'
import { RequestRef } from './ErrorPanel'
import OpsLink from './OpsLink'

type Result = { query: string; data: Correlate } | null

/** What a match is, in words, with the safe label the server sent (owner emails come masked). */
function MatchText({ match }: { match: CorrelateMatch }) {
  const { t } = useTranslation()
  return (
    <span className="flex min-w-0 flex-col">
      <span className="text-[13px] font-medium">{t(`operations.correlate.type.${match.type}`)}</span>
      <span className="text-muted-foreground truncate font-mono text-xs">
        {match.label || shortId(match.id, 24)}
        {match.workspace && match.type !== 'workspace' ? ` · ${match.workspace}` : ''}
        {match.ts ? (
          <span title={utcTitle(match.ts)}>
            {' · '}
            {fmtStamp(match.ts)}
          </span>
        ) : null}
      </span>
    </span>
  )
}

/**
 * The ID box in the header (§2.2, FR-50): paste a request, run, track or doc
 * ID, `ws/doc`, a workspace or an owner email. One match drills straight to
 * its inspector; several are listed (at most 10); none says so. Lookups are
 * audited on the server with the input hashed.
 */
export default function CorrelateBar() {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const [value, setValue] = useState('')
  const [pending, setPending] = useState(false)
  const [result, setResult] = useState<Result>(null)
  const [error, setError] = useState<OpsError | null>(null)
  const [invalid, setInvalid] = useState(false)
  const controller = useRef<AbortController | null>(null)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    const id = cleanCorrelateInput(value)
    setInvalid(!id)
    setError(null)
    setResult(null)
    if (!id) return
    controller.current?.abort()
    const c = new AbortController()
    controller.current = c
    setPending(true)
    try {
      const data = await opsApi.correlate(id, c.signal)
      if (controller.current !== c) return
      const targets = data.matches.map((m) => correlateView(nav.view, m)).filter((v) => v !== null)
      if (data.matches.length === 1 && targets.length === 1) {
        nav.setView(targets[0])
        setValue('')
        return
      }
      setResult({ query: id, data })
    } catch (err) {
      if (isAbort(err) || controller.current !== c) return
      setError(toOpsError(err))
    } finally {
      if (controller.current === c) {
        controller.current = null
        setPending(false)
      }
    }
  }

  const dismiss = () => {
    setResult(null)
    setError(null)
    setInvalid(false)
  }

  const matches = result?.data.matches ?? []
  return (
    <div
      className="relative flex w-full max-w-[640px] flex-col gap-1.5"
      onBlur={(e) => {
        // Focus left the box and its results: the list closes like a menu.
        if (result && !e.currentTarget.contains(e.relatedTarget as Node | null)) setResult(null)
      }}
    >
      <form role="search" onSubmit={submit} className="flex items-center gap-2">
        <label htmlFor="ops-correlate" className="sr-only">
          {t('operations.correlate.label')}
        </label>
        <div className="relative min-w-0 flex-1">
          <SearchIcon className="text-muted-foreground pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2" aria-hidden="true" />
          <Input
            id="ops-correlate"
            value={value}
            onChange={(e) => {
              setValue(e.target.value)
              if (invalid) setInvalid(false)
            }}
            onKeyDown={(e) => {
              if (e.key === 'Escape' && (result || error)) {
                e.preventDefault()
                dismiss()
              }
            }}
            placeholder={t('operations.correlate.placeholder')}
            autoComplete="off"
            spellCheck={false}
            maxLength={MAX_CORRELATE}
            aria-invalid={invalid || undefined}
            aria-describedby={invalid ? 'ops-correlate-invalid' : undefined}
            className="h-9 w-full ps-8 text-[13px]"
          />
        </div>
        <button
          type="submit"
          className="bg-primary text-primary-foreground inline-flex min-h-9 items-center gap-1.5 rounded-lg px-3 text-[13px] hover:opacity-90 disabled:opacity-50"
          disabled={pending || !value.trim()}
          aria-busy={pending || undefined}
        >
          {pending ? <Loader2Icon className="size-3.5 animate-spin" aria-hidden="true" /> : null}
          {t('operations.correlate.go')}
        </button>
      </form>
      {invalid ? (
        <p id="ops-correlate-invalid" className="m-0 text-xs text-[color:var(--ops-crit)]">
          {t('operations.correlate.invalid')}
        </p>
      ) : null}
      {error ? (
        <div role="alert" className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-[color:var(--ops-crit)]">{t(error.messageKey)}</span>
          {error.requestId ? <RequestRef requestId={error.requestId} /> : null}
        </div>
      ) : null}
      {result ? (
        <div className="bg-background ops-fade-in absolute top-full right-0 left-0 z-20 mt-1 flex flex-col gap-1 rounded-lg border p-2 shadow-lg" role="status">
          <div className="flex items-center justify-between gap-2">
            <span className="text-muted-foreground text-xs">
              {matches.length ? t('operations.correlate.matches', { count: matches.length }) : t('operations.correlate.none', { query: shortId(result.query, 40) })}
            </span>
            <button type="button" className="hover:bg-muted rounded p-1" aria-label={t('operations.correlate.close')} onClick={dismiss}>
              <XIcon className="size-3.5" aria-hidden="true" />
            </button>
          </div>
          {matches.length ? (
            <ul className="m-0 flex list-none flex-col p-0">
              {matches.map((m, i) => {
                const to = correlateView(nav.view, m)
                return (
                  <li key={`${m.type}-${m.id}-${i}`}>
                    {to ? (
                      <OpsLink
                        to={to}
                        onNavigate={() => {
                          nav.setView(to)
                          dismiss()
                          setValue('')
                        }}
                        className="hover:bg-muted flex items-center gap-2 rounded-md px-2 py-1.5"
                      >
                        <MatchText match={m} />
                      </OpsLink>
                    ) : (
                      <span className="flex px-2 py-1.5">
                        <MatchText match={m} />
                      </span>
                    )}
                  </li>
                )
              })}
            </ul>
          ) : (
            <p className="text-muted-foreground m-0 text-xs">{t('operations.correlate.noneHint')}</p>
          )}
        </div>
      ) : null}
    </div>
  )
}
