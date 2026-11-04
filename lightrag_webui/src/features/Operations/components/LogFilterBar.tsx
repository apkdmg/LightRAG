import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { LinkIcon, PlusIcon, SlidersHorizontalIcon, XIcon } from 'lucide-react'
import Input from '@/components/ui/Input'
import { cn } from '@/lib/utils'
import { useDebounce } from '@/hooks/useDebounce'
import { useOpsShell } from '../context'
import { DEFAULT_LOG_FILTER, ID_FIELDS, IdField, LOG_LEVELS, LOG_WINDOWS, LogFilter, activeFields, effectiveWindow, filterKey, normalizeFilter, textSearchWindow } from '../logs'

/** Typing settles for this long before the URL and the query change. */
const TYPE_DEBOUNCE_MS = 300

const selectClass = 'border-input bg-background text-foreground h-9 rounded-md border px-2 text-[13px]'
const fieldLabel = 'text-muted-foreground flex flex-col gap-0.5 text-xs'

/**
 * The Logs filter bar (§2.5): level, window, workspace, module, text (24 h
 * at most), and the ID filters behind "+ request_id"-style buttons. The
 * filter lives in the URL (`olq`), so "Copy link" hands a colleague the same
 * view. Text fields commit after typing settles; selects at once. On narrow
 * screens the fields fold behind "Filters (n)".
 */
export default function LogFilterBar({ filter, onChange }: { filter: LogFilter; onChange: (next: LogFilter) => void }) {
  const { t } = useTranslation()
  const [draft, setDraft] = useState(filter)
  const [open, setOpen] = useState(false)
  const [shownIds, setShownIds] = useState<IdField[]>([])
  const committed = useRef(filterKey(filter))

  // A filter from outside (a pivot, a pasted link) replaces what is being edited.
  const incoming = filterKey(filter)
  const [seenIncoming, setSeenIncoming] = useState(incoming)
  if (seenIncoming !== incoming) {
    setSeenIncoming(incoming)
    if (incoming !== filterKey(draft)) setDraft(filter)
  }

  const commit = (next: LogFilter) => {
    const clean = normalizeFilter(next)
    committed.current = filterKey(clean)
    onChange(clean)
  }

  // Only a settled edit commits; the URL catching up with a select change must not replay old text.
  const debounced = useDebounce(draft, TYPE_DEBOUNCE_MS)
  const settled = useRef(filterKey(debounced))
  useEffect(() => {
    const k = filterKey(debounced)
    if (k === settled.current) return
    settled.current = k
    if (k === committed.current || k === incoming) return
    committed.current = k
    onChange(normalizeFilter(debounced))
  }, [debounced, incoming, onChange])

  const setText = (k: keyof LogFilter, v: string) => setDraft((d) => ({ ...d, [k]: v }))
  const setNow = (patch: Partial<LogFilter>) => {
    const next = { ...draft, ...patch }
    setDraft(next)
    commit(next)
  }

  const active = activeFields(filter)
  const ids = ID_FIELDS.filter((k) => k !== 'fingerprint')
  const visibleIds = ids.filter((k) => !!draft[k] || shownIds.includes(k))
  // A valid pg_trgm index lets a text search span the whole 7 days (V7); otherwise it is capped at 24 h.
  const textWindow = textSearchWindow(useOpsShell().status?.features)
  const capped = !!filter.q && effectiveWindow(filter, textWindow) !== filter.window

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href)
      toast.success(t('operations.logs.linkCopied'))
    } catch {
      toast.error(t('operations.logs.linkCopyFailed'))
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        className="hover:bg-muted inline-flex min-h-9 items-center gap-1.5 self-start rounded-lg border px-3 text-[13px] md:hidden"
        aria-expanded={open}
        aria-controls="ops-log-filters"
        onClick={() => setOpen((o) => !o)}
      >
        <SlidersHorizontalIcon className="size-3.5" aria-hidden="true" />
        {t('operations.logs.filters', { count: active.length + (filter.level !== DEFAULT_LOG_FILTER.level ? 1 : 0) })}
      </button>
      <div id="ops-log-filters" className={cn('flex-wrap items-end gap-3', open ? 'flex' : 'hidden md:flex')}>
        <label className={fieldLabel}>
          {t('operations.logs.level')}
          <select value={draft.level} onChange={(e) => setNow({ level: e.target.value as LogFilter['level'] })} className={selectClass}>
            {LOG_LEVELS.map((l) => (
              <option key={l} value={l}>
                {t(`operations.logs.levelOption.${l}`)}
              </option>
            ))}
          </select>
        </label>
        <label className={fieldLabel}>
          {t('operations.logs.window.label')}
          <select value={draft.window} onChange={(e) => setNow({ window: e.target.value as LogFilter['window'] })} className={selectClass}>
            {LOG_WINDOWS.map((w) => (
              <option key={w} value={w}>
                {t(`operations.logs.window.${w}`)}
              </option>
            ))}
          </select>
        </label>
        <label className={fieldLabel}>
          {t('operations.logs.workspace')}
          <Input value={draft.workspace} onChange={(e) => setText('workspace', e.target.value)} placeholder={t('operations.logs.workspacePlaceholder')} className="h-9 w-44" maxLength={255} />
        </label>
        <label className={fieldLabel}>
          {t('operations.logs.module')}
          <Input value={draft.module} onChange={(e) => setText('module', e.target.value)} placeholder={t('operations.logs.modulePlaceholder')} className="h-9 w-40" maxLength={128} />
        </label>
        <label className={fieldLabel}>
          {t('operations.logs.text')}
          <Input
            type="search"
            value={draft.q}
            onChange={(e) => setText('q', e.target.value)}
            placeholder={textWindow === '7d' ? t('operations.logs.textPlaceholderWeek') : t('operations.logs.textPlaceholder')}
            className="h-9 w-56"
            maxLength={200}
            aria-describedby="ops-log-q-note"
          />
        </label>
        {visibleIds.map((k) => (
          <label key={k} className={fieldLabel}>
            {t(`operations.logs.id.${k}`)}
            <Input value={draft[k]} onChange={(e) => setText(k, e.target.value)} className="h-9 w-48 font-mono text-xs" maxLength={255} spellCheck={false} />
          </label>
        ))}
        <div className="flex flex-wrap gap-1.5">
          {ids
            .filter((k) => !visibleIds.includes(k))
            .map((k) => (
              <button
                key={k}
                type="button"
                className="hover:bg-muted text-muted-foreground inline-flex min-h-9 items-center gap-1 rounded-lg border border-dashed px-2.5 text-xs"
                onClick={() => setShownIds((s) => [...s, k])}
              >
                <PlusIcon className="size-3" aria-hidden="true" />
                {t(`operations.logs.id.${k}`)}
              </button>
            ))}
        </div>
      </div>
      <p id="ops-log-q-note" className={cn('text-muted-foreground m-0 text-xs', !capped && 'sr-only')}>
        {textWindow === '7d' ? t('operations.logs.textWindowShort') : t('operations.logs.textWindow')}
      </p>
      {active.length ? (
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <span className="text-muted-foreground">{t('operations.logs.active')}</span>
          {active.map((k) => (
            <span key={k} className="bg-muted inline-flex max-w-[280px] items-center gap-1 rounded-full border py-0.5 ps-2.5 pe-1">
              <span className="truncate">
                {t(`operations.logs.field.${k}`)}={filter[k]}
              </span>
              <button
                type="button"
                className="hover:bg-background rounded-full p-0.5"
                aria-label={t('operations.logs.removeFilter', { field: t(`operations.logs.field.${k}`) })}
                onClick={() => {
                  setShownIds((s) => s.filter((x) => x !== k))
                  setNow({ [k]: '' })
                }}
              >
                <XIcon className="size-3" aria-hidden="true" />
              </button>
            </span>
          ))}
          <span className="ms-auto inline-flex gap-1.5">
            <button type="button" className="hover:bg-muted inline-flex min-h-8 items-center gap-1 rounded-lg border px-2.5 text-[13px]" onClick={copyLink}>
              <LinkIcon className="size-3.5" aria-hidden="true" />
              {t('operations.logs.copyLink')}
            </button>
            <button
              type="button"
              className="hover:bg-muted min-h-8 rounded-lg border px-2.5 text-[13px]"
              onClick={() => {
                setShownIds([])
                setNow({ ...DEFAULT_LOG_FILTER, level: draft.level, window: draft.window })
              }}
            >
              {t('operations.logs.clear')}
            </button>
          </span>
        </div>
      ) : null}
    </div>
  )
}
