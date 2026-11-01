import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { Check, ChevronsUpDown, Loader2 } from 'lucide-react'

import Button from '@/components/ui/Button'
import { Card, CardContent } from '@/components/ui/Card'
import {
  Command,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator
} from '@/components/ui/Command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import { listAdminWorkspaces, type AdminWorkspace } from '@/api/lightrag'
import { useDebounce } from '@/hooks/useDebounce'
import { cn, errorMessage } from '@/lib/utils'

// Enough to pick from recent workspaces; anything else is found by searching.
const RESULT_LIMIT = 20
const SEARCH_DEBOUNCE_MS = 250

interface SearchResults {
  query: string
  workspaces: AdminWorkspace[]
  total: number
}

interface ManageWorkspaceSelectorProps {
  value: string | null // null = the admin's own workspace
  onChange: (workspace: string | null) => void
  serverDefaultAvailable: boolean
  serverDefaultSelected: boolean
  onServerDefaultSelectedChange: (selected: boolean) => void
}

/**
 * Admin-only: pick whose workspace the provider settings below apply to.
 *
 * A deployment can have thousands of workspaces, so nothing loads until the list
 * is opened, and the server does the searching.
 */
export default function ManageWorkspaceSelector({
  value,
  onChange,
  serverDefaultAvailable,
  serverDefaultSelected,
  onServerDefaultSelectedChange
}: ManageWorkspaceSelectorProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const [results, setResults] = useState<SearchResults | null>(null)
  const [labels, setLabels] = useState<Record<string, string>>({})
  const query = useDebounce(search, SEARCH_DEBOUNCE_MS).trim()

  useEffect(() => {
    if (!open) return
    let cancelled = false
    listAdminWorkspaces(1, RESULT_LIMIT, query || undefined)
      .then((res) => {
        if (!cancelled) setResults({ query, workspaces: res.workspaces, total: res.total })
      })
      .catch((err) => {
        if (!cancelled) {
          setResults({ query, workspaces: [], total: 0 })
          toast.error(errorMessage(err) || t('providerSettings.admin.loadWorkspacesFailed'))
        }
      })
    return () => {
      cancelled = true
    }
  }, [open, query, t])

  const handleOpenChange = (next: boolean) => {
    setOpen(next)
    if (!next) setSearch('')
  }

  const selectServerDefault = () => {
    onServerDefaultSelectedChange(true)
    onChange(null)
    handleOpenChange(false)
  }

  const selectWorkspace = (workspace: string | null, label?: string) => {
    if (workspace && label) setLabels((l) => ({ ...l, [workspace]: label }))
    onServerDefaultSelectedChange(false)
    onChange(workspace)
    handleOpenChange(false)
  }

  const current = results?.query === query ? results : null
  const typed = search.trim()
  const selectedLabel = serverDefaultSelected
    ? t('providerSettings.server.selectorLabel')
    : value === null
      ? t('providerSettings.admin.myWorkspace')
      : (labels[value] ?? value)

  const checkMark = (selected: boolean) => (
    <Check className={cn('mr-2 size-4 shrink-0', selected ? 'opacity-100' : 'opacity-0')} />
  )

  return (
    <Card>
      <CardContent className="flex flex-col gap-2 pt-6">
        <label className="text-sm font-medium">{t('providerSettings.admin.manageWorkspace')}</label>
        <Popover open={open} onOpenChange={handleOpenChange}>
          <PopoverTrigger asChild>
            <Button
              variant="outline"
              role="combobox"
              aria-expanded={open}
              className="w-full justify-between font-normal"
            >
              <span className="truncate">{selectedLabel}</span>
              <ChevronsUpDown className="size-4 shrink-0 opacity-50" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-(--radix-popover-trigger-width) p-0" align="start">
            <Command shouldFilter={false}>
              <CommandInput
                value={search}
                onValueChange={setSearch}
                placeholder={t('providerSettings.admin.searchPlaceholder')}
              />
              <CommandList>
                {!typed && (
                  <>
                    <CommandGroup>
                      {serverDefaultAvailable && (
                        <CommandItem value="__server__" onSelect={selectServerDefault}>
                          {checkMark(serverDefaultSelected)}
                          {t('providerSettings.server.selectorLabel')}
                        </CommandItem>
                      )}
                      <CommandItem value="__self__" onSelect={() => selectWorkspace(null)}>
                        {checkMark(!serverDefaultSelected && value === null)}
                        {t('providerSettings.admin.myWorkspace')}
                      </CommandItem>
                    </CommandGroup>
                    <CommandSeparator />
                  </>
                )}

                {!current && (
                  <p className="flex items-center gap-2 px-3 py-3 text-sm text-muted-foreground">
                    <Loader2 className="size-4 animate-spin" />
                    {t('providerSettings.loading')}
                  </p>
                )}

                {current && current.workspaces.length > 0 && (
                  <CommandGroup
                    heading={
                      query
                        ? t('providerSettings.admin.matching', { count: current.total })
                        : t('providerSettings.admin.recent')
                    }
                  >
                    {current.workspaces.map((w) => {
                      const label = `${w.owner_username} (${w.workspace_id})`
                      return (
                        <CommandItem
                          key={w.workspace_id}
                          value={w.workspace_id}
                          onSelect={() => selectWorkspace(w.workspace_id, label)}
                        >
                          {checkMark(!serverDefaultSelected && value === w.workspace_id)}
                          <span className="truncate">{label}</span>
                        </CommandItem>
                      )
                    })}
                  </CommandGroup>
                )}

                {current && current.workspaces.length === 0 && typed && (
                  <CommandGroup heading={t('providerSettings.admin.noMatch')}>
                    <CommandItem value="__typed__" onSelect={() => selectWorkspace(typed)}>
                      {t('providerSettings.admin.useTypedId', { workspace: typed })}
                    </CommandItem>
                  </CommandGroup>
                )}

                {current && current.total > current.workspaces.length && (
                  <p className="border-t px-3 py-2 text-xs text-muted-foreground">
                    {t('providerSettings.admin.showingOf', {
                      shown: current.workspaces.length,
                      total: current.total
                    })}
                  </p>
                )}
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>
        {value && (
          <p className="text-xs text-amber-600 dark:text-amber-400">
            {t('providerSettings.admin.actingOn', { workspace: value })}
          </p>
        )}
      </CardContent>
    </Card>
  )
}
