import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { Loader2 } from 'lucide-react'

import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import Input from '@/components/ui/Input'
import { listOwnProviderWorkspaces, type OwnProviderWorkspaceList } from '@/api/lightrag'
import { useDebounce } from '@/hooks/useDebounce'
import { errorMessage } from '@/lib/utils'

const PAGE_SIZE = 20
const SEARCH_DEBOUNCE_MS = 250

interface OwnProviderWorkspacesDialogProps {
  onClose: () => void
  onManage: (workspace: string) => void
}

/** The workspaces a server default doesn't reach, a page at a time. */
export default function OwnProviderWorkspacesDialog({
  onClose,
  onManage
}: OwnProviderWorkspacesDialogProps) {
  const { t } = useTranslation()
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [list, setList] = useState<(OwnProviderWorkspaceList & { query: string }) | null>(null)
  const query = useDebounce(search, SEARCH_DEBOUNCE_MS).trim()

  useEffect(() => {
    let cancelled = false
    listOwnProviderWorkspaces(page, PAGE_SIZE, query || undefined)
      .then((res) => {
        if (!cancelled) setList({ ...res, query })
      })
      .catch((err) => {
        if (!cancelled) toast.error(errorMessage(err))
      })
    return () => {
      cancelled = true
    }
  }, [page, query])

  const current = list && list.query === query && list.page === page ? list : null
  const lastPage = list ? Math.max(Math.ceil(list.total / PAGE_SIZE), 1) : 1

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{t('providerSettings.server.ownListTitle')}</DialogTitle>
          <DialogDescription>{t('providerSettings.server.ownListDescription')}</DialogDescription>
        </DialogHeader>

        <Input
          value={search}
          onChange={(e) => {
            setSearch(e.target.value)
            setPage(1)
          }}
          placeholder={t('providerSettings.server.ownListSearch')}
          autoComplete="off"
        />

        <div className="max-h-96 min-h-40 overflow-auto rounded-md border border-border/60">
          {!current && (
            <p className="flex items-center gap-2 p-3 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              {t('providerSettings.loading')}
            </p>
          )}
          {current?.workspaces.length === 0 && (
            <p className="p-3 text-sm text-muted-foreground">
              {t('providerSettings.admin.noMatch')}
            </p>
          )}
          {current?.workspaces.map((w) => (
            <div
              key={w.workspace_id}
              className="flex items-center justify-between gap-3 border-b border-border/40 px-3 py-2 last:border-b-0"
            >
              <div className="min-w-0">
                <p className="truncate font-mono text-sm">{w.workspace_id}</p>
                <p className="text-xs text-muted-foreground">
                  {w.slots.map((slot) => t(`providerSettings.${slot}Title`)).join(', ')}
                </p>
              </div>
              <Button variant="outline" size="sm" onClick={() => onManage(w.workspace_id)}>
                {t('providerSettings.server.manage')}
              </Button>
            </div>
          ))}
        </div>

        <DialogFooter className="items-center sm:justify-between">
          <span className="text-xs text-muted-foreground">
            {list && t('providerSettings.server.ownListCount', { count: list.total })}
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => p - 1)}
              disabled={page <= 1}
            >
              {t('providerSettings.server.previous')}
            </Button>
            <span className="text-xs text-muted-foreground">
              {t('providerSettings.server.pageOf', { page, total: lastPage })}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => p + 1)}
              disabled={page >= lastPage}
            >
              {t('providerSettings.server.next')}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
