import { useState, useCallback, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { AlertTriangleIcon, Loader2Icon, Trash2Icon } from 'lucide-react'

import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter
} from '@/components/ui/Dialog'
import { errorMessage } from '@/lib/utils'
import {
  getOwnWorkspace,
  deleteOwnWorkspace,
  DELETE_WORKSPACE_CONFIRMATION,
  type OwnWorkspace
} from '@/api/lightrag'
import { navigationService } from '@/services/navigation'

/**
 * Danger zone: lets the owner permanently delete their own workspace.
 * Hidden when the server has no multi-tenant workspaces or the session is not
 * the owner's (API key, impersonation).
 */
export default function DeleteWorkspaceCard() {
  const { t } = useTranslation()
  const [workspace, setWorkspace] = useState<OwnWorkspace | null>(null)
  const [open, setOpen] = useState(false)
  const [confirmText, setConfirmText] = useState('')
  const [isDeleting, setIsDeleting] = useState(false)

  useEffect(() => {
    let cancelled = false
    getOwnWorkspace()
      .then((data) => {
        if (!cancelled) setWorkspace(data)
      })
      .catch(() => {
        // Not available on this server (single workspace) or not signed in
      })
    return () => {
      cancelled = true
    }
  }, [])

  const handleOpenChange = useCallback((newOpen: boolean) => {
    setOpen(newOpen)
    if (!newOpen) {
      setConfirmText('')
    }
  }, [])

  const handleOpen = useCallback(async () => {
    // Re-check: the recent sign-in window may have run out since page load
    try {
      setWorkspace(await getOwnWorkspace())
    } catch {
      // Keep the last known state; the delete request checks again
    }
  }, [])

  const handleDelete = useCallback(async () => {
    if (confirmText !== DELETE_WORKSPACE_CONFIRMATION || isDeleting) return
    setIsDeleting(true)
    try {
      await deleteOwnWorkspace()
      toast.success(t('workspaceSettings.delete.success'))
      navigationService.logout()
    } catch (err) {
      const message = errorMessage(err)
      if (message.includes('Sign in again')) {
        setWorkspace((ws) => (ws ? { ...ws, sign_in_seconds_left: 0 } : ws))
      } else if (message.startsWith('409')) {
        toast.error(t('workspaceSettings.delete.busy'))
      } else {
        toast.error(t('workspaceSettings.delete.error', { error: message }))
      }
      setConfirmText('')
    } finally {
      setIsDeleting(false)
    }
  }, [confirmText, isDeleting, t])

  if (!workspace?.can_delete) {
    return null
  }

  const needsSignIn = workspace.sign_in_seconds_left <= 0

  return (
    <Card className="border-red-500/40">
      <CardHeader>
        <CardTitle className="text-red-500 dark:text-red-400">
          {t('workspaceSettings.delete.cardTitle')}
        </CardTitle>
        <CardDescription>
          {t('workspaceSettings.delete.cardDescription', { workspace: workspace.workspace_id })}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Dialog open={open} onOpenChange={handleOpenChange}>
          <DialogTrigger asChild>
            <Button variant="destructive" size="sm" onClick={handleOpen}>
              <Trash2Icon /> {t('workspaceSettings.delete.button')}
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2 font-bold text-red-500 dark:text-red-400">
                <AlertTriangleIcon className="h-5 w-5" />
                {t('workspaceSettings.delete.title')}
              </DialogTitle>
              <DialogDescription className="pt-2">
                {t('workspaceSettings.delete.description', { workspace: workspace.workspace_id })}
              </DialogDescription>
            </DialogHeader>

            <div className="mb-2 font-semibold text-red-500 dark:text-red-400">
              {t('workspaceSettings.delete.warning')}
            </div>

            {needsSignIn ? (
              <p className="text-sm">{t('workspaceSettings.delete.signInAgain')}</p>
            ) : (
              <div className="space-y-2">
                <label htmlFor="delete-workspace-confirm" className="text-sm font-medium">
                  {t('workspaceSettings.delete.confirmPrompt', {
                    phrase: DELETE_WORKSPACE_CONFIRMATION
                  })}
                </label>
                <Input
                  id="delete-workspace-confirm"
                  value={confirmText}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setConfirmText(e.target.value)}
                  placeholder={DELETE_WORKSPACE_CONFIRMATION}
                  autoComplete="off"
                  className="w-full"
                  disabled={isDeleting}
                />
              </div>
            )}

            <DialogFooter>
              <Button variant="outline" onClick={() => handleOpenChange(false)} disabled={isDeleting}>
                {t('common.cancel')}
              </Button>
              {needsSignIn ? (
                <Button onClick={() => navigationService.logout()}>
                  {t('workspaceSettings.delete.signInButton')}
                </Button>
              ) : (
                <Button
                  variant="destructive"
                  onClick={handleDelete}
                  disabled={confirmText !== DELETE_WORKSPACE_CONFIRMATION || isDeleting}
                >
                  {isDeleting ? (
                    <>
                      <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
                      {t('workspaceSettings.delete.deleting')}
                    </>
                  ) : (
                    t('workspaceSettings.delete.confirmButton')
                  )}
                </Button>
              )}
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  )
}
