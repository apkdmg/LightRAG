import { useEffect, useState, useRef } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useAuthStore } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { handleOAuth2Callback } from '@/api/lightrag'
import { toast } from 'sonner'
import { useTranslation } from 'react-i18next'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import { Loader2, AlertCircle } from 'lucide-react'

const OAuth2Callback = () => {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { login } = useAuthStore()
  const { t } = useTranslation()
  const [error, setError] = useState<string | null>(null)
  const callbackProcessed = useRef(false)

  useEffect(() => {
    const processCallback = async () => {
      // Prevent double processing in React strict mode
      if (callbackProcessed.current) return
      callbackProcessed.current = true

      const code = searchParams.get('code')
      const state = searchParams.get('state')
      const errorParam = searchParams.get('error')
      const errorDescription = searchParams.get('error_description')

      // Handle Keycloak error response
      if (errorParam) {
        const errorMsg = errorDescription || errorParam
        setError(errorMsg)
        toast.error(t('login.ssoError', 'SSO login failed: ') + errorMsg)
        setTimeout(() => navigate('/login'), 3000)
        return
      }

      if (!code || !state) {
        setError('Missing authorization code or state parameter')
        toast.error(t('login.ssoError', 'Invalid SSO callback'))
        setTimeout(() => navigate('/login'), 3000)
        return
      }

      try {
        const response = await handleOAuth2Callback(code, state)

        // Get previous username for comparison
        const previousUsername = localStorage.getItem('LIGHTRAG-PREVIOUS-USER')
        const isSameUser = previousUsername === response.username

        // Clear chat history if different user
        if (!isSameUser) {
          console.log('Different user logging in via SSO, clearing chat history')
          useSettingsStore.getState().setRetrievalHistory([])
        }

        // Update previous username
        localStorage.setItem('LIGHTRAG-PREVIOUS-USER', response.username)

        // Login with SSO mode flag
        login(
          response.access_token,
          false,  // isGuest
          true,   // isSSO
          response.core_version || null,
          response.api_version || null,
          response.webui_title || null,
          response.webui_description || null
        )

        // Set version check flag
        if (response.core_version || response.api_version) {
          sessionStorage.setItem('VERSION_CHECKED_FROM_LOGIN', 'true')
        }

        toast.success(t('login.ssoSuccess', 'SSO login successful'))
        navigate('/')
      } catch (error) {
        console.error('OAuth2 callback failed:', error)
        const errorMsg = error instanceof Error ? error.message : 'Failed to complete SSO login'
        setError(errorMsg)
        toast.error(t('login.ssoError', 'Failed to complete SSO login'))
        setTimeout(() => navigate('/login'), 3000)
      }
    }

    processCallback()
  }, [searchParams, login, navigate, t])

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-gradient-to-br from-emerald-50 to-teal-100 dark:from-gray-900 dark:to-gray-800">
      <Card className="w-full max-w-[400px] shadow-lg mx-4">
        <CardHeader className="flex items-center justify-center pb-4 pt-6">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            {error ? (
              <>
                <AlertCircle className="h-5 w-5 text-red-500" />
                {t('login.ssoError', 'SSO Error')}
              </>
            ) : (
              t('login.ssoProcessing', 'Processing SSO Login...')
            )}
          </h2>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center px-8 pb-8">
          {error ? (
            <div className="text-center">
              <p className="text-red-500 mb-4 text-sm">{error}</p>
              <p className="text-muted-foreground text-sm">
                {t('login.redirectingToLogin', 'Redirecting to login page...')}
              </p>
            </div>
          ) : (
            <>
              <Loader2 className="h-8 w-8 animate-spin text-emerald-500 mb-4" />
              <p className="text-muted-foreground text-sm">
                {t('login.ssoVerifying', 'Verifying your credentials...')}
              </p>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export default OAuth2Callback
