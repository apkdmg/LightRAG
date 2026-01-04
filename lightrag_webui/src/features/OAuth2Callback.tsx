import { useEffect, useState, useRef } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useAuthStore } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { toast } from 'sonner'
import { useTranslation } from 'react-i18next'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import { Loader2, AlertCircle } from 'lucide-react'

// Helper function to get cookie value by name
const getCookie = (name: string): string | null => {
  const cookies = document.cookie.split(';')
  for (const cookie of cookies) {
    const [cookieName, ...cookieValueParts] = cookie.trim().split('=')
    if (cookieName === name) {
      // Join with '=' in case the value contains '=' characters
      const cookieValue = cookieValueParts.join('=')
      try {
        return decodeURIComponent(cookieValue)
      } catch {
        return cookieValue
      }
    }
  }
  return null
}

// Helper function to delete a cookie
const deleteCookie = (name: string): void => {
  document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;`
}

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

      const errorParam = searchParams.get('error')
      const errorDescription = searchParams.get('error_description')

      // Handle error response (from Keycloak or backend)
      if (errorParam) {
        const errorMsg = errorDescription || errorParam
        setError(errorMsg)
        toast.error(t('login.ssoError', 'SSO login failed: ') + errorMsg)
        setTimeout(() => navigate('/login'), 3000)
        return
      }

      // Check for success indicator and read user metadata from cookie
      const success = searchParams.get('success')
      const userCookie = getCookie('lightrag_user')

      if (success !== 'true' || !userCookie) {
        setError('Missing authentication data')
        toast.error(t('login.ssoError', 'Invalid SSO callback'))
        setTimeout(() => navigate('/login'), 3000)
        return
      }

      try {
        // Parse user metadata from cookie (non-sensitive data)
        // The actual token is in HTTP-only cookie and will be sent automatically
        const userData = JSON.parse(userCookie)
        const username = userData.username

        if (!username) {
          throw new Error('Missing username in authentication data')
        }

        // Get previous username for comparison
        const previousUsername = localStorage.getItem('LIGHTRAG-PREVIOUS-USER')
        const isSameUser = previousUsername === username

        // Clear chat history if different user
        if (!isSameUser) {
          console.log('Different user logging in via SSO, clearing chat history')
          useSettingsStore.getState().setRetrievalHistory([])
        }

        // Update previous username
        localStorage.setItem('LIGHTRAG-PREVIOUS-USER', username)

        // Login with SSO mode flag
        // Note: Token is stored in HTTP-only cookie and sent automatically with requests
        // We pass a placeholder token here; the actual auth is cookie-based
        login(
          'cookie-based-auth',  // Placeholder - actual token is in HTTP-only cookie
          false,  // isGuest
          true,   // isSSO
          userData.core_version || null,
          userData.api_version || null,
          userData.webui_title || null,
          userData.webui_description || null
        )

        // Set version check flag
        if (userData.core_version || userData.api_version) {
          sessionStorage.setItem('VERSION_CHECKED_FROM_LOGIN', 'true')
        }

        // Clean up the user metadata cookie (optional, for cleanliness)
        deleteCookie('lightrag_user')

        toast.success(t('login.ssoSuccess', 'SSO login successful'))

        // Use setTimeout to ensure React has processed the auth state update
        // before navigating. This prevents the AppRouter's redirect effect
        // from firing before isAuthenticated is true.
        setTimeout(() => {
          navigate('/', { replace: true })
        }, 100)
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
