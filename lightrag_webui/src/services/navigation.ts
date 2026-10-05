import { NavigateFunction } from 'react-router-dom';
import { useAuthStore, useBackendState } from '@/stores/state';
import { useGraphStore } from '@/stores/graph';
import { useSettingsStore } from '@/stores/settings';
import { SearchHistoryManager } from '@/utils/SearchHistoryManager';

class NavigationService {
  private navigate: NavigateFunction | null = null;

  setNavigate(navigate: NavigateFunction) {
    this.navigate = navigate;
  }

  /**
   * Reset all application state to ensure a clean environment.
   * This function should be called when:
   * 1. User logs out
   * 2. Authentication token expires
   * 3. Direct access to login page
   *
   * @param preserveHistory If true, chat history will be preserved. Default is false.
   */
  resetAllApplicationState(preserveHistory = false) {
    console.log('Resetting all application state...');

    // Reset graph state
    const graphStore = useGraphStore.getState();
    const sigma = graphStore.sigmaInstance;
    graphStore.reset();
    graphStore.setGraphDataFetchAttempted(false);
    graphStore.setLabelsFetchAttempted(false);
    graphStore.setSigmaInstance(null);
    graphStore.setIsFetching(false); // Reset isFetching state to prevent data loading issues

    // Reset backend state
    useBackendState.getState().clear();

    // Clear the user's data (chat and prompt history, graph searches, API key)
    // unless it should be kept for the same user signing back in.
    if (!preserveHistory) {
      clearUserData();
    }

    // Clear authentication state
    sessionStorage.clear();

    if (sigma) {
      sigma.getGraph().clear();
      sigma.kill();
      useGraphStore.getState().setSigmaInstance(null);
    }
  }

  /**
   * Explicit logout: clear all of the user's data from this browser, then
   * sign out. Nothing from this session is kept for whoever uses the
   * browser next.
   */
  logout() {
    localStorage.removeItem('LIGHTRAG-PREVIOUS-USER');
    this.resetAllApplicationState(false);
    useAuthStore.getState().logout();

    if (this.navigate) {
      this.navigate('/login');
    }
  }

  /**
   * Navigate to login page and reset application state.
   *
   * Used when the session ends without an explicit logout (expired or
   * rejected token): the user's data is kept so the same user can carry on
   * after signing back in, and is cleared at sign-in if a different user
   * signs in instead.
   */
  navigateToLogin() {
    if (!this.navigate) {
      console.error('Navigation function not set');
      return;
    }

    // Store current username before logout for comparison during next login
    const currentUsername = useAuthStore.getState().username;
    if (currentUsername) {
      localStorage.setItem('LIGHTRAG-PREVIOUS-USER', currentUsername);
    }

    // Reset application state but preserve history
    // History will be cleared on next login if the user changes
    this.resetAllApplicationState(true);
    useAuthStore.getState().logout();

    this.navigate('/login');
  }

  navigateToHome() {
    if (!this.navigate) {
      console.error('Navigation function not set');
      return;
    }

    this.navigate('/');
  }
}

/**
 * Remove all per-user data stored in this browser.
 */
export function clearUserData() {
  useSettingsStore.getState().clearUserData();
  SearchHistoryManager.clearHistory();
}

/**
 * Called after a successful sign-in. If someone other than the previous user
 * signed in, clear the previous user's data before it can be shown.
 */
export function handleSignedInUser(username: string) {
  const previousUsername = localStorage.getItem('LIGHTRAG-PREVIOUS-USER');
  if (previousUsername !== username) {
    clearUserData();
  }
  localStorage.setItem('LIGHTRAG-PREVIOUS-USER', username);
}

export const navigationService = new NavigationService();
