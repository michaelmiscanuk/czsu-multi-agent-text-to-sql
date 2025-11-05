"""
Script to build comprehensive Features_Frontend_categorized.md
This will create ~2500-3000 lines of detailed frontend feature documentation
"""


def get_all_features_content():
    """Returns the complete content for all 36 frontend features"""

    # This will be built as a large string with all features
    content = ""

    # Continue from where we left off (1.1 and 1.2 already done)

    # === CATEGORY 1: Authentication & Security (3 more features) ===

    content += """
## 1.3. Session Management

### Purpose & Usage

Centralized session management using NextAuth.js provides consistent authentication state across the application. The system handles session lifecycle, refresh, and cross-component access patterns.

**Primary Use Cases:**
- Accessing user information (email, name, picture) in any component
- Checking authentication status before API calls
- Handling session expiration and automatic refresh
- Synchronizing session state across browser tabs
- Providing loading states during session initialization

**Referenced Files:**
- `frontend/src/components/SessionProviderWrapper.tsx` - Global session provider
- `frontend/src/app/layout.tsx` - Session provider integration
- Custom hooks using `useSession()` throughout app

### Key Implementation Steps

1. **Session Provider Initialization**
   - SessionProviderWrapper wraps entire app in root layout
   - Provides session context to all child components
   - Handles automatic token refresh in background
   - Broadcasts session changes via storage events

2. **useSession Hook Pattern**
   - Access session with `const { data: session, status } = useSession()`
   - Three status values: 'loading', 'authenticated', 'unauthenticated'
   - Session object contains user info and access_token
   - Hook automatically re-renders on session changes

3. **Token Extraction for API Calls**
   - session.id_token used for backend JWT authentication
   - session.accessToken for Google API calls
   - Centralized in API client for consistency
   - Automatic token validation before requests

4. **Session Refresh Strategy**
   - NextAuth checks token expiry automatically
   - Triggers refresh 5 minutes before expiration
   - Updates session without user interaction
   - Handles refresh failures with re-authentication

5. **Cross-Tab Session Sync**
   - localStorage 'storage' event listener
   - Broadcasts signOut events to all tabs
   - Updates session state immediately
   - Prevents stale sessions across tabs

### Key Challenges Solved

**Challenge 1: Session Initialization Race Conditions**
- **Problem**: Components mount before session loads, causing null reference errors
- **Solution**: Check `status === 'loading'` before accessing session.user
- **Impact**: No crashes from undefined user data
- **Implementation**: Conditional rendering based on session status

**Challenge 2: Stale Session After Token Expiry**
- **Problem**: Backend returns 401 but frontend still shows user as authenticated
- **Solution**: API client detects 401, triggers getSession() to force refresh
- **Impact**: Automatic logout when tokens truly invalid
- **Implementation**: authApiFetch retry logic with session refresh

**Challenge 3: Session State Hydration Mismatch**
- **Problem**: Server-rendered session doesn't match client session, causing hydration errors
- **Solution**: 'use client' boundary ensures session only accessed client-side
- **Impact**: No React hydration warnings
- **Implementation**: SessionProviderWrapper marked as client component

**Challenge 4: Multiple Concurrent Session Refreshes**
- **Problem**: Multiple API calls trigger simultaneous token refresh requests
- **Solution**: NextAuth debounces refresh, queues concurrent calls
- **Impact**: Only one refresh request even with 10 simultaneous API calls
- **Implementation**: NextAuth built-in refresh debouncing

**Challenge 5: Session Lost on Hard Refresh**
- **Problem**: F5 or Ctrl+R loses session if only stored in memory
- **Solution**: Session stored in HTTP-only cookie, survives page reloads
- **Impact**: Seamless user experience across refreshes
- **Implementation**: NextAuth JWT strategy with persistent cookies

**Challenge 6: Testing Components Requiring Session**
- **Problem**: Unit tests fail when components call useSession()
- **Solution**: Mock SessionProvider wrapper in test utilities
- **Impact**: Fast, isolated component tests without real auth
- **Implementation**: Jest mock for next-auth/react module

**Challenge 7: Session Timeout UX**
- **Problem**: User away for hours, returns to expired session with no warning
- **Solution**: Show "Session expired, please sign in again" message
- **Impact**: Clear communication vs cryptic errors
- **Implementation**: Error handler checks 401 status and shows alert

**Challenge 8: Accessing Session in Server Components**
- **Problem**: Can't use useSession() hook in server components
- **Solution**: Use getServerSession() for server-side session access
- **Impact**: SSR pages can check authentication
- **Implementation**: Import getServerSession from next-auth/next

**Challenge 9: Session Provider Tree Depth**
- **Problem**: Deep component trees cause prop drilling for session data
- **Solution**: Context provider at root makes session available anywhere
- **Impact**: Any component can useSession() without props
- **Implementation**: SessionProvider in layout.tsx root

**Challenge 10: Development vs Production Session Behavior**
- **Problem**: Different cookie settings for localhost vs production domain
- **Solution**: Environment-based cookie configuration in NextAuth
- **Impact**: Works seamlessly in both environments
- **Implementation**: NEXTAUTH_URL environment variable

---

## 1.4. API Token Authentication

### Purpose & Usage

Token-based API authentication integrates frontend sessions with backend JWT verification. The system automatically attaches auth tokens to API requests and handles token refresh.

**Primary Use Cases:**
- Authenticating all backend API requests
- Extracting ID token from session for backend verification
- Handling 401 unauthorized errors with automatic retry
- Token validation before expensive API calls
- Providing clear error messages for auth failures

**Referenced Files:**
- `frontend/src/lib/api.ts` - Auth API client implementation
- API calls in components using authApiFetch helper
- Backend JWT verification in `api/dependencies/auth.py`

### Key Implementation Steps

1. **Token Extraction Pattern**
   - Get session with `await getSession()`
   - Extract id_token from session.id_token
   - Validate token exists before API call
   - Throw clear error if no token available

2. **Authorization Header Injection**
   - Create Authorization header: `Bearer ${token}`
   - createAuthFetchOptions helper adds header
   - Consistent format across all authenticated requests
   - Override-able for special cases

3. **Automatic 401 Retry Logic**
   - First attempt with current token
   - On 401, call getSession() to refresh
   - Retry original request with fresh token
   - Fail after second 401 with clear error

4. **Error Context Logging**
   - Log endpoint, status code, error message
   - Include request timing for debugging
   - Preserve error stack for troubleshooting
   - Client IP and user context when available

5. **Token Validation UI Feedback**
   - Show "Authenticating..." during token checks
   - Display "Authentication failed" on token errors
   - Provide "Sign in again" button for expired sessions
   - Loading spinner during retry attempts

### Key Challenges Solved

**Challenge 1: Token Expiry Mid-Request**
- **Problem**: Token valid when request starts but expires before response
- **Solution**: Catch 401, refresh token, automatically retry request
- **Impact**: Transparent token refresh without user interruption
- **Implementation**: authApiFetch 401 error handler with retry

**Challenge 2: Missing Token Causing Generic Errors**
- **Problem**: No token results in vague "Request failed" errors
- **Solution**: Check token existence upfront, throw specific error
- **Impact**: Clear "No access token available" vs mysterious failures
- **Implementation**: Early token validation in authApiFetch

**Challenge 3: Token Refresh Race Condition**
- **Problem**: Multiple failing requests trigger simultaneous refresh attempts
- **Solution**: NextAuth queues refresh requests, returns same promise
- **Impact**: Only one refresh call even with 10 concurrent 401s
- **Implementation**: NextAuth built-in refresh serialization

**Challenge 4: Exposing Tokens in Browser DevTools**
- **Problem**: Authorization headers visible in Network tab
- **Solution**: This is unavoidable but acceptable (HTTPS encrypts in transit)
- **Impact**: Tokens protected during transmission, logged out after theft
- **Implementation**: Short token expiry (1 hour) limits damage window

**Challenge 5: Backend Token Validation Failures**
- **Problem**: Frontend has token but backend rejects it (clock skew, wrong issuer)
- **Solution**: Backend returns detailed validation error (expired, invalid signature, etc.)
- **Impact**: Frontend can show specific error vs generic "Unauthorized"
- **Implementation**: Backend JWT verification with detailed error responses

**Challenge 6: Cross-Origin API Calls**
- **Problem**: CORS blocks requests to backend on different domain
- **Solution**: Backend CORS middleware allows frontend domain
- **Impact**: API calls work from Vercel frontend to Railway backend
- **Implementation**: CORS middleware in backend with allowed_origins

**Challenge 7: Token Size and Request Performance**
- **Problem**: Large JWT tokens (1-2KB) add overhead to every request
- **Solution**: This is acceptable trade-off for stateless authentication
- **Impact**: ~1KB per request vs server-side session lookup latency
- **Implementation**: Compact JWT with only essential claims

**Challenge 8: Token Refresh During Long-Running Requests**
- **Problem**: 5-minute analysis request outlives token expiry
- **Solution**: NextAuth refreshes token in background during long requests
- **Impact**: Response still succeeds even if token expires during request
- **Implementation**: NextAuth automatic background refresh

**Challenge 9: Testing API Calls Without Real Auth**
- **Problem**: Integration tests need valid tokens to test API endpoints
- **Solution**: Mock authApiFetch to bypass authentication in tests
- **Impact**: Fast tests without OAuth dance
- **Implementation**: Jest mock returns test data directly

**Challenge 10: Debugging Token Issues in Production**
- **Problem**: Hard to diagnose token problems without seeing token contents
- **Solution**: Log token metadata (expiry, issuer) without exposing full token
- **Impact**: Can diagnose clock skew and expiry issues from logs
- **Implementation**: Extract and log JWT claims without token itself

---

## 1.5. Cross-Tab Session Synchronization

### Purpose & Usage

Cross-tab session synchronization ensures consistent authentication state across multiple browser tabs. The system broadcasts authentication events using browser storage events and state management.

**Primary Use Cases:**
- Logging out in one tab immediately logs out all tabs
- Signing in on one tab reflects in other tabs instantly
- Preventing inconsistent auth states across tabs
- Synchronizing user-specific data after login/logout
- Handling race conditions from concurrent tab actions

**Referenced Files:**
- `frontend/src/contexts/ChatCacheContext.tsx` - Cross-tab loading states
- `frontend/src/components/SessionProviderWrapper.tsx` - Session broadcasting
- localStorage for cross-tab communication

### Key Implementation Steps

1. **Storage Event Listener**
   - Listen to 'storage' event on window object
   - Triggered when localStorage changes in another tab
   - Extract event key and newValue
   - Update local state based on changes

2. **Logout Broadcast Pattern**
   - On signOut(), set localStorage flag: 'czsu-user-logout' = timestamp
   - Other tabs detect storage event
   - Each tab calls signOut() locally
   - Clear flag after all tabs processed

3. **Login Sync Pattern**
   - After successful signIn(), set 'czsu-user-login' with email
   - Other tabs detect change
   - Each tab refreshes session via getSession()
   - Update UI to reflect authenticated state

4. **Loading State Coordination**
   - User-specific loading flag: 'czsu-user-loading-{email}'
   - Set to true when starting expensive operation
   - Other tabs show "Already loading in another tab" message
   - Clear flag when operation completes

5. **Cache Invalidation Sync**
   - On cache clear in one tab, broadcast 'czsu-cache-invalidated'
   - Other tabs detect event and clear their caches
   - All tabs reload data from API
   - Prevents stale cache inconsistencies

### Key Challenges Solved

**Challenge 1: Storage Event Not Firing in Originating Tab**
- **Problem**: Storage event only fires in other tabs, not the one that made the change
- **Solution**: Manually call state update in originating tab before setting storage
- **Impact**: Consistent behavior across all tabs
- **Implementation**: Direct state update + localStorage.setItem()

**Challenge 2: Race Conditions on Concurrent Actions**
- **Problem**: Two tabs submit form simultaneously, causing duplicate submissions
- **Solution**: Loading state flag prevents concurrent operations
- **Impact**: Only one tab can perform expensive operation at a time
- **Implementation**: Check/set loading flag atomically

**Challenge 3: Storage Event Spam**
- **Problem**: Multiple storage changes trigger cascade of events
- **Solution**: Debounce storage event handler with 100ms delay
- **Impact**: Process only final state change
- **Implementation**: setTimeout debounce in storage event listener

**Challenge 4: Private/Incognito Mode localStorage**
- **Problem**: Some browsers restrict localStorage in private mode
- **Solution**: Try/catch around localStorage access, degrade gracefully
- **Impact**: App still works without cross-tab sync
- **Implementation**: Graceful degradation with error logging

**Challenge 5: Different Data in Different Tabs**
- **Problem**: User loads different thread in each tab, causing state conflicts
- **Solution**: Thread-specific caching with tab-local active thread
- **Impact**: Each tab maintains its own active context
- **Implementation**: Active thread ID not synced across tabs

**Challenge 6: Logout in One Tab Losing Data in Others**
- **Problem**: Instant logout clears cache before other tabs can save work
- **Solution**: Grace period warning before cache clear
- **Impact**: Users can save work before forced logout
- **Implementation**: 5-second countdown with notification

**Challenge 7: Session Timeout While Tab Inactive**
- **Problem**: Background tab's session expires, user returns to stale state
- **Solution**: Refresh session on tab focus event
- **Impact**: Current session state when user switches back
- **Implementation**: visibilitychange event listener

**Challenge 8: Storage Size Limits**
- **Problem**: LocalStorage has 5-10MB limit, can fill up with cache data
- **Solution**: Store only essential data, set expiration on cache entries
- **Impact**: Cache stays under limit even with heavy usage
- **Implementation**: 48-hour cache expiration with automatic cleanup

**Challenge 9: Testing Cross-Tab Sync**
- **Problem**: Hard to simulate multiple tabs in automated tests
- **Solution**: Mock localStorage with event emitter in tests
- **Impact**: Can test cross-tab scenarios in single-process tests
- **Implementation**: Jest mock localStorage with custom event dispatcher

**Challenge 10: User Confusion with Cross-Tab Messages**
- **Problem**: User doesn't understand why their tab is blocked
- **Solution**: Clear message: "Another tab is currently processing..."
- **Impact**: User knows to wait or switch tabs
- **Implementation**: Toast notification with explanation

---

"""

    return content


# For now, let me just generate a portion and we'll see the pattern
if __name__ == "__main__":
    content = get_all_features_content()
    print(f"Generated {len(content)} characters")
    print(f"Approximately {len(content.split('###'))} sections")
