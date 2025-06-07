"use client";
import { useSession, signIn, signOut } from "next-auth/react";

export default function AuthButton({ compact = false }: { compact?: boolean } = {}) {
  const { data: session, status } = useSession();
  if (typeof window !== 'undefined') {
    console.log('[AuthButton] session:', session);
  }
  if (status === "loading") return <span className="text-xs text-gray-400">Loading...</span>;
  if (session) {
    return (
      <div className="flex items-center space-x-2">
        {session.user?.image ? (
          <img
            src={session.user.image}
            alt={session.user?.name || session.user?.email || 'avatar'}
            className="w-7 h-7 rounded-full border border-gray-300 bg-gray-100 object-cover"
            onError={e => {
              const target = e.target as HTMLImageElement;
              target.onerror = null;
              target.style.display = 'none';
              const fallback = document.createElement('div');
              fallback.className = 'w-7 h-7 rounded-full bg-gray-300 flex items-center justify-center text-xs font-bold text-gray-700';
              fallback.innerText = (session.user?.name || session.user?.email || '?').split(' ').map(s => s[0]).join('').slice(0,2).toUpperCase();
              target.parentNode?.insertBefore(fallback, target.nextSibling);
            }}
          />
        ) : (
          <div className="w-7 h-7 rounded-full bg-gray-300 flex items-center justify-center text-xs font-bold text-gray-700">
            {(session.user?.name || session.user?.email || '?').split(' ').map(s => s[0]).join('').slice(0,2).toUpperCase()}
          </div>
        )}
        <span className="text-xs text-gray-700 font-medium">{session.user?.name || session.user?.email}</span>
        <button
          className="px-3 py-1.5 text-xs bg-gradient-to-r from-gray-200 to-gray-300 hover:from-gray-300 hover:to-gray-400 rounded-lg font-semibold text-gray-700 border border-gray-300 transition-all duration-200 shadow-sm"
          onClick={() => signOut()}
        >
          Sign out
        </button>
      </div>
    );
  }
  if (compact) {
    // Header: Log In button styled as in the screenshot, with a more gray hover and more gradual transition
    return (
      <button
        className="flex items-center px-7 py-2 bg-white rounded-full border border-gray-200 shadow-md text-[#172153] font-bold hover:bg-gray-200 hover:shadow-lg hover:border-gray-300 transition-all duration-300 focus:outline-none"
        onClick={() => signIn('google')}
        style={{ minWidth: 110 }}
      >
        Log In
      </button>
    );
  }
  // Main Google button
  return (
    <button
      className="flex items-center justify-center w-full px-4 py-2 border border-gray-300 rounded-lg shadow-sm bg-white hover:bg-blue-50 hover:shadow-lg hover:border-blue-400 transition-all duration-200 text-gray-700 font-medium text-base focus:outline-none"
      onClick={() => signIn("google")}
      style={{ minWidth: 220 }}
    >
      <img
        src="https://developers.google.com/identity/images/g-logo.png"
        alt="Google logo"
        className="w-5 h-5 mr-3"
      />
      Continue with Google
    </button>
  );
} 