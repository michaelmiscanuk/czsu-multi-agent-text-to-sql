"use client";
import { useSession, signIn, signOut } from "next-auth/react";

export default function AuthButton() {
  const { data: session, status } = useSession();
  if (status === "loading") return <span className="text-xs text-gray-400">Loading...</span>;
  if (session) {
    return (
      <div className="flex items-center space-x-2">
        {session.user?.image && (
          <img src={session.user.image} alt="avatar" className="w-7 h-7 rounded-full border border-gray-300" />
        )}
        <span className="text-xs text-white font-medium">{session.user?.name || session.user?.email}</span>
        <button
          className="px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded font-semibold text-gray-700 border border-gray-300 transition-all duration-200"
          onClick={() => signOut()}
        >
          Sign out
        </button>
      </div>
    );
  }
  return (
    <button
      className="flex items-center justify-center w-full px-4 py-2 border border-gray-300 rounded-lg shadow-sm bg-white hover:bg-gray-50 transition-all duration-200 text-gray-700 font-medium text-base"
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