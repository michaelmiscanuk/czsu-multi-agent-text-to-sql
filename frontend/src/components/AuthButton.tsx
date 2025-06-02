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
        <span className="text-xs text-gray-700 font-medium">{session.user?.name || session.user?.email}</span>
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
      className="px-3 py-1 text-xs bg-gradient-to-r from-blue-500 to-blue-400 hover:from-blue-600 hover:to-blue-500 text-white rounded-full font-semibold shadow border border-blue-200 transition-all duration-200"
      onClick={() => signIn("google")}
    >
      Sign in with Google
    </button>
  );
} 