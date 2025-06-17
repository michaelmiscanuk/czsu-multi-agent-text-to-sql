"use client";
import { useSession } from "next-auth/react";
import { useRouter, usePathname } from "next/navigation";
import { useEffect } from "react";
import AuthButton from "./AuthButton";

// Public routes that don't require authentication
const PUBLIC_ROUTES = ["/", "/contacts", "/login"];

// Routes that should allow navigation but protect content behind login
const PROTECTED_ROUTES = ["/chat", "/catalog", "/data"];

export default function AuthGuard({ children }: { children: React.ReactNode }) {
  const { status } = useSession();
  const pathname = usePathname();

  const isPublicRoute = PUBLIC_ROUTES.includes(pathname);
  const isProtectedRoute = PROTECTED_ROUTES.includes(pathname);

  if (status === "loading") {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-gray-600">Loading...</div>
      </div>
    );
  }

  // For public routes or authenticated users, show content normally
  if (isPublicRoute || status === "authenticated") {
    return <>{children}</>;
  }

  // For protected routes when unauthenticated, show login requirement
  if (isProtectedRoute && status === "unauthenticated") {
    const getPageTitle = () => {
      switch (pathname) {
        case "/chat":
          return "Chat";
        case "/catalog":
          return "Catalog";
        case "/data":
          return "Data";
        default:
          return "This Page";
      }
    };

    return (
      <div className="flex flex-col items-center justify-center min-h-[500px] p-8">
        <div className="max-w-md text-center space-y-6">
          <div className="text-6xl mb-4">🔒</div>
          <h1 className="text-2xl font-bold text-gray-800 mb-2">
            {getPageTitle()} - Login Required
          </h1>
          <p className="text-gray-600 mb-6">
            You need to sign in to access the {getPageTitle().toLowerCase()} page.
          </p>
          <div className="space-y-4">
            <AuthButton compact={false} />
            <p className="text-sm text-gray-500">
              Sign in to start using our multi-agent text-to-SQL system
            </p>
          </div>
        </div>
      </div>
    );
  }

  // For any other unauthenticated routes, redirect to login
  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <div className="text-gray-600">Redirecting to login...</div>
    </div>
  );
} 