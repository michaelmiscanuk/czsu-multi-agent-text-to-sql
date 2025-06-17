"use client";
import { useSession } from "next-auth/react";
import { useRouter, usePathname } from "next/navigation";
import { useEffect } from "react";

// Public routes that don't require authentication
const PUBLIC_ROUTES = ["/", "/contacts", "/login"];

export default function AuthGuard({ children }: { children: React.ReactNode }) {
  const { status } = useSession();
  const router = useRouter();
  const pathname = usePathname();

  const isPublicRoute = PUBLIC_ROUTES.includes(pathname);

  useEffect(() => {
    if (status === "unauthenticated" && !isPublicRoute) {
      router.replace("/login");
    }
  }, [status, router, pathname, isPublicRoute]);

  if (status === "loading") {
    return <div>Loading...</div>;
  }

  // Allow access to public routes even when unauthenticated
  if (isPublicRoute || status === "authenticated") {
    return <>{children}</>;
  }

  // For protected routes when unauthenticated, show loading while redirecting
  return <div>Loading...</div>;
} 