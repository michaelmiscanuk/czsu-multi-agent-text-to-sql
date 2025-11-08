"use client";
import { usePathname, useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { useEffect } from "react";
import Header from '@/components/Header';
import AuthGuard from "@/components/AuthGuard";
import { ChatCacheProvider } from '@/contexts/ChatCacheContext';

// Public routes that don't require authentication
const PUBLIC_ROUTES = ["/", "/contacts", "/login", "/terms-of-use"];

// Routes that should allow navigation but protect content behind login
const PROTECTED_ROUTES = ["/chat", "/catalog", "/data"];

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  // Client-side auth redirect only for unknown routes
  const pathname = usePathname();
  const { status } = useSession();
  const router = useRouter();

  const isPublicRoute = PUBLIC_ROUTES.includes(pathname);
  const isProtectedRoute = PROTECTED_ROUTES.includes(pathname);

  useEffect(() => {
    // Only redirect if it's not a public route AND not a protected route
    // This lets AuthGuard handle protected routes
    if (status === "unauthenticated" && !isPublicRoute && !isProtectedRoute) {
      router.replace("/login");
    }
  }, [status, router, pathname, isPublicRoute, isProtectedRoute]);

  const isChatPage = pathname === "/chat";
  const isCatalogPage = pathname === "/catalog";
  const isDataPage = pathname === "/data";
  const shouldHideFooter = isChatPage || isCatalogPage || isDataPage;

  return (
    <ChatCacheProvider>
      <div className="min-h-screen w-full bg-gradient-to-br from-blue-100 via-blue-50 to-blue-200 flex flex-col">
        <div className="sticky top-0 z-50"><Header /></div>
        <main className="main-container-unified">
          <AuthGuard>{children}</AuthGuard>
        </main>
        {!shouldHideFooter && (
          <footer className="w-full text-center text-gray-400 text-sm py-4 mt-4">
            &copy; {new Date().getFullYear()} Michael Miscanuk. All rights reserved. Data from the{' '}
            <a 
              href="https://csu.gov.cz/podminky_pro_vyuzivani_a_dalsi_zverejnovani_statistickych_udaju_csu" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-gray-600 underline"
            >
              Czech Statistical Office (CZSU)
            </a>
            .
            {' | '}
            <a href="/terms-of-use" className="text-gray-400 hover:text-gray-600 underline">
              Terms of Use
            </a>
          </footer>
        )}
      </div>
    </ChatCacheProvider>
  );
} 