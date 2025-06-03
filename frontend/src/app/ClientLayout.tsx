"use client";
import { usePathname, useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { useEffect } from "react";
import Header from '@/components/Header';
import AuthGuard from "@/components/AuthGuard";

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  // Client-side auth redirect
  const pathname = usePathname();
  const { status } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (pathname !== "/login" && status === "unauthenticated") {
      router.replace("/login");
    }
  }, [status, router, pathname]);

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-blue-100 via-blue-50 to-blue-200 flex flex-col">
      <div className="sticky top-0 z-50"><Header /></div>
      <main className="flex justify-center flex-1 py-8 px-2">
        <AuthGuard>{children}</AuthGuard>
      </main>
      <footer className="w-full text-center text-gray-400 text-sm py-4 mt-4">
        &copy; {new Date().getFullYear()} Michael Miscanuk. Data from the Czech Statistical Office (CZSU).
      </footer>
    </div>
  );
} 