"use client";
import AuthButton from "@/components/AuthButton";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function LoginPage() {
  const { status } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (status === "authenticated") {
      router.replace("/chat");
    }
  }, [status, router]);

  return (
    <div className="w-full max-w-md mx-auto bg-white rounded-2xl shadow-2xl border border-gray-100 min-h-[60vh] flex flex-col items-center justify-center p-8 mt-12">
      <div className="flex flex-col items-center mb-8">
        <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center mb-4">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10 text-gray-400">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118A7.5 7.5 0 0112 15.75a7.5 7.5 0 017.5 4.368" />
          </svg>
        </div>
        <h2 className="text-2xl font-bold mb-2">Login to Your Account</h2>
        <p className="text-gray-600 text-sm">Sign in to access the CZSU Data Explorer</p>
      </div>
      <div className="w-full flex flex-col items-center">
        <AuthButton />
      </div>
    </div>
  );
} 