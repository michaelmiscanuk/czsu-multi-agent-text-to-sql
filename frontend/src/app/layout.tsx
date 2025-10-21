import type { Metadata } from "next";
import "./globals.css";
import Header from '@/components/Header';
import SessionProviderWrapper from "@/components/SessionProviderWrapper";
import AuthGuard from "@/components/AuthGuard";
import ClientLayout from "./ClientLayout";

export const metadata: Metadata = {
  title: "CZSU Multi-Agent Text-to-SQL",
  description: "CZSU Multi-Agent Text-to-SQL",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className="antialiased"
        suppressHydrationWarning={true}
      >
        <SessionProviderWrapper>
          <ClientLayout>{children}</ClientLayout>
        </SessionProviderWrapper>
      </body>
    </html>
  );
}
