import type { Metadata } from "next";
// Font configuration - change this import to switch fonts globally
import { Inter } from "next/font/google";
import "./globals.css";
import Header from '@/components/Header';
import SessionProviderWrapper from "@/components/SessionProviderWrapper";
import AuthGuard from "@/components/AuthGuard";
import ClientLayout from "./ClientLayout";

// Font setup - change the font import above and variable name below to switch fonts
const primaryFont = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

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
        className={`${primaryFont.variable} antialiased`}
        suppressHydrationWarning={true}
      >
        <SessionProviderWrapper>
          <ClientLayout>{children}</ClientLayout>
        </SessionProviderWrapper>
      </body>
    </html>
  );
}
