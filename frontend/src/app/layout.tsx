import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Header from '@/components/Header';
import SessionProviderWrapper from "@/components/SessionProviderWrapper";
import AuthGuard from "@/components/AuthGuard";
import ClientLayout from "./ClientLayout";

const inter = Inter({
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
        className={`${inter.variable} antialiased`}
      >
        <SessionProviderWrapper>
          <ClientLayout>{children}</ClientLayout>
        </SessionProviderWrapper>
      </body>
    </html>
  );
}
