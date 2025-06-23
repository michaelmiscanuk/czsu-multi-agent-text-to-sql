/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_GOOGLE_CLIENT_ID: string
  readonly VITE_GOOGLE_CLIENT_SECRET: string
  readonly VITE_API_BASE_URL: string
  readonly VITE_NEXTAUTH_URL: string
  readonly VITE_NEXTAUTH_SECRET: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
} 