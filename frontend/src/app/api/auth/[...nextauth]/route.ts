import NextAuth from "next-auth";
import GoogleProvider from "next-auth/providers/google";

async function refreshAccessToken(token: any) {
  try {
    const url =
      "https://oauth2.googleapis.com/token?" +
      new URLSearchParams({
        client_id: process.env.GOOGLE_CLIENT_ID!,
        client_secret: process.env.GOOGLE_CLIENT_SECRET!,
        grant_type: "refresh_token",
        refresh_token: token.refreshToken,
      });

    const response = await fetch(url, {
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      method: "POST",
    });

    const refreshedTokens = await response.json();

    if (!response.ok) throw refreshedTokens;

    return {
      ...token,
      accessToken: refreshedTokens.access_token,
      id_token: refreshedTokens.id_token,
      expires_at: Date.now() + refreshedTokens.expires_in * 1000,
      refreshToken: refreshedTokens.refresh_token ?? token.refreshToken,
    };
  } catch (error) {
    console.error("Error refreshing access token", error);
    return {
      ...token,
      error: "RefreshAccessTokenError",
    };
  }
}

const handler = NextAuth({
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
      authorization: {
        params: {
          scope: "openid email profile",
          access_type: "offline",
          prompt: "consent",
        },
      },
      profile(profile) {
        return {
          id: profile.sub,
          name: profile.name,
          email: profile.email,
          image: profile.picture,
        };
      },
    }),
  ],
  callbacks: {
    async jwt({ token, account, user, profile }) {
      // On initial sign in
      if (account) {
        token.accessToken = account.access_token;
        token.refreshToken = account.refresh_token;
        token.id_token = account.id_token;
        token.expires_at = account.expires_at ? account.expires_at * 1000 : Date.now() + 3600 * 1000; // fallback 1h
        // Add user info for session
        if (account?.provider === 'google') {
          token.picture = (profile && typeof profile === 'object' && 'picture' in profile && typeof profile.picture === 'string') ? profile.picture : (typeof account?.picture === 'string' ? account.picture : undefined);
          token.name = (profile && typeof profile === 'object' && 'name' in profile && typeof profile.name === 'string') ? profile.name : (typeof account?.name === 'string' ? account.name : undefined);
          token.email = (profile && typeof profile === 'object' && 'email' in profile && typeof profile.email === 'string') ? profile.email : (typeof account?.email === 'string' ? account.email : undefined);
        }
      }
      // If token has expired, refresh it
      if (typeof token.expires_at === 'number' && Date.now() > token.expires_at) {
        return await refreshAccessToken(token);
      }
      return token;
    },
    async session({ session, token, user }) {
      (session as any).accessToken = token.accessToken;
      (session as any).refreshToken = token.refreshToken;
      (session as any).id_token = token.id_token;
      // Ensure user info is present for avatar
      session.user = session.user || {};
      if (token.picture) session.user.image = token.picture;
      if (token.name) session.user.name = token.name;
      if (token.email) session.user.email = token.email;
      return session;
    },
  },
});

export { handler as GET, handler as POST }; 