'use server';
import { NextRequest } from "next/server";

export async function GET(
  req: NextRequest,
  context: any
) {
  const { width, height } = await context.params;
  const w = parseInt(width, 10) || 100;
  const h = parseInt(height, 10) || 100;

  // Create a simple SVG as a placeholder
  const svg = `<svg width='${w}' height='${h}' xmlns='http://www.w3.org/2000/svg'><rect width='100%' height='100%' fill='#e5e7eb'/><text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' fill='#9ca3af' font-size='20'>${w}x${h}</text></svg>`;
  return new Response(svg, {
    status: 200,
    headers: {
      'Content-Type': 'image/svg+xml',
      'Cache-Control': 'public, max-age=3600',
    },
  });
} 