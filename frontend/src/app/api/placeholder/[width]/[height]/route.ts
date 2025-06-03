import { NextRequest } from "next/server";

export async function GET(req: NextRequest, { params }: { params: { width: string, height: string } }) {
  const width = parseInt(params.width, 10) || 100;
  const height = parseInt(params.height, 10) || 100;

  // Create a simple SVG as a placeholder
  const svg = `<svg width='${width}' height='${height}' xmlns='http://www.w3.org/2000/svg'><rect width='100%' height='100%' fill='#e5e7eb'/><text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' fill='#9ca3af' font-size='20'>${width}x${height}</text></svg>`;
  return new Response(svg, {
    status: 200,
    headers: {
      'Content-Type': 'image/svg+xml',
      'Cache-Control': 'public, max-age=3600',
    },
  });
} 