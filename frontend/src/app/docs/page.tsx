'use client';

import { useEffect, useState } from 'react';
import SwaggerUI from 'swagger-ui-react';
import 'swagger-ui-react/swagger-ui.css';

export default function DocsPage() {
  const [spec, setSpec] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSpec = async () => {
      try {
        // In development, API is on localhost:8000
        // In production, use the deployed API URL
        const apiUrl = process.env.NODE_ENV === 'production'
          ? 'https://www.multiagent-texttosql-prototype.online/api'
          : 'http://localhost:8000';

        const response = await fetch(`${apiUrl}/openapi.json`);
        if (!response.ok) {
          throw new Error(`Failed to fetch OpenAPI spec: ${response.status}`);
        }
        const data = await response.json();
        setSpec(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An unknown error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchSpec();
  }, []);

  if (loading) {
    return <div className="flex justify-center items-center min-h-screen">Loading API documentation...</div>;
  }

  if (error) {
    return (
      <div className="flex flex-col justify-center items-center min-h-screen">
        <h1 className="text-2xl font-bold text-red-600 mb-4">Error Loading Documentation</h1>
        <p className="text-gray-600">{error}</p>
        <p className="text-sm text-gray-500 mt-4">
          Make sure the API server is running on localhost:8000 (development) or properly configured for production.
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <SwaggerUI spec={spec} />
    </div>
  );
}