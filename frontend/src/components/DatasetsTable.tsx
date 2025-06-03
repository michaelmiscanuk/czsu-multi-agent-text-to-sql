import React, { useEffect, useState } from 'react';
import { removeDiacritics } from './utils';
import { useSession } from "next-auth/react";

// To support both local dev and production, set NEXT_PUBLIC_API_BASE in .env.local to your backend URL (e.g., http://localhost:8000) for local dev.
// In production, leave it unset to use relative paths.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '';

interface CatalogRow {
  selection_code: string;
  extended_description: string;
}

interface CatalogResponse {
  results: CatalogRow[];
  total: number;
  page: number;
  page_size: number;
}

interface CatalogTableProps {
  onRowClick?: (selection_code: string) => void;
}

const CATALOG_PAGE_KEY = 'czsu-catalog-page';
const CATALOG_FILTER_KEY = 'czsu-catalog-filter';

const CatalogTable: React.FC<CatalogTableProps> = ({ onRowClick }) => {
  const { data: session } = useSession();
  const [data, setData] = useState<CatalogRow[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [filter, setFilter] = useState('');
  const [loading, setLoading] = useState(false);

  // Restore page and filter from localStorage on mount
  useEffect(() => {
    const savedPage = localStorage.getItem(CATALOG_PAGE_KEY);
    const savedFilter = localStorage.getItem(CATALOG_FILTER_KEY);
    if (savedPage) {
      const pageNum = Number(savedPage);
      if (!isNaN(pageNum)) setPage(pageNum);
    }
    if (typeof savedFilter === 'string') setFilter(savedFilter);
  }, []);

  // Persist page and filter to localStorage
  useEffect(() => {
    localStorage.setItem(CATALOG_PAGE_KEY, String(page));
  }, [page]);
  useEffect(() => {
    localStorage.setItem(CATALOG_FILTER_KEY, filter);
  }, [filter]);

  useEffect(() => {
    console.log('[DatasetsTable] Session:', JSON.stringify(session, null, 2));
    setLoading(true);
    const handleError = (err: any) => {
      setData([]);
      setTotal(0);
      setLoading(false);
    };
    // Helper to build fetch options
    const getFetchOptions = () =>
      session?.id_token
        ? { headers: { Authorization: `Bearer ${session.id_token}` } }
        : undefined;
    const fetchOptions = getFetchOptions();
    console.log('[DatasetsTable] Fetch options:', JSON.stringify(fetchOptions, null, 2));
    // If there is a filter, fetch all catalog and filter client-side
    if (filter) {
      fetch(`${API_BASE}/catalog?page=1&page_size=10000`, fetchOptions)
        .then(res => res.ok ? res.json() : Promise.reject(res))
        .then((res: CatalogResponse) => {
          const normWords = removeDiacritics(filter.toLowerCase()).split(/\s+/).filter(Boolean);
          const filteredResults = res.results.filter(row => {
            const haystack = removeDiacritics((row.selection_code + ' ' + row.extended_description).toLowerCase());
            return normWords.every(word => haystack.includes(word));
          });
          setData(filteredResults.slice((page - 1) * 10, page * 10));
          setTotal(filteredResults.length);
          setLoading(false);
        })
        .catch(handleError);
    } else {
      // No filter: use backend pagination
      const params = new URLSearchParams({ page: page.toString() });
      fetch(`${API_BASE}/catalog?${params.toString()}`, fetchOptions)
        .then(res => res.ok ? res.json() : Promise.reject(res))
        .then((res: CatalogResponse) => {
          setData(res.results);
          setTotal(res.total);
          setLoading(false);
        })
        .catch(handleError);
    }
  }, [page, filter, session?.id_token]);

  const totalPages = Math.ceil(total / 10);

  // Reset filter and page
  const handleReset = () => {
    setFilter('');
    setPage(1);
  };

  return (
    <div className="flex flex-col h-full p-6">
      <div className="mb-4 flex items-center">
        <input
          className="border border-gray-300 rounded px-3 py-2 w-80 mr-2"
          placeholder="Filter by keyword..."
          aria-label="Filter catalog by keyword"
          value={filter}
          onChange={e => { setPage(1); setFilter(e.target.value); }}
        />
        {filter && (
          <button
            className="text-gray-400 hover:text-gray-700 text-lg font-bold px-2 py-1 focus:outline-none"
            title="Clear filter"
            aria-label="Clear filter"
            tabIndex={0}
            onClick={handleReset}
            style={{ lineHeight: 1 }}
          >
            ×
          </button>
        )}
        <span className="text-gray-500 text-sm ml-4">{total} records</span>
      </div>
      <div className="flex-1 overflow-auto">
        <div className="overflow-x-auto rounded shadow border border-gray-200 bg-white">
          <table className="min-w-full text-xs">
            <thead className="bg-blue-100 sticky top-0 z-10">
              <tr>
                <th className="px-4 py-2 border-b text-left font-semibold text-gray-700 whitespace-nowrap">Selection Code</th>
                <th className="px-4 py-2 border-b text-left font-semibold text-gray-700 whitespace-nowrap">Extended Description</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={2} className="text-center py-8">Loading...</td></tr>
              ) : data.length === 0 ? (
                <tr><td colSpan={2} className="text-center py-8">No records found.</td></tr>
              ) : (
                data.map((row, i) => (
                  <tr key={row.selection_code} className={i % 2 === 0 ? "bg-white" : "bg-blue-50"}>
                    <td className="px-4 py-2 border-b font-mono text-xs text-blue-900">
                      {onRowClick ? (
                        <button
                          className="text-blue-600 underline hover:text-blue-800 cursor-pointer p-0 bg-transparent border-0 outline-none"
                          style={{ textDecoration: 'underline' }}
                          onClick={() => onRowClick(row.selection_code)}
                        >
                          {row.selection_code}
                        </button>
                      ) : (
                        row.selection_code
                      )}
                    </td>
                    <td className="px-4 py-2 border-b text-sm whitespace-pre-line text-gray-800">{row.extended_description}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
      <div className="mt-4 flex justify-between items-center">
        <button
          className="px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 text-sm"
          onClick={() => setPage(p => Math.max(1, p - 1))}
          disabled={page === 1}
        >Previous</button>
        <span className="text-gray-600 text-sm">Page {page} of {totalPages || 1}</span>
        <button
          className="px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 text-sm"
          onClick={() => setPage(p => Math.min(totalPages, p + 1))}
          disabled={page === totalPages || totalPages === 0}
        >Next</button>
      </div>
    </div>
  );
};

export default CatalogTable; 