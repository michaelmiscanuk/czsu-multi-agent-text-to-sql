import React, { useEffect, useState } from 'react';
import { removeDiacritics } from './utils';
import { useSession } from "next-auth/react";
import { authApiFetch } from '@/lib/api';
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
  const [isRestored, setIsRestored] = useState(false);

  // Restore page and filter from localStorage on mount
  useEffect(() => {
    const savedPage = localStorage.getItem(CATALOG_PAGE_KEY);
    const savedFilter = localStorage.getItem(CATALOG_FILTER_KEY);
    if (savedPage) {
      const pageNum = Number(savedPage);
      if (!isNaN(pageNum)) setPage(pageNum);
    }
    if (savedFilter) setFilter(savedFilter);
    setIsRestored(true);
  }, []);

  // Persist page and filter to localStorage
  useEffect(() => {
    localStorage.setItem(CATALOG_PAGE_KEY, String(page));
  }, [page]);
  useEffect(() => {
    localStorage.setItem(CATALOG_FILTER_KEY, filter);
  }, [filter]);

  useEffect(() => {
    // Don't fetch data until localStorage restoration is complete
    if (!isRestored) return;
    
    console.log('[DatasetsTable] Session:', JSON.stringify(session, null, 2));
    setLoading(true);
    const handleError = (err: any) => {
      setData([]);
      setTotal(0);
      setLoading(false);
    };
    // If there is a filter, fetch all catalog and filter client-side
    if (filter) {
      if (!session?.id_token) {
        handleError(new Error('No authentication token'));
        return;
      }
      authApiFetch<CatalogResponse>('/catalog?page=1&page_size=10000', session.id_token)
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
      if (!session?.id_token) {
        handleError(new Error('No authentication token'));
        return;
      }
      const params = new URLSearchParams({ page: page.toString() });
      authApiFetch<CatalogResponse>(`/catalog?${params.toString()}`, session.id_token)
        .then((res: CatalogResponse) => {
          setData(res.results);
          setTotal(res.total);
          setLoading(false);
        })
        .catch(handleError);
    }
  }, [page, filter, session?.id_token, isRestored]);

  const totalPages = Math.ceil(total / 10);

  // Reset filter and page
  const handleReset = () => {
    setFilter('');
    setPage(1);
  };

  return (
    <div className="table-container">
      <div className="filter-section flex items-center">
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
      
      <div className="table-section">
        <div className="overflow-x-auto rounded shadow border border-gray-200 bg-white">
          <table className="min-w-full table-content-font">
            <thead className="bg-blue-100 sticky top-0 z-10">
              <tr>
                <th className="px-4 py-2 border-b text-left table-header-font text-gray-700 whitespace-nowrap">Selection Code</th>
                <th className="px-4 py-2 border-b text-left table-header-font text-gray-700 whitespace-nowrap">Extended Description</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={2} className="text-center py-8 table-content-font">Loading...</td></tr>
              ) : data.length === 0 ? (
                <tr><td colSpan={2} className="text-center py-8 table-content-font">No records found.</td></tr>
              ) : (
                data.map((row, i) => (
                  <tr key={row.selection_code} className={i % 2 === 0 ? "bg-white" : "bg-blue-50"}>
                    <td className="px-4 py-2 border-b font-mono table-content-font text-blue-900">
                      {onRowClick ? (
                        <button
                          className="dataset-code-badge"
                          onClick={() => onRowClick(row.selection_code)}
                        >
                          {row.selection_code}
                        </button>
                      ) : (
                        row.selection_code
                      )}
                    </td>
                    <td className="px-4 py-2 border-b table-description-font table-cell-readable text-gray-800">{row.extended_description}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
      
      <div className="pagination-section flex items-center justify-between">
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