import React, { useEffect, useState } from 'react';
import { removeDiacritics } from './utils';

// To support both local dev and production, set NEXT_PUBLIC_API_BASE in .env.local to your backend URL (e.g., http://localhost:8000) for local dev.
// In production, leave it unset to use relative paths.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '';

interface DatasetRow {
  selection_code: string;
  extended_description: string;
}

interface DatasetsResponse {
  results: DatasetRow[];
  total: number;
  page: number;
  page_size: number;
}

interface DatasetsTableProps {
  onRowClick?: (selection_code: string) => void;
}

const DatasetsTable: React.FC<DatasetsTableProps> = ({ onRowClick }) => {
  const [data, setData] = useState<DatasetRow[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [filter, setFilter] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    // If there is a filter, fetch all datasets and filter client-side
    if (filter) {
      fetch(`${API_BASE}/datasets?page=1&page_size=10000`)
        .then(res => res.json())
        .then((res: DatasetsResponse) => {
          const normWords = removeDiacritics(filter.toLowerCase()).split(/\s+/).filter(Boolean);
          const filteredResults = res.results.filter(row => {
            const haystack = removeDiacritics((row.selection_code + ' ' + row.extended_description).toLowerCase());
            return normWords.every(word => haystack.includes(word));
          });
          setData(filteredResults.slice((page - 1) * 10, page * 10));
          setTotal(filteredResults.length);
          setLoading(false);
        });
    } else {
      // No filter: use backend pagination
      const params = new URLSearchParams({ page: page.toString() });
      fetch(`${API_BASE}/datasets?${params.toString()}`)
        .then(res => res.json())
        .then((res: DatasetsResponse) => {
          setData(res.results);
          setTotal(res.total);
          setLoading(false);
        });
    }
  }, [page, filter]);

  const totalPages = Math.ceil(total / 10);

  return (
    <div className="flex flex-col h-full p-6">
      <div className="mb-4 flex items-center">
        <input
          className="border border-gray-300 rounded px-3 py-2 w-80 mr-4"
          placeholder="Filter by keyword..."
          value={filter}
          onChange={e => { setPage(1); setFilter(e.target.value); }}
        />
        <span className="text-gray-500 text-sm">{total} records</span>
      </div>
      <div className="flex-1 overflow-auto">
        <table className="min-w-full border border-gray-200 rounded">
          <thead>
            <tr className="bg-gray-100">
              <th className="px-4 py-2 border-b text-left">Selection Code</th>
              <th className="px-4 py-2 border-b text-left">Extended Description</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={2} className="text-center py-8">Loading...</td></tr>
            ) : data.length === 0 ? (
              <tr><td colSpan={2} className="text-center py-8">No records found.</td></tr>
            ) : (
              data.map(row => (
                <tr
                  key={row.selection_code}
                  className={"hover:bg-gray-50"}
                >
                  <td className="px-4 py-2 border-b font-mono text-xs">
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
                  <td className="px-4 py-2 border-b text-sm whitespace-pre-line">{row.extended_description}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
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

export default DatasetsTable; 