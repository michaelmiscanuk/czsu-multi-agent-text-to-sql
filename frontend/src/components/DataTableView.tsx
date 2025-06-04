import React, { useEffect, useRef } from 'react';
import { removeDiacritics } from './utils';
import { useSession } from "next-auth/react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '';

type DataTableViewProps = {
  search: string;
  setSearch: (s: string) => void;
  selectedTable: string | null;
  setSelectedTable: (t: string | null) => void;
  columns: string[];
  setColumns: (c: string[]) => void;
  rows: any[][];
  setRows: (r: any[][]) => void;
  selectedColumn: string | null;
  setSelectedColumn: (c: string | null) => void;
  columnFilters: { [col: string]: string };
  setColumnFilters: (f: { [col: string]: string }) => void;
  pendingTableSearch?: string | null;
  setPendingTableSearch?: (s: string | null) => void;
};

const SELECTED_TABLE_KEY = 'czsu-data-selectedTable';
const COLUMN_FILTERS_KEY = 'czsu-data-columnFilters';
const SELECTED_COLUMN_KEY = 'czsu-data-selectedColumn';
const SEARCH_KEY = 'czsu-data-search';

const DataTableView: React.FC<DataTableViewProps> = ({
  search,
  setSearch,
  selectedTable,
  setSelectedTable,
  columns,
  setColumns,
  rows,
  setRows,
  selectedColumn,
  setSelectedColumn,
  columnFilters,
  setColumnFilters,
  pendingTableSearch,
  setPendingTableSearch,
}) => {
  const [suggestions, setSuggestions] = React.useState<{ selection_code: string, short_description: string }[]>([]);
  const [showSuggestions, setShowSuggestions] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const [tableLoading, setTableLoading] = React.useState(false);
  const [allTables, setAllTables] = React.useState<{ selection_code: string, short_description: string }[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const { data: session } = useSession();

  // Prefill search box if pendingTableSearch changes
  React.useEffect(() => {
    if (pendingTableSearch) {
      setSearch(pendingTableSearch);
      setShowSuggestions(true);
      inputRef.current?.focus();
    }
  }, [pendingTableSearch]);

  // On mount, if search is non-empty and selectedTable is null, set selectedTable to search
  React.useEffect(() => {
    if (search && !selectedTable) {
      setSelectedTable(search);
    }
  }, [search, selectedTable, setSelectedTable]);

  // Fetch all tables on mount (for combo box)
  React.useEffect(() => {
    const getFetchOptions = () =>
      session?.id_token
        ? { headers: { Authorization: `Bearer ${session.id_token}` } }
        : undefined;
    fetch(`${API_BASE}/data-tables`, getFetchOptions())
      .then(res => res.ok ? res.json() : Promise.reject(res))
      .then(data => setAllTables(data.tables || []))
      .catch(() => setAllTables([]));
  }, [session?.id_token]);

  // Fetch table suggestions as user types or when search is empty
  useEffect(() => {
    if (search.trim() === '') {
      setSuggestions(allTables);
      return;
    }
    setLoading(true);
    let filteredTables: { selection_code: string, short_description: string }[];
    let normWords: string[];
    if (search.startsWith('*')) {
      // Only search in selection_code
      normWords = removeDiacritics(search.slice(1).toLowerCase()).split(/\s+/).filter(Boolean) as string[];
      filteredTables = allTables.filter(table => {
        const haystack = removeDiacritics(table.selection_code.toLowerCase());
        return normWords.every((word: string) => haystack.includes(word));
      });
    } else {
      // Search in both selection_code and short_description
      normWords = removeDiacritics(search.toLowerCase()).split(/\s+/).filter(Boolean) as string[];
      filteredTables = allTables.filter(table => {
        const haystack = removeDiacritics((table.selection_code + ' ' + (table.short_description || '')).toLowerCase());
        return normWords.every((word: string) => haystack.includes(word));
      });
    }
    const sortedTables = filteredTables.slice().sort((a: { selection_code: string, short_description: string }, b: { selection_code: string, short_description: string }) => a.selection_code.localeCompare(b.selection_code, 'cs', { sensitivity: 'base' }));
    setSuggestions(sortedTables);
    setLoading(false);
  }, [search, allTables]);

  // Fetch table data when a table is selected
  useEffect(() => {
    if (!selectedTable) {
      setColumns([]);
      setRows([]);
      setSelectedColumn(null);
      setColumnFilters({});
      return;
    }
    setTableLoading(true);
    const getFetchOptions = () =>
      session?.id_token
        ? { headers: { Authorization: `Bearer ${session.id_token}` } }
        : undefined;
    const url = `${API_BASE}/data-table?table=${encodeURIComponent(selectedTable)}`;
    const fetchOptions = getFetchOptions();
    console.log('[DataTableView] Fetching table:', selectedTable, url, JSON.stringify(fetchOptions, null, 2));
    fetch(url, fetchOptions)
      .then(res => res.ok ? res.json() : Promise.reject(res))
      .then(data => {
        console.log('[DataTableView] Received data:', JSON.stringify(data, null, 2));
        setColumns(data.columns || []);
        setRows(data.rows || []);
        setSelectedColumn(data.columns && data.columns.length > 0 ? data.columns[0] : null);
        setColumnFilters({});
        setTableLoading(false);
      })
      .catch((err) => {
        console.error('[DataTableView] Error fetching table:', JSON.stringify(err, null, 2));
        setTableLoading(false);
      });
  }, [selectedTable, setColumns, setRows, setSelectedColumn, setColumnFilters, session?.id_token]);

  // Auto-select and load the table if pendingTableSearch matches a suggestion exactly
  React.useEffect(() => {
    if (pendingTableSearch && suggestions.length > 0) {
      const match = suggestions.find(s => s.selection_code === pendingTableSearch);
      if (match) {
        setSelectedTable(match.selection_code);
        setSearch(match.selection_code);
        setShowSuggestions(false);
        if (setPendingTableSearch) setPendingTableSearch(null);
      }
    }
    // Only run when suggestions or pendingTableSearch changes
  }, [pendingTableSearch, suggestions, setSelectedTable, setPendingTableSearch]);

  // Handle suggestion click
  const handleSuggestionClick = (table: string) => {
    setSelectedTable(table);
    setSearch(table);
    setShowSuggestions(false);
    if (setPendingTableSearch) setPendingTableSearch(null);
  };

  // Handle input focus/blur
  const handleBlur = () => {
    setTimeout(() => setShowSuggestions(false), 100); // Delay to allow click
  };

  // Handle column filter change and remember per column
  const handleColumnFilterChange = (col: string, value: string) => {
    setColumnFilters({ ...columnFilters, [col]: value });
  };

  // Handle column selection change
  const handleColumnSelect = (col: string) => {
    setSelectedColumn(col);
  };

  // Clear all filters
  const handleClearFilters = () => {
    setColumnFilters({});
  };

  // Enhanced filter logic for numeric 'value' column
  const filteredRows = React.useMemo(() => {
    if (!columns.length || Object.values(columnFilters).every(v => !v)) return rows;
    return rows.filter(row =>
      columns.every((col, idx) => {
        const filter = columnFilters[col];
        if (!filter) return true;
        if (col === 'value') {
          // Numeric filter: support >, >=, <, <=, !=, =, ==, or just a number (equals)
          const match = filter.trim().match(/^(>=|<=|!=|>|<|=|==)?\s*(-?\d+(?:\.\d+)?)/);
          if (match) {
            const op = match[1] || '==';
            const num = parseFloat(match[2]);
            const cellNum = parseFloat(row[idx]);
            if (isNaN(cellNum)) return false;
            switch (op) {
              case '>': return cellNum > num;
              case '>=': return cellNum >= num;
              case '<': return cellNum < num;
              case '<=': return cellNum <= num;
              case '!=': return cellNum !== num;
              case '=':
              case '==': return cellNum === num;
              default: return cellNum === num;
            }
          } else {
            // fallback: substring match (diacritics-insensitive, multi-word)
            const normWords = removeDiacritics(filter.toLowerCase()).split(/\s+/).filter(Boolean);
            const haystack = removeDiacritics(String(row[idx]).toLowerCase());
            return normWords.every(word => haystack.includes(word));
          }
        } else {
          // Default: diacritics-insensitive, multi-word substring match
          if (row[idx] === null) return false;
          const normWords = removeDiacritics(filter.toLowerCase()).split(/\s+/).filter(Boolean);
          const haystack = removeDiacritics(String(row[idx]).toLowerCase());
          return normWords.every(word => haystack.includes(word));
        }
      })
    );
  }, [rows, columns, columnFilters]);

  return (
    <div className="flex flex-col h-full p-6">
      <div className="mb-4 flex flex-col relative z-30">
        <div className="flex items-center w-full">
          <input
            ref={inputRef}
            className="border border-gray-300 rounded px-3 py-2 w-112 mr-2 shadow-sm focus:ring-2 focus:ring-blue-200 focus:border-blue-400 transition"
            placeholder="Search for a table..."
            aria-label="Search for a table"
            value={search}
            onChange={e => {
              const value = e.target.value;
              setSearch(value);
              setShowSuggestions(true);
              if (value.trim() === '') {
                setSuggestions(allTables);
                setSelectedTable(null);
              }
            }}
            onFocus={() => {
              setShowSuggestions(true);
              if (!search.trim()) {
                setSuggestions(allTables);
              }
            }}
            onBlur={handleBlur}
          />
          {search && (
            <button
              className="text-gray-400 hover:text-gray-700 text-lg font-bold px-2 py-1 focus:outline-none ml-2"
              title="Clear filter"
              aria-label="Clear filter"
              tabIndex={0}
              onClick={() => {
                setSearch('');
                setSelectedTable(null);
                setSuggestions(allTables);
              }}
              style={{ lineHeight: 1 }}
            >
              ×
            </button>
          )}
        </div>
        <div className="flex items-center mt-2 space-x-4">
          <span
            className="text-gray-400 text-[10px] font-normal block"
            style={{ lineHeight: 1, fontFamily: 'var(--font-inter, Inter, system-ui, sans-serif)' }}
            title="Starting with * searches only for codes."
          >
            Starting with * searches only for codes.
          </span>
          <span className="text-gray-500 text-xs">{allTables.length} tables</span>
        </div>
        {showSuggestions && suggestions.length > 0 && (
          <ul className="absolute left-0 top-full z-40 bg-white border border-gray-200 rounded w-112 mt-1 max-h-60 overflow-auto shadow-lg">
            {suggestions.map(table => (
              <li
                key={table.selection_code}
                className="px-4 py-2 hover:bg-gray-100 cursor-pointer text-sm"
                onMouseDown={() => handleSuggestionClick(table.selection_code)}
              >
                <span className="font-mono text-xs text-blue-900">{table.selection_code}</span>
                {table.short_description && (
                  <span className="ml-2 text-gray-700">- {table.short_description}</span>
                )}
              </li>
            ))}
          </ul>
        )}
        {loading && <div className="absolute right-3 top-2 text-xs text-gray-400">Loading...</div>}
      </div>
      {selectedTable && (
        <div className="mb-2 flex items-center">
          <span className="font-mono text-xs bg-gray-100 border border-gray-200 rounded px-2 py-1 mr-2">Table code: {selectedTable}</span>
        </div>
      )}
      {selectedTable && columns.length > 0 && (
        <div className="flex-1 overflow-auto">
          {tableLoading ? (
            <div className="text-center py-8">Loading table...</div>
          ) : columns.length === 0 ? (
            <div className="text-center py-8">No data found for this table.</div>
          ) : (
            <div className="overflow-x-auto rounded shadow border border-gray-200 bg-white">
              <table className="min-w-full text-xs">
                <thead className="bg-blue-100 sticky top-0 z-10">
                  <tr>
                    {columns.map(col => (
                      <th key={col} className="px-4 py-2 border-b text-left font-semibold text-gray-700 whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                  </tr>
                  <tr>
                    {columns.map(col => (
                      <th key={col + '-filter'} className="px-4 py-1 border-b bg-blue-50">
                        <input
                          className="border border-gray-300 rounded px-2 py-1 text-xs w-full"
                          placeholder={`Filter...`}
                          value={columnFilters[col] || ''}
                          onChange={e => handleColumnFilterChange(col, e.target.value)}
                          title={col === 'value' ? 'You can filter using >, <, >=, <=, =, !=, etc. (e.g. "> 10000")' : undefined}
                        />
                        {col === 'value' && (
                          <span
                            className="text-gray-400 text-[10px] font-normal block mt-1"
                            style={{ lineHeight: 1, fontFamily: 'var(--font-inter, Inter, system-ui, sans-serif)' }}
                          >
                            <span title='You can filter using &gt;, &lt;, &gt;=, &lt;=, =, !=, etc. (e.g. "> 10000")'>
                              e.g. &gt; 10000, &lt;= 500
                            </span>
                          </span>
                        )}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredRows.map((row, i) => (
                    <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-blue-50"}>
                      {row.map((cell, j) => (
                        <td key={j} className="px-4 py-2 border-b whitespace-pre-line text-gray-800">
                          {cell !== null ? String(cell) : ''}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DataTableView; 