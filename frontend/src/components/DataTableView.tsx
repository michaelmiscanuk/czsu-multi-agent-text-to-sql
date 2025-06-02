import React, { useEffect, useRef } from 'react';
import { removeDiacritics } from './utils';

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
  const [suggestions, setSuggestions] = React.useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const [tableLoading, setTableLoading] = React.useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Restore state from localStorage on mount
  React.useEffect(() => {
    const savedTable = localStorage.getItem(SELECTED_TABLE_KEY);
    const savedFilters = localStorage.getItem(COLUMN_FILTERS_KEY);
    const savedCol = localStorage.getItem(SELECTED_COLUMN_KEY);
    if (savedTable) setSelectedTable(savedTable);
    if (savedFilters) {
      try {
        setColumnFilters(JSON.parse(savedFilters));
      } catch {}
    }
    if (savedCol) setSelectedColumn(savedCol);
  }, [setSelectedTable, setColumnFilters, setSelectedColumn]);

  // Persist state to localStorage
  React.useEffect(() => {
    if (selectedTable) {
      localStorage.setItem(SELECTED_TABLE_KEY, selectedTable);
    } else {
      localStorage.removeItem(SELECTED_TABLE_KEY);
    }
  }, [selectedTable]);
  React.useEffect(() => {
    localStorage.setItem(COLUMN_FILTERS_KEY, JSON.stringify(columnFilters));
  }, [columnFilters]);
  React.useEffect(() => {
    if (selectedColumn) {
      localStorage.setItem(SELECTED_COLUMN_KEY, selectedColumn);
    } else {
      localStorage.removeItem(SELECTED_COLUMN_KEY);
    }
  }, [selectedColumn]);

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

  // Fetch table suggestions as user types
  useEffect(() => {
    if (search.trim() === '') {
      setSuggestions([]);
      return;
    }
    setLoading(true);
    fetch(`${API_BASE}/data-tables?q=${encodeURIComponent(search)}`)
      .then(res => res.json())
      .then(data => {
        const normSearch = removeDiacritics(search.toLowerCase());
        const filteredTables = (data.tables || []).filter((table: string) =>
          removeDiacritics(table.toLowerCase()).startsWith(normSearch)
        );
        const sortedTables = filteredTables.slice().sort((a: string, b: string) => a.localeCompare(b, 'cs', { sensitivity: 'base' }));
        setSuggestions(sortedTables);
        setLoading(false);
      });
  }, [search]);

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
    fetch(`${API_BASE}/data-table?table=${encodeURIComponent(selectedTable)}`)
      .then(res => res.json())
      .then(data => {
        setColumns(data.columns || []);
        setRows(data.rows || []);
        setSelectedColumn(data.columns && data.columns.length > 0 ? data.columns[0] : null);
        setColumnFilters({});
        setTableLoading(false);
      });
  }, [selectedTable, setColumns, setRows, setSelectedColumn, setColumnFilters]);

  // Auto-select and load the table if pendingTableSearch matches a suggestion exactly
  React.useEffect(() => {
    if (pendingTableSearch && suggestions.length > 0) {
      const match = suggestions.find(s => s === pendingTableSearch);
      if (match) {
        setSelectedTable(match);
        setSearch(match);
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
      <div className="mb-4 relative flex items-center">
        <input
          ref={inputRef}
          className="border border-gray-300 rounded px-3 py-2 w-96"
          placeholder="Search for a table..."
          value={search}
          onChange={e => {
            const value = e.target.value;
            setSearch(value);
            setShowSuggestions(true);
            // Only clear selectedTable if the search is cleared (empty string)
            if (value.trim() === '') {
              setSelectedTable(null);
            }
          }}
          onFocus={() => setShowSuggestions(true)}
          onBlur={handleBlur}
        />
        {(selectedTable || Object.values(columnFilters).some(Boolean) || selectedColumn) && (
          <button
            className="ml-2 text-gray-400 hover:text-gray-700 text-lg font-bold px-2 py-1 focus:outline-none"
            title="Reset all"
            onClick={() => {
              setSearch('');
              setSelectedTable(null);
              setColumns([]);
              setRows([]);
              setSelectedColumn(null);
              setColumnFilters({});
              if (setPendingTableSearch) setPendingTableSearch(null);
              localStorage.removeItem(SELECTED_TABLE_KEY);
              localStorage.removeItem(COLUMN_FILTERS_KEY);
              localStorage.removeItem(SELECTED_COLUMN_KEY);
            }}
            style={{ lineHeight: 1 }}
          >
            ×
          </button>
        )}
        {showSuggestions && suggestions.length > 0 && (
          <ul className="absolute z-10 bg-white border border-gray-200 rounded w-96 mt-1 max-h-60 overflow-auto shadow-lg">
            {suggestions.map(table => (
              <li
                key={table}
                className="px-4 py-2 hover:bg-gray-100 cursor-pointer text-sm"
                onMouseDown={() => handleSuggestionClick(table)}
              >
                {table}
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
        <div className="mb-4 flex items-center space-x-2">
          <label className="text-sm text-gray-700">Column:</label>
          <select
            className="border border-gray-300 rounded px-2 py-1 text-sm"
            value={selectedColumn || ''}
            onChange={e => handleColumnSelect(e.target.value)}
          >
            {columns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
          <input
            className="border border-gray-300 rounded px-2 py-1 text-sm w-64"
            placeholder={selectedColumn === 'value' ? `Filter in column (e.g. > 10000, <= 500)` : `Filter in column...`}
            value={selectedColumn ? columnFilters[selectedColumn] || '' : ''}
            onChange={e => selectedColumn && handleColumnFilterChange(selectedColumn, e.target.value)}
          />
          <button
            className="ml-2 text-gray-400 hover:text-gray-700 text-lg font-bold px-2 py-1 focus:outline-none"
            title="Clear all filters"
            onClick={handleClearFilters}
            style={{ lineHeight: 1 }}
          >
            ×
          </button>
        </div>
      )}
      {selectedTable && (
        <div className="flex-1 overflow-auto">
          {tableLoading ? (
            <div className="text-center py-8">Loading table...</div>
          ) : columns.length === 0 ? (
            <div className="text-center py-8">No data found for this table.</div>
          ) : (
            <table className="min-w-full border border-gray-200 rounded text-xs">
              <thead>
                <tr className="bg-gray-100">
                  {columns.map(col => (
                    <th key={col} className="px-2 py-1 border-b text-left whitespace-nowrap">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredRows.map((row, i) => (
                  <tr key={i} className="hover:bg-gray-50">
                    {row.map((cell, j) => (
                      <td key={j} className="px-2 py-1 border-b whitespace-pre-line">{cell !== null ? String(cell) : ''}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
};

export default DataTableView; 