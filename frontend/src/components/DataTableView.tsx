import React, { useEffect, useRef } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '';

type DataTableViewProps = {
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

const DataTableView: React.FC<DataTableViewProps> = ({
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
  const [search, setSearch] = React.useState('');
  const [suggestions, setSuggestions] = React.useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const [tableLoading, setTableLoading] = React.useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Prefill search box if pendingTableSearch changes
  React.useEffect(() => {
    if (pendingTableSearch) {
      setSearch(pendingTableSearch);
      setShowSuggestions(true);
      inputRef.current?.focus();
    }
  }, [pendingTableSearch]);

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
        setSuggestions(data.tables || []);
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
            // fallback: substring match
            return String(row[idx]).toLowerCase().includes(filter.toLowerCase());
          }
        } else {
          // Default: substring match
          return row[idx] !== null && String(row[idx]).toLowerCase().includes(filter.toLowerCase());
        }
      })
    );
  }, [rows, columns, columnFilters]);

  return (
    <div className="flex flex-col h-full p-6">
      <div className="mb-4 relative">
        <input
          ref={inputRef}
          className="border border-gray-300 rounded px-3 py-2 w-96"
          placeholder="Search for a table..."
          value={search}
          onChange={e => {
            setSearch(e.target.value);
            setShowSuggestions(true);
            setSelectedTable(null);
          }}
          onFocus={() => setShowSuggestions(true)}
          onBlur={handleBlur}
        />
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