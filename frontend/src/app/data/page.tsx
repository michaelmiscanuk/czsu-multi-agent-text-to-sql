"use client";
import DataTableView from '../../components/DataTableView';
import { useSearchParams } from 'next/navigation';
import React from 'react';

const SEARCH_KEY = 'czsu-data-search';
const SELECTED_TABLE_KEY = 'czsu-data-selectedTable';
const COLUMN_FILTERS_KEY = 'czsu-data-columnFilters';
const SELECTED_COLUMN_KEY = 'czsu-data-selectedColumn';

export default function DataPage() {
  const searchParams = useSearchParams();
  const table = searchParams.get('table');

  // State lifted up
  const [search, setSearch] = React.useState('');
  const [selectedTable, setSelectedTable] = React.useState<string | null>(null);
  const [columns, setColumns] = React.useState<string[]>([]);
  const [rows, setRows] = React.useState<any[][]>([]);
  const [selectedColumn, setSelectedColumn] = React.useState<string | null>(null);
  const [columnFilters, setColumnFilters] = React.useState<{ [col: string]: string }>({});
  const [pendingTableSearch, setPendingTableSearch] = React.useState<string | null>(null);

  // Only restore from localStorage on first mount
  const didRestoreRef = React.useRef(false);
  React.useEffect(() => {
    if (!didRestoreRef.current) {
      const savedSearch = localStorage.getItem(SEARCH_KEY);
      const savedTable = localStorage.getItem(SELECTED_TABLE_KEY);
      const savedCol = localStorage.getItem(SELECTED_COLUMN_KEY);
      const savedFilters = localStorage.getItem(COLUMN_FILTERS_KEY);
      console.log('[DataPage] Restoring from localStorage:', { savedSearch, savedTable, savedCol, savedFilters });
      setSearch(savedSearch || '');
      setSelectedTable(savedTable || table);
      setSelectedColumn(savedCol || null);
      setColumnFilters(savedFilters ? JSON.parse(savedFilters) : {});
      didRestoreRef.current = true;
      console.log('[DataPage] State after restore:', {
        search: savedSearch || '',
        selectedTable: savedTable || table,
        selectedColumn: savedCol || null,
        columnFilters: savedFilters ? JSON.parse(savedFilters) : {}
      });
    }
    // eslint-disable-next-line
  }, []);

  // Persist to localStorage on state change
  React.useEffect(() => {
    console.log('[DataPage] Persisting search to localStorage:', search);
    localStorage.setItem(SEARCH_KEY, search);
  }, [search]);
  React.useEffect(() => {
    console.log('[DataPage] Persisting selectedTable to localStorage:', selectedTable);
    if (selectedTable) {
      localStorage.setItem(SELECTED_TABLE_KEY, selectedTable);
    } else {
      localStorage.removeItem(SELECTED_TABLE_KEY);
    }
  }, [selectedTable]);
  React.useEffect(() => {
    console.log('[DataPage] Persisting selectedColumn to localStorage:', selectedColumn);
    if (selectedColumn) {
      localStorage.setItem(SELECTED_COLUMN_KEY, selectedColumn);
    } else {
      localStorage.removeItem(SELECTED_COLUMN_KEY);
    }
  }, [selectedColumn]);
  React.useEffect(() => {
    console.log('[DataPage] Persisting columnFilters to localStorage:', columnFilters);
    localStorage.setItem(COLUMN_FILTERS_KEY, JSON.stringify(columnFilters));
  }, [columnFilters]);

  return (
    <div className="w-full max-w-5xl mx-auto">
      <DataTableView
        search={search}
        setSearch={setSearch}
        selectedTable={selectedTable}
        setSelectedTable={setSelectedTable}
        columns={columns}
        setColumns={setColumns}
        rows={rows}
        setRows={setRows}
        selectedColumn={selectedColumn}
        setSelectedColumn={setSelectedColumn}
        columnFilters={columnFilters}
        setColumnFilters={setColumnFilters}
        pendingTableSearch={pendingTableSearch}
        setPendingTableSearch={setPendingTableSearch}
      />
    </div>
  );
} 