"use client";
import DataTableView from '../../components/DataTableView';
import { useSearchParams } from 'next/navigation';
import React from 'react';

export default function DataPage() {
  const searchParams = useSearchParams();
  const table = searchParams.get('table');
  const [selectedTable, setSelectedTable] = React.useState<string | null>(table);
  const [columns, setColumns] = React.useState<string[]>([]);
  const [rows, setRows] = React.useState<any[][]>([]);
  const [selectedColumn, setSelectedColumn] = React.useState<string | null>(null);
  const [columnFilters, setColumnFilters] = React.useState<{ [col: string]: string }>({});
  const [pendingTableSearch, setPendingTableSearch] = React.useState<string | null>(null);

  React.useEffect(() => {
    setSelectedTable(table);
  }, [table]);

  return (
    <div className="w-full max-w-5xl mx-auto">
      <DataTableView
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