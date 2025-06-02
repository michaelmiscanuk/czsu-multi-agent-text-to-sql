"use client";
import DatasetsTable from '../../components/DatasetsTable';
import { useRouter } from 'next/navigation';

export default function DatasetsPage() {
  const router = useRouter();
  const handleDatasetRowClick = (selection_code: string) => {
    router.push(`/data?table=${encodeURIComponent(selection_code)}`);
  };
  return (
    <div className="w-full max-w-5xl mx-auto">
      <DatasetsTable onRowClick={handleDatasetRowClick} />
    </div>
  );
} 