"use client";
import CatalogTable from '../../components/DatasetsTable';
import { useRouter } from 'next/navigation';

export default function CatalogPage() {
  const router = useRouter();
  const handleCatalogRowClick = (selection_code: string) => {
    router.push(`/data?table=${encodeURIComponent(selection_code)}`);
  };
  return (
    <div className="w-full max-w-6xl mx-auto">
      <CatalogTable onRowClick={handleCatalogRowClick} />
    </div>
  );
} 