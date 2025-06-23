<template>
  <div class="unified-white-block-system">
    <div class="table-container">
      <DatasetsTable :on-row-click="handleCatalogRowClick" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'
import DatasetsTable from '@/components/DatasetsTable.vue'

const router = useRouter()

const handleCatalogRowClick = (selection_code: string) => {
  router.push(`/data?table=${encodeURIComponent(selection_code)}`)
}
</script> 