<template>
  <div class="table-container">
    <div class="filter-section flex items-center">
      <input
        v-model="filter"
        class="border border-gray-300 rounded px-3 py-2 w-80 mr-2"
        placeholder="Filter by keyword..."
        aria-label="Filter catalog by keyword"
        @input="handleFilterChange"
      />
      <button
        v-if="filter"
        class="text-gray-400 hover:text-gray-700 text-lg font-bold px-2 py-1 focus:outline-none"
        title="Clear filter"
        aria-label="Clear filter"
        @click="handleReset"
        style="line-height: 1"
      >
        ×
      </button>
      <span class="text-gray-500 text-sm ml-4">{{ total }} records</span>
    </div>
    
    <div class="table-section">
      <div class="overflow-x-auto rounded shadow border border-gray-200 bg-white">
        <table class="min-w-full table-content-font">
          <thead class="bg-blue-100 sticky top-0 z-10">
            <tr>
              <th class="px-4 py-2 border-b text-left table-header-font text-gray-700 whitespace-nowrap">
                Selection Code
              </th>
              <th class="px-4 py-2 border-b text-left table-header-font text-gray-700 whitespace-nowrap">
                Extended Description
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-if="loading">
              <td colspan="2" class="text-center py-8 table-content-font">Loading...</td>
            </tr>
            <tr v-else-if="data.length === 0">
              <td colspan="2" class="text-center py-8 table-content-font">No records found.</td>
            </tr>
            <template v-else>
              <tr
                v-for="(row, i) in data"
                :key="row.selection_code"
                :class="i % 2 === 0 ? 'bg-white' : 'bg-blue-50'"
              >
                <td class="px-4 py-2 border-b font-mono table-content-font text-blue-900">
                  <button
                    v-if="onRowClick"
                    class="dataset-code-badge"
                    @click="() => onRowClick?.(row.selection_code)"
                  >
                    {{ row.selection_code }}
                  </button>
                  <span v-else>{{ row.selection_code }}</span>
                </td>
                <td class="px-4 py-2 border-b table-description-font table-cell-readable text-gray-800">
                  {{ row.extended_description }}
                </td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </div>
    
    <div class="pagination-section flex items-center justify-between">
      <button
        class="px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 text-sm disabled:opacity-50 disabled:cursor-not-allowed"
        :disabled="page === 1"
        @click="setPage(Math.max(1, page - 1))"
      >
        Previous
      </button>
      <span class="text-gray-600 text-sm">Page {{ page }} of {{ totalPages || 1 }}</span>
      <button
        class="px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 text-sm disabled:opacity-50 disabled:cursor-not-allowed"
        :disabled="page === totalPages || totalPages === 0"
        @click="setPage(Math.min(totalPages, page + 1))"
      >
        Next
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { API_CONFIG } from '@/lib/api'

// To support both local dev and production, set VITE_API_BASE_URL in .env.local to your backend URL (e.g., http://localhost:8000) for local dev.
// In production, leave it unset to use relative paths.
const API_BASE = API_CONFIG.baseUrl || 'https://czsu-multi-agent-text-to-sql.onrender.com'

interface CatalogRow {
  selection_code: string
  extended_description: string
}

interface CatalogResponse {
  results: CatalogRow[]
  total: number
  page: number
  page_size: number
}

interface Props {
  onRowClick?: (selection_code: string) => void
}

const props = defineProps<Props>()

const authStore = useAuthStore()

// Reactive state
const data = ref<CatalogRow[]>([])
const total = ref(0)
const page = ref(1)
const filter = ref('')
const loading = ref(false)
const isRestored = ref(false)

// Constants
const CATALOG_PAGE_KEY = 'czsu-catalog-page'
const CATALOG_FILTER_KEY = 'czsu-catalog-filter'

// Utility function to remove diacritics (from utils.ts)
function removeDiacritics(str: string): string {
  return str.normalize('NFD').replace(/\p{Diacritic}/gu, '')
}

// Computed
const totalPages = computed(() => Math.ceil(total.value / 10))

// Methods
const setPage = (newPage: number) => {
  page.value = newPage
}

const handleFilterChange = () => {
  page.value = 1
}

const handleReset = () => {
  filter.value = ''
  page.value = 1
}

const fetchData = async () => {
  if (!isRestored.value) return

  console.log('[DatasetsTable] Session:', JSON.stringify(authStore.session, null, 2))
  loading.value = true
  
  const handleError = (err: any) => {
    console.error('Error fetching catalog data:', err)
    data.value = []
    total.value = 0
    loading.value = false
  }

  // Helper to build fetch options
  const getFetchOptions = () =>
    authStore.session?.idToken
      ? { headers: { Authorization: `Bearer ${authStore.session.idToken}` } }
      : undefined
  
  const fetchOptions = getFetchOptions()
  console.log('[DatasetsTable] Fetch options:', JSON.stringify(fetchOptions, null, 2))
  
  try {
    if (filter.value) {
      // Fetch all data and filter client-side
      const response = await fetch(`${API_BASE}/catalog?page=1&page_size=10000`, fetchOptions)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      
      const res: CatalogResponse = await response.json()
      
      const normWords = removeDiacritics(filter.value.toLowerCase()).split(/\s+/).filter(Boolean)
      const filteredResults = res.results.filter(row => {
        const haystack = removeDiacritics((row.selection_code + ' ' + row.extended_description).toLowerCase())
        return normWords.every(word => haystack.includes(word))
      })
      
      data.value = filteredResults.slice((page.value - 1) * 10, page.value * 10)
      total.value = filteredResults.length
    } else {
      // Use backend pagination
      const params = new URLSearchParams({ page: page.value.toString() })
      const response = await fetch(`${API_BASE}/catalog?${params.toString()}`, fetchOptions)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      
      const res: CatalogResponse = await response.json()
      
      data.value = res.results
      total.value = res.total
    }
  } catch (error) {
    handleError(error)
  } finally {
    loading.value = false
  }
}

// Restore state from localStorage
onMounted(() => {
  const savedPage = localStorage.getItem(CATALOG_PAGE_KEY)
  const savedFilter = localStorage.getItem(CATALOG_FILTER_KEY)
  
  if (savedPage) {
    const pageNum = Number(savedPage)
    if (!isNaN(pageNum)) {
      page.value = pageNum
    }
  }
  
  if (savedFilter) {
    filter.value = savedFilter
  }
  
  isRestored.value = true
})

// Persist state to localStorage
watch(page, (newPage) => {
  localStorage.setItem(CATALOG_PAGE_KEY, String(newPage))
})

watch(filter, (newFilter) => {
  localStorage.setItem(CATALOG_FILTER_KEY, newFilter)
})

// Fetch data when dependencies change
watch([page, filter, () => authStore.session?.idToken, isRestored], fetchData, { immediate: false })
</script> 