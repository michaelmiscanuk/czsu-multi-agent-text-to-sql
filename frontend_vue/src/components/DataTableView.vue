<template>
  <div class="table-container">
    <div class="filter-section flex flex-col relative z-30">
      <div class="flex items-center justify-between w-full">
        <div class="flex items-center">
          <input
            ref="inputRef"
            class="border border-gray-300 rounded px-3 py-2 w-112 mr-2 shadow-sm focus:ring-2 focus:ring-blue-200 focus:border-blue-400 transition"
            placeholder="Search for a table..."
            aria-label="Search for a table"
            :value="search"
            @input="handleSearchInput"
            @focus="handleInputFocus"
            @blur="handleBlur"
          />
          <button
            v-if="search"
            class="text-gray-400 hover:text-gray-700 text-lg font-bold px-2 py-1 focus:outline-none"
            title="Clear filter"
            aria-label="Clear filter"
            tabindex="0"
            @click="handleClearSearch"
            style="line-height: 1"
          >
            ×
          </button>
        </div>
        <button
          v-if="selectedTable"
          class="dataset-code-badge"
          @click="handleTableCodeClick"
          :title="`Go to catalog and filter by ${selectedTable}`"
        >
          {{ selectedTable }}
        </button>
      </div>
      
      <div class="flex items-center mt-2 space-x-4">
        <span
          class="text-gray-400 text-[10px] font-normal block"
          style="line-height: 1; font-family: var(--table-font-family); margin-left: 1rem"
          title="Starting with * searches only for codes."
        >
          Starting with * searches only for codes.
        </span>
        <span class="text-gray-500 text-xs">{{ allTables.length }} tables</span>
      </div>
      
      <ul 
        v-if="showSuggestions && suggestions.length > 0"
        class="absolute left-0 top-full z-40 bg-white border border-gray-200 rounded w-112 mt-1 max-h-60 overflow-auto shadow-lg"
      >
        <li
          v-for="table in suggestions"
          :key="table.selection_code"
          class="px-4 py-2 hover:bg-gray-100 cursor-pointer text-sm"
          @mousedown="() => handleSuggestionClick(table.selection_code)"
        >
          <span class="font-mono text-xs text-blue-900">{{ table.selection_code }}</span>
          <span v-if="table.short_description" class="ml-2 text-gray-700">
            - {{ table.short_description }}
          </span>
        </li>
      </ul>
      
      <div v-if="loading" class="absolute right-3 top-2 text-xs text-gray-400">
        Loading...
      </div>
    </div>
    
    <div class="table-section">
      <div class="flex-1 overflow-auto">
        <div v-if="tableLoading" class="text-center py-8">
          Loading table...
        </div>
        <div v-else-if="columns.length === 0" class="text-center py-8">
          No data found for this table.
        </div>
        <div v-else class="overflow-x-auto rounded shadow border border-gray-200 bg-white">
          <table class="min-w-full table-content-font">
            <thead class="bg-blue-100 sticky top-0 z-10">
              <tr>
                <th
                  v-for="col in columns"
                  :key="col"
                  class="px-4 py-2 border-b text-left table-header-font text-gray-700 whitespace-nowrap cursor-pointer select-none group"
                  @click="() => handleSort(col)"
                  tabindex="0"
                  :aria-sort="getAriaSortValue(col)"
                  :title="`Sort by ${col}`"
                  style="user-select: none"
                >
                  <span class="flex items-center">
                    {{ col }}
                    <span class="ml-1">
                      <svg 
                        v-if="getSortIcon(col) === 'asc'"
                        width="12" 
                        height="12" 
                        viewBox="0 0 12 12" 
                        class="inline" 
                        aria-label="Sorted ascending"
                      >
                        <polygon points="6,3 11,9 1,9" fill="black" />
                      </svg>
                      <svg 
                        v-else-if="getSortIcon(col) === 'desc'"
                        width="12" 
                        height="12" 
                        viewBox="0 0 12 12" 
                        class="inline" 
                        aria-label="Sorted descending"
                      >
                        <polygon points="1,3 11,3 6,9" fill="black" />
                      </svg>
                      <svg 
                        v-else
                        width="12" 
                        height="12" 
                        viewBox="0 0 12 12" 
                        class="inline" 
                        aria-label="Not sorted"
                      >
                        <polygon points="2,4 10,4 6,8" fill="#bbb" />
                      </svg>
                    </span>
                  </span>
                </th>
              </tr>
              <tr>
                <th 
                  v-for="col in columns" 
                  :key="col + '-filter'" 
                  class="px-4 py-1 border-b bg-blue-50"
                >
                  <input
                    class="border border-gray-300 rounded px-2 py-1 table-content-font w-full"
                    placeholder="Filter..."
                    :value="columnFilters[col] || ''"
                    @input="(e) => handleColumnFilterChange(col, (e.target as HTMLInputElement).value)"
                    v-bind="col === 'value' ? { title: 'You can filter using >, <, >=, <=, =, !=, etc. (e.g. > 10000)' } : {}"
                  />
                  <span
                    v-if="col === 'value'"
                    class="text-gray-400 text-[10px] font-normal block mt-1"
                    style="line-height: 1; font-family: var(--table-font-family)"
                  >
                    <span title='You can filter using >, <, >=, <=, =, !=, etc. (e.g. > 10000)'>
                      e.g. > 10000, <= 500
                    </span>
                  </span>
                </th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(row, i) in sortedRows"
                :key="i"
                :class="i % 2 === 0 ? 'bg-white' : 'bg-blue-50'"
              >
                <td
                  v-for="(cell, j) in row"
                  :key="j"
                  class="px-4 py-2 border-b table-content-font table-cell-readable text-gray-800"
                >
                  {{ cell !== null ? String(cell) : '' }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { storeToRefs } from 'pinia'
import { useAuthStore } from '@/stores/auth'
import { authApiFetch } from '@/lib/api'

// Utility function to remove diacritics (from utils.ts)
function removeDiacritics(str: string): string {
  return str.normalize('NFD').replace(/\p{Diacritic}/gu, '')
}

// Types
interface TableSuggestion {
  selection_code: string
  short_description: string
}

interface DataTableResponse {
  columns: string[]
  rows: any[][]
}

interface TablesResponse {
  tables: TableSuggestion[]
}

interface SortConfig {
  column: string | null
  direction: 'asc' | 'desc' | null
}

// Props interface matching the original React component exactly
interface Props {
  search: string
  selectedTable: string | null
  columns: string[]
  rows: any[][]
  selectedColumn: string | null
  columnFilters: { [col: string]: string }
  pendingTableSearch?: string | null
}

const props = defineProps<Props>()

// Emits interface matching the original React component exactly
const emit = defineEmits<{
  'update:search': [value: string]
  'update:selectedTable': [value: string | null]
  'update:columns': [value: string[]]
  'update:rows': [value: any[][]]
  'update:selectedColumn': [value: string | null]
  'update:columnFilters': [value: { [col: string]: string }]
  'update:pendingTableSearch': [value: string | null]
}>()

// Composables
const router = useRouter()
const authStore = useAuthStore()

// Refs
const inputRef = ref<HTMLInputElement>()
const suggestions = ref<TableSuggestion[]>([])
const showSuggestions = ref(false)
const loading = ref(false)
const tableLoading = ref(false)
const allTables = ref<TableSuggestion[]>([])
const sortConfig = ref<SortConfig>({ column: null, direction: null })

// Constants matching the original
const SELECTED_TABLE_KEY = 'czsu-data-selectedTable'
const COLUMN_FILTERS_KEY = 'czsu-data-columnFilters'
const SELECTED_COLUMN_KEY = 'czsu-data-selectedColumn'
const SEARCH_KEY = 'czsu-data-search'

// Computed values
const search = computed({
  get: () => props.search,
  set: (value: string) => emit('update:search', value)
})

const selectedTable = computed({
  get: () => props.selectedTable,
  set: (value: string | null) => emit('update:selectedTable', value)
})

const columns = computed({
  get: () => props.columns,
  set: (value: string[]) => emit('update:columns', value)
})

const rows = computed({
  get: () => props.rows,
  set: (value: any[][]) => emit('update:rows', value)
})

const selectedColumn = computed({
  get: () => props.selectedColumn,
  set: (value: string | null) => emit('update:selectedColumn', value)
})

const columnFilters = computed({
  get: () => props.columnFilters,
  set: (value: { [col: string]: string }) => emit('update:columnFilters', value)
})

const pendingTableSearch = computed({
  get: () => props.pendingTableSearch,
  set: (value: string | null) => emit('update:pendingTableSearch', value)
})

// Enhanced filter logic for numeric 'value' column
const filteredRows = computed(() => {
  if (!columns.value.length || Object.values(columnFilters.value).every(v => !v)) {
    return rows.value
  }

  return rows.value.filter(row =>
    columns.value.every((col, idx) => {
      const filter = columnFilters.value[col]
      if (!filter) return true

      if (col === 'value') {
        // Numeric filter: support >, >=, <, <=, !=, =, ==, or just a number (equals)
        const match = filter.trim().match(/^(>=|<=|!=|>|<|=|==)?\s*(-?\d+(?:\.\d+)?)/);
        if (match && match[2]) {
          const op = match[1] || '=='
          const num = parseFloat(match[2])
          const cellNum = parseFloat(row[idx])
          if (isNaN(cellNum)) return false
          
          switch (op) {
            case '>': return cellNum > num
            case '>=': return cellNum >= num
            case '<': return cellNum < num
            case '<=': return cellNum <= num
            case '!=': return cellNum !== num
            case '=':
            case '==': return cellNum === num
            default: return cellNum === num
          }
        } else {
          // fallback: substring match (diacritics-insensitive, multi-word)
          const normWords = removeDiacritics(filter.toLowerCase()).split(/\s+/).filter(Boolean)
          const haystack = removeDiacritics(String(row[idx]).toLowerCase())
          return normWords.every(word => haystack.includes(word))
        }
      } else {
        // Default: diacritics-insensitive, multi-word substring match
        if (row[idx] === null) return false
        const normWords = removeDiacritics(filter.toLowerCase()).split(/\s+/).filter(Boolean)
        const haystack = removeDiacritics(String(row[idx]).toLowerCase())
        return normWords.every(word => haystack.includes(word))
      }
    })
  )
})

// Sorting logic
const sortedRows = computed(() => {
  if (!sortConfig.value.column || !sortConfig.value.direction) {
    return filteredRows.value
  }

  const colIdx = columns.value.indexOf(sortConfig.value.column)
  if (colIdx === -1) return filteredRows.value

  const sorted = [...filteredRows.value].sort((a, b) => {
    const aVal = a[colIdx]
    const bVal = b[colIdx]
    
    // Try numeric sort if both are numbers
    if (!isNaN(parseFloat(aVal)) && !isNaN(parseFloat(bVal))) {
      return sortConfig.value.direction === 'asc'
        ? parseFloat(aVal) - parseFloat(bVal)
        : parseFloat(bVal) - parseFloat(aVal)
    }
    
    // Fallback to string sort (diacritics-insensitive)
    const aStr = removeDiacritics(String(aVal)).toLowerCase()
    const bStr = removeDiacritics(String(bVal)).toLowerCase()
    if (aStr < bStr) return sortConfig.value.direction === 'asc' ? -1 : 1
    if (aStr > bStr) return sortConfig.value.direction === 'asc' ? 1 : -1
    return 0
  })
  
  return sorted
})

// Methods
const handleSearchInput = (e: Event) => {
  const target = e.target as HTMLInputElement
  const value = target.value
  search.value = value
  showSuggestions.value = true
  
  if (value.trim() === '') {
    suggestions.value = allTables.value
    selectedTable.value = null
  }
}

const handleInputFocus = () => {
  showSuggestions.value = true
  if (!search.value.trim()) {
    suggestions.value = allTables.value
  }
}

const handleBlur = () => {
  setTimeout(() => {
    showSuggestions.value = false
  }, 100) // Delay to allow click
}

const handleClearSearch = () => {
  search.value = ''
  selectedTable.value = null
  suggestions.value = allTables.value
}

const handleSuggestionClick = (table: string) => {
  selectedTable.value = table
  search.value = table
  showSuggestions.value = false
  if (pendingTableSearch.value !== null) {
    pendingTableSearch.value = null
  }
}

const handleColumnFilterChange = (col: string, value: string) => {
  columnFilters.value = { ...columnFilters.value, [col]: value }
}

const handleSort = (col: string) => {
  if (sortConfig.value.column !== col) {
    sortConfig.value = { column: col, direction: 'asc' }
  } else if (sortConfig.value.direction === 'asc') {
    sortConfig.value = { column: col, direction: 'desc' }
  } else if (sortConfig.value.direction === 'desc') {
    sortConfig.value = { column: null, direction: null }
  } else {
    sortConfig.value = { column: col, direction: 'asc' }
  }
}

const handleTableCodeClick = () => {
  if (selectedTable.value) {
    // Set the catalog filter in localStorage before navigation
    localStorage.setItem('czsu-catalog-filter', selectedTable.value)
    localStorage.setItem('czsu-catalog-page', '1') // Reset to first page
    
    // Navigate to catalog
    router.push('/catalog')
  }
}

const getSortIcon = (col: string): 'asc' | 'desc' | 'none' => {
  if (sortConfig.value.column === col) {
    return sortConfig.value.direction || 'none'
  }
  return 'none'
}

const getAriaSortValue = (col: string): 'none' | 'ascending' | 'descending' | 'other' => {
  if (sortConfig.value.column === col) {
    if (sortConfig.value.direction === 'asc') return 'ascending'
    if (sortConfig.value.direction === 'desc') return 'descending'
  }
  return 'none'
}

// Fetch all tables on mount (for combo box)
const fetchAllTables = async () => {
  try {
    const token = await authStore.getValidToken()
    if (!token) return

    const response = await authApiFetch<TablesResponse>('/data-tables', token)
    allTables.value = response.tables || []
  } catch (error) {
    console.error('[DataTableView] Error fetching tables:', error)
    allTables.value = []
  }
}

// Fetch table data when a table is selected
const fetchTableData = async () => {
  if (!selectedTable.value) {
    columns.value = []
    rows.value = []
    selectedColumn.value = null
    columnFilters.value = {}
    return
  }

  tableLoading.value = true
  
  try {
    const token = await authStore.getValidToken()
    if (!token) {
      console.error('[DataTableView] No authentication token available')
      return
    }

    const url = `/data-table?table=${encodeURIComponent(selectedTable.value)}`
    console.log('[DataTableView] Fetching table:', selectedTable.value, url)
    
    const response = await authApiFetch<DataTableResponse>(url, token)
    console.log('[DataTableView] Received data:', response)
    
    columns.value = response.columns || []
    rows.value = response.rows || []
    const firstColumn = response.columns && response.columns.length > 0 ? response.columns[0] : null
    if (firstColumn !== undefined) {
      selectedColumn.value = firstColumn
    }
    columnFilters.value = {}
  } catch (error) {
    console.error('[DataTableView] Error fetching table:', error)
  } finally {
    tableLoading.value = false
  }
}

// Watchers

// Prefill search box if pendingTableSearch changes
watch(() => props.pendingTableSearch, (newValue) => {
  if (newValue) {
    search.value = newValue
    showSuggestions.value = true
    nextTick(() => {
      inputRef.value?.focus()
    })
  }
})

// On mount, if search is non-empty and selectedTable is null, set selectedTable to search
watch([() => props.search, () => props.selectedTable], ([newSearch, newSelectedTable]) => {
  if (newSearch && !newSelectedTable) {
    selectedTable.value = newSearch
  }
})

// Fetch table suggestions as user types or when search is empty
watch([() => props.search, allTables], ([newSearch, newAllTables]) => {
  if (newSearch.trim() === '') {
    suggestions.value = newAllTables
    return
  }
  
  loading.value = true
  
  let filteredTables: TableSuggestion[]
  let normWords: string[]
  
  if (newSearch.startsWith('*')) {
    // Only search in selection_code
    normWords = removeDiacritics(newSearch.slice(1).toLowerCase()).split(/\s+/).filter(Boolean)
    filteredTables = newAllTables.filter(table => {
      const haystack = removeDiacritics(table.selection_code.toLowerCase())
      return normWords.every((word: string) => haystack.includes(word))
    })
  } else {
    // Search in both selection_code and short_description
    normWords = removeDiacritics(newSearch.toLowerCase()).split(/\s+/).filter(Boolean)
    filteredTables = newAllTables.filter(table => {
      const haystack = removeDiacritics((table.selection_code + ' ' + (table.short_description || '')).toLowerCase())
      return normWords.every((word: string) => haystack.includes(word))
    })
  }
  
  const sortedTables = filteredTables.slice().sort((a, b) => 
    a.selection_code.localeCompare(b.selection_code, 'cs', { sensitivity: 'base' })
  )
  
  suggestions.value = sortedTables
  loading.value = false
})

// Auto-select and load the table if pendingTableSearch matches a suggestion exactly
watch([() => props.pendingTableSearch, suggestions], ([newPendingTableSearch, newSuggestions]) => {
  if (newPendingTableSearch && newSuggestions.length > 0) {
    const match = newSuggestions.find(s => s.selection_code === newPendingTableSearch)
    if (match) {
      selectedTable.value = match.selection_code
      search.value = match.selection_code
      showSuggestions.value = false
      if (pendingTableSearch.value !== null) {
        pendingTableSearch.value = null
      }
    }
  }
})

// Fetch table data when selectedTable changes
watch(() => props.selectedTable, fetchTableData)

// Initialize on mount
onMounted(() => {
  fetchAllTables()
})
</script>

<style scoped>
.w-112 {
  width: 28rem; /* 448px */
}
</style> 