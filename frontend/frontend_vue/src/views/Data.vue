<template>
  <div class="unified-white-block-system">
    <div class="table-container">
      <DataTableView
        :search="search"
        @update:search="setSearch"
        :selectedTable="selectedTable"
        @update:selectedTable="setSelectedTable"
        :columns="columns"
        @update:columns="setColumns"
        :rows="rows"
        @update:rows="setRows"
        :selectedColumn="selectedColumn"
        @update:selectedColumn="setSelectedColumn"
        :columnFilters="columnFilters"
        @update:columnFilters="setColumnFilters"
        :pendingTableSearch="pendingTableSearch"
        @update:pendingTableSearch="setPendingTableSearch"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import DataTableView from '@/components/DataTableView.vue'

// Constants matching the original
const SEARCH_KEY = 'czsu-data-search'
const SELECTED_TABLE_KEY = 'czsu-data-selectedTable'
const COLUMN_FILTERS_KEY = 'czsu-data-columnFilters'
const SELECTED_COLUMN_KEY = 'czsu-data-selectedColumn'

// Get route to access query parameters
const route = useRoute()

// State lifted up - exactly matching the React version
const search = ref('')
const selectedTable = ref<string | null>(null)
const columns = ref<string[]>([])
const rows = ref<any[][]>([])
const selectedColumn = ref<string | null>(null)
const columnFilters = ref<{ [col: string]: string }>({})
const pendingTableSearch = ref<string | null>(null)

// Track if we've restored from localStorage yet
const didRestore = ref(false)

// State setters to match React pattern
const setSearch = (value: string) => {
  search.value = value
}

const setSelectedTable = (value: string | null) => {
  selectedTable.value = value
}

const setColumns = (value: string[]) => {
  columns.value = value
}

const setRows = (value: any[][]) => {
  rows.value = value
}

const setSelectedColumn = (value: string | null) => {
  selectedColumn.value = value
}

const setColumnFilters = (value: { [col: string]: string }) => {
  columnFilters.value = value
}

const setPendingTableSearch = (value: string | null) => {
  pendingTableSearch.value = value
}

// Only restore from localStorage on first mount
onMounted(() => {
  if (!didRestore.value) {
    const savedSearch = localStorage.getItem(SEARCH_KEY)
    const savedTable = localStorage.getItem(SELECTED_TABLE_KEY)
    const savedCol = localStorage.getItem(SELECTED_COLUMN_KEY)
    const savedFilters = localStorage.getItem(COLUMN_FILTERS_KEY)
    
    console.log('[DataPage] Restoring from localStorage:', { 
      savedSearch, 
      savedTable, 
      savedCol, 
      savedFilters 
    })

    // Get table param from route query
    const table = route.query.table as string

    // If table param is present, use it for search and selection
    if (table) {
      setSearch(table)
      setSelectedTable(table)
      setPendingTableSearch(table)
    } else {
      setSearch(savedSearch || '')
      setSelectedTable(savedTable || null)
      setPendingTableSearch(null)
    }
    
    setSelectedColumn(savedCol || null)
    setColumnFilters(savedFilters ? JSON.parse(savedFilters) : {})
    didRestore.value = true
    
    console.log('[DataPage] State after restore:', {
      search: table || savedSearch || '',
      selectedTable: table || savedTable || null,
      selectedColumn: savedCol || null,
      columnFilters: savedFilters ? JSON.parse(savedFilters) : {}
    })
  }
})

// Persist to localStorage on state changes
watch(search, (newSearch) => {
  console.log('[DataPage] Persisting search to localStorage:', newSearch)
  localStorage.setItem(SEARCH_KEY, newSearch)
})

watch(selectedTable, (newSelectedTable) => {
  console.log('[DataPage] Persisting selectedTable to localStorage:', newSelectedTable)
  if (newSelectedTable) {
    localStorage.setItem(SELECTED_TABLE_KEY, newSelectedTable)
  } else {
    localStorage.removeItem(SELECTED_TABLE_KEY)
  }
})

watch(selectedColumn, (newSelectedColumn) => {
  console.log('[DataPage] Persisting selectedColumn to localStorage:', newSelectedColumn)
  if (newSelectedColumn) {
    localStorage.setItem(SELECTED_COLUMN_KEY, newSelectedColumn)
  } else {
    localStorage.removeItem(SELECTED_COLUMN_KEY)
  }
})

watch(columnFilters, (newColumnFilters) => {
  console.log('[DataPage] Persisting columnFilters to localStorage:', newColumnFilters)
  localStorage.setItem(COLUMN_FILTERS_KEY, JSON.stringify(newColumnFilters))
}, { deep: true })
</script> 