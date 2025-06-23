import { ref, onMounted, onUnmounted, type Ref } from 'vue';

interface UseInfiniteScrollOptions {
  threshold?: number;
  rootMargin?: string;
}

interface UseInfiniteScrollReturn {
  isLoading: Ref<boolean>;
  error: Ref<string | null>;
  hasMore: Ref<boolean>;
  loadMore: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setHasMore: (hasMore: boolean) => void;
  observerRef: Ref<HTMLDivElement | null>;
}

export function useInfiniteScroll(
  onLoadMore: () => Promise<void>,
  options: UseInfiniteScrollOptions = {}
): UseInfiniteScrollReturn {
  const { threshold = 1.0, rootMargin = '0px' } = options;
  
  const isLoading = ref(false);
  const error = ref<string | null>(null);
  const hasMore = ref(true);
  const observerRef = ref<HTMLDivElement | null>(null);
  
  let observer: IntersectionObserver | null = null;
  
  const loadMore = async () => {
    if (isLoading.value || !hasMore.value) return;
    
    try {
      setLoading(true);
      setError(null);
      await onLoadMore();
    } catch (err) {
      console.error('[useInfiniteScroll] Load more error:', err);
      setError(err instanceof Error ? err.message : 'Failed to load more items');
    } finally {
      setLoading(false);
    }
  };
  
  const setLoading = (loading: boolean) => {
    isLoading.value = loading;
  };
  
  const setError = (errorMessage: string | null) => {
    error.value = errorMessage;
  };
  
  const setHasMore = (more: boolean) => {
    hasMore.value = more;
  };
  
  const handleIntersection = (entries: IntersectionObserverEntry[]) => {
    const target = entries[0];
    if (target.isIntersecting && hasMore.value && !isLoading.value) {
      loadMore();
    }
  };
  
  onMounted(() => {
    if (!observerRef.value) return;
    
    observer = new IntersectionObserver(handleIntersection, {
      threshold,
      rootMargin,
    });
    
    observer.observe(observerRef.value);
  });
  
  onUnmounted(() => {
    if (observer && observerRef.value) {
      observer.unobserve(observerRef.value);
      observer.disconnect();
    }
  });
  
  return {
    isLoading,
    error,
    hasMore,
    loadMore,
    setLoading,
    setError,
    setHasMore,
    observerRef,
  };
} 