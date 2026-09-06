import { useCallback, useEffect, useRef, useState } from "react";

export interface PollingState<T> {
  item: T | null;
  error: string | null;
  polling: boolean;
  refresh: () => Promise<void>;
}

export function usePolling<T>(
  fetcher: () => Promise<T>,
  isTerminal: (item: T) => boolean,
  intervalMs: number,
  enabled: boolean,
): PollingState<T> {
  const [item, setItem] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [polling, setPolling] = useState(false);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const refresh = useCallback(async () => {
    try {
      const next = await fetcherRef.current();
      setItem(next);
      setError(null);
      setPolling(!isTerminal(next));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setPolling(false);
    }
  }, [isTerminal]);

  useEffect(() => {
    if (!enabled) return;
    void refresh();
  }, [enabled, refresh]);

  useEffect(() => {
    if (!enabled || !polling) return;
    const timer = window.setInterval(() => {
      void refresh();
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [enabled, polling, intervalMs, refresh]);

  return { item, error, polling, refresh };
}
