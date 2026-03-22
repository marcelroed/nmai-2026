import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import type { AppData, GroundTruthAnalysis, ReplayAnalysis } from "./types";

type AppDataState = {
  data: AppData | null;
  error: string | null;
  isLoading: boolean;
};

const AppDataContext = createContext<AppDataState | null>(null);
const seedAnalysisCache = new Map<string, SeedAnalysisPayload>();

type SeedAnalysisPayload = {
  groundTruth: GroundTruthAnalysis | null;
  replay: ReplayAnalysis | null;
};

type SeedAnalysisState = SeedAnalysisPayload & {
  error: string | null;
  isLoading: boolean;
};

function seedAnalysisKey(roundId: string, seedIdx: number) {
  return `${roundId}:${seedIdx}`;
}

async function fetchJsonOrNull<T>(url: string): Promise<T | null> {
  const response = await fetch(url);

  if (response.status === 404) {
    return null;
  }

  if (!response.ok) {
    throw new Error(`Failed to load ${url} (${response.status})`);
  }

  return (await response.json()) as T;
}

export function AppDataProvider(props: { children: ReactNode }) {
  const [data, setData] = useState<AppData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    fetch("/generated/rounds.json")
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Failed to load rounds.json (${response.status})`);
        }

        return (await response.json()) as AppData;
      })
      .then((payload) => {
        if (cancelled) {
          return;
        }

        setData(payload);
        setError(null);
      })
      .catch((caught: Error) => {
        if (!cancelled) {
          setError(caught.message);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <AppDataContext.Provider
      value={{
        data,
        error,
        isLoading: data === null && error === null,
      }}
    >
      {props.children}
    </AppDataContext.Provider>
  );
}

export function useAppData() {
  const value = useContext(AppDataContext);

  if (value === null) {
    throw new Error("useAppData must be used inside AppDataProvider");
  }

  return value;
}

export function useSeedAnalysis(roundId: string | null, seedIdx: number) {
  const [state, setState] = useState<SeedAnalysisState>({
    groundTruth: null,
    replay: null,
    error: null,
    isLoading: false,
  });

  useEffect(() => {
    if (roundId === null || Number.isNaN(seedIdx)) {
      setState({
        groundTruth: null,
        replay: null,
        error: null,
        isLoading: false,
      });
      return;
    }

    const key = seedAnalysisKey(roundId, seedIdx);
    const cached = seedAnalysisCache.get(key);

    if (cached) {
      setState({
        ...cached,
        error: null,
        isLoading: false,
      });
      return;
    }

    let cancelled = false;

    setState((current) => ({
      ...current,
      error: null,
      isLoading: true,
    }));

    Promise.all([
      fetchJsonOrNull<GroundTruthAnalysis>(
        `/generated/analysis/${roundId}/ground-truth-seed-${seedIdx}.json`,
      ),
      fetchJsonOrNull<ReplayAnalysis>(`/generated/analysis/${roundId}/replay-seed-${seedIdx}.json`),
    ])
      .then(([groundTruth, replay]) => {
        if (cancelled) {
          return;
        }

        const payload = { groundTruth, replay };
        seedAnalysisCache.set(key, payload);
        setState({
          ...payload,
          error: null,
          isLoading: false,
        });
      })
      .catch((caught: Error) => {
        if (!cancelled) {
          setState({
            groundTruth: null,
            replay: null,
            error: caught.message,
            isLoading: false,
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [roundId, seedIdx]);

  return state;
}
