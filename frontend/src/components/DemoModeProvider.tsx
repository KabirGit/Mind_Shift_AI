"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useSyncExternalStore
} from "react";

export type AppMode = "demo" | "live";

type DemoModeContextValue = {
  mode: AppMode;
  setMode: (mode: AppMode) => void;
};

const DemoModeContext = createContext<DemoModeContextValue | null>(null);
const STORAGE_KEY = "mind-shift-ai-mode";
const MODE_CHANGE_EVENT = "mind-shift-ai-mode-change";

function readStoredMode(): AppMode {
  if (typeof window === "undefined") return "demo";
  const stored = window.localStorage.getItem(STORAGE_KEY);
  return stored === "demo" || stored === "live" ? stored : "demo";
}

function subscribeToMode(callback: () => void) {
  window.addEventListener("storage", callback);
  window.addEventListener(MODE_CHANGE_EVENT, callback);
  return () => {
    window.removeEventListener("storage", callback);
    window.removeEventListener(MODE_CHANGE_EVENT, callback);
  };
}

function getServerMode(): AppMode {
  return "demo";
}

export function DemoModeProvider({ children }: { children: React.ReactNode }) {
  const mode = useSyncExternalStore(subscribeToMode, readStoredMode, getServerMode);
  const setMode = useCallback((nextMode: AppMode) => {
    window.localStorage.setItem(STORAGE_KEY, nextMode);
    window.dispatchEvent(new Event(MODE_CHANGE_EVENT));
  }, []);

  const value = useMemo<DemoModeContextValue>(
    () => ({
      mode,
      setMode
    }),
    [mode, setMode]
  );

  return (
    <DemoModeContext.Provider value={value}>{children}</DemoModeContext.Provider>
  );
}

export function useDemoMode() {
  const context = useContext(DemoModeContext);
  if (!context) {
    throw new Error("useDemoMode must be used inside DemoModeProvider");
  }
  return context;
}
