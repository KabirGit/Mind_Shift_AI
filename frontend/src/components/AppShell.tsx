"use client";

import Link from "next/link";

import { useDemoMode } from "@/components/DemoModeProvider";

type AppShellProps = {
  children: React.ReactNode;
};

export function AppShell({ children }: AppShellProps) {
  const { mode, setMode } = useDemoMode();
  const isDemo = mode === "demo";

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <header className="sticky top-0 z-20 border-b border-line/80 bg-canvas/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
          <Link href="/chat" className="flex items-center gap-3 font-semibold">
            <span className="grid size-8 place-items-center rounded-full bg-coral text-white">
              *
            </span>
            <span>Mind Shift AI</span>
          </Link>
          <nav className="flex rounded-xl border border-line bg-[#fffdf8] p-1 text-sm font-semibold">
            <Link
              className="rounded-lg px-4 py-2 text-ink hover:border-coral hover:text-ink"
              href="/chat"
            >
              Chat
            </Link>
            <Link
              className="rounded-lg px-4 py-2 text-ink hover:border-coral hover:text-ink"
              href="/dashboard"
            >
              Insights
            </Link>
          </nav>
        </div>
        <div className="border-t border-line/70 bg-[#fffdf8]">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-5 py-3 text-sm">
            <div className="flex min-w-0 flex-wrap items-center gap-3">
              <span
                className={`rounded-full px-3 py-1 text-xs font-bold uppercase tracking-[0.12em] ${
                  isDemo ? "bg-coral text-white" : "bg-night text-canvas"
                }`}
              >
                {isDemo ? "Demo" : "Live"}
              </span>
              <p className="max-w-3xl leading-6 text-body">
                {isDemo
                  ? "Demo data - this shows a simulated persona's 30 days of journaling with Mind Shift AI, and what the analytics look like after real use."
                  : "Live mode - new entries use the real pipeline, including storage, retrieval, safety checks, and the final LLM response."}
              </p>
            </div>
            <button
              className="rounded-lg border border-coral bg-coral px-4 py-2 font-semibold text-white hover:bg-coralDark"
              onClick={() => setMode(isDemo ? "live" : "demo")}
            >
              {isDemo ? "Try it yourself with a live entry" : "Return to demo"}
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-5 py-8">{children}</main>
    </div>
  );
}
