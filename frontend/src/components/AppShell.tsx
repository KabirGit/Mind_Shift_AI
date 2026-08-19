import Link from "next/link";

type AppShellProps = {
  children: React.ReactNode;
};

export function AppShell({ children }: AppShellProps) {
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
      </header>
      <main className="mx-auto max-w-7xl px-5 py-8">{children}</main>
    </div>
  );
}
