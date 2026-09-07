"use client";

import {
  Activity,
  BookOpen,
  BrainCircuit,
  CheckCircle2,
  Gauge,
  GitBranch,
  LockKeyhole,
  ShieldCheck,
  Wrench
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/Card";
import { useDemoMode } from "@/components/DemoModeProvider";
import {
  type DemoJournalEntries,
  type Observability,
  getDemoJournalEntries,
  getObservability
} from "@/lib/api";
import { titleCase } from "@/lib/format";

export default function ObservabilityPage() {
  const { mode } = useDemoMode();
  const [snapshot, setSnapshot] = useState<Observability | null>(null);
  const [journals, setJournals] = useState<DemoJournalEntries | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [observability, journalData] = await Promise.all([
          getObservability(mode),
          mode === "demo" ? getDemoJournalEntries() : Promise.resolve(null)
        ]);
        if (cancelled) return;
        setSnapshot(observability);
        setJournals(journalData);
      } catch (exc) {
        if (!cancelled) {
          setError(exc instanceof Error ? exc.message : "Observability request failed.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [mode]);

  const diagnostics = snapshot?.diagnostics;
  const headlineMetrics = useMemo(() => {
    if (!snapshot || !diagnostics) return [];
    return [
      {
        label: "Journal coverage",
        value: `${snapshot.dataset.entry_count} entries`,
        detail: `${snapshot.dataset.days_covered} distinct days`
      },
      {
        label: "Retrieval precision@3",
        value: formatPercent(numberValue(diagnostics.retrieval_precision.precision_at_k)),
        detail: `${numberValue(diagnostics.retrieval_precision.n_samples_used)} deterministic samples`
      },
      {
        label: "Emotion confidence",
        value: formatPercent(numberValue(diagnostics.emotion_confidence.mean_confidence)),
        detail: `${numberValue(diagnostics.emotion_confidence.low_confidence_ratio)} low-confidence ratio`
      },
      {
        label: "Golden evaluation",
        value: `${snapshot.evaluation.passed ?? "-"}/${snapshot.evaluation.case_count}`,
        detail: titleCase(snapshot.evaluation.status)
      }
    ];
  }, [snapshot, diagnostics]);

  return (
    <AppShell>
      <div className="mb-7 grid gap-5 lg:grid-cols-[1.4fr_.6fr] lg:items-end">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.12em] text-coral">
            Observability
          </p>
          <h1 className="mt-1 max-w-4xl font-display text-5xl leading-tight text-ink">
            Evidence that the system works—not just a polished response.
          </h1>
          <p className="mt-3 max-w-3xl text-base leading-7 text-body">
            This page connects the 30-day persona to retrieval, analytics, bounded agent
            behavior, model calls, safety routes, evaluation coverage, and redacted traces.
          </p>
        </div>
        <div className="rounded-xl border border-line bg-[#fffdf8] p-4 text-sm leading-6 text-body">
          <p className="font-semibold text-ink">Evidence source</p>
          <p>{snapshot?.source ?? (loading ? "Loading snapshot…" : "Unavailable")}</p>
          {snapshot ? (
            <p className="mt-1 text-xs text-muted">
              Generated {new Date(snapshot.generated_at).toLocaleString()}
            </p>
          ) : null}
        </div>
      </div>

      {error ? (
        <div className="mb-5 rounded-xl border border-[#c64545] bg-[#fff1ed] p-4 text-sm text-ink">
          {error}
        </div>
      ) : null}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {headlineMetrics.map((metric) => (
          <Card key={metric.label} tone="light">
            <p className="text-xs font-bold uppercase tracking-[0.12em] text-muted">
              {metric.label}
            </p>
            <p className="mt-2 font-display text-4xl text-ink">{metric.value}</p>
            <p className="mt-1 text-sm text-body">{metric.detail}</p>
          </Card>
        ))}
      </div>

      {snapshot ? (
        <>
          <div className="mt-6 grid gap-6 xl:grid-cols-[1.1fr_.9fr]">
            <Card tone="dark">
              <SectionHeading icon={<GitBranch />} inverse title="Request path" />
              <div className="mt-5 grid gap-3 sm:grid-cols-2">
                {snapshot.request_flow.map((stage, index) => (
                  <div
                    className="flex gap-3 rounded-lg border border-white/15 bg-white/5 p-3"
                    key={stage}
                  >
                    <span className="grid size-7 shrink-0 place-items-center rounded-full bg-coral text-xs font-bold text-white">
                      {index + 1}
                    </span>
                    <p className="text-sm leading-6 text-[#e6e0d7]">{stage}</p>
                  </div>
                ))}
              </div>
            </Card>

            <Card tone="light">
              <SectionHeading icon={<BrainCircuit />} title="Model boundary" />
              <dl className="mt-5 space-y-3">
                {Object.entries(snapshot.model_boundary).map(([key, value]) => (
                  <KeyValue key={key} label={key} value={String(value)} />
                ))}
              </dl>
              <div className="mt-5 rounded-lg border border-line bg-canvas p-3 text-sm leading-6 text-body">
                The hosted model phrases decisions and final guidance. Safety, retrieval,
                scoring, reversibility, and tool limits remain deterministic Python.
              </div>
            </Card>
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-3">
            {snapshot.route_contracts.map((route) => (
              <Card key={route.mode} tone={route.mode === "safety" ? "coral" : "light"}>
                <div className="flex items-center justify-between gap-3">
                  <h2 className="font-display text-3xl">{titleCase(route.mode)}</h2>
                  {route.mode === "safety" ? <ShieldCheck className="size-6" /> : <Activity className="size-6 text-coral" />}
                </div>
                <p className="mt-2 text-sm leading-6 opacity-90">{route.when}</p>
                <div className="mt-4 grid grid-cols-2 gap-3">
                  <Metric label="LLM calls" value={route.logical_llm_calls} />
                  <Metric label="Max tools" value={route.maximum_tool_calls} />
                </div>
                <p className="mt-4 text-sm font-semibold leading-6">{route.guarantee}</p>
              </Card>
            ))}
          </div>

          <Card className="mt-6" tone="light">
            <SectionHeading icon={<CheckCircle2 />} title="Capability proof matrix" />
            <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {snapshot.capabilities.map((capability) => (
                <div className="rounded-lg border border-line bg-canvas p-4" key={capability.name}>
                  <div className="flex items-start justify-between gap-3">
                    <p className="font-semibold text-ink">{capability.name}</p>
                    <span
                      className={`rounded-full px-2 py-1 text-[10px] font-bold uppercase tracking-[0.1em] ${
                        capability.status === "proven"
                          ? "bg-[#dff4e5] text-[#27663c]"
                          : "bg-[#f4eadf] text-[#7c5a34]"
                      }`}
                    >
                      {titleCase(capability.status)}
                    </span>
                  </div>
                  <p className="mt-2 text-sm leading-6 text-body">{capability.evidence}</p>
                  <p className="mt-2 text-xs font-semibold text-muted">{capability.component}</p>
                </div>
              ))}
            </div>
          </Card>

          <div className="mt-6 grid gap-6 xl:grid-cols-[.8fr_1.2fr]">
            <Card tone="light">
              <SectionHeading icon={<Gauge />} title="Context policy" />
              <dl className="mt-5 space-y-3">
                {Object.entries(snapshot.context_policy).map(([key, value]) => (
                  <KeyValue
                    key={key}
                    label={key}
                    value={typeof value === "boolean" ? (value ? "Yes" : "No") : String(value)}
                  />
                ))}
              </dl>
            </Card>

            <Card tone="light">
              <SectionHeading icon={<Wrench />} title="Recent redacted traces" />
              <div className="mt-5 space-y-3">
                {snapshot.traces.map((trace) => (
                  <div className="rounded-lg border border-line bg-canvas p-4" key={trace.trace_id}>
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="rounded-full bg-night px-3 py-1 text-xs font-bold text-white">
                          {titleCase(trace.mode)}
                        </span>
                        <span className="text-sm font-semibold text-ink">
                          {titleCase(trace.status)} · {titleCase(trace.outcome)}
                        </span>
                      </div>
                      <span className="font-mono text-xs text-muted">{trace.trace_id}</span>
                    </div>
                    <div className="mt-3 grid grid-cols-2 gap-2 text-sm sm:grid-cols-5">
                      <TraceMetric label="LLM calls" value={trace.logical_llm_calls} />
                      <TraceMetric label="Attempts" value={trace.provider_attempts} />
                      <TraceMetric label="Tools" value={trace.tools_called.length} />
                      <TraceMetric label="Memories" value={trace.memories_retrieved} />
                      <TraceMetric label="Latency" value={`${Math.round(trace.latency_ms)} ms`} />
                    </div>
                    {trace.tools_called.length ? (
                      <p className="mt-3 text-xs leading-5 text-muted">
                        Tools: {trace.tools_called.join(" → ")}. Termination: {titleCase(trace.agent_termination_reason ?? "none")}.
                      </p>
                    ) : null}
                  </div>
                ))}
              </div>
            </Card>
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <Card tone="dark">
              <SectionHeading icon={<CheckCircle2 />} inverse title="Golden regression suite" />
              <p className="mt-3 font-display text-5xl">
                {snapshot.evaluation.passed ?? "-"}/{snapshot.evaluation.case_count}
              </p>
              <p className="mt-2 text-sm leading-6 text-[#d7d1c7]">
                Deterministic production-pipeline cases passed in this frozen demo snapshot.
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                {snapshot.evaluation.coverage.map((item) => (
                  <span className="rounded-full border border-white/20 px-3 py-1 text-xs text-[#e6e0d7]" key={item}>
                    {item}
                  </span>
                ))}
              </div>
            </Card>
            <Card tone="light">
              <SectionHeading icon={<LockKeyhole />} title="Privacy boundary" />
              <div className="mt-5 space-y-3">
                {Object.entries(snapshot.privacy).map(([key, value]) => (
                  <KeyValue
                    key={key}
                    label={key}
                    value={typeof value === "boolean" ? (value ? "Yes" : "No") : String(value)}
                  />
                ))}
              </div>
            </Card>
          </div>

          {journals ? <JournalEvidence journals={journals} /> : null}
        </>
      ) : loading ? (
        <Card className="mt-6" tone="light">Loading system evidence…</Card>
      ) : null}
    </AppShell>
  );
}

function JournalEvidence({ journals }: { journals: DemoJournalEntries }) {
  const representative = [0, 7, 14, 21, 26, 29]
    .map((index) => journals.entries[index])
    .filter(Boolean);
  return (
    <Card className="mt-6" tone="light">
      <SectionHeading icon={<BookOpen />} title="30-day journal evidence" />
      <p className="mt-3 max-w-4xl text-sm leading-6 text-body">
        Every day contains a detailed synthetic entry plus emotion, sentiment, topics,
        habits, and people metadata. The six cards show the story arc; expand the audit
        below to inspect all {journals.entry_count} records.
      </p>
      <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {representative.map((entry) => (
          <JournalCard entry={entry} key={entry.id} />
        ))}
      </div>
      <details className="mt-5 rounded-lg border border-line bg-canvas p-4">
        <summary className="cursor-pointer font-semibold text-ink">
          Inspect all {journals.days_covered} dated journal records
        </summary>
        <div className="mt-4 grid gap-3 lg:grid-cols-2">
          {journals.entries.map((entry) => (
            <JournalCard compact entry={entry} key={entry.id} />
          ))}
        </div>
      </details>
    </Card>
  );
}

function JournalCard({
  entry,
  compact = false
}: {
  entry: DemoJournalEntries["entries"][number];
  compact?: boolean;
}) {
  return (
    <article className="rounded-lg border border-line bg-[#fffdf8] p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="font-semibold text-ink">{entry.date}</p>
        <span className="rounded-full bg-[#f4eadf] px-3 py-1 text-xs font-bold text-coralDark">
          {titleCase(entry.emotion)} · {formatPercent(entry.emotion_confidence)}
        </span>
      </div>
      <p className={`mt-3 text-sm leading-6 text-body ${compact ? "line-clamp-4" : ""}`}>
        {entry.text}
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        {[...entry.topics, ...entry.habits, ...entry.people].map((tag) => (
          <span className="rounded-full border border-line px-2 py-1 text-[11px] text-muted" key={`${entry.id}-${tag}`}>
            {titleCase(tag)}
          </span>
        ))}
      </div>
    </article>
  );
}

function SectionHeading({
  title,
  icon,
  inverse = false
}: {
  title: string;
  icon: React.ReactNode;
  inverse?: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-coral [&>svg]:size-5">{icon}</span>
      <h2 className={`font-display text-3xl ${inverse ? "text-white" : "text-ink"}`}>{title}</h2>
    </div>
  );
}

function KeyValue({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-line/70 pb-2 text-sm last:border-0">
      <dt className="text-body">{titleCase(label.replace(/_/g, " "))}</dt>
      <dd className="max-w-[62%] text-right font-semibold text-ink">{value}</dd>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border border-current/15 p-3">
      <p className="text-xs uppercase tracking-[0.1em] opacity-70">{label}</p>
      <p className="mt-1 font-display text-3xl">{value}</p>
    </div>
  );
}

function TraceMetric({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <p className="text-[10px] font-bold uppercase tracking-[0.1em] text-muted">{label}</p>
      <p className="font-semibold text-ink">{value}</p>
    </div>
  );
}

function numberValue(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}
