"use client";

import {
  Activity,
  ArrowDownToLine,
  Brain,
  LineChart as LineChartIcon,
  Search
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/Card";
import { EmptyState } from "@/components/EmptyState";
import { ProgressBar } from "@/components/ProgressBar";
import {
  type DashboardSummary,
  type Diagnostics,
  type GoalProgress,
  type GraphQuery,
  type GrowthSnapshot,
  type TimelineEvent,
  getDashboardSummary,
  getDiagnostics,
  getGoals,
  getGrowth,
  getPredictions,
  getTimeline,
  queryGraph,
  weeklyReportUrl
} from "@/lib/api";
import { percent, signed, titleCase } from "@/lib/format";

type PredictionsState = Awaited<ReturnType<typeof getPredictions>>;
type GrowthState = { snapshots: GrowthSnapshot[]; narrative: string };

const RANGES = ["Last 7 days", "Last 30 days", "All time"];

export default function DashboardPage() {
  const [range, setRange] = useState(RANGES[1]);
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [goals, setGoals] = useState<GoalProgress[]>([]);
  const [predictions, setPredictions] = useState<PredictionsState | null>(null);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [growth, setGrowth] = useState<GrowthState | null>(null);
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [graphNode, setGraphNode] = useState("");
  const [graph, setGraph] = useState<GraphQuery | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [summaryData, goalsData, predictionData, timelineData, growthData, diag] =
          await Promise.all([
            getDashboardSummary(range),
            getGoals(),
            getPredictions(),
            getTimeline(),
            getGrowth(),
            getDiagnostics()
          ]);
        if (cancelled) return;
        setSummary(summaryData);
        setGoals(goalsData.goals);
        setPredictions(predictionData);
        setTimeline(timelineData.events);
        setGrowth(growthData);
        setDiagnostics(diag);
      } catch (exc) {
        if (!cancelled) {
          setError(exc instanceof Error ? exc.message : "Dashboard request failed.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [range]);

  const topicRows = useMemo(() => {
    return Object.entries(summary?.recurring_topics ?? {})
      .map(([topic, count]) => ({ topic: titleCase(topic), count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 8);
  }, [summary]);

  const emotionRows = useMemo(() => {
    const byDate = new Map<string, Record<string, string | number>>();
    for (const point of summary?.emotion_over_time ?? []) {
      const row = byDate.get(point.date) ?? { date: point.date };
      row[titleCase(point.emotion)] = point.count;
      byDate.set(point.date, row);
    }
    return Array.from(byDate.values());
  }, [summary]);

  async function runGraphSearch() {
    const node = graphNode.trim();
    if (!node) return;
    setGraph(await queryGraph(node));
  }

  return (
    <AppShell>
      <div className="mb-7 flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.12em] text-coral">
            Insights
          </p>
          <h1 className="font-display text-5xl leading-tight text-ink">
            Conclusions before charts.
          </h1>
          <p className="mt-2 max-w-2xl text-body">
            The dashboard reads your existing analytics endpoints and keeps raw evidence
            close, but secondary.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            className="rounded-lg border border-line bg-[#fffdf8] px-4 py-2 font-semibold text-ink"
            onChange={(event) => setRange(event.target.value)}
            value={range}
          >
            {RANGES.map((item) => (
              <option key={item}>{item}</option>
            ))}
          </select>
          <a
            className="inline-flex items-center gap-2 rounded-lg bg-coral px-4 py-2 font-semibold text-white hover:bg-coralDark"
            href={weeklyReportUrl()}
          >
            <ArrowDownToLine className="size-4" />
            Weekly PDF
          </a>
        </div>
      </div>

      {error ? (
        <div className="mb-5 rounded-xl border border-[#c64545] bg-[#fff1ed] p-4 text-sm text-ink">
          {error}
        </div>
      ) : null}

      <div className="grid gap-4 md:grid-cols-3">
        <Card tone="coral">
          <p className="text-sm font-semibold uppercase tracking-[0.12em]">
            Mood direction
          </p>
          <h2 className="mt-2 font-display text-3xl">
            {predictions ? titleCase(predictions.sentiment_forecast.direction) : "Loading"}
          </h2>
          <p className="mt-2 text-sm leading-6">
            {predictions?.sentiment_forecast.explanation ?? "Reading recent entries."}
          </p>
        </Card>
        <Card tone="dark">
          <p className="text-sm font-semibold uppercase tracking-[0.12em] text-[#a09d96]">
            Stress pattern
          </p>
          <h2 className="mt-2 font-display text-3xl">
            {predictions ? titleCase(predictions.burnout_risk.risk_level) : "Loading"}
          </h2>
          <p className="mt-2 text-sm leading-6 text-[#d7d1c7]">
            {predictions?.burnout_risk.explanation ?? "Reading risk signals."}
          </p>
        </Card>
        <Card>
          <p className="text-sm font-semibold uppercase tracking-[0.12em] text-muted">
            Entry window
          </p>
          <h2 className="mt-2 font-display text-3xl">
            {summary?.lookback_days === 36500 ? "All time" : `${summary?.lookback_days ?? 0} days`}
          </h2>
          <p className="mt-2 text-sm leading-6 text-body">
            {summary?.triggers.length ?? 0} triggers, {summary?.habits.length ?? 0} habit
            signals, {summary?.relationships.length ?? 0} people patterns.
          </p>
        </Card>
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-[1.2fr_.8fr]">
        <Card className="min-h-[360px]" tone="light">
          <SectionTitle icon={<Activity />} title="Emotional rhythm" />
          {emotionRows.length ? (
            <ResponsiveContainer height={280} width="100%">
              <AreaChart data={emotionRows}>
                <CartesianGrid stroke="#d2c3b2" strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fill: "#6c6a64", fontSize: 12 }} />
                <YAxis tick={{ fill: "#6c6a64", fontSize: 12 }} />
                <Tooltip />
                {Object.keys(emotionRows[0])
                  .filter((key) => key !== "date")
                  .map((key, index) => (
                    <Area
                      dataKey={key}
                      fill={index % 2 ? "#5db8a6" : "#cc785c"}
                      fillOpacity={0.2}
                      key={key}
                      stroke={index % 2 ? "#5db8a6" : "#cc785c"}
                    />
                  ))}
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState
              detail="Write dated entries and the emotion series will appear here."
              title={loading ? "Loading rhythm" : "No emotion series yet"}
            />
          )}
        </Card>

        <Card tone="light">
          <SectionTitle icon={<LineChartIcon />} title="Top topics" />
          {topicRows.length ? (
            <ResponsiveContainer height={280} width="100%">
              <BarChart data={topicRows}>
                <CartesianGrid stroke="#d2c3b2" strokeDasharray="3 3" />
                <XAxis dataKey="topic" tick={{ fill: "#6c6a64", fontSize: 12 }} />
                <YAxis tick={{ fill: "#6c6a64", fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#cc785c" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState
              detail="Recurring topics will show once the backend sees repeated themes."
              title="No topics yet"
            />
          )}
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <InsightList title="Triggers" rows={summary?.triggers ?? []} />
        <InsightList title="Habits" rows={summary?.habits ?? []} />
        <InsightList title="People" rows={summary?.relationships ?? []} />
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-[.8fr_1.2fr]">
        <Card>
          <SectionTitle icon={<Brain />} title="Goal progress" />
          {goals.length ? (
            <div className="mt-4 space-y-4">
              {goals.map((goal) => (
                <div key={goal.goal_keyword}>
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <p className="font-semibold">{titleCase(goal.goal_keyword)}</p>
                    <span className="text-sm text-muted">
                      {percent(goal.estimated_progress)}
                    </span>
                  </div>
                  <ProgressBar value={goal.estimated_progress} />
                  <p className="mt-2 text-sm leading-6 text-body">{goal.explanation}</p>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState
              detail="Repeated goals such as job search, fitness, or education will appear here."
              title="No tracked goals yet"
            />
          )}
        </Card>

        <Card tone="light">
          <SectionTitle icon={<Search />} title="Knowledge graph search" />
          <div className="mt-4 flex gap-3">
            <input
              className="min-w-0 flex-1 rounded-lg border border-line bg-[#fffdf8] px-4 py-2 text-ink outline-none focus:border-coral"
              onChange={(event) => setGraphNode(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void runGraphSearch();
              }}
              placeholder="career, money, Alice..."
              value={graphNode}
            />
            <button
              className="rounded-lg bg-coral px-4 py-2 font-semibold text-white hover:bg-coralDark"
              onClick={() => void runGraphSearch()}
            >
              Search
            </button>
          </div>
          {graph ? (
            <div className="mt-4 rounded-xl border border-line bg-canvas p-4">
              <p className="font-semibold">{graph.summary}</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {graph.neighbors.map((neighbor) => (
                  <span
                    className="rounded-full border border-line bg-[#fffdf8] px-3 py-1 text-sm"
                    key={neighbor}
                  >
                    {neighbor}
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <p className="mt-4 text-sm text-muted">
              Search one node to see how it connects to your topics, people, and habits.
            </p>
          )}
        </Card>
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <Card tone="light">
          <SectionTitle title="My Journey" />
          {timeline.length ? (
            <div className="mt-4 space-y-4">
              {timeline.slice(0, 8).map((event) => (
                <div className="border-l-2 border-coral pl-4" key={event.timestamp}>
                  <p className="text-sm font-semibold text-muted">
                    {event.timestamp.slice(0, 10)} · {titleCase(event.event_type)}
                  </p>
                  <h3 className="mt-1 font-semibold text-ink">{event.title}</h3>
                  <p className="mt-1 text-sm leading-6 text-body">{event.description}</p>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState
              detail="Significant journal moments will appear as the timeline grows."
              title="No journey events yet"
            />
          )}
        </Card>

        <Card tone="light">
          <SectionTitle title="Growth over time" />
          <p className="mt-1 text-sm leading-6 text-body">
            {growth?.narrative ?? "Loading growth narrative."}
          </p>
          {growth?.snapshots.length ? (
            <ResponsiveContainer height={250} width="100%">
              <AreaChart data={growth.snapshots}>
                <CartesianGrid stroke="#d2c3b2" strokeDasharray="3 3" />
                <XAxis dataKey="period_label" tick={{ fill: "#6c6a64", fontSize: 12 }} />
                <YAxis tick={{ fill: "#6c6a64", fontSize: 12 }} />
                <Tooltip />
                <Area
                  dataKey="avg_sentiment"
                  fill="#5db8a6"
                  fillOpacity={0.25}
                  stroke="#5db8a6"
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState
              detail="Monthly snapshots need entries across time."
              title="No growth chart yet"
            />
          )}
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1.1fr_.9fr]">
        <Card tone="coral">
          <h2 className="font-display text-3xl">Insights list</h2>
          {summary?.insights.length ? (
            <ul className="mt-4 space-y-3">
              {summary.insights.map((insight) => (
                <li className="rounded-lg bg-white/15 p-3 leading-6" key={insight}>
                  {insight}
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-3 text-sm leading-6">
              Keep journaling and the deterministic insight engine will summarize the
              recurring story here.
            </p>
          )}
        </Card>
        <Card tone="dark">
          <h2 className="font-display text-3xl">Diagnostics</h2>
          <pre className="mt-4 overflow-auto rounded-lg bg-nightLift p-4 font-mono text-xs leading-6 text-[#d7d1c7]">
            {JSON.stringify(diagnostics ?? {}, null, 2)}
          </pre>
        </Card>
      </div>
    </AppShell>
  );
}

function SectionTitle({
  title,
  icon
}: {
  title: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2">
      {icon ? <span className="text-coral [&>svg]:size-5">{icon}</span> : null}
      <h2 className="font-display text-3xl text-ink">{title}</h2>
    </div>
  );
}

function InsightList({
  title,
  rows
}: {
  title: string;
  rows: Array<{
    confidence: number;
    explanation: string;
    trend?: string;
    topic?: string;
    habit?: string;
    person?: string;
    delta?: number;
    avg_sentiment?: number;
  }>;
}) {
  return (
    <Card tone="light">
      <h2 className="font-display text-3xl">{title}</h2>
      {rows.length ? (
        <div className="mt-4 space-y-3">
          {rows.map((row) => {
            const name = row.topic ?? row.habit ?? row.person ?? "Pattern";
            return (
              <div className="rounded-xl border border-line bg-canvas p-3" key={name}>
                <div className="flex items-center justify-between gap-3">
                  <p className="font-semibold">{titleCase(name)}</p>
                  <span className="rounded-full bg-card px-2 py-1 text-xs font-semibold text-ink">
                    {percent(row.confidence)}
                  </span>
                </div>
                <p className="mt-2 text-sm leading-6 text-body">{row.explanation}</p>
                {typeof row.delta === "number" ? (
                  <p className="mt-1 text-xs font-semibold text-muted">
                    Delta {signed(row.delta)}
                  </p>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : (
        <EmptyState
          detail={`No ${title.toLowerCase()} have enough repeated evidence yet.`}
          title={`No ${title.toLowerCase()} yet`}
        />
      )}
    </Card>
  );
}
