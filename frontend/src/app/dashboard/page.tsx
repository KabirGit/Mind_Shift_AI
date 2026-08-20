"use client";

import {
  ArrowDownToLine,
  Brain,
  CalendarDays,
  HeartHandshake,
  Search,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Users
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
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
import { RelationshipGraph } from "@/components/RelationshipGraph";
import {
  type DashboardStory,
  type Diagnostics,
  type GraphQuery,
  type PeopleGraph,
  type TimelineEvent,
  getDashboardStory,
  getDiagnostics,
  getPeopleGraph,
  getTimeline,
  queryGraph,
  weeklyReportUrl
} from "@/lib/api";
import { percent, signed, titleCase } from "@/lib/format";

const RANGES = ["Last 7 days", "Last 30 days", "All time"];

export default function DashboardPage() {
  const [range, setRange] = useState(RANGES[1]);
  const [story, setStory] = useState<DashboardStory | null>(null);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [peopleGraph, setPeopleGraph] = useState<PeopleGraph | null>(null);
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
        const [storyData, timelineData, diag, peopleGraphData] = await Promise.all([
          getDashboardStory(range),
          getTimeline(),
          getDiagnostics(),
          getPeopleGraph()
        ]);
        if (cancelled) return;
        setStory(storyData);
        setTimeline(timelineData.events);
        setDiagnostics(diag);
        setPeopleGraph(peopleGraphData);
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

  const thresholds = story?.thresholds;
  const working = useMemo(() => {
    if (!story || !thresholds) return [];
    // Mirror the backend confidence gate so stale or malformed payloads cannot render claims.
    return story.top_working.filter(
      (item) => item.confidence >= thresholds.min_insight_confidence
    );
  }, [story, thresholds]);

  const draining = useMemo(() => {
    if (!story || !thresholds) return [];
    // Mirror the backend confidence gate so trend claims stay evidence-backed.
    return story.top_draining.filter(
      (item) => item.confidence >= thresholds.min_insight_confidence
    );
  }, [story, thresholds]);

  const people = useMemo(() => {
    if (!story || !thresholds) return [];
    // Three mentions is the minimum repeated-person threshold supplied by the API.
    return story.people.filter(
      (person) => person.mention_count >= thresholds.min_mention_count
    );
  }, [story, thresholds]);

  const weekRows = useMemo(() => {
    return (story?.weekly_buckets ?? []).slice(-4).map((bucket) => ({
      ...bucket,
      sentiment_label: signed(bucket.avg_sentiment),
      topic: titleCase(bucket.top_topic)
    }));
  }, [story]);

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
            Dashboard
          </p>
          <h1 className="font-display text-5xl leading-tight text-ink">
            Your Month in Review
          </h1>
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

      <Card className="min-h-[260px]" tone="coral">
        {story ? (
          <MonthHero story={story} />
        ) : (
          <p className="text-sm leading-6">{loading ? "Loading review." : "No review yet."}</p>
        )}
      </Card>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <WorkingSection rows={working} />
        <DrainingSection rows={draining} />
        <PeopleSection rows={people} />
      </div>

      {story?.rhythm ? (
        <Card className="mt-6" tone="light">
          <SectionTitle icon={<CalendarDays />} title="Your Rhythm" />
          <p className="mt-3 text-lg font-semibold text-ink">
            {titleCase(story.rhythm.topic)} tends to feel {story.rhythm.delta >= 0 ? "lighter" : "heavier"} on{" "}
            {story.rhythm.day_time_crossing ?? story.rhythm.peak_day_of_week}.
          </p>
          <p className="mt-2 text-sm leading-6 text-body">{story.rhythm.explanation}</p>
          <EvidenceLine
            confidence={story.rhythm.confidence}
            detail={`Delta ${signed(story.rhythm.delta)} against baseline ${signed(
              story.rhythm.baseline_avg_sentiment
            )}`}
          />
        </Card>
      ) : null}

      <div className="mt-6 grid gap-6 xl:grid-cols-[1.05fr_.95fr]">
        <Card tone="light">
          <SectionTitle icon={<TrendingUp />} title="Week by Week" />
          {weekRows.length ? (
            <>
              <ResponsiveContainer height={260} width="100%">
                <BarChart data={weekRows}>
                  <CartesianGrid stroke="#d2c3b2" strokeDasharray="3 3" />
                  <XAxis dataKey="label" tick={{ fill: "#6c6a64", fontSize: 12 }} />
                  <YAxis domain={[-1, 1]} tick={{ fill: "#6c6a64", fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="avg_sentiment" fill="#5db8a6" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {weekRows.map((bucket) => (
                  <div className="rounded-lg border border-line bg-canvas p-3" key={bucket.label}>
                    <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">
                      {bucket.label}
                    </p>
                    <p className="mt-1 font-semibold text-ink">{bucket.sentiment_label}</p>
                    <p className="text-xs text-body">
                      {bucket.entry_count} entries - {titleCase(bucket.dominant_emotion)} - {bucket.topic}
                    </p>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <EmptyState
              detail="Weekly comparisons need dated entries inside the selected range."
              title="No weekly buckets yet"
            />
          )}
        </Card>

        <LookingAhead story={story} />
      </div>

      <details className="mt-6 rounded-xl border border-line bg-[#fffdf8] p-5 text-ink">
        <summary className="cursor-pointer font-display text-3xl">Explore Further</summary>
        <div className="mt-5 grid gap-6 xl:grid-cols-[1.1fr_.9fr]">
          <Card tone="light">
            <SectionTitle icon={<Search />} title="Relationship graph" />
            <div className="mt-4">
              <RelationshipGraph graph={peopleGraph} />
            </div>
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
              <div className="mt-4 rounded-lg border border-line bg-canvas p-4">
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
            ) : null}
          </Card>

          <Card tone="dark">
            <h2 className="font-display text-3xl">Diagnostics</h2>
            <pre className="mt-4 overflow-auto rounded-lg bg-nightLift p-4 font-mono text-xs leading-6 text-[#d7d1c7]">
              {JSON.stringify(diagnostics ?? {}, null, 2)}
            </pre>
          </Card>
        </div>

        <Card className="mt-6" tone="light">
          <SectionTitle title="Full timeline" />
          {timeline.length ? (
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              {timeline.map((event) => (
                <div className="border-l-2 border-coral pl-4" key={event.timestamp}>
                  <p className="text-sm font-semibold text-muted">
                    {event.timestamp.slice(0, 10)} - {titleCase(event.event_type)}
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
      </details>
    </AppShell>
  );
}

function MonthHero({ story }: { story: DashboardStory }) {
  const headline = story.headline;
  if (!headline.has_sufficient_data) {
    return (
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.12em] text-white/80">
          {headline.entry_count} of {headline.minimum_entry_count} entries
        </p>
        <h2 className="mt-3 font-display text-4xl">
          Not enough data yet to summarize this month with confidence.
        </h2>
        <p className="mt-3 max-w-2xl leading-7 text-white/90">
          Keep journaling and this review will synthesize mood, habits, people, rhythm,
          goals, and forecast signals once the range has enough entries.
        </p>
      </div>
    );
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1.4fr_.6fr]">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.12em] text-white/80">
          {headline.entry_count} entries - {headline.days_in_range} days
        </p>
        <h2 className="mt-3 font-display text-4xl leading-tight">
          Your mood shifted from {titleCase(headline.dominant_emotion_start)} to{" "}
          {titleCase(headline.dominant_emotion_end)} over the past{" "}
          {headline.days_in_range} days.
        </h2>
        <p className="mt-3 max-w-3xl leading-7 text-white/90">
          Sentiment moved {signed(headline.sentiment_delta)} from baseline{" "}
          {signed(headline.baseline_sentiment)} to current{" "}
          {signed(headline.current_sentiment)}. Recovery speed moved from{" "}
          {headline.recovery_speed_days_start.toFixed(2)} to{" "}
          {headline.recovery_speed_days_end.toFixed(2)} day(s).
        </p>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-white/85">
          {headline.growth_narrative}
        </p>
      </div>
      <div className="rounded-lg border border-white/30 bg-white/10 p-4">
        <p className="text-sm font-semibold uppercase tracking-[0.12em] text-white/80">
          Growth score
        </p>
        <p className="mt-2 font-display text-5xl">{percent(headline.growth_score)}</p>
        <ProgressBar value={headline.growth_score} />
      </div>
    </div>
  );
}

function WorkingSection({ rows }: { rows: DashboardStory["top_working"] }) {
  return (
    <Card tone="light">
      <SectionTitle icon={<HeartHandshake />} title="What's Working For You" />
      {rows.length ? (
        <div className="mt-4 space-y-3">
          {rows.map((habit) => (
            <div className="rounded-lg border border-line bg-canvas p-3" key={habit.habit}>
              <p className="font-semibold text-ink">
                {titleCase(habit.habit)} days trend {signed(habit.delta)} higher than other days.
              </p>
              <p className="mt-2 text-sm leading-6 text-body">{habit.explanation}</p>
              <EvidenceLine
                confidence={habit.confidence}
                detail={`${habit.mention_count} mentions - ${habit.streak_length} day streak`}
              />
            </div>
          ))}
        </div>
      ) : (
        <EmptyState
          detail="Not enough data yet to confirm a pattern - keep journaling."
          title="No confirmed support pattern"
        />
      )}
    </Card>
  );
}

function DrainingSection({ rows }: { rows: DashboardStory["top_draining"] }) {
  return (
    <Card tone="light">
      <SectionTitle icon={<TrendingDown />} title="What Tends To Weigh On You" />
      {rows.length ? (
        <div className="mt-4 space-y-3">
          {rows.map((trigger) => (
            <div className="rounded-lg border border-line bg-canvas p-3" key={trigger.topic}>
              <p className="font-semibold text-ink">
                {titleCase(trigger.topic)} appears with average sentiment{" "}
                {signed(trigger.avg_sentiment)}.
              </p>
              <p className="mt-2 text-sm leading-6 text-body">{trigger.explanation}</p>
              <EvidenceLine
                confidence={trigger.confidence}
                detail={`${trigger.frequency} mentions - ${titleCase(trigger.dominant_emotion)}`}
              />
            </div>
          ))}
        </div>
      ) : (
        <EmptyState
          detail="Not enough data yet to confirm a draining pattern - keep journaling."
          title="No confirmed pressure pattern"
        />
      )}
    </Card>
  );
}

function PeopleSection({ rows }: { rows: DashboardStory["people"] }) {
  return (
    <Card tone="light">
      <SectionTitle icon={<Users />} title="People In Your Life" />
      {rows.length ? (
        <div className="mt-4 space-y-3">
          {rows.map((person) => (
            <div className="rounded-lg border border-line bg-canvas p-3" key={person.person}>
              <p className="font-semibold text-ink">
                {person.person} shows up as {person.relationship_type} with sentiment{" "}
                {signed(person.avg_sentiment)}.
              </p>
              <p className="mt-2 text-sm leading-6 text-body">{person.explanation}</p>
              <EvidenceLine
                confidence={person.confidence}
                detail={`${person.mention_count} mentions - ${titleCase(person.sentiment_trend)}`}
              />
            </div>
          ))}
        </div>
      ) : (
        <EmptyState
          detail="Not enough repeated person mentions yet to summarize a relationship pattern."
          title="No recurring people pattern"
        />
      )}
    </Card>
  );
}

function LookingAhead({ story }: { story: DashboardStory | null }) {
  return (
    <Card tone="dark">
      <SectionTitle icon={<Brain />} inverse title="Looking Ahead" />
      {story ? (
        <div className="mt-4 space-y-5">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.12em] text-[#a09d96]">
              Mood direction
            </p>
            <h3 className="mt-1 font-display text-3xl">
              {titleCase(story.forecast.sentiment_forecast.direction)}
            </h3>
            <p className="mt-2 text-sm leading-6 text-[#d7d1c7]">
              {story.forecast.sentiment_forecast.explanation}
            </p>
            {story.forecast.sentiment_forecast.forecast_accuracy_note ? (
              <p className="mt-2 text-xs leading-5 text-[#d7d1c7]">
                {story.forecast.sentiment_forecast.forecast_accuracy_note}
              </p>
            ) : null}
          </div>

          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.12em] text-[#a09d96]">
              Stress pattern
            </p>
            <h3 className="mt-1 font-display text-3xl">
              {titleCase(story.forecast.burnout_risk.risk_level)}
            </h3>
            <p className="mt-2 text-sm leading-6 text-[#d7d1c7]">
              {story.forecast.burnout_risk.explanation}
            </p>
          </div>

          {story.goals.length ? (
            <div className="space-y-4">
              {story.goals.slice(0, 3).map((goal) => (
                <div key={goal.goal_keyword}>
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <p className="font-semibold">{titleCase(goal.goal_keyword)}</p>
                    <span className="text-sm text-[#d7d1c7]">
                      {percent(goal.estimated_progress)}
                    </span>
                  </div>
                  <ProgressBar value={goal.estimated_progress} />
                  <p className="mt-2 text-sm leading-6 text-[#d7d1c7]">{goal.explanation}</p>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState
              detail="Repeated goal mentions will appear here once there is enough history."
              title="No tracked goals yet"
            />
          )}

          {story.highlight_memory ? (
            <div className="rounded-lg border border-white/20 bg-white/10 p-3">
              <p className="inline-flex items-center gap-2 text-sm font-semibold">
                <Sparkles className="size-4" />
                Highlight memory
              </p>
              <p className="mt-2 text-sm leading-6 text-[#d7d1c7]">
                {story.highlight_memory.timestamp.slice(0, 10)} - {story.highlight_memory.title}
              </p>
            </div>
          ) : null}
        </div>
      ) : (
        <p className="mt-4 text-sm text-[#d7d1c7]">Loading forecast.</p>
      )}
    </Card>
  );
}

function SectionTitle({
  title,
  icon,
  inverse = false
}: {
  title: string;
  icon?: React.ReactNode;
  inverse?: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      {icon ? <span className="text-coral [&>svg]:size-5">{icon}</span> : null}
      <h2 className={`font-display text-3xl ${inverse ? "text-canvas" : "text-ink"}`}>
        {title}
      </h2>
    </div>
  );
}

function EvidenceLine({
  confidence,
  detail
}: {
  confidence: number;
  detail: string;
}) {
  return (
    <p className="mt-2 text-xs font-semibold uppercase tracking-[0.08em] text-muted">
      {detail} - Confidence {percent(confidence)}
    </p>
  );
}
