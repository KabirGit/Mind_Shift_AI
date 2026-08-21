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
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/Card";
import { useDemoMode } from "@/components/DemoModeProvider";
import { EmptyState } from "@/components/EmptyState";
import { ProgressBar } from "@/components/ProgressBar";
import { RelationshipGraph } from "@/components/RelationshipGraph";
import {
  type DashboardStory,
  type GraphQuery,
  type PeopleGraph,
  type TimelineEvent,
  getDashboardStory,
  getPeopleGraph,
  getTimeline,
  queryGraph,
  weeklyReportUrl
} from "@/lib/api";
import { percent, titleCase } from "@/lib/format";
import {
  deltaPlain,
  moodScore,
  sentimentLabel,
  sentimentSummary
} from "@/lib/presentation";

const RANGES = ["Last 7 days", "Last 30 days", "All time"];

export default function DashboardPage() {
  const { mode } = useDemoMode();
  const [range, setRange] = useState(RANGES[1]);
  const [story, setStory] = useState<DashboardStory | null>(null);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
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
        const [storyData, timelineData, peopleGraphData] = await Promise.all([
          getDashboardStory(range, mode),
          getTimeline(mode),
          getPeopleGraph(mode)
        ]);
        if (cancelled) return;
        setStory(storyData);
        setTimeline(timelineData.events);
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
  }, [range, mode]);

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
      sentiment_label: bucket.mood_label ?? sentimentLabel(bucket.avg_sentiment),
      mood_score: bucket.mood_score ?? moodScore(bucket.avg_sentiment),
      topic: titleCase(bucket.top_topic)
    }));
  }, [story]);

  async function runGraphSearch() {
    const node = graphNode.trim();
    if (!node) return;
    setGraph(await queryGraph(node, mode));
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

      {story ? (
        <Card className="mt-6" tone="light">
          <SectionTitle icon={<CalendarDays />} title="Your Rhythm" />
          <p className="mt-3 text-lg font-semibold text-ink">
            {recoverySentence(story)}
          </p>
          <p className="mt-2 text-sm leading-6 text-body">
            {story.rhythm
              ? rhythmSentence(story)
              : "No day or time pattern cleared the confidence gate for this range."}
          </p>
          <EvidenceLine
            confidence={story.rhythm?.confidence ?? 1}
            detail={`Recovery speed ${story.headline.recovery_speed_days_start.toFixed(2)} to ${story.headline.recovery_speed_days_end.toFixed(2)} day(s)`}
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
                  <YAxis domain={[0, 100]} tick={{ fill: "#6c6a64", fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="mood_score" fill="#5db8a6" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {weekRows.map((bucket) => (
                  <div className="rounded-lg border border-line bg-canvas p-3" key={bucket.label}>
                    <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">
                      {bucket.label}
                    </p>
                    <p className="mt-1 font-semibold text-ink">
                      {bucket.mood_score}% - {titleCase(bucket.sentiment_label)}
                    </p>
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
        <div className="mt-5">
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
        </div>

        <Card className="mt-6" tone="light">
          <SectionTitle title="Full timeline" />
          {timeline.length ? (
            <TimelineMoodChart events={timeline} />
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
  const baselineMood = headline.baseline_mood_score ?? moodScore(headline.baseline_sentiment);
  const currentMood = headline.current_mood_score ?? moodScore(headline.current_sentiment);
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
          Your month moved from {titleCase(headline.dominant_emotion_start)} to{" "}
          {titleCase(headline.dominant_emotion_end)}, with mood score moving from{" "}
          {baselineMood}% to {currentMood}%.
        </h2>
        <p className="mt-3 max-w-3xl leading-7 text-white/90">
          {headline.sentiment_delta_summary ??
            `Mood score moved from ${baselineMood}% to ${currentMood}%.`} Recovery
          speed moved from{" "}
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
                {habit.delta_summary ??
                  `${titleCase(habit.habit)} days look ${deltaPlain(habit.delta)}.`}
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
                {titleCase(trigger.display_label ?? trigger.topic)} appears as a{" "}
                {trigger.sentiment_summary ?? sentimentSummary(trigger.avg_sentiment)} pattern.
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
                {person.impact_summary ??
                  `${person.person} appears across ${person.mention_count} entries with ${trendPhrase(
                    person.sentiment_trend
                  )}.`}
              </p>
              <p className="mt-2 text-sm leading-6 text-body">{person.explanation}</p>
              <EvidenceLine
                confidence={person.confidence}
                detail={`${person.mention_count} mentions - ${titleCase(
                  person.relationship_type === "unknown" ? "connection" : person.relationship_type
                )} - ${titleCase(person.sentiment_trend)}`}
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
              Predicted mood score:{" "}
              {story.forecast.sentiment_forecast.predicted_mood_score ??
                moodScore(story.forecast.sentiment_forecast.predicted_sentiment)}
              %. {forecastPlainSentence(story.forecast.sentiment_forecast.direction)}
            </p>
            <p className="mt-2 text-xs leading-5 text-[#d7d1c7]">
              {story.forecast.sentiment_forecast.metric_note}
            </p>
            {story.forecast.sentiment_forecast.forecast_accuracy_note ? (
              <details className="mt-2 text-xs leading-5 text-[#d7d1c7]">
                <summary className="cursor-pointer font-semibold">Forecast details</summary>
                <p className="mt-1">{story.forecast.sentiment_forecast.explanation}</p>
                <p className="mt-1">
                  {story.forecast.sentiment_forecast.forecast_accuracy_note}
                </p>
              </details>
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
            <p className="mt-2 text-xs leading-5 text-[#d7d1c7]">
              {story.forecast.burnout_risk.metric_note}
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
                  <p className="mt-1 text-xs leading-5 text-[#d7d1c7]">
                    {goal.metric_note}
                  </p>
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

function TimelineMoodChart({ events }: { events: TimelineEvent[] }) {
  const points = events.map((event, index) => ({
    ...event,
    point: index + 1,
    date: event.timestamp.slice(5, 10),
    mood_score: event.mood_score ?? moodScore(event.sentiment)
  }));
  const labels = points
    .filter((event) => event.event_type !== "normal")
    .sort((a, b) => b.significance_score - a.significance_score)
    .slice(0, 5);

  return (
    <div className="mt-4">
      <ResponsiveContainer height={300} width="100%">
        <LineChart data={points} margin={{ bottom: 16, left: 24, right: 88, top: 52 }}>
          <CartesianGrid stroke="#d2c3b2" strokeDasharray="3 3" />
          <XAxis
            dataKey="point"
            padding={{ left: 32, right: 48 }}
            tick={{ fill: "#6c6a64", fontSize: 12 }}
            tickFormatter={(value) => points[Number(value) - 1]?.date ?? ""}
          />
          <YAxis domain={[-5, 105]} tick={{ fill: "#6c6a64", fontSize: 12 }} />
          <Tooltip
            formatter={(value) => [`${value}%`, "Mood score"]}
            labelFormatter={(value) => {
              const point = points[Number(value) - 1];
              return point ? `${point.timestamp.slice(0, 10)} - ${point.title}` : "";
            }}
          />
          <Line
            dataKey="mood_score"
            dot={false}
            stroke="#cc785c"
            strokeWidth={3}
            type="monotone"
          />
          {labels.map((event) => (
            <ReferenceDot
              fill={event.event_type === "positive_peak" ? "#3d915e" : "#c64545"}
              key={event.timestamp}
              r={5}
              ifOverflow="visible"
              stroke="#fffdf8"
              strokeWidth={2}
              x={event.point}
              y={event.mood_score}
              label={{
                fill: "#292722",
                fontSize: 11,
                fontWeight: 700,
                position: "top",
                value: titleCase(event.event_type.replace(/_/g, " "))
              }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        {labels.map((event) => (
          <p className="rounded-lg border border-line bg-canvas p-3 text-sm text-body" key={event.timestamp}>
            <span className="font-semibold text-ink">{event.timestamp.slice(0, 10)}</span>{" "}
            {event.title}: {event.description}
          </p>
        ))}
      </div>
    </div>
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

function recoverySentence(story: DashboardStory): string {
  const start = story.headline.recovery_speed_days_start;
  const end = story.headline.recovery_speed_days_end;
  if (end + 0.05 < start) {
    return `Recovery after heavier entries became faster, moving from ${start.toFixed(2)} to ${end.toFixed(2)} day(s).`;
  }
  if (end > start + 0.05) {
    return `Recovery after heavier entries slowed, moving from ${start.toFixed(2)} to ${end.toFixed(2)} day(s).`;
  }
  return `Recovery after heavier entries stayed about steady at ${end.toFixed(2)} day(s).`;
}

function rhythmSentence(story: DashboardStory): string {
  if (!story.rhythm) return "";
  const feltDelta = deltaPlain(story.rhythm.delta);
  return `${titleCase(story.rhythm.topic)} has a ${feltDelta} cadence in this range; the exact day/time signal is treated as supporting detail rather than the main takeaway.`;
}

function forecastPlainSentence(direction: string): string {
  if (direction === "improving") {
    return "Recent entries point to a lighter near-term mood direction.";
  }
  if (direction === "declining") {
    return "Recent entries point to a heavier near-term mood direction.";
  }
  return "Recent entries point to a steady near-term mood direction.";
}

function trendPhrase(trend: string): string {
  const article = /^[aeiou]/i.test(trend) ? "an" : "a";
  return `${article} ${trend} trend`;
}
