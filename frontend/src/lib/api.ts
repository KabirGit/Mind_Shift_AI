const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/$/, "");

const fallbackApiUrl =
  process.env.NODE_ENV === "production"
    ? "https://ai-reflection-intelligence-platform-eei6.onrender.com"
    : "http://127.0.0.1:8501";

export const API_BASE_URL = configuredApiUrl || fallbackApiUrl;
export type EndpointMode = "demo" | "live";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type EmotionResult = {
  emotion: string;
  confidence: number;
  all_emotions?: Array<{ emotion: string; score: number }>;
};

export type ChatResponse = {
  emotion: EmotionResult;
  response: string;
  memory_replay?: Record<string, unknown> | null;
  crisis: {
    flagged: boolean;
    matched_terms: string[];
    resources?: string;
  };
  retrieved_memories: Array<Record<string, unknown>>;
  packet?: {
    insights: string[];
    reflection_prompts: string[];
    proactive_alerts: string[];
    memory_replay?: Record<string, unknown> | null;
  } | null;
  prompt?: string | null;
};

export type EmotionPoint = {
  date: string;
  emotion: string;
  count: number;
};

export type TriggerStat = {
  topic: string;
  frequency: number;
  avg_sentiment: number;
  dominant_emotion: string;
  trend: string;
  confidence: number;
  explanation: string;
};

export type HabitCorrelation = {
  habit: string;
  mention_count: number;
  avg_sentiment_when_mentioned: number;
  avg_sentiment_other_days: number;
  delta: number;
  correlation_label: string;
  streak_length: number;
  consistency_percentage: number;
  confidence: number;
  explanation: string;
};

export type DemoChatHistory = {
  mode: "demo";
  persona: string;
  generated_with_live_llm: boolean;
  messages: Array<{
    role: "user" | "assistant";
    content: string;
    emotion?: EmotionResult;
    memory_replay?: Record<string, unknown> | null;
    crisis?: ChatResponse["crisis"];
    retrieved_memories?: Array<Record<string, unknown>>;
    prompt?: string | null;
  }>;
};

export type RelationshipProfile = {
  person: string;
  mention_count: number;
  avg_sentiment: number;
  dominant_emotion: string;
  last_mentioned: string;
  trend: string;
  relationship_type: string;
  relationship_type_confidence: number;
  relationship_type_ambiguity: string;
  closeness_score: number;
  sentiment_trend: string;
  co_mentioned_with: Record<string, number>;
  confidence: number;
  explanation: string;
};

export type DashboardSummary = {
  range: string;
  lookback_days: number;
  emotion_over_time: EmotionPoint[];
  recurring_topics: Record<string, number>;
  triggers: TriggerStat[];
  habits: HabitCorrelation[];
  relationships: RelationshipProfile[];
  insights: string[];
};

export type DashboardHeadline = {
  baseline_sentiment: number;
  current_sentiment: number;
  sentiment_delta: number;
  dominant_emotion_start: string;
  dominant_emotion_end: string;
  recovery_speed_days_start: number;
  recovery_speed_days_end: number;
  entry_count: number;
  days_in_range: number;
  growth_score: number;
  growth_narrative: string;
  has_sufficient_data: boolean;
  minimum_entry_count: number;
};

export type GoalProgress = {
  goal_keyword: string;
  first_mentioned: string;
  last_mentioned: string;
  mention_count: number;
  avg_sentiment: number;
  sentiment_trend: string;
  phase: string;
  estimated_progress: number;
  confidence: number;
  explanation: string;
};

export type SentimentForecast = {
  horizon_days: number;
  predicted_sentiment: number;
  direction: string;
  confidence: number;
  forecast_accuracy_note?: string;
  explanation: string;
};

export type BurnoutRisk = {
  risk_level: string;
  score: number;
  contributing_factors: string[];
  confidence: number;
  explanation: string;
};

export type TemporalPattern = {
  topic: string;
  peak_day_of_week: string;
  peak_time_of_day?: string | null;
  day_time_crossing?: string | null;
  day_time_sample_size: number;
  peak_day_avg_sentiment: number;
  peak_time_avg_sentiment?: number | null;
  baseline_avg_sentiment: number;
  delta: number;
  confidence: number;
  explanation: string;
};

export type TimelineEvent = {
  timestamp: string;
  title: string;
  description: string;
  emotion: string;
  sentiment: number;
  baseline_sentiment?: number | null;
  primary_person?: string | null;
  significance_score: number;
  event_type: string;
};

export type GrowthSnapshot = {
  period_label: string;
  entry_count: number;
  avg_sentiment: number;
  dominant_emotion: string;
  top_topic: string;
  snapshot_date: string;
};

export type WeeklyBucket = {
  label: string;
  avg_sentiment: number;
  dominant_emotion: string;
  top_topic: string;
  entry_count: number;
};

export type DashboardStory = {
  range: string;
  lookback_days: number;
  headline: DashboardHeadline;
  top_working: HabitCorrelation[];
  top_draining: TriggerStat[];
  people: RelationshipProfile[];
  rhythm?: TemporalPattern | null;
  weekly_buckets: WeeklyBucket[];
  forecast: {
    sentiment_forecast: SentimentForecast;
    burnout_risk: BurnoutRisk;
  };
  goals: GoalProgress[];
  highlight_memory?: TimelineEvent | null;
  thresholds: {
    min_insight_confidence: number;
    min_mention_count: number;
    min_entry_count: number;
  };
};

export type GraphQuery = {
  node: string;
  summary: string;
  neighbors: string[];
  edge_data: Array<Record<string, unknown>>;
};

export type PeopleGraphNode = {
  id: string;
  label: string;
  type: "user" | "person";
  relationship_type: string;
  mention_count: number;
};

export type PeopleGraphEdge = {
  source: string;
  target: string;
  sentiment: number;
  weight: number;
  closeness_score: number;
};

export type PeopleGraph = {
  nodes: PeopleGraphNode[];
  edges: PeopleGraphEdge[];
};

export type Diagnostics = {
  retrieval_precision: Record<string, unknown>;
  emotion_confidence: Record<string, unknown>;
  latency: Record<string, unknown>;
};

export function sendChat(
  text: string,
  chatHistory: ChatMessage[],
  topK = 3
): Promise<ChatResponse> {
  return apiFetch<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify({ text, chat_history: chatHistory, top_k: topK })
  });
}

function demoPrefix(mode: EndpointMode): string {
  return mode === "demo" ? "/api/demo" : "/api";
}

export function getDashboardSummary(
  range: string,
  mode: EndpointMode = "live"
): Promise<DashboardSummary> {
  return apiFetch<DashboardSummary>(
    `${demoPrefix(mode)}/dashboard/summary?range=${encodeURIComponent(range)}`
  );
}

export function getDashboardStory(
  range: string,
  mode: EndpointMode = "live"
): Promise<DashboardStory> {
  return apiFetch<DashboardStory>(
    `${demoPrefix(mode)}/dashboard/story?range=${encodeURIComponent(range)}`
  );
}

export function getGoals(mode: EndpointMode = "live"): Promise<{ goals: GoalProgress[] }> {
  return apiFetch<{ goals: GoalProgress[] }>(`${demoPrefix(mode)}/dashboard/goals`);
}

export function getPredictions(mode: EndpointMode = "live"): Promise<{
  sentiment_forecast: SentimentForecast;
  burnout_risk: BurnoutRisk;
}> {
  return apiFetch(`${demoPrefix(mode)}/dashboard/predictions`);
}

export function getTimeline(mode: EndpointMode = "live"): Promise<{ events: TimelineEvent[] }> {
  return apiFetch<{ events: TimelineEvent[] }>(`${demoPrefix(mode)}/dashboard/timeline`);
}

export function getGrowth(mode: EndpointMode = "live"): Promise<{
  snapshots: GrowthSnapshot[];
  narrative: string;
}> {
  return apiFetch(`${demoPrefix(mode)}/dashboard/growth`);
}

export function queryGraph(
  node: string,
  mode: EndpointMode = "live"
): Promise<GraphQuery> {
  return apiFetch<GraphQuery>(
    `${demoPrefix(mode)}/graph/query?node=${encodeURIComponent(node)}`
  );
}

export function getPeopleGraph(mode: EndpointMode = "live"): Promise<PeopleGraph> {
  return apiFetch<PeopleGraph>(`${demoPrefix(mode)}/graph/people`);
}

export function getDiagnostics(mode: EndpointMode = "live"): Promise<Diagnostics> {
  return apiFetch<Diagnostics>(`${demoPrefix(mode)}/diagnostics`);
}

export function getDemoChatHistory(): Promise<DemoChatHistory> {
  return apiFetch<DemoChatHistory>("/api/demo/chat-history");
}

export function weeklyReportUrl(): string {
  return `${API_BASE_URL}/api/report/weekly`;
}
