const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/$/, "");

const fallbackApiUrl =
  process.env.NODE_ENV === "production"
    ? "https://ai-reflection-intelligence-platform-eei6.onrender.com"
    : "http://127.0.0.1:8501";

export const API_BASE_URL = configuredApiUrl || fallbackApiUrl;

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
  explanation: string;
};

export type BurnoutRisk = {
  risk_level: string;
  score: number;
  contributing_factors: string[];
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

export function getDashboardSummary(range: string): Promise<DashboardSummary> {
  return apiFetch<DashboardSummary>(
    `/api/dashboard/summary?range=${encodeURIComponent(range)}`
  );
}

export function getGoals(): Promise<{ goals: GoalProgress[] }> {
  return apiFetch<{ goals: GoalProgress[] }>("/api/dashboard/goals");
}

export function getPredictions(): Promise<{
  sentiment_forecast: SentimentForecast;
  burnout_risk: BurnoutRisk;
}> {
  return apiFetch("/api/dashboard/predictions");
}

export function getTimeline(): Promise<{ events: TimelineEvent[] }> {
  return apiFetch<{ events: TimelineEvent[] }>("/api/dashboard/timeline");
}

export function getGrowth(): Promise<{
  snapshots: GrowthSnapshot[];
  narrative: string;
}> {
  return apiFetch("/api/dashboard/growth");
}

export function queryGraph(node: string): Promise<GraphQuery> {
  return apiFetch<GraphQuery>(`/api/graph/query?node=${encodeURIComponent(node)}`);
}

export function getPeopleGraph(): Promise<PeopleGraph> {
  return apiFetch<PeopleGraph>("/api/graph/people");
}

export function getDiagnostics(): Promise<Diagnostics> {
  return apiFetch<Diagnostics>("/api/diagnostics");
}

export function weeklyReportUrl(): string {
  return `${API_BASE_URL}/api/report/weekly`;
}
