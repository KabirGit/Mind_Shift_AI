export function moodScore(sentiment: number): number {
  return Math.round((Math.max(-1, Math.min(1, sentiment)) + 1) * 50);
}

export function sentimentLabel(sentiment: number): string {
  if (sentiment >= 0.55) return "strongly positive";
  if (sentiment >= 0.2) return "positive";
  if (sentiment > -0.2) return "mixed";
  if (sentiment > -0.55) return "heavy";
  return "very heavy";
}

export function sentimentSummary(sentiment: number): string {
  return `${sentimentLabel(sentiment)} mood score ${moodScore(sentiment)}%`;
}

export function deltaPlain(delta: number): string {
  const direction = delta > 0.05 ? "lighter" : delta < -0.05 ? "heavier" : "steady";
  const magnitude = Math.abs(delta);
  if (direction === "steady") return "about steady";
  if (magnitude < 0.15) return `slightly ${direction}`;
  if (magnitude < 0.35) return `noticeably ${direction}`;
  return `strongly ${direction}`;
}
