export function titleCase(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\w\S*/g, (word) => word[0].toUpperCase() + word.slice(1));
}

export function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}
