type ProgressBarProps = {
  value: number;
};

export function ProgressBar({ value }: ProgressBarProps) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div className="h-2 overflow-hidden rounded-full bg-[#d9c9b5]">
      <div className="h-full rounded-full bg-coral" style={{ width: `${pct}%` }} />
    </div>
  );
}
