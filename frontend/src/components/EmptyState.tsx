type EmptyStateProps = {
  title: string;
  detail: string;
};

export function EmptyState({ title, detail }: EmptyStateProps) {
  return (
    <div className="rounded-xl border border-dashed border-line bg-[#fffdf8] p-6 text-center">
      <p className="font-semibold text-ink">{title}</p>
      <p className="mt-1 text-sm text-muted">{detail}</p>
    </div>
  );
}
