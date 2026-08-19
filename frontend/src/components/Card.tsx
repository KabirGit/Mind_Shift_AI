import clsx from "clsx";

type CardProps = {
  children: React.ReactNode;
  className?: string;
  tone?: "cream" | "light" | "dark" | "coral";
};

export function Card({ children, className, tone = "cream" }: CardProps) {
  return (
    <section
      className={clsx(
        "rounded-xl border p-5 shadow-soft",
        tone === "cream" && "border-[#c9b8a5] bg-card text-ink",
        tone === "light" && "border-line bg-[#fffdf8] text-ink",
        tone === "dark" && "border-night bg-night text-canvas",
        tone === "coral" && "border-coral bg-coral text-white",
        className
      )}
    >
      {children}
    </section>
  );
}
