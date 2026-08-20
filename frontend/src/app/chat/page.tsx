"use client";

import { AlertTriangle, ChevronDown, FileText, Send } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/Card";
import { useDemoMode } from "@/components/DemoModeProvider";
import {
  type ChatMessage,
  type ChatResponse,
  getDemoChatHistory,
  sendChat
} from "@/lib/api";
import { percent, titleCase } from "@/lib/format";

type ThreadItem = ChatMessage & {
  emotion?: ChatResponse["emotion"];
  memoryReplay?: Record<string, unknown> | null;
  crisis?: ChatResponse["crisis"];
  retrieved?: Array<Record<string, unknown>>;
  prompt?: string | null;
};

export default function ChatPage() {
  const { mode } = useDemoMode();
  const [thread, setThread] = useState<ThreadItem[]>([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const isDemo = mode === "demo";

  useEffect(() => {
    let cancelled = false;
    async function loadDemo() {
      if (!isDemo) {
        setThread([]);
        setError(null);
        return;
      }
      setPending(true);
      setError(null);
      try {
        const demo = await getDemoChatHistory();
        if (cancelled) return;
        setThread(
          demo.messages.map((message) => ({
            role: message.role,
            content: message.content,
            emotion: message.emotion,
            memoryReplay: message.memory_replay,
            crisis: message.crisis,
            retrieved: message.retrieved_memories,
            prompt: message.prompt
          }))
        );
      } catch (exc) {
        if (!cancelled) {
          setError(exc instanceof Error ? exc.message : "Demo chat request failed.");
        }
      } finally {
        if (!cancelled) setPending(false);
      }
    }
    void loadDemo();
    return () => {
      cancelled = true;
    };
  }, [isDemo]);

  async function submit() {
    const text = input.trim();
    if (!text || pending || isDemo) return;

    const visibleHistory = thread.map(({ role, content }) => ({ role, content }));
    const userItem: ThreadItem = { role: "user", content: text };
    setThread((items) => [...items, userItem]);
    setInput("");
    setPending(true);
    setError(null);

    try {
      const result = await sendChat(text, visibleHistory);
      setThread((items) => [
        ...items,
        {
          role: "assistant",
          content: result.response,
          emotion: result.emotion,
          memoryReplay: result.memory_replay,
          crisis: result.crisis,
          retrieved: result.retrieved_memories,
          prompt: result.prompt
        }
      ]);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Chat request failed.");
    } finally {
      setPending(false);
    }
  }

  async function readFile(file: File) {
    const text = await file.text();
    setInput((current) => `${current}${current ? "\n\n" : ""}${text.slice(0, 5000)}`);
  }

  const latestAssistant = [...thread].reverse().find((item) => item.role === "assistant");
  const alerts = latestAssistant?.crisis?.flagged
    ? [latestAssistant.crisis.resources ?? "Crisis language was detected."]
    : [];

  return (
    <AppShell>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
        <section className="min-h-[calc(100vh-9rem)] rounded-2xl border border-line bg-cream p-4 shadow-soft">
          <div className="mb-4 flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.12em] text-coral">
                Journal
              </p>
              <h1 className="font-display text-4xl leading-tight text-ink">
                A steady place to write
              </h1>
            </div>
            <button
              className="rounded-lg border border-line bg-[#fffdf8] px-4 py-2 text-sm font-semibold text-ink"
              disabled={isDemo}
              onClick={() => setThread([])}
            >
              {isDemo ? "Read-only demo" : "Reset"}
            </button>
          </div>

          {alerts.map((alert) => (
            <div
              className="mb-4 flex gap-3 rounded-xl border border-[#c64545] bg-[#fff1ed] p-4 text-sm text-ink"
              key={alert}
            >
              <AlertTriangle className="mt-0.5 size-5 shrink-0 text-[#c64545]" />
              <p>{alert}</p>
            </div>
          ))}

          <div className="h-[52vh] overflow-y-auto rounded-xl border border-line bg-[#fffdf8] p-4">
            {thread.length === 0 ? (
              <div className="grid h-full place-items-center text-center">
                <div>
                  <p className="font-display text-2xl text-ink">
                    Start with what happened today.
                  </p>
                  <p className="mt-2 max-w-md text-sm leading-6 text-muted">
                    The response will use your memory, emotion signal, and deterministic
                    reflection engines without changing the backend pipeline.
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {thread.map((item, index) => (
                  <article
                    className={`rounded-xl border p-4 ${
                      item.role === "user"
                        ? "ml-auto max-w-[82%] border-coral bg-coral text-white"
                        : "mr-auto max-w-[88%] border-line bg-canvas text-ink"
                    }`}
                    key={`${item.role}-${index}`}
                  >
                    <p className="whitespace-pre-wrap leading-7">{item.content}</p>
                    {item.emotion ? <EmotionStrip emotion={item.emotion} /> : null}
                    {item.memoryReplay ? (
                      <details className="mt-3 rounded-lg border border-line bg-[#fffdf8] p-3 text-sm text-ink">
                        <summary className="cursor-pointer font-semibold">
                          Memory replay
                        </summary>
                        <pre className="mt-2 whitespace-pre-wrap font-mono text-xs text-body">
                          {JSON.stringify(item.memoryReplay, null, 2)}
                        </pre>
                      </details>
                    ) : null}
                    {showDebug && item.role === "assistant" ? (
                      <details className="mt-3 rounded-lg border border-line bg-[#fffdf8] p-3 text-sm text-ink">
                        <summary className="cursor-pointer font-semibold">
                          Retrieved context and prompt
                        </summary>
                        <pre className="mt-2 whitespace-pre-wrap font-mono text-xs text-body">
                          {JSON.stringify(
                            { retrieved: item.retrieved, prompt: item.prompt },
                            null,
                            2
                          )}
                        </pre>
                      </details>
                    ) : null}
                  </article>
                ))}
              </div>
            )}
          </div>

          {error ? (
            <p className="mt-3 rounded-lg border border-[#c64545] bg-[#fff1ed] px-3 py-2 text-sm text-ink">
              {error}
            </p>
          ) : null}

          <div className="mt-4 rounded-xl border border-line bg-[#fffdf8] p-3">
            {isDemo ? (
              <div className="rounded-lg border border-line bg-canvas p-4 text-sm leading-6 text-body">
                This is a frozen demo transcript. Use the banner CTA to switch to live
                mode and write a new entry through the real pipeline.
              </div>
            ) : null}
            <textarea
              className="min-h-28 w-full resize-none bg-transparent p-2 text-base leading-7 text-ink outline-none placeholder:text-muted disabled:opacity-60"
              disabled={isDemo}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                  void submit();
                }
              }}
              placeholder="Write your current journal entry..."
              value={input}
            />
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3">
              <div className="flex items-center gap-3">
                <input
                  accept=".txt,.md,.csv"
                  className="hidden"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void readFile(file);
                  }}
                  ref={fileRef}
                  type="file"
                />
                <button
                  className="inline-flex items-center gap-2 rounded-lg border border-line px-3 py-2 text-sm font-semibold text-ink disabled:opacity-60"
                  disabled={isDemo}
                  onClick={() => fileRef.current?.click()}
                >
                  <FileText className="size-4" />
                  Attach text
                </button>
                <label className="inline-flex items-center gap-2 text-sm font-semibold text-ink">
                  <input
                    checked={showDebug}
                    onChange={(event) => setShowDebug(event.target.checked)}
                    type="checkbox"
                  />
                  Debug context
                </label>
              </div>
              <button
                className="inline-flex items-center gap-2 rounded-lg bg-coral px-4 py-2 text-sm font-semibold text-white hover:bg-coralDark disabled:opacity-60"
                disabled={pending || !input.trim() || isDemo}
                onClick={() => void submit()}
              >
                <Send className="size-4" />
                {pending ? "Reflecting" : "Send"}
              </button>
            </div>
          </div>
        </section>

        <aside className="space-y-4">
          <Card tone="dark">
            <p className="text-sm font-semibold uppercase tracking-[0.12em] text-[#a09d96]">
              Current signal
            </p>
            <h2 className="mt-2 font-display text-3xl">
              {latestAssistant?.emotion
                ? titleCase(latestAssistant.emotion.emotion)
                : "Not read yet"}
            </h2>
            <p className="mt-2 text-sm leading-6 text-[#d7d1c7]">
              {thread.length} session messages. The backend keeps the crisis check first
              and the LLM call last.
            </p>
          </Card>
          <Card>
            <div className="flex items-center justify-between">
              <h2 className="font-display text-2xl">Secondary emotions</h2>
              <ChevronDown className="size-5 text-muted" />
            </div>
            {latestAssistant?.emotion?.all_emotions?.length ? (
              <div className="mt-4 space-y-3">
                {latestAssistant.emotion.all_emotions.slice(0, 4).map((item) => (
                  <div key={item.emotion}>
                    <div className="mb-1 flex justify-between text-sm font-semibold">
                      <span>{titleCase(item.emotion)}</span>
                      <span>{percent(item.score)}</span>
                    </div>
                    <div className="h-2 rounded-full bg-[#d9c9b5]">
                      <div
                        className="h-2 rounded-full bg-coral"
                        style={{ width: `${Math.round(item.score * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-3 text-sm leading-6 text-muted">
                Send an entry to see the emotional spread.
              </p>
            )}
          </Card>
        </aside>
      </div>
    </AppShell>
  );
}

function EmotionStrip({ emotion }: { emotion: ChatResponse["emotion"] }) {
  return (
    <div className="mt-3 flex flex-wrap gap-2 text-xs font-semibold">
      <span className="rounded-full border border-line bg-[#fffdf8] px-3 py-1 text-ink">
        Mood: {titleCase(emotion.emotion)}
      </span>
      <span className="rounded-full border border-line bg-[#fffdf8] px-3 py-1 text-ink">
        Confidence: {percent(emotion.confidence)}
      </span>
    </div>
  );
}
