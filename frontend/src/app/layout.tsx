import type { Metadata } from "next";

import { DemoModeProvider } from "@/components/DemoModeProvider";

import "./globals.css";

export const metadata: Metadata = {
  title: "Mind Shift AI",
  description: "AI journaling assistant with memory and reflection analytics"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <DemoModeProvider>{children}</DemoModeProvider>
      </body>
    </html>
  );
}
