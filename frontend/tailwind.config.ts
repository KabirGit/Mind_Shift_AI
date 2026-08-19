import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#faf9f5",
        cream: "#f0e7da",
        card: "#e6dac8",
        ink: "#141413",
        body: "#3d3d3a",
        muted: "#6c6a64",
        coral: "#cc785c",
        coralDark: "#a9583e",
        night: "#181715",
        nightLift: "#252320",
        line: "#d2c3b2",
        teal: "#5db8a6",
        amber: "#e8a55a"
      },
      fontFamily: {
        display: ["Georgia", "Times New Roman", "serif"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"]
      },
      boxShadow: {
        soft: "0 18px 45px rgba(20, 20, 19, 0.08)"
      }
    }
  },
  plugins: []
};

export default config;
