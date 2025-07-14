import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Combines class names using clsx and resolves Tailwind conflicts with twMerge.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Utility to simulate delay in async functions (e.g., mock API).
 * @param ms Number of milliseconds to wait
 */
export const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Truncates a long string and appends "..." if it exceeds the limit.
 * @param text Input string
 * @param limit Max characters (default 120)
 */
export const truncate = (text: string, limit = 120): string =>
  text.length > limit ? text.slice(0, limit) + "..." : text;

/**
 * Maps technology tags to Tailwind background color classes.
 * Customize as needed based on your tag set.
 */
export const getTagColor = (tag: string): string => {
  const colors: Record<string, string> = {
    React: "bg-blue-500",
    TypeScript: "bg-cyan-600",
    "Node.js": "bg-green-600",
    Figma: "bg-pink-500",
    Firebase: "bg-orange-500",
    Python: "bg-yellow-500",
    TailwindCSS: "bg-sky-500",
    MongoDB: "bg-emerald-600",
    Express: "bg-gray-700",
    Redux: "bg-purple-500",
    HTML: "bg-orange-600",
    CSS: "bg-blue-400",
    JavaScript: "bg-yellow-400",
    "UI/UX": "bg-indigo-500",
  };

  return colors[tag] || "bg-gray-500";
};
