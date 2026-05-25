import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VG Landscape Visualizer",
  description: "See what your Martha's Vineyard property could look like before any work begins.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
