"use client";

import { useState, useRef, useCallback, useEffect } from "react";

const STYLES = [
  { id: "natural_meadow", label: "Natural meadow", desc: "Wildflower fields with native grasses" },
  { id: "classic_lawn",   label: "Classic lawn", desc: "Manicured green lawn with edged beds" },
  { id: "coastal_garden", label: "Coastal garden", desc: "Beach grasses, dune planting, weathered fences" },
  { id: "formal_garden",  label: "Formal garden", desc: "Geometric beds, boxwood hedges, symmetry" },
  { id: "shade_garden",   label: "Shade garden", desc: "Hostas, ferns, moss, dappled light" },
  { id: "pollinator",     label: "Pollinator garden", desc: "Native flowering plants for bees and butterflies" },
];

const VARIATIONS = 3;

interface Result { url: string; style: string; prompt: string; }

export default function Home() {
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [styleId, setStyleId] = useState<string>(STYLES[0].id);
  const [extraPrompt, setExtraPrompt] = useState("");
  const [dragging, setDragging] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<Result[]>([]);

  const setUploadedFile = useCallback((f: File) => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setError(null);
    setResults([]);
  }, [preview]);

  // Revoke any outstanding preview URL on unmount.
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f?.type.startsWith("image/")) setUploadedFile(f);
  };

  const submit = async () => {
    if (!file) return;
    setSubmitting(true);
    setError(null);
    setResults([]);
    try {
      const fd = new FormData();
      fd.append("image", file);
      fd.append("styleId", styleId);
      fd.append("extraPrompt", extraPrompt);
      fd.append("variations", String(VARIATIONS));
      const res = await fetch("/api/visualize", { method: "POST", body: fd });
      if (!res.ok) {
        // Try to parse the JSON body for a readable error; fall back to status text.
        let message = res.statusText;
        try {
          const body = (await res.json()) as { error?: string };
          if (body?.error) message = body.error;
        } catch {
          // body wasn't JSON
        }
        throw new Error(message);
      }
      const data = (await res.json()) as { results: Result[] };
      setResults(data.results);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed.");
    } finally {
      setSubmitting(false);
    }
  };

  const selectedStyle = STYLES.find((s) => s.id === styleId)!;

  return (
    <div className="page">
      <h1 className="title">VG Landscape Visualizer</h1>
      <p className="subtitle">
        See what your Martha&apos;s Vineyard property could look like before any work begins.
        Upload a yard photo, pick a style, generate variations.
      </p>

      <div className="layout">
        <div className="panel">
          <h2>Source photo</h2>
          <div
            className={`dropzone ${dragging ? "drag" : ""}`}
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            {preview ? (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img src={preview} alt="" />
            ) : (
              <>
                <p><strong>Drop a yard photo here</strong>, or click to browse.</p>
                <p className="hint">JPG / PNG, up to 25 MB</p>
              </>
            )}
            <input
              ref={fileRef}
              type="file"
              accept="image/jpeg,image/png"
              style={{ display: "none" }}
              onChange={(e) => { const f = e.target.files?.[0]; if (f) setUploadedFile(f); e.target.value = ""; }}
            />
          </div>

          <div className="field" style={{ marginTop: 20 }}>
            <label>Style</label>
            <div className="styles">
              {STYLES.map((s) => (
                <button
                  type="button"
                  key={s.id}
                  className={`style${s.id === styleId ? " active" : ""}`}
                  onClick={() => setStyleId(s.id)}
                >
                  <strong>{s.label}</strong>
                  <span>{s.desc}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="field">
            <label>Additional instructions (optional)</label>
            <textarea
              value={extraPrompt}
              onChange={(e) => setExtraPrompt(e.target.value)}
              placeholder="e.g. add a flagstone walkway leading to the front door, with low boxwood edging"
            />
          </div>

          <button className="btn" onClick={submit} disabled={!file || submitting}>
            {submitting ? "Generating…" : `Generate ${VARIATIONS} variations`}
          </button>
          {error && <div className="error" style={{ marginTop: 12 }}>{error}</div>}
        </div>

        <div className="panel">
          <h2>Results</h2>
          {results.length === 0 ? (
            <div className="empty">
              {submitting
                ? `Generating ${VARIATIONS} ${selectedStyle.label.toLowerCase()} variations… this can take up to 3 minutes.`
                : "Variations will appear here after you click Generate."}
            </div>
          ) : (
            <div className="results">
              {results.map((r, i) => (
                <div className="result" key={r.url + i}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={r.url} alt={r.style} />
                  <div className="meta">{r.style} · variation {i + 1}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
