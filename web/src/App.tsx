import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, SfxApiClient, localStorageKeyStore } from "./api/client";
import type { Prediction, SfxRequestOptions } from "./api/types";
import { Dropzone } from "./components/Dropzone";
import { ParamsForm } from "./components/ParamsForm";
import { PredictionDetail } from "./components/PredictionDetail";
import { PredictionList } from "./components/PredictionList";

const TERMINAL = new Set(["succeeded", "failed", "canceled"]);
const POLL_MS = 2500;

const DEFAULT_OPTIONS: SfxRequestOptions = {
  prompt: "",
  audio_mode: "replace",
  video_handling: "copy",
};

export default function App() {
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [apiKey, setApiKey] = useState(() => localStorageKeyStore.get());
  const [options, setOptions] = useState<SfxRequestOptions>(DEFAULT_OPTIONS);
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const client = useMemo(() => new SfxApiClient(apiKey), [apiKey]);

  const refreshList = useCallback(async () => {
    if (!apiKey) return;
    try {
      const rows = await client.list({ limit: 25 });
      setPredictions(rows);
      setError(null);
    } catch (cause) {
      setError(cause instanceof ApiError ? `${cause.errorCode}: ${cause.message}` : String(cause));
    }
  }, [apiKey, client]);

  useEffect(() => {
    void refreshList();
  }, [refreshList]);

  useEffect(() => {
    const active = predictions.filter((prediction) => !TERMINAL.has(prediction.status));
    if (active.length === 0) return;
    const timer = window.setInterval(() => {
      void refreshList();
      void Promise.all(
        active.map(async (prediction) => {
          try {
            return await client.get(prediction.id);
          } catch {
            return prediction;
          }
        }),
      ).then((updated) => {
        setPredictions((previous) =>
          previous.map((row) => updated.find((candidate) => candidate.id === row.id) ?? row),
        );
      });
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [predictions, client]);

  const selected = predictions.find((prediction) => prediction.id === selectedId) ?? null;

  const submit = async () => {
    if (!options.prompt.trim()) {
      setError("prompt is required");
      return;
    }
    if (!file && !videoUrl.trim()) {
      setError("choose a file or paste a video URL");
      return;
    }
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const created = file
        ? await client.createMultipart(file, options)
        : await client.createFromUrl(videoUrl.trim(), options);
      setPredictions((previous) => [created, ...previous.filter((row) => row.id !== created.id)]);
      setSelectedId(created.id);
      setNotice(`submitted ${created.id}`);
    } catch (cause) {
      setError(cause instanceof ApiError ? `${cause.errorCode}: ${cause.message}` : String(cause));
    } finally {
      setBusy(false);
    }
  };

  const cancel = async (id: string) => {
    try {
      const updated = await client.cancel(id);
      setPredictions((previous) => previous.map((row) => (row.id === id ? updated : row)));
    } catch (cause) {
      setError(cause instanceof ApiError ? `${cause.errorCode}: ${cause.message}` : String(cause));
    }
  };

  const remove = async (id: string) => {
    try {
      await client.remove(id);
      setPredictions((previous) => previous.filter((row) => row.id !== id));
      if (selectedId === id) setSelectedId(null);
    } catch (cause) {
      setError(cause instanceof ApiError ? `${cause.errorCode}: ${cause.message}` : String(cause));
    }
  };

  const saveAs = async (prediction: Prediction) => {
    const url = prediction.output?.video ?? prediction.output?.audio ?? prediction.output?.url;
    if (!url) return;
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${prediction.id}.mp4`;
    anchor.click();
  };

  const applyKey = () => {
    localStorageKeyStore.set(apiKeyInput.trim());
    setApiKey(apiKeyInput.trim());
    setNotice("API key saved to this browser");
  };

  const clearKey = () => {
    localStorageKeyStore.clear();
    setApiKeyInput("");
    setApiKey("");
    setPredictions([]);
    setSelectedId(null);
  };

  return (
    <div className="shell">
      <header className="topbar">
        <div className="topbar__brand">
          Video→Video <strong>SFX</strong> Studio
        </div>
        <div className="topbar__key">
          {apiKey ? (
            <button type="button" className="btn btn--ghost" onClick={clearKey}>
              key ····{apiKey.slice(-4)} — change
            </button>
          ) : (
            <>
              <input
                type="password"
                placeholder="API key (vsfx_…)"
                value={apiKeyInput}
                onChange={(event) => setApiKeyInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") applyKey();
                }}
              />
              <button type="button" className="btn" onClick={applyKey} disabled={!apiKeyInput.trim()}>
                Save key
              </button>
            </>
          )}
        </div>
      </header>

      {error ? <div className="alert alert--error">{error}</div> : null}
      {notice ? <div className="alert alert--info">{notice}</div> : null}

      <main className="layout">
        <div className="layout__left">
          <Dropzone file={file} onFile={setFile} videoUrl={videoUrl} onVideoUrl={setVideoUrl} />
          <ParamsForm
            options={options}
            onChange={(patch) => setOptions((previous) => ({ ...previous, ...patch }))}
            disabled={busy}
          />
          <button type="button" className="btn btn--primary btn--block" onClick={() => void submit()} disabled={busy || !apiKey}>
            {busy ? "Submitting…" : "Generate sound"}
          </button>
        </div>
        <div className="layout__right">
          <section className="panel">
            <h2 className="panel__title">Predictions</h2>
            <PredictionList
              predictions={predictions}
              selectedId={selectedId}
              onSelect={setSelectedId}
              onCancel={(id: string) => void cancel(id)}
              onDelete={(id: string) => void remove(id)}
            />
          </section>
          {selected ? <PredictionDetail prediction={selected} onDownload={(p) => void saveAs(p)} /> : null}
        </div>
      </main>

      <footer className="footer">
        <span>±10 ms A/V sync · faststart MP4 · HMAC-signed webhooks</span>
      </footer>
    </div>
  );
}
