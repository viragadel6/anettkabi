import { useMemo } from "react";
import { resolveOutputUrl } from "../api/client";
import type { Prediction } from "../api/types";
import { StatusBadge } from "./StatusBadge";

export interface PredictionDetailProps {
  prediction: Prediction;
  onDownload: (prediction: Prediction) => void;
}

export function PredictionDetail({ prediction, onDownload }: PredictionDetailProps) {
  const outputUrl = useMemo(() => resolveOutputUrl(prediction), [prediction]);
  const isAudio = typeof outputUrl === "string" && outputUrl.includes(".wav");
  return (
    <section className="panel">
      <div className="detail__head">
        <h2 className="panel__title">Prediction {prediction.id}</h2>
        <StatusBadge status={prediction.status} />
      </div>
      <dl className="detail__grid">
        <dt>Prompt</dt>
        <dd>{prediction.prompt ?? "-"}</dd>
        <dt>Created</dt>
        <dd>{prediction.created_at ?? "-"}</dd>
        <dt>Completed</dt>
        <dd>{prediction.completed_at ?? "-"}</dd>
        {prediction.error ? (
          <>
            <dt>Error</dt>
            <dd className="warning">
              {prediction.error.error_code ?? "error"}: {prediction.error.message ?? ""}
            </dd>
          </>
        ) : null}
        {prediction.metrics && Object.keys(prediction.metrics).length > 0 ? (
          <>
            <dt>Metrics</dt>
            <dd>
              {Object.entries(prediction.metrics)
                .map(([key, value]) => `${key}=${typeof value === "number" ? value.toFixed(2) : String(value)}`)
                .join(" · ")}
            </dd>
          </>
        ) : null}
      </dl>
      {prediction.status === "succeeded" && outputUrl ? (
        <div className="detail__player">
          {isAudio ? (
            <audio controls src={outputUrl} className="player" preload="metadata" />
          ) : (
            <video controls src={outputUrl} className="player" preload="metadata" playsInline />
          )}
          <div className="detail__actions">
            <a className="btn" href={outputUrl} download={`${prediction.id}${isAudio ? ".wav" : ".mp4"}`}>
              Download
            </a>
            <button type="button" className="btn btn--ghost" onClick={() => onDownload(prediction)}>
              Save as…
            </button>
          </div>
        </div>
      ) : null}
    </section>
  );
}
