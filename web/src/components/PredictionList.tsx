import type { Prediction } from "../api/types";
import { StatusBadge } from "./StatusBadge";

export interface PredictionListProps {
  predictions: Prediction[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onCancel: (id: string) => void;
  onDelete: (id: string) => void;
}

export function PredictionList({ predictions, selectedId, onSelect, onCancel, onDelete }: PredictionListProps) {
  if (predictions.length === 0) {
    return <p className="muted">No predictions yet. Design your first sound.</p>;
  }
  return (
    <ul className="plist">
      {predictions.map((prediction) => (
        <li
          key={prediction.id}
          className={prediction.id === selectedId ? "plist__item plist__item--selected" : "plist__item"}
          onClick={() => onSelect(prediction.id)}
        >
          <div className="plist__row">
            <StatusBadge status={prediction.status} />
            <span className="plist__id">{prediction.id}</span>
            <span className="plist__time">{prediction.created_at?.replace("T", " ").slice(0, 19) ?? ""}</span>
          </div>
          <div className="plist__row plist__row--meta">
            <span className="plist__prompt">{prediction.prompt ?? ""}</span>
            <span className="plist__actions">
              {prediction.status === "queued" || prediction.status === "processing" || prediction.status === "starting" ? (
                <button
                  type="button"
                  className="btn btn--ghost"
                  onClick={(event) => {
                    event.stopPropagation();
                    onCancel(prediction.id);
                  }}
                >
                  cancel
                </button>
              ) : null}
              <button
                type="button"
                className="btn btn--ghost btn--danger"
                onClick={(event) => {
                  event.stopPropagation();
                  onDelete(prediction.id);
                }}
              >
                delete
              </button>
            </span>
          </div>
        </li>
      ))}
    </ul>
  );
}
