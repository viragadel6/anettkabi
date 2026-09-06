import type { PredictionStatus } from "../api/types";

const STYLES: Record<string, string> = {
  starting: "badge badge--starting",
  queued: "badge badge--queued",
  processing: "badge badge--processing",
  succeeded: "badge badge--succeeded",
  failed: "badge badge--failed",
  canceled: "badge badge--canceled",
};

export function StatusBadge({ status }: { status: PredictionStatus }) {
  const cls = STYLES[status] ?? "badge";
  return <span className={cls}>{status}</span>;
}
