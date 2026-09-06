import { useRef, useState } from "react";
import type { DragEvent } from "react";

export interface DropzoneProps {
  file: File | null;
  onFile: (file: File | null) => void;
  videoUrl: string;
  onVideoUrl: (url: string) => void;
}

const ACCEPTED = [".mp4", ".mov", ".webm", ".mkv"];
const MAX_BYTES = 512 * 1024 * 1024;

export function Dropzone({ file, onFile, videoUrl, onVideoUrl }: DropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);

  const accept = (candidate: File | null) => {
    if (candidate === null) {
      onFile(null);
      return;
    }
    const extension = candidate.name.slice(candidate.name.lastIndexOf(".")).toLowerCase();
    if (!ACCEPTED.includes(extension)) {
      setWarning(`unsupported extension ${extension}; allowed: ${ACCEPTED.join(", ")}`);
      return;
    }
    if (candidate.size > MAX_BYTES) {
      setWarning(`file exceeds ${Math.floor(MAX_BYTES / 1024 / 1024)} MB; use uploads + URL mode`);
      return;
    }
    setWarning(null);
    onFile(candidate);
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files?.[0];
    accept(dropped ?? null);
  };

  return (
    <section className="panel">
      <h2 className="panel__title">1 · Source video</h2>
      <div
        className={dragging ? "dropzone dropzone--active" : "dropzone"}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
        }}
      >
        {file ? (
          <div>
            <strong>{file.name}</strong>
            <div className="muted">{Math.ceil(file.size / 1024 / 1024)} MB · click or drop to replace</div>
          </div>
        ) : (
          <div>
            Drop a video here, or click to browse
            <div className="muted">mp4 · mov · webm · mkv, up to {Math.floor(MAX_BYTES / 1024 / 1024)} MB</div>
          </div>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="video/mp4,video/quicktime,video/webm,video/x-matroska"
        hidden
        onChange={(event) => accept(event.target.files?.[0] ?? null)}
      />
      {warning ? <p className="warning">{warning}</p> : null}
      <div className="urlrow">
        <input
          type="url"
          placeholder="…or paste an https:// video URL"
          value={videoUrl}
          onChange={(event) => onVideoUrl(event.target.value)}
        />
      </div>
    </section>
  );
}
