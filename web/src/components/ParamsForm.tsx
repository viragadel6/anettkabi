import type { AudioMode, SfxRequestOptions, VideoHandling } from "../api/types";

export interface ParamsFormProps {
  options: SfxRequestOptions;
  onChange: (patch: Partial<SfxRequestOptions>) => void;
  disabled?: boolean;
}

function NumberField({
  label,
  value,
  placeholder,
  onValue,
  step = "any",
  min,
  max,
}: {
  label: string;
  value: number | undefined;
  placeholder: string;
  onValue: (value: number | undefined) => void;
  step?: string;
  min?: number;
  max?: number;
}) {
  return (
    <label className="field">
      <span className="field__label">{label}</span>
      <input
        type="number"
        step={step}
        min={min}
        max={max}
        placeholder={placeholder}
        value={value ?? ""}
        onChange={(event) => {
          const raw = event.target.value;
          onValue(raw === "" ? undefined : Number(raw));
        }}
      />
    </label>
  );
}

export function ParamsForm({ options, onChange }: ParamsFormProps) {
  return (
    <section className="panel">
      <h2 className="panel__title">2 · Sound design</h2>
      <label className="field">
        <span className="field__label">Prompt</span>
        <textarea
          rows={3}
          placeholder="e.g. cinematic whoosh into deep sub-bass impact, tight tail"
          value={options.prompt}
          onChange={(event) => onChange({ prompt: event.target.value })}
        />
      </label>
      <label className="field">
        <span className="field__label">Negative prompt</span>
        <input
          type="text"
          placeholder="e.g. music, speech"
          value={options.negative_prompt ?? ""}
          onChange={(event) => onChange({ negative_prompt: event.target.value || undefined })}
        />
      </label>
      <div className="grid grid--3">
        <NumberField label="Seed" placeholder="random" value={options.seed} onValue={(seed) => onChange({ seed })} step="1" min={0} />
        <NumberField
          label="Steps"
          placeholder="32"
          value={options.num_inference_steps}
          onValue={(num_inference_steps) => onChange({ num_inference_steps })}
          step="1"
          min={1}
        />
        <NumberField
          label="Guidance"
          placeholder="3.0"
          value={options.guidance_scale}
          onValue={(guidance_scale) => onChange({ guidance_scale })}
          step="0.1"
          min={0}
        />
        <NumberField label="Start (s)" placeholder="0" value={options.start_time} onValue={(start_time) => onChange({ start_time })} step="0.01" min={0} />
        <NumberField label="Duration (s)" placeholder="full" value={options.duration} onValue={(duration) => onChange({ duration })} step="0.1" min={0.1} />
        <NumberField
          label="Loudness (LUFS)"
          placeholder="-14"
          value={options.target_loudness_lufs}
          onValue={(target_loudness_lufs) => onChange({ target_loudness_lufs })}
          step="0.5"
        />
      </div>
      <div className="grid grid--2">
        <label className="field">
          <span className="field__label">Audio mode</span>
          <select value={options.audio_mode ?? "replace"} onChange={(event) => onChange({ audio_mode: event.target.value as AudioMode })}>
            <option value="replace">replace</option>
            <option value="mix">mix</option>
            <option value="duck">duck</option>
          </select>
        </label>
        <label className="field">
          <span className="field__label">Video handling</span>
          <select
            value={options.video_handling ?? "copy"}
            onChange={(event) => onChange({ video_handling: event.target.value as VideoHandling })}
          >
            <option value="copy">copy (fast)</option>
            <option value="reencode">reencode</option>
          </select>
        </label>
        <NumberField
          label="SFX gain (dB)"
          placeholder="0"
          value={options.sfx_gain_db}
          onValue={(sfx_gain_db) => onChange({ sfx_gain_db })}
          step="0.5"
        />
        <NumberField
          label="Original gain (dB)"
          placeholder="0"
          value={options.original_audio_gain_db}
          onValue={(original_audio_gain_db) => onChange({ original_audio_gain_db })}
          step="0.5"
        />
        <NumberField
          label="Duck threshold (dB)"
          placeholder="-30"
          value={options.duck_threshold_db}
          onValue={(duck_threshold_db) => onChange({ duck_threshold_db })}
          step="1"
        />
        <NumberField
          label="Duck ratio"
          placeholder="0.25"
          value={options.duck_ratio}
          onValue={(duck_ratio) => onChange({ duck_ratio })}
          step="0.05"
          min={0}
          max={1}
        />
        <NumberField
          label="Duck attack (ms)"
          placeholder="20"
          value={options.duck_attack_ms}
          onValue={(duck_attack_ms) => onChange({ duck_attack_ms })}
          step="1"
          min={0}
        />
        <NumberField
          label="Duck release (ms)"
          placeholder="250"
          value={options.duck_release_ms}
          onValue={(duck_release_ms) => onChange({ duck_release_ms })}
          step="1"
          min={0}
        />
      </div>
      <label className="check">
        <input
          type="checkbox"
          checked={options.return_audio_only ?? false}
          onChange={(event) => onChange({ return_audio_only: event.target.checked || undefined })}
        />
        <span>Return audio only (WAV)</span>
      </label>
    </section>
  );
}
