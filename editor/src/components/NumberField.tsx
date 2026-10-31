import { useEffect, useRef, useState } from 'react';

export function NumberField({ label, displayLabel, value, min = -32768, max = 32767, step = 1, help, onCommit }: {
  label: string; displayLabel?: string; value: number; min?: number; max?: number; step?: number; help?: string; onCommit: (value: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));
  const cancelCommit = useRef(false);
  useEffect(() => setDraft(String(value)), [value]);
  function commit() {
    if (cancelCommit.current) { cancelCommit.current = false; return; }
    const number = Number(draft);
    if (draft.trim() && Number.isFinite(number) && number >= min && number <= max && Number.isInteger(number / step)) {
      if (number !== value) onCommit(number);
    }
  }
  return <label className="number-field"><span>{displayLabel ?? label}</span><input type="number" required aria-label={label} title={`${help ?? `Change ${label.toLowerCase()} (${min} to ${max}).`} Enter applies; Escape cancels; leaving the field also applies a valid value.`} value={draft} min={min} max={max} step={step}
    onChange={event => setDraft(event.target.value)} onBlur={commit} onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur(); if (event.key === 'Escape') { cancelCommit.current = true; setDraft(String(value)); event.currentTarget.blur(); } }} /></label>;
}
