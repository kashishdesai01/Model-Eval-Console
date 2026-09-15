import { percent } from '../lib/format';

export function ModelOutput({
  label,
  probability,
  raw,
}: {
  label?: number | null;
  probability: number | null;
  raw?: string | null;
}) {
  if (raw !== undefined && raw !== null) {
    return (
      <div>
        <strong>{label === -1 ? 'Invalid output' : label === 1 ? 'Positive' : 'Negative'}</strong>
        <div className="subtle">
          Generated: <code>{raw || '(empty)'}</code>
        </div>
        <small className="subtle">No classifier probability</small>
      </div>
    );
  }
  return (
    <span className="mono">
      {probability === null ? 'Unavailable' : `P(positive): ${percent(probability)}`}
    </span>
  );
}
