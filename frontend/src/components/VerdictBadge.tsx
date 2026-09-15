import { Check, CircleHelp, X } from 'lucide-react';

export function VerdictBadge({ verdict }: { verdict: 'pass' | 'fail' | 'inconclusive' }) {
  const Icon = verdict === 'pass' ? Check : verdict === 'fail' ? X : CircleHelp;
  return (
    <span className={`verdict-badge ${verdict}`}>
      <Icon size={14} />
      {verdict.toUpperCase()}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`status-badge ${status}`}>
      <span className="status-dot" />
      {status}
    </span>
  );
}
