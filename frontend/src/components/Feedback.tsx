import { AlertCircle, LoaderCircle } from 'lucide-react';
import type { ReactNode } from 'react';

export function Loading({ label = 'Loading evaluation data…' }: { label?: string }) {
  return (
    <div className="loading" role="status">
      <LoaderCircle className="spin" size={18} />
      {label}
    </div>
  );
}

export function ErrorMessage({ error }: { error: Error | null }) {
  if (!error) return null;
  return (
    <div className="error-message" role="alert">
      <AlertCircle size={18} />
      <span>{error.message}</span>
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">◇</div>
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
