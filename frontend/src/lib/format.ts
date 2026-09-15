export const percent = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
export const points = (value: number) => `${value > 0 ? '+' : ''}${(value * 100).toFixed(2)}`;
export const shortId = (value: string) => value.slice(0, 8);
export const time = (value: string) =>
  new Date(value).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
export const metricNumber = (
  metrics: Record<string, unknown> | null | undefined,
  key: string,
): number | undefined =>
  typeof metrics?.[key] === 'number' ? (metrics[key] as number) : undefined;
