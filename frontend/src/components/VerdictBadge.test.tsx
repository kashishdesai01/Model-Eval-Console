import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { VerdictBadge } from './VerdictBadge';

describe('VerdictBadge', () => {
  it.each(['pass', 'fail', 'inconclusive'] as const)(
    'communicates %s without relying on color',
    (verdict) => {
      render(<VerdictBadge verdict={verdict} />);
      expect(screen.getByText(verdict.toUpperCase())).toBeVisible();
    },
  );
});
