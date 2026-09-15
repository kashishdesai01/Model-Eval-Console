import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ModelOutput } from './ModelOutput';

describe('Model output evidence', () => {
  it('shows invalid raw answers without inventing a probability', () => {
    render(<ModelOutput label={-1} probability={null} raw="unclear" />);
    expect(screen.getByText('Invalid output')).toBeInTheDocument();
    expect(screen.getByText('unclear')).toBeInTheDocument();
    expect(screen.getByText('No classifier probability')).toBeInTheDocument();
    expect(screen.queryByText(/P\(positive\)/)).not.toBeInTheDocument();
  });
  it('identifies classifier scores as positive-class probabilities', () => {
    render(<ModelOutput label={1} probability={0.9} />);
    expect(screen.getByText('P(positive): 90.0%')).toBeInTheDocument();
  });
});
