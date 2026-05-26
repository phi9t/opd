import { useMemo, useState } from 'react'
import type { StepMetrics, TokenSample } from './types'

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const idx = Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)))
  return sorted[idx]
}

function tokenBg(kl: number, scale: number): string {
  if (scale <= 0) return 'transparent'
  const norm = Math.max(0, Math.min(1, kl / scale))
  return `rgba(227, 77, 77, ${norm.toFixed(3)})`
}

function Sample({ sample, scale }: { sample: TokenSample; scale: number }) {
  return (
    <div className="token-sample">
      <div className="token-sample-prompt" title="prompt">
        <span className="token-sample-label">prompt</span>
        <span>{sample.prompt}</span>
      </div>
      <div className="token-sample-body">
        {sample.tokens.map((tok, i) => {
          const kl = sample.kl[i] ?? 0
          return (
            <span
              key={i}
              className="token-cell"
              style={{ backgroundColor: tokenBg(kl, scale) }}
              title={`token=${tok}\nreverse-KL=${kl.toFixed(4)}`}
            >
              {tok}
            </span>
          )
        })}
      </div>
    </div>
  )
}

export default function TokenHeatmap({ steps }: { steps: StepMetrics[] }) {
  const stepsWithSamples = useMemo(
    () => steps.filter((s) => (s.samples?.length ?? 0) > 0),
    [steps],
  )

  const [stepIdx, setStepIdx] = useState(0)
  const [scaleMode, setScaleMode] = useState<'per-step' | 'global'>('per-step')

  const globalScale = useMemo(() => {
    const allKl = stepsWithSamples.flatMap((s) =>
      (s.samples ?? []).flatMap((sample) => sample.kl.map(Math.abs)),
    )
    return percentile(allKl, 0.95)
  }, [stepsWithSamples])

  if (stepsWithSamples.length === 0) {
    return (
      <section className="card-view chart-card" aria-label="Per-token KL heatmap">
        <h2 className="section-title">Per-token reverse-KL</h2>
        <p className="section-desc">
          No token samples are recorded in this run. Re-export after running with{' '}
          <code>log_token_samples &gt; 0</code> to enable this view.
        </p>
      </section>
    )
  }

  const safeIdx = Math.min(stepIdx, stepsWithSamples.length - 1)
  const current = stepsWithSamples[safeIdx]
  const samples = current.samples ?? []
  const stepKl = samples.flatMap((s) => s.kl.map(Math.abs))
  const scale =
    scaleMode === 'global' ? globalScale : percentile(stepKl, 0.95) || 1e-6

  return (
    <section className="card-view chart-card" aria-label="Per-token KL heatmap">
      <div className="token-header">
        <div>
          <h2 className="section-title">Per-token reverse-KL</h2>
          <p className="section-desc">
            Each token is shaded by{' '}
            <code>log π_student − log π_teacher</code> at that position. Brighter
            red = the student diverged more from the teacher there. Hover a token
            for the exact value.
          </p>
        </div>
        <div className="token-legend">
          <span className="token-legend-label">0</span>
          <span className="token-legend-bar" />
          <span className="token-legend-label">{scale.toFixed(3)}</span>
        </div>
      </div>

      <div className="token-controls">
        <label className="token-control">
          <span className="token-control-label">
            step{' '}
            <strong>{current.step}</strong>{' '}
            <span className="text-muted">
              ({safeIdx + 1}/{stepsWithSamples.length})
            </span>
          </span>
          <input
            type="range"
            min={0}
            max={stepsWithSamples.length - 1}
            value={safeIdx}
            onChange={(e) => setStepIdx(Number(e.target.value))}
            aria-label="Training step"
          />
        </label>

        <div className="token-scale-toggle" role="radiogroup" aria-label="Scale">
          <button
            type="button"
            role="radio"
            aria-checked={scaleMode === 'per-step'}
            className={scaleMode === 'per-step' ? 'active' : ''}
            onClick={() => setScaleMode('per-step')}
          >
            per-step
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={scaleMode === 'global'}
            className={scaleMode === 'global' ? 'active' : ''}
            onClick={() => setScaleMode('global')}
          >
            global
          </button>
        </div>
      </div>

      <div className="token-samples">
        {samples.map((sample, i) => (
          <Sample key={i} sample={sample} scale={scale} />
        ))}
      </div>
    </section>
  )
}
