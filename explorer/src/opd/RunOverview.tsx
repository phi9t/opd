import { Cpu, Layers, Zap } from 'lucide-react'
import type { RunBundle } from './types'

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms.toFixed(1)}ms`
}

export default function RunOverview({ bundle }: { bundle: RunBundle }) {
  const last = bundle.steps[bundle.steps.length - 1]
  const totalMs = bundle.steps.reduce(
    (sum, s) => sum + s.gen_ms + s.teacher_ms + s.train_ms + s.sync_ms,
    0,
  )

  return (
    <section className="card-view" aria-label="Run overview">
      <div className="dataset-header-section">
        <div className="dataset-title-meta">
          <h2>{bundle.run_id}</h2>
          <p>
            {bundle.tier} tier · {bundle.runtime} runtime · {bundle.loss_mode}{' '}
            loss · teacher signal: {bundle.teacher_signal}
          </p>
        </div>
        <div className="dataset-attributes">
          <span className="attr-badge">
            <Cpu size={14} aria-hidden="true" />
            {bundle.device}
          </span>
          <span className="attr-badge">
            <Layers size={14} aria-hidden="true" />
            student {bundle.models.student_hidden_size} / teacher{' '}
            {bundle.models.teacher_hidden_size}
          </span>
          <span className="attr-badge">
            <Zap size={14} aria-hidden="true" />
            vocab {bundle.models.vocab_size.toLocaleString()}
          </span>
        </div>
      </div>

      <div className="opd-stats-row">
        <div className="stat-chip">
          <span>Steps</span>
          <span className="stat-value">{bundle.steps.length}</span>
        </div>
        <div className="stat-chip">
          <span>Wall time (sum)</span>
          <span className="stat-value">{formatMs(totalMs)}</span>
        </div>
        {last ? (
          <>
            <div className="stat-chip">
              <span>Latest mean KL</span>
              <span className="stat-value">{last.mean_kl.toExponential(3)}</span>
            </div>
            <div className="stat-chip">
              <span>Latest loss</span>
              <span className="stat-value">{last.loss.toExponential(3)}</span>
            </div>
            <div className="stat-chip">
              <span>Latest grad norm</span>
              <span className="stat-value">{last.grad_norm.toFixed(3)}</span>
            </div>
          </>
        ) : null}
      </div>
    </section>
  )
}
