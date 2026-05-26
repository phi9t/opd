import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { StepMetrics } from './types'

const PHASE_COLORS = {
  gen_ms: '#6366f1',
  teacher_ms: '#06b6d4',
  train_ms: '#10b981',
  sync_ms: '#f59e0b',
} as const

export default function StepTimeline({ steps }: { steps: StepMetrics[] }) {
  const data = steps.map((s) => ({
    step: s.step,
    gen_ms: s.gen_ms,
    teacher_ms: s.teacher_ms,
    train_ms: s.train_ms,
    sync_ms: s.sync_ms,
  }))

  return (
    <section className="card-view chart-card" aria-label="Step phase timeline">
      <h2 className="section-title">Phase timing per step</h2>
      <p className="section-desc">
        Stacked wall-clock milliseconds: generation, teacher, training, and weight
        sync.
      </p>
      <div className="chart-plot">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis
              dataKey="step"
              tick={{ fill: '#9ca3af', fontSize: 12 }}
              label={{ value: 'Step', position: 'insideBottom', offset: -4, fill: '#9ca3af' }}
            />
            <YAxis
              tick={{ fill: '#9ca3af', fontSize: 12 }}
              label={{
                value: 'ms',
                angle: -90,
                position: 'insideLeft',
                fill: '#9ca3af',
              }}
            />
            <Tooltip
              contentStyle={{
                background: '#111827',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 8,
              }}
              formatter={(value) => {
                const n = typeof value === 'number' ? value : Number(value)
                return Number.isFinite(n) ? [`${n.toFixed(1)} ms`, ''] : ['', '']
              }}
            />
            <Legend />
            <Bar
              dataKey="gen_ms"
              name="Generation"
              stackId="phases"
              fill={PHASE_COLORS.gen_ms}
            />
            <Bar
              dataKey="teacher_ms"
              name="Teacher"
              stackId="phases"
              fill={PHASE_COLORS.teacher_ms}
            />
            <Bar
              dataKey="train_ms"
              name="Train"
              stackId="phases"
              fill={PHASE_COLORS.train_ms}
            />
            <Bar
              dataKey="sync_ms"
              name="Sync"
              stackId="phases"
              fill={PHASE_COLORS.sync_ms}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
