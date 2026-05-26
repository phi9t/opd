import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { StepMetrics } from './types'

export default function LearningCurves({ steps }: { steps: StepMetrics[] }) {
  const data = steps.map((s) => ({
    step: s.step,
    mean_kl: s.mean_kl,
    loss: s.loss,
  }))

  return (
    <section className="card-view chart-card" aria-label="Learning curves">
      <h2 className="section-title">Learning curves</h2>
      <p className="section-desc">Mean KL and training loss across steps.</p>
      <div className="chart-plot">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="step" tick={{ fill: '#9ca3af', fontSize: 12 }} />
            <YAxis
              tick={{ fill: '#9ca3af', fontSize: 12 }}
              tickFormatter={(v: number) => v.toExponential(1)}
            />
            <Tooltip
              contentStyle={{
                background: '#111827',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 8,
              }}
              formatter={(value, name) => {
                const n = typeof value === 'number' ? value : Number(value)
                if (!Number.isFinite(n) || name == null) return ['', '']
                const label = String(name) === 'mean_kl' ? 'Mean KL' : 'Loss'
                return [n.toExponential(4), label]
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="mean_kl"
              name="Mean KL"
              stroke="#8b5cf6"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="loss"
              name="Loss"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
