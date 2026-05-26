import { useCallback, useEffect, useState } from 'react'
import Glossary from './Glossary'
import LearningCurves from './LearningCurves'
import RunOverview from './RunOverview'
import StepTimeline from './StepTimeline'
import type { RunBundle, RunIndexEntry } from './types'

const INDEX_URL = '/data/index.json'

export default function OpdExplorer() {
  const [index, setIndex] = useState<RunIndexEntry[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [bundle, setBundle] = useState<RunBundle | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)

    fetch(INDEX_URL)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load index (${res.status})`)
        return res.json() as Promise<RunIndexEntry[]>
      })
      .then((entries) => {
        if (cancelled) return
        setIndex(entries)
        if (entries.length > 0) {
          setSelectedId((prev) => prev ?? entries[0].id)
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load runs')
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [])

  const loadRun = useCallback((entry: RunIndexEntry) => {
    setLoading(true)
    setError(null)
    setSelectedId(entry.id)

    fetch(`/data/${entry.path}`)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load run (${res.status})`)
        return res.json() as Promise<RunBundle>
      })
      .then((data) => setBundle(data))
      .catch((err: unknown) => {
        setBundle(null)
        setError(err instanceof Error ? err.message : 'Failed to load run')
      })
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    const entry = index.find((e) => e.id === selectedId)
    if (entry) loadRun(entry)
  }, [index, selectedId, loadRun])

  if (loading && index.length === 0) {
    return (
      <div className="loading-container" role="status">
        <div className="spinner" aria-hidden="true" />
        <p>Loading training runs…</p>
      </div>
    )
  }

  if (error && !bundle) {
    return (
      <div className="loading-container" role="alert">
        <p className="text-danger">{error}</p>
      </div>
    )
  }

  return (
    <div className="dashboard-grid">
      <aside className="sidebar-panel">
        <h2 className="sidebar-title">Training runs</h2>
        <ul className="dataset-list" role="list">
          {index.map((entry) => (
            <li key={entry.id}>
              <button
                type="button"
                className={`dataset-item ${entry.id === selectedId ? 'active' : ''}`}
                onClick={() => loadRun(entry)}
                aria-pressed={entry.id === selectedId}
              >
                <div className="dataset-item-left">
                  <span className="dataset-name-label">{entry.id}</span>
                  <span className="dataset-desc-label">{entry.tier} tier</span>
                </div>
                <div className="dataset-item-right">
                  <span className="dataset-domain-badge">{entry.tier}</span>
                </div>
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <div className="main-view-panel">
        {loading && !bundle ? (
          <div className="loading-container" role="status">
            <div className="spinner" aria-hidden="true" />
            <p>Loading run metrics…</p>
          </div>
        ) : bundle ? (
          <>
            <RunOverview bundle={bundle} />
            <div className="chart-grid">
              <StepTimeline steps={bundle.steps} />
              <LearningCurves steps={bundle.steps} />
            </div>
            <Glossary terms={bundle.glossary} />
          </>
        ) : (
          <p className="text-muted">Select a run to view metrics.</p>
        )}
      </div>
    </div>
  )
}
