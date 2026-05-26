export default function Glossary({ terms }: { terms: Record<string, string> }) {
  const entries = Object.entries(terms).sort(([a], [b]) => a.localeCompare(b))

  if (entries.length === 0) {
    return <p className="text-muted">No glossary terms.</p>
  }

  return (
    <section className="card-view" aria-label="Glossary">
      <h2 className="section-title">Glossary</h2>
      <dl className="vocab-gloss-list">
        {entries.map(([term, definition]) => (
          <div key={term} className="vocab-gloss-item">
            <dt>{term}</dt>
            <dd>{definition}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
