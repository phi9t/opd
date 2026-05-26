import { Fragment, type ReactNode } from 'react'
import { BlockMath, InlineMath } from 'react-katex'

// Split on $$...$$ (block) and $...$ (inline) and render each segment.
function renderWithMath(text: string): ReactNode {
  const parts = text.split(/(\$\$[^$]+\$\$|\$[^$]+\$)/g)
  return parts.map((part, i) => {
    if (part.startsWith('$$') && part.endsWith('$$')) {
      return <BlockMath key={i} math={part.slice(2, -2)} />
    }
    if (part.startsWith('$') && part.endsWith('$') && part.length > 2) {
      return <InlineMath key={i} math={part.slice(1, -1)} />
    }
    return <Fragment key={i}>{part}</Fragment>
  })
}

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
            <dd>{renderWithMath(definition)}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
