import type { Citation } from '../types'

export function Sources({ citations }: { citations: Citation[] }) {
  const groups = new Map<string, { citation: Citation; pages: Set<number> }>()
  for (const citation of citations) {
    const key = JSON.stringify([citation.source_url, citation.source_title, citation.source_type])
    if (!groups.has(key)) groups.set(key, { citation, pages: new Set() })
    if (citation.page !== null) groups.get(key)!.pages.add(citation.page)
  }
  return <section className="sources" aria-labelledby="sources-title">
    <h3 id="sources-title">Источники <span>{groups.size}</span></h3>
    <ol className="source-list">
      {[...groups.entries()].map(([key, { citation, pages }]) => <li key={key}>
        <a href={citation.source_url} target="_blank" rel="noopener noreferrer">
          <span className="source-title">{citation.source_title} <span aria-hidden="true">↗</span></span>
          <span className="source-meta">{citation.source_type === 'pdf'
            ? `PDF · стр. ${[...pages].sort((a, b) => a - b).join(', ')}`
            : 'Официальная страница'}</span>
          <span className="sr-only"> (откроется в новой вкладке)</span>
        </a>
      </li>)}
    </ol>
  </section>
}
