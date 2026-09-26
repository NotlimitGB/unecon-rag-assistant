import type { AnswerResponse, Citation } from '../types'

export const health = { status: 'ok', service: 'unecon-rag-assistant' }
export const citation: Citation = {
  context_id: 'C1', chunk_id: 'internal-chunk', source_id: 'internal-source',
  source_title: 'Правила приёма', source_url: 'https://unecon.ru/rules.pdf', source_type: 'pdf', page: 2,
}
export const answer = (query = 'Когда приём?'): AnswerResponse => ({
  query, status: 'answered', answer: 'Приём документов начинается 20 июня.', retrieval_mode: 'reranked', citations: [citation],
})
export const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), {
  status, headers: { 'Content-Type': 'application/json' },
})
export function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}
