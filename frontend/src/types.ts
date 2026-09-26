export interface HealthResponse {
  status: 'ok'
  service: 'unecon-rag-assistant'
}

export interface Citation {
  context_id: string
  chunk_id: string
  source_id: string
  source_title: string
  source_url: string
  source_type: 'html' | 'pdf'
  page: number | null
}

export interface AnswerResponse {
  query: string
  status: 'answered' | 'insufficient_evidence'
  answer: string
  retrieval_mode: 'dense' | 'reranked'
  citations: Citation[]
}
