import type { AnswerResponse, Citation, HealthResponse } from './types'

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '')

export const errorMessages = {
  unavailable: 'Сервис временно недоступен. Попробуйте ещё раз.',
  network: 'Не удалось связаться с сервисом. Проверьте подключение и попробуйте ещё раз.',
  validation: 'Проверьте текст вопроса и попробуйте ещё раз.',
  response: 'Не удалось обработать ответ сервиса. Попробуйте ещё раз.',
} as const

export class ApiError extends Error {
  constructor(kind: keyof typeof errorMessages) {
    super(errorMessages[kind])
    this.name = 'ApiError'
  }
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function exactKeys(value: Record<string, unknown>, keys: string[]) {
  return Object.keys(value).length === keys.length && keys.every((key) => key in value)
}

function nonempty(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

function officialUrl(value: string) {
  try {
    const url = new URL(value)
    // URL normalizes an empty username away; reject even an empty userinfo section.
    const authority = value.trim().match(/^https:\/\/([^/?#]*)/i)?.[1]
    return url.protocol === 'https:' &&
      (url.hostname === 'unecon.ru' || url.hostname.endsWith('.unecon.ru')) &&
      !!authority && !authority.includes('@') && !url.username && !url.password && !url.port && !url.hash
  } catch {
    return false
  }
}

function isCitation(value: unknown): value is Citation {
  if (!record(value) || !exactKeys(value, [
    'context_id', 'chunk_id', 'source_id', 'source_title', 'source_url', 'source_type', 'page',
  ])) return false
  if (!['context_id', 'chunk_id', 'source_id', 'source_title', 'source_url'].every(
    (key) => nonempty(value[key]),
  ) || !officialUrl(value.source_url as string)) return false
  return (value.source_type === 'html' && value.page === null) ||
    (value.source_type === 'pdf' && Number.isInteger(value.page) && (value.page as number) > 0)
}

export function isAnswerResponse(value: unknown): value is AnswerResponse {
  if (!record(value) || !exactKeys(value, ['query', 'status', 'answer', 'retrieval_mode', 'citations'])) return false
  return nonempty(value.query) && nonempty(value.answer) &&
    (value.retrieval_mode === 'dense' || value.retrieval_mode === 'reranked') &&
    Array.isArray(value.citations) && value.citations.every(isCitation) &&
    ((value.status === 'answered' && value.citations.length > 0) ||
      (value.status === 'insufficient_evidence' && value.citations.length === 0))
}

async function request(path: string, init: RequestInit): Promise<unknown> {
  let response: Response
  try {
    response = await fetch(`${apiBaseUrl}${path}`, init)
  } catch (error) {
    if (init.signal?.aborted || (error instanceof Error && error.name === 'AbortError')) throw error
    throw new ApiError('network')
  }
  if (!response.ok) {
    throw new ApiError(response.status === 503 ? 'unavailable' : response.status === 422 ? 'validation' : 'response')
  }
  try {
    return await response.json()
  } catch (error) {
    if (init.signal?.aborted) throw error
    throw new ApiError('response')
  }
}

export async function checkBackendHealth(signal: AbortSignal): Promise<HealthResponse> {
  const payload = await request('/api/v1/health', { signal })
  if (!record(payload) || !exactKeys(payload, ['status', 'service']) ||
    payload.status !== 'ok' || payload.service !== 'unecon-rag-assistant') throw new ApiError('response')
  return payload as unknown as HealthResponse
}

export async function askQuestion(question: string, signal?: AbortSignal): Promise<AnswerResponse> {
  const payload = await request('/api/v1/answer', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }), signal,
  })
  if (!isAnswerResponse(payload) || payload.query !== question) throw new ApiError('response')
  return payload
}
