import { afterEach, describe, expect, it, vi } from 'vitest'
import { askQuestion, checkBackendHealth, errorMessages, isAnswerResponse } from './api'
import { answer, citation, health, json } from './test/fixtures'

afterEach(() => vi.unstubAllGlobals())

describe('API contract', () => {
  it('checks exact health and passes its signal', async () => {
    const fetch = vi.fn().mockResolvedValue(json(health))
    vi.stubGlobal('fetch', fetch)
    const signal = new AbortController().signal
    await expect(checkBackendHealth(signal)).resolves.toEqual(health)
    expect(fetch).toHaveBeenCalledWith('http://localhost:8000/api/v1/health', { signal })
  })
  it('rejects malformed health', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({ status: 'ok' })))
    await expect(checkBackendHealth(new AbortController().signal)).rejects.toThrow(errorMessages.response)
  })
  it('posts only question and returns canonical PDF/HTML answer', async () => {
    const payload = answer()
    payload.citations.push({ ...citation, source_type: 'html', page: null, source_url: 'https://priem.unecon.ru/' })
    const fetch = vi.fn().mockResolvedValue(json(payload))
    vi.stubGlobal('fetch', fetch)
    const signal = new AbortController().signal
    await expect(askQuestion(payload.query, signal)).resolves.toEqual(payload)
    expect(fetch).toHaveBeenCalledWith('http://localhost:8000/api/v1/answer', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: payload.query }), signal,
    })
  })
  it('accepts neutral refusal only with no citations', () => {
    expect(isAnswerResponse({ ...answer(), status: 'insufficient_evidence', citations: [] })).toBe(true)
    expect(isAnswerResponse({ ...answer(), status: 'insufficient_evidence' })).toBe(false)
  })
  it.each([
    'http://unecon.ru/a', 'https://evil.ru/a', 'https://unecon.ru.evil.ru/',
    'https://evilunecon.ru/', 'javascript:alert(1)', '/relative',
    'https://name:secret@unecon.ru/', 'https://@unecon.ru/', 'https://unecon.ru:444/a', 'https://unecon.ru/a#page=2',
  ])('rejects unsafe URL %s', (source_url) => {
    expect(isAnswerResponse({ ...answer(), citations: [{ ...citation, source_url }] })).toBe(false)
  })
  it('allows explicit HTTPS port and official subdomain without rewriting', () => {
    expect(isAnswerResponse({ ...answer(), citations: [{ ...citation, source_url: 'https://www.unecon.ru:443/a' }] })).toBe(true)
  })
  it.each([
    null, [], {}, { ...answer(), extra: 1 }, { ...answer(), query: '' },
    { ...answer(), answer: ' ' }, { ...answer(), status: 'other' },
    { ...answer(), retrieval_mode: 'other' }, { ...answer(), citations: [] },
    ...[0, -1, 1.5, '2', null].map((page) => ({ ...answer(), citations: [{ ...citation, page }] })),
    { ...answer(), citations: [{ ...citation, source_type: 'html' }] },
    { ...answer(), citations: [{ ...citation, source_title: '' }] },
    { ...answer(), citations: [{ ...citation, secret: 'hidden' }] },
  ])('strictly rejects malformed response %#', (payload) => {
    expect(isAnswerResponse(payload)).toBe(false)
  })
  it.each([[422, 'validation'], [503, 'unavailable'], [500, 'response'], [404, 'response']] as const)(
    'maps HTTP %s to safe message', async (status, kind) => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({ detail: 'PRIVATE' }, status)))
      await expect(askQuestion('test')).rejects.toThrow(errorMessages[kind])
    },
  )
  it('maps connection error without leaking exception', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('PRIVATE')))
    await expect(askQuestion('test')).rejects.toThrow(errorMessages.network)
  })
  it('rejects damaged JSON and a mismatched question', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(new Response('{')).mockResolvedValueOnce(json(answer('other'))))
    await expect(askQuestion('test')).rejects.toThrow(errorMessages.response)
    await expect(askQuestion('test')).rejects.toThrow(errorMessages.response)
  })
  it('preserves abort for caller to ignore', async () => {
    const controller = new AbortController()
    controller.abort()
    const error = new DOMException('Aborted', 'AbortError')
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(error))
    await expect(askQuestion('test', controller.signal)).rejects.toBe(error)
  })
})
