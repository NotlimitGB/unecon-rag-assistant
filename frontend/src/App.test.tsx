import { StrictMode } from 'react'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { errorMessages } from './api'
import { answer, citation, deferred, health, json } from './test/fixtures'

const disclosure = 'Ответ сформирован ИИ на основе официальных материалов СПбГЭУ. Важную информацию рекомендуем уточнять в приёмной комиссии.'
const input = () => screen.getByRole('textbox', { name: 'Ваш вопрос' })
const send = () => screen.getByRole('button', { name: /Отправить/ })
function setup(response: () => Promise<Response> = async () => json(answer())) {
  const fetch = vi.fn((url: string) => url.endsWith('/health') ? Promise.resolve(json(health)) : response())
  vi.stubGlobal('fetch', fetch)
  const view = render(<App />)
  fireEvent.click(screen.getByRole('button', { name: 'Открыть помощника абитуриента' }))
  return { fetch, ...view }
}
async function ready() {
  await waitFor(() => expect(screen.queryByText('Проверяем доступность помощника…')).not.toBeInTheDocument())
}
function write(value: string) { fireEvent.change(input(), { target: { value } }) }
async function submit(value = 'Когда приём?') {
  await ready(); write(value); fireEvent.click(send())
}
afterEach(() => vi.unstubAllGlobals())

describe('applicant interface', () => {
  it('shows scope, empty state and exact disclosure; examples only fill and focus', async () => {
    const { fetch } = setup()
    expect(send()).toBeDisabled()
    expect(screen.getByRole('dialog', { name: 'Помощник абитуриента' })).toBeInTheDocument()
    expect(screen.getByText(/Приёмная кампания 2026/)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Чем я могу помочь?' })).toBeInTheDocument()
    await ready()
    const examples = within(screen.getByLabelText('Примеры вопросов')).getAllByRole('button')
    expect(examples).toHaveLength(3)
    for (const example of examples) {
      fireEvent.click(example)
      expect(input()).toHaveFocus()
      expect(input()).not.toHaveValue('')
    }
    expect(fetch).toHaveBeenCalledTimes(1)
    expect(screen.getByText(disclosure)).toBeInTheDocument()
  })
  it('shows question immediately, clears draft, blocks duplicate pending and restores focus', async () => {
    const pending = deferred<Response>()
    const { fetch } = setup(() => pending.promise)
    await ready(); input().focus(); write('  Когда приём?  '); fireEvent.keyDown(input(), { key: 'Enter' })
    expect(screen.getByText('Когда приём?')).toBeInTheDocument()
    expect(input()).toHaveValue('')
    expect(input()).toHaveAttribute('readonly')
    expect(screen.getByText('Ищу информацию в официальных материалах…')).toBeInTheDocument()
    expect(screen.getByText(disclosure)).toBeInTheDocument()
    fireEvent.keyDown(input(), { key: 'Enter' })
    fireEvent.submit(input().closest('form')!)
    expect(fetch).toHaveBeenCalledTimes(2)
    await act(async () => pending.resolve(json(answer())))
    expect(screen.getByText(answer().answer)).toBeInTheDocument()
    expect(input()).not.toHaveAttribute('readonly')
    expect(input()).toHaveFocus()
    expect(screen.queryByText('internal-chunk')).not.toBeInTheDocument()
    expect(screen.queryByText('reranked')).not.toBeInTheDocument()
  })
  it('groups sources in first appearance order with sorted unique pages and unchanged links', async () => {
    const payload = answer()
    payload.citations = [citation, { ...citation, source_title: 'Стоимость', source_url: 'https://unecon.ru/tuition/', source_type: 'html', page: null },
      { ...citation, page: 1 }, citation]
    setup(async () => json(payload)); await submit()
    const links = await within(screen.getByRole('dialog')).findAllByRole('link')
    expect(links).toHaveLength(2)
    expect(links[0]).toHaveTextContent('Правила приёма')
    expect(links[0]).toHaveTextContent('PDF · стр. 1, 2')
    expect(links[1]).toHaveTextContent('Официальная страница')
    expect(links[1]).not.toHaveTextContent('стр.')
    for (const link of links) {
      expect(link).toHaveAttribute('target', '_blank')
      expect(link).toHaveAttribute('rel', 'noopener noreferrer')
    }
    expect(links[0]).toHaveAttribute('href', citation.source_url)
  })
  it('renders refusal verbatim without sources or error styling', async () => {
    setup(async () => json({ ...answer(), status: 'insufficient_evidence', answer: 'Недостаточно данных для ответа.', citations: [] }))
    await submit()
    expect(await screen.findByText('Недостаточно данных для ответа.')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /Источники/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
  it('retries the submitted question without losing a new draft; next question replaces result', async () => {
    let attempt = 0
    const { fetch } = setup(async () => ++attempt === 1 ? json({ detail: 'PRIVATE' }, 503) : json(answer(attempt === 2 ? 'Когда приём?' : 'Новый вопрос')))
    await submit()
    expect(await screen.findByRole('alert')).toHaveTextContent(errorMessages.unavailable)
    write('Новый вопрос')
    fireEvent.click(screen.getByRole('button', { name: 'Попробовать ещё раз' }))
    await screen.findByText(answer().answer)
    expect(input()).toHaveValue('Новый вопрос')
    expect(fetch.mock.calls[2]).toBeDefined()
    const calls = vi.mocked(globalThis.fetch).mock.calls
    expect(calls[2][1]?.body).toBe(JSON.stringify({ question: 'Когда приём?' }))
    fireEvent.click(send())
    await screen.findByText('Новый вопрос')
    expect(screen.queryByText('Когда приём?')).not.toBeInTheDocument()
    expect(screen.queryByText('PRIVATE')).not.toBeInTheDocument()
  })
  it.each(['', ' \n\t '])('blocks blank input %j', async (value) => {
    const { fetch } = setup(); await ready(); write(value)
    expect(send()).toBeDisabled(); fireEvent.submit(input().closest('form')!)
    expect(fetch).toHaveBeenCalledTimes(1)
  })
  it.each(['я', '😀'])('counts Unicode code points for %s at 2000/2001 before trimming', async (character) => {
    const { fetch } = setup(); await ready()
    write(character.repeat(1599)); expect(screen.queryByText(/\/ 2000/)).not.toBeInTheDocument()
    write(character.repeat(2000)); expect(send()).toBeEnabled(); expect(screen.getByText('2000 / 2000')).toBeInTheDocument()
    write(character.repeat(2000) + ' '); expect(send()).toBeDisabled()
    expect(input()).toHaveValue(character.repeat(2000) + ' ')
    expect(screen.getByRole('alert')).toHaveTextContent('не больше 2000')
    fireEvent.keyDown(input(), { key: 'Enter' }); expect(fetch).toHaveBeenCalledTimes(1)
  })
  it('Enter submits but Shift+Enter and IME composition do not', async () => {
    const { fetch } = setup(); await ready(); write('Когда приём?')
    fireEvent.keyDown(input(), { key: 'Enter', shiftKey: true })
    fireEvent.keyDown(input(), { key: 'Enter', isComposing: true })
    fireEvent.keyDown(input(), { key: 'Enter', keyCode: 229 })
    expect(fetch).toHaveBeenCalledTimes(1)
    fireEvent.keyDown(input(), { key: 'Enter' })
    await screen.findByText(answer().answer)
    expect(fetch).toHaveBeenCalledTimes(2)
  })
  it('submits all 2000 emoji without truncation', async () => {
    const question = '😀'.repeat(2000)
    setup(async () => json(answer(question)))
    await submit(question)
    await screen.findByText(answer().answer)
    expect(vi.mocked(globalThis.fetch).mock.calls[1][1]?.body).toBe(JSON.stringify({ question }))
  })
  it('allows separate health retry with no automatic answer request', async () => {
    const fetch = vi.fn().mockRejectedValueOnce(new Error('PRIVATE')).mockResolvedValueOnce(json(health))
    vi.stubGlobal('fetch', fetch); render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Открыть помощника абитуриента' }))
    const retry = await screen.findByRole('button', { name: 'Повторить проверку' })
    write('Когда приём?'); expect(send()).toBeDisabled()
    fireEvent.click(retry); await ready(); expect(send()).toBeEnabled()
    expect(fetch).toHaveBeenCalledTimes(2)
  })
  it('aborts health on unmount and ignores its stale completion in StrictMode', async () => {
    const stale = deferred<Response>()
    const fetch = vi.fn().mockReturnValueOnce(stale.promise).mockResolvedValueOnce(json(health))
    vi.stubGlobal('fetch', fetch)
    const { unmount } = render(<StrictMode><App /></StrictMode>)
    fireEvent.click(screen.getByRole('button', { name: 'Открыть помощника абитуриента' }))
    await ready(); write('Когда приём?')
    expect(fetch.mock.calls[0][1].signal.aborted).toBe(true)
    await act(async () => stale.resolve(json({}, 503)))
    expect(send()).toBeEnabled()
    unmount(); expect(fetch.mock.calls[1][1].signal.aborted).toBe(true)
  })
  it('aborts answer on unmount; late completion cannot appear in a new mount', async () => {
    const stale = deferred<Response>()
    const { unmount } = setup(() => stale.promise)
    await submit()
    const signal = vi.mocked(globalThis.fetch).mock.calls[1][1]?.signal
    unmount(); expect(signal?.aborted).toBe(true)
    setup(); await ready()
    await act(async () => stale.resolve(json(answer())))
    expect(screen.queryByText(answer().answer)).not.toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
  it('does not steal focus outside the form when request ends', async () => {
    const pending = deferred<Response>()
    setup(() => pending.promise); await submit()
    const outside = document.createElement('a'); outside.href = '#'; outside.textContent = 'outside'
    document.body.append(outside); outside.focus()
    await act(async () => pending.resolve(json(answer())))
    expect(outside).toHaveFocus(); outside.remove()
  })
  it('displays markup as literal text and never creates script elements', async () => {
    const literal = '<script>alert(1)</script>\n**Текст**'
    setup(async () => json({ ...answer(), answer: literal })); await submit()
    await screen.findByText(/<script>alert/)
    expect(document.querySelector('.answer-text')?.textContent).toBe(literal)
    expect(document.querySelector('script')).toBeNull()
  })
  it.each([
    [() => Promise.resolve(json({}, 422)), errorMessages.validation],
    [() => Promise.reject(new Error('PRIVATE')), errorMessages.network],
    [() => Promise.resolve(new Response('{')), errorMessages.response],
    [() => Promise.resolve(json({}, 500)), errorMessages.response],
  ] as const)('shows only safe error for failure %#', async (response, message) => {
    setup(response); await submit()
    expect(await screen.findByRole('alert')).toHaveTextContent(message)
    expect(document.body).not.toHaveTextContent('PRIVATE')
    expect(screen.getByText(disclosure)).toBeInTheDocument()
  })
})
