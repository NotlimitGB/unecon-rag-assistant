import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { answer, deferred, health, json } from './test/fixtures'

const open = () => fireEvent.click(screen.getByRole('button', { name: 'Открыть помощника абитуриента' }))
const close = () => fireEvent.click(screen.getByRole('button', { name: 'Закрыть помощника' }))
const input = () => screen.getByRole('textbox', { name: 'Ваш вопрос' })
async function ready() {
  await waitFor(() => expect(screen.queryByText('Проверяем доступность помощника…')).not.toBeInTheDocument())
}
function setup(response: () => Promise<Response> = async () => json(answer())) {
  const fetch = vi.fn((url: string) => url.endsWith('/health') ? Promise.resolve(json(health)) : response())
  vi.stubGlobal('fetch', fetch)
  return { fetch, ...render(<App />) }
}
function submit() {
  fireEvent.change(input(), { target: { value: 'Когда приём?' } })
  fireEvent.click(screen.getByRole('button', { name: /Отправить/ }))
}
afterEach(() => vi.unstubAllGlobals())

describe('floating widget shell', () => {
  it('starts closed on an honest host with working section anchors', async () => {
    setup()
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Поступлениев СПбГЭУ')
    expect(screen.getByText('Демонстрационная страница интеграции справочного модуля')).toBeVisible()
    const launcher = screen.getByRole('button', { name: 'Открыть помощника абитуриента' })
    expect(launcher).toHaveAttribute('aria-expanded', 'false')
    expect(launcher).toHaveAttribute('aria-controls', 'assistant-panel')
    expect(document.getElementById('assistant-panel')).not.toBeVisible()
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Отправить' })).not.toBeInTheDocument()
    for (const link of within(screen.getByRole('navigation')).getAllByRole('link')) {
      expect(document.querySelector(link.getAttribute('href')!)).not.toBeNull()
    }
    await ready()
  })
  it('opens a non-modal named panel and focuses the input on a ready desktop', async () => {
    setup(); await ready(); open()
    expect(screen.getByRole('dialog', { name: 'Помощник абитуриента' })).toBeVisible()
    expect(screen.getByRole('dialog')).not.toHaveAttribute('aria-modal', 'true')
    expect(screen.getByRole('button', { name: 'Свернуть помощника абитуриента' })).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText('Бакалавриат и специалитет · 2026')).toBeVisible()
    expect(input()).toHaveFocus()
    expect(screen.getByRole('dialog').querySelector('.widget-content')).toBeInTheDocument()
    expect(screen.getByRole('dialog').querySelector('.composer-section')).toBeInTheDocument()
    expect(document.querySelectorAll('.ai-disclosure')).toHaveLength(1)
    expect(document.querySelector('.ai-disclosure')?.previousElementSibling?.tagName).toBe('FORM')
  })
  it('close and Escape return focus, while normal host focus does not close the panel', async () => {
    setup(); await ready(); open(); close()
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Открыть помощника абитуриента' })).toHaveFocus()
    open()
    const link = screen.getByRole('link', { name: 'Контакты' })
    link.focus(); expect(link).toHaveFocus()
    expect(screen.getByRole('dialog')).toBeVisible()
    fireEvent.keyDown(window, { key: 'Escape', isComposing: true })
    expect(screen.getByRole('dialog')).toBeVisible()
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Открыть помощника абитуриента' })).toHaveFocus()
  })
  it('preserves the draft and health without extra requests', async () => {
    const { fetch } = setup(); await ready(); open()
    fireEvent.change(input(), { target: { value: 'Когда начинается приём?' } })
    close(); open()
    expect(input()).toHaveValue('Когда начинается приём?')
    expect(fetch).toHaveBeenCalledTimes(1)
  })
  it('preserves answer and sources after closing without repeating retrieval', async () => {
    const { fetch } = setup(); await ready(); open(); submit()
    await screen.findByText(answer().answer)
    close(); open()
    expect(screen.getByText(answer().answer)).toBeVisible()
    expect(within(screen.getByRole('dialog')).getByRole('link', { name: /Правила приёма/ })).toBeVisible()
    expect(fetch).toHaveBeenCalledTimes(2)
  })
  it('preserves an error and allows explicit retry after reopening', async () => {
    const { fetch } = setup(async () => json({ detail: 'PRIVATE' }, 503))
    await ready(); open(); submit(); await screen.findByRole('alert')
    close(); open()
    expect(screen.getByRole('alert')).toHaveTextContent('Сервис временно недоступен.')
    expect(fetch).toHaveBeenCalledTimes(2)
    fireEvent.click(screen.getByRole('button', { name: 'Попробовать ещё раз' }))
    await screen.findByRole('alert'); expect(fetch).toHaveBeenCalledTimes(3)
  })
  it('keeps pending work alive across close, reopen and background completion without stealing focus', async () => {
    const pending = deferred<Response>()
    const { fetch } = setup(() => pending.promise)
    await ready(); open(); submit()
    const signal = vi.mocked(globalThis.fetch).mock.calls[1][1]?.signal
    close(); expect(signal?.aborted).toBe(false)
    open(); expect(screen.getByText('Ищу информацию в официальных материалах…')).toBeVisible()
    close()
    await act(async () => pending.resolve(json(answer())))
    expect(screen.getByRole('button', { name: 'Открыть помощника абитуриента' })).toHaveFocus()
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    open(); expect(screen.getByText(answer().answer)).toBeVisible()
    expect(fetch).toHaveBeenCalledTimes(2)
  })
  it('still aborts a hidden pending request when the widget unmounts', async () => {
    const pending = deferred<Response>()
    const { unmount } = setup(() => pending.promise)
    await ready(); open(); submit()
    const signal = vi.mocked(globalThis.fetch).mock.calls[1][1]?.signal
    close(); unmount(); expect(signal?.aborted).toBe(true)
    await act(async () => pending.reject(new DOMException('aborted', 'AbortError')))
  })
  it('focuses the title on mobile without opening the keyboard', async () => {
    vi.mocked(window.matchMedia).mockReturnValue({ matches: true } as MediaQueryList)
    setup(); await ready(); open()
    expect(screen.getByRole('heading', { name: 'Помощник абитуриента' })).toHaveFocus()
    expect(input()).not.toHaveFocus()
    close(); expect(screen.getByRole('button', { name: 'Открыть помощника абитуриента' })).toHaveFocus()
  })
  it('does not steal focus when health completes after opening', async () => {
    const pending = deferred<Response>()
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(pending.promise))
    render(<App />); open()
    expect(screen.getByRole('heading', { name: 'Помощник абитуриента' })).toHaveFocus()
    const link = screen.getByRole('link', { name: 'Контакты' }); link.focus()
    await act(async () => pending.resolve(json(health)))
    expect(link).toHaveFocus()
  })
  it('updates mobile viewport bounds and cleans up listeners', async () => {
    const viewport = Object.assign(new EventTarget(), { height: 700, offsetTop: 0 })
    const remove = vi.spyOn(viewport, 'removeEventListener')
    const removeWindow = vi.spyOn(window, 'removeEventListener')
    vi.stubGlobal('visualViewport', viewport)
    const { unmount, container } = setup(); await ready(); open()
    const widget = container.querySelector<HTMLElement>('.assistant-widget')!
    expect(widget.style.getPropertyValue('--widget-viewport-height')).toBe('700px')
    viewport.height = 410; viewport.offsetTop = 15
    act(() => viewport.dispatchEvent(new Event('resize')))
    expect(widget.style.getPropertyValue('--widget-viewport-height')).toBe('410px')
    expect(widget.style.getPropertyValue('--widget-viewport-top')).toBe('15px')
    viewport.offsetTop = 22; act(() => viewport.dispatchEvent(new Event('scroll')))
    expect(widget.style.getPropertyValue('--widget-viewport-top')).toBe('22px')
    unmount()
    expect(remove).toHaveBeenCalledWith('resize', expect.any(Function))
    expect(remove).toHaveBeenCalledWith('scroll', expect.any(Function))
    expect(removeWindow).toHaveBeenCalledWith('keydown', expect.any(Function))
  })
})
