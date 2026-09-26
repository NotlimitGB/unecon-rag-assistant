import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { ApiError, askQuestion, checkBackendHealth, errorMessages } from '../api'
import type { AnswerResponse } from '../types'
import { Header } from './Header'
import { EmptyState } from './EmptyState'
import { Answer } from './Answer'
import { ErrorNotice } from './ErrorNotice'
import { QuestionForm } from './QuestionForm'
import { WidgetLauncher } from './WidgetLauncher'
import './AssistantWidget.css'

type Result = { state: 'idle' } | { state: 'pending'; question: string } |
  { state: 'success'; question: string; response: AnswerResponse } |
  { state: 'error'; question: string; message: string }

export function AssistantWidget() {
  const [isOpen, setIsOpen] = useState(false)
  const [health, setHealth] = useState<'loading' | 'ready' | 'error'>('loading')
  const [healthAttempt, setHealthAttempt] = useState(0)
  const [draft, setDraft] = useState('')
  const [result, setResult] = useState<Result>({ state: 'idle' })
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const formRef = useRef<HTMLFormElement>(null)
  const answerController = useRef<AbortController | null>(null)
  const rootRef = useRef<HTMLDivElement>(null)
  const launcherRef = useRef<HTMLButtonElement>(null)
  const titleRef = useRef<HTMLHeadingElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const wasOpen = useRef(false)

  useLayoutEffect(() => {
    if (isOpen && !wasOpen.current) {
      const canFocusInput = !window.matchMedia('(max-width: 600px)').matches && health === 'ready' && result.state !== 'pending'
      if (canFocusInput) inputRef.current?.focus({ preventScroll: true })
      else titleRef.current?.focus({ preventScroll: true })
    } else if (!isOpen && wasOpen.current) {
      launcherRef.current?.focus({ preventScroll: true })
    }
    wasOpen.current = isOpen
  }, [isOpen, health, result.state])

  useEffect(() => {
    if (!isOpen) return
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape' && !event.isComposing && !event.defaultPrevented) {
        event.preventDefault()
        setIsOpen(false)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isOpen])

  useEffect(() => {
    const viewport = window.visualViewport
    if (!viewport) return
    function updateViewport() {
      rootRef.current?.style.setProperty('--widget-viewport-height', `${viewport!.height}px`)
      rootRef.current?.style.setProperty('--widget-viewport-top', `${viewport!.offsetTop}px`)
    }
    updateViewport()
    viewport.addEventListener('resize', updateViewport)
    viewport.addEventListener('scroll', updateViewport)
    return () => {
      viewport.removeEventListener('resize', updateViewport)
      viewport.removeEventListener('scroll', updateViewport)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void checkBackendHealth(controller.signal).then(() => {
      if (!controller.signal.aborted) setHealth('ready')
    }).catch(() => {
      if (!controller.signal.aborted) setHealth('error')
    })
    return () => controller.abort()
  }, [healthAttempt])

  useEffect(() => () => { answerController.current?.abort() }, [])

  async function submit(raw: string, clearDraft: boolean) {
    const question = raw.trim()
    if (!question || [...raw].length > 2000 || health !== 'ready' || answerController.current) return
    const controller = new AbortController()
    answerController.current = controller
    if (clearDraft) setDraft('')
    setResult({ state: 'pending', question })
    if (contentRef.current) contentRef.current.scrollTop = 0
    try {
      const response = await askQuestion(question, controller.signal)
      if (!controller.signal.aborted && answerController.current === controller) {
        setResult({ state: 'success', question, response })
      }
    } catch (error) {
      if (!controller.signal.aborted && answerController.current === controller) {
        setResult({ state: 'error', question, message: error instanceof ApiError ? error.message : errorMessages.response })
      }
    } finally {
      if (!controller.signal.aborted && answerController.current === controller) {
        answerController.current = null
        if (wasOpen.current && formRef.current?.contains(document.activeElement)) inputRef.current?.focus({ preventScroll: true })
      }
    }
  }

  return <div className="assistant-widget" ref={rootRef} data-open={isOpen}>
    <WidgetLauncher isOpen={isOpen} buttonRef={launcherRef} onToggle={() => setIsOpen((open) => !open)} />
    <section id="assistant-panel" className="widget-panel" role="dialog" aria-labelledby="widget-title" hidden={!isOpen}>
      <Header titleRef={titleRef} onClose={() => setIsOpen(false)} />
      <div className="widget-content" ref={contentRef}>
      {health !== 'ready' && <div className="connection-notice">
        <p role="status">{health === 'loading' ? 'Проверяем доступность помощника…' : 'Не удалось подключиться к помощнику. Попробуйте проверить соединение ещё раз.'}</p>
        {health === 'error' && <button type="button" className="text-button" onClick={() => {
          setHealth('loading'); setHealthAttempt((attempt) => attempt + 1)
        }}>Повторить проверку</button>}
      </div>}
      {result.state === 'idle' ? <EmptyState onExample={(question) => {
        setDraft(question); inputRef.current?.focus()
      }} /> : <section className="result" aria-label="Результат запроса" aria-busy={result.state === 'pending'}>
        <div className="submitted-question"><span className="eyebrow">Ваш вопрос</span><p>{result.question}</p></div>
        {result.state === 'pending' && <div className="loading-card" role="status" aria-live="polite">
          <span className="loading-dot" aria-hidden="true" />
          <div><p>Ищу информацию в официальных материалах…</p><span>Первый ответ может занять немного больше времени.</span></div>
        </div>}
        {result.state === 'success' && <Answer response={result.response} />}
        {result.state === 'error' && <ErrorNotice message={result.message}
          onRetry={() => { void submit(result.question, false) }} disabled={health !== 'ready'} />}
      </section>}
      </div>
      <QuestionForm draft={draft} onChange={setDraft} onSubmit={() => { void submit(draft, true) }}
        pending={result.state === 'pending'} ready={health === 'ready'} inputRef={inputRef} formRef={formRef} />
    </section>
  </div>
}
