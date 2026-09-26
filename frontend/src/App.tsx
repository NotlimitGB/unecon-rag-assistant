import { useEffect, useRef, useState } from 'react'
import { ApiError, askQuestion, checkBackendHealth, errorMessages } from './api'
import type { AnswerResponse } from './types'
import { Header } from './components/Header'
import { EmptyState } from './components/EmptyState'
import { Answer } from './components/Answer'
import { ErrorNotice } from './components/ErrorNotice'
import { QuestionForm } from './components/QuestionForm'
import './App.css'

type Result = { state: 'idle' } | { state: 'pending'; question: string } |
  { state: 'success'; question: string; response: AnswerResponse } |
  { state: 'error'; question: string; message: string }

function App() {
  const [health, setHealth] = useState<'loading' | 'ready' | 'error'>('loading')
  const [healthAttempt, setHealthAttempt] = useState(0)
  const [draft, setDraft] = useState('')
  const [result, setResult] = useState<Result>({ state: 'idle' })
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const formRef = useRef<HTMLFormElement>(null)
  const answerController = useRef<AbortController | null>(null)

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
        if (formRef.current?.contains(document.activeElement)) inputRef.current?.focus()
      }
    }
  }

  return <div className="page-shell">
    <Header />
    <main>
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
      <QuestionForm draft={draft} onChange={setDraft} onSubmit={() => { void submit(draft, true) }}
        pending={result.state === 'pending'} ready={health === 'ready'} inputRef={inputRef} formRef={formRef} />
    </main>
    <footer className="page-footer">Официальные материалы <span aria-hidden="true">·</span> Проверяемые источники</footer>
  </div>
}

export default App
