import { useEffect, useState } from 'react'
import { checkBackendHealth } from './api'
import './App.css'

type ConnectionState = 'loading' | 'success' | 'error'

const connectionMessages: Record<ConnectionState, string> = {
  loading: 'Проверка подключения...',
  success: 'Сервер доступен',
  error: 'Сервер недоступен',
}

function App() {
  const [connectionState, setConnectionState] = useState<ConnectionState>('loading')

  useEffect(() => {
    const controller = new AbortController()

    void checkBackendHealth(controller.signal)
      .then(() => {
        if (!controller.signal.aborted) {
          setConnectionState('success')
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setConnectionState('error')
        }
      })

    return () => controller.abort()
  }, [])

  return (
    <main className="page-shell">
      <section className="welcome-card" aria-labelledby="project-title">
        <div className="brand-mark" aria-hidden="true">
          У
        </div>
        <p className="eyebrow">Справочная система для абитуриентов</p>
        <h1 id="project-title">UNEcon RAG Assistant</h1>
        <p className="description">
          Техническая основа проекта для работы с официальными материалами университета.
        </p>

        <div
          className={`connection-status connection-status--${connectionState}`}
          role="status"
          aria-live="polite"
        >
          <span className="connection-dot" aria-hidden="true" />
          <span>{connectionMessages[connectionState]}</span>
        </div>

        <p className="project-note">Сейчас доступна проверка соединения с сервером приложения.</p>
      </section>
      <footer className="page-footer">
        Разработка справочного модуля с контекстным дополнением для абитуриентов
      </footer>
    </main>
  )
}

export default App
