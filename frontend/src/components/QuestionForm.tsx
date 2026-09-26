import type { RefObject } from 'react'

export function QuestionForm({ draft, onChange, onSubmit, pending, ready, inputRef, formRef }: {
  draft: string; onChange: (value: string) => void; onSubmit: () => void
  pending: boolean; ready: boolean
  inputRef: RefObject<HTMLTextAreaElement | null>; formRef: RefObject<HTMLFormElement | null>
}) {
  const count = [...draft].length
  const tooLong = count > 2000
  const canSubmit = ready && !pending && !tooLong && draft.trim().length > 0
  return <div className="composer-section">
    <form ref={formRef} onSubmit={(event) => { event.preventDefault(); if (canSubmit) onSubmit() }}>
      <label htmlFor="question">Ваш вопрос</label>
      <div className={`composer ${tooLong ? 'composer--invalid' : ''}`}>
        <textarea id="question" ref={inputRef} value={draft} rows={3}
          placeholder="Например, какие документы нужны для поступления?"
          readOnly={pending} aria-invalid={tooLong}
          aria-describedby={`question-hint${count >= 1600 ? ' question-count' : ''}${tooLong ? ' question-error' : ''}`}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing && event.keyCode !== 229) {
              event.preventDefault()
              if (canSubmit) event.currentTarget.form?.requestSubmit()
            }
          }} />
        <div className="composer-actions">
          <span className="keyboard-hint">Enter — отправить <span>· Shift+Enter — новая строка</span></span>
          {count >= 1600 && <span id="question-count" className="character-count">{count} / 2000</span>}
          <button className="submit-button" type="submit" disabled={!canSubmit}>
            {pending ? 'Ожидаем ответ' : 'Отправить'} <span aria-hidden="true">↑</span>
          </button>
        </div>
      </div>
      {tooLong && <p id="question-error" className="field-error" role="alert">Вопрос должен содержать не больше 2000 символов. Сократите текст.</p>}
      <p id="question-hint" className="form-hint">Каждый вопрос рассматривается отдельно. Укажите все важные детали в одном сообщении.</p>
    </form>
    <p className="ai-disclosure">Ответ сформирован ИИ на основе официальных материалов СПбГЭУ. Важную информацию рекомендуем уточнять в приёмной комиссии.</p>
  </div>
}
