import type { AnswerResponse } from '../types'
import { Sources } from './Sources'

export function Answer({ response }: { response: AnswerResponse }) {
  return <article className="answer-card" aria-labelledby="answer-title">
    <h2 id="answer-title">{response.status === 'answered' ? 'Ответ по официальным материалам' : 'Недостаточно сведений'}</h2>
    <p className="answer-text">{response.answer}</p>
    {response.status === 'answered' && <Sources citations={response.citations} />}
  </article>
}
