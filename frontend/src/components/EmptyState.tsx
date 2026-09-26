const examples = [
  ['Сроки приёма', 'Когда начинается приём документов?'],
  ['Вступительные испытания', 'Какие экзамены нужны для поступления на Экономику?'],
  ['Стоимость обучения', 'Сколько стоит обучение на Экономике?'],
  ['Контакты', 'Как связаться с отделом по работе с договорами?'],
]

export function EmptyState({ onExample }: { onExample: (question: string) => void }) {
  return <section className="empty-state" aria-labelledby="welcome-title">
    <p className="eyebrow">Поступление — шаг за шагом</p>
    <h2 id="welcome-title">Задайте вопрос<br className="desktop-break" /> о поступлении</h2>
    <p className="intro">Помогу найти информацию в официальных материалах СПбГЭУ за 2026 год.
      В ответе будут ссылки на источники — вы сможете проверить детали.</p>
    <div className="examples" aria-label="Примеры вопросов">
      {examples.map(([label, question]) => <button type="button" className="example" key={label}
        onClick={() => onExample(question)}>
        <span className="example-label">{label}</span>
        <span className="example-question">{question}<span aria-hidden="true">↗</span></span>
      </button>)}
    </div>
  </section>
}
