const sections = [
  { id: 'programs', number: '01', title: 'Бакалавриат и специалитет', text: 'Знакомство с направлениями подготовки — первый шаг к выбору образовательного пути.', label: 'Образовательные программы' },
  { id: 'documents', number: '02', title: 'Документы и правила приёма', text: 'При подготовке к поступлению сверяйтесь с действующими официальными документами университета.', label: 'Документы' },
  { id: 'exams', number: '03', title: 'Вступительные испытания', text: 'Требования к испытаниям зависят от выбранной программы. Помощник поможет найти соответствующий источник.', label: 'Испытания' },
  { id: 'tuition', number: '04', title: 'Платное обучение', text: 'Информацию об условиях и стоимости обучения важно проверять по официальным материалам.', label: 'Платное обучение' },
]

export function DemoHostPage() {
  return <div className="demo-host">
    <header className="demo-header">
      <a className="demo-brand" href="#admissions">СПбГЭУ<span>Поступающим</span></a>
      <nav aria-label="Разделы демонстрационной страницы">
        <a href="#admissions">Поступление</a>
        <a href="#programs">Образовательные программы</a>
        <a href="#documents">Документы</a>
        <a href="#contacts">Контакты</a>
      </nav>
    </header>
    <main className="demo-main">
      <section className="demo-hero" id="admissions">
        <p className="demo-kicker">Приёмная кампания 2026</p>
        <h1>Поступление<br />в СПбГЭУ</h1>
        <p className="demo-intro">Информация для поступающих на программы бакалавриата и специалитета.</p>
        <a className="demo-section-link" href="#programs">С чего начать <span aria-hidden="true">↓</span></a>
        <div className="demo-hero-art" aria-hidden="true"><span>2026</span></div>
      </section>
      <section className="demo-information" aria-labelledby="demo-guide-title">
        <div className="demo-section-heading"><p className="demo-kicker">Ваш следующий шаг</p><h2 id="demo-guide-title">О поступлении</h2></div>
        <div className="demo-cards">{sections.map((section) => <section className="demo-card" id={section.id} key={section.id}>
          <span className="demo-card-number">{section.number}</span>
          <h3>{section.title}</h3><p>{section.text}</p>
        </section>)}</div>
      </section>
      <section className="demo-contact" id="contacts" aria-labelledby="demo-contact-title">
        <p className="demo-kicker">Официальная информация</p>
        <h2 id="demo-contact-title">Остались вопросы?</h2>
        <p>Откройте помощника в правом нижнем углу. Он найдёт информацию в материалах приёмной кампании и покажет источники. Важные детали уточняйте в приёмной комиссии.</p>
      </section>
    </main>
    <footer className="demo-footer"><span>СПбГЭУ · Поступающим</span><p>Демонстрационная страница интеграции справочного модуля</p></footer>
  </div>
}
