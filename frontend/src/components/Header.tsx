import type { RefObject } from 'react'

export function Header({ titleRef, onClose }: {
  titleRef: RefObject<HTMLHeadingElement | null>; onClose: () => void
}) {
  return <header className="widget-header">
    <div>
      <h2 id="widget-title" tabIndex={-1} ref={titleRef}>Помощник абитуриента</h2>
      <p>Бакалавриат и специалитет · 2026</p>
    </div>
    <button type="button" className="widget-close" aria-label="Закрыть помощника" onClick={onClose}>
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="m5 5 10 10M15 5 5 15" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" /></svg>
    </button>
  </header>
}
