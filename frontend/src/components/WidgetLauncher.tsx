import type { RefObject } from 'react'

export function WidgetLauncher({ isOpen, onToggle, buttonRef }: {
  isOpen: boolean; onToggle: () => void; buttonRef: RefObject<HTMLButtonElement | null>
}) {
  return <button type="button" className="widget-launcher" ref={buttonRef} onClick={onToggle}
    aria-label={isOpen ? 'Свернуть помощника абитуриента' : 'Открыть помощника абитуриента'}
    aria-expanded={isOpen} aria-controls="assistant-panel">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M20 11.5a8 8 0 0 1-8 8H5l-3 2v-10a8 8 0 0 1 8-8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      <path d="m17 2 1.4 3.6L22 7l-3.6 1.4L17 12l-1.4-3.6L12 7l3.6-1.4L17 2Z" fill="currentColor" />
      <path d="M7 12h3m-3 3h7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
    <span>Помощник</span>
  </button>
}
