export function ErrorNotice({ message, onRetry, disabled = false }: {
  message: string; onRetry: () => void; disabled?: boolean
}) {
  return <div className="error-notice">
    <p role="alert">{message}</p>
    <button type="button" className="text-button" onClick={onRetry} disabled={disabled}>Попробовать ещё раз</button>
  </div>
}
