/** Keep the focused control inside our same-origin engine iframe, too. */
export function captureFocus(): HTMLElement | null {
  let current = document.activeElement;
  while (current?.tagName === 'IFRAME') {
    const nested = (current as HTMLIFrameElement).contentDocument?.activeElement;
    if (!nested) break;
    current = nested;
  }
  return current && typeof (current as HTMLElement).focus === 'function' ? current as HTMLElement : null;
}

export function restoreFocus(target: HTMLElement | null) {
  if (!target?.isConnected) return;
  // Firefox also needs the containing browsing context to regain focus when
  // the control was already active inside an iframe before opening a panel.
  const frames: HTMLElement[] = [];
  let owner = target.ownerDocument.defaultView;
  while (owner?.frameElement) {
    const frame = owner.frameElement as HTMLElement;
    frames.unshift(frame);
    owner = frame.ownerDocument.defaultView;
  }
  for (const frame of frames) frame.focus({ preventScroll: true });
  target.focus({ preventScroll: true });
}
