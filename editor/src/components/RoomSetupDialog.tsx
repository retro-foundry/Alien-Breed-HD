import { useEffect, useId, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { captureFocus, restoreFocus } from '../focus';

/** Present the same validated inspector forms in a workspace-sized dialog. */
export function RoomSetupDialog({ title, children, onClose }: { title: string; children: ReactNode; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null), heading = useId();
  useEffect(() => {
    const previous = captureFocus(), element = dialog.current!;
    element.showModal();
    return () => { element.close(); restoreFocus(previous); };
  }, []);
  return createPortal(<dialog ref={dialog} className="room-setup-dialog" aria-labelledby={heading}
    onCancel={event => { event.preventDefault(); onClose(); }} onKeyDown={event => event.stopPropagation()}>
    <header><div><h2 id={heading}>{title}</h2><p>Apply/Add saves one undoable edit. Escape or Close discards only unapplied settings.</p></div>
      <button type="button" aria-label="Close setup" onClick={onClose}>Close</button></header>
    <div className="room-setup-dialog-body">{children}</div>
  </dialog>, document.body);
}
