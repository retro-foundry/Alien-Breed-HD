import { useEffect, useId, useState } from 'react';
import { createPortal } from 'react-dom';

/** Share native control descriptions with mouse and keyboard users. */
export function ControlTooltips() {
  const id = useId();
  const [tip, setTip] = useState<{ text: string; left: number; top: number; above: boolean; width: number; host: HTMLElement } | null>(null);
  useEffect(() => {
    let active: Element | null = null, nativeTitle = '';
    const restore = () => {
      if (!active) return;
      if (active.getAttribute('title') === '') active.setAttribute('title', nativeTitle);
      const descriptions = (active.getAttribute('aria-describedby') ?? '').split(/\s+/).filter(value => value && value !== id);
      if (descriptions.length) active.setAttribute('aria-describedby', descriptions.join(' ')); else active.removeAttribute('aria-describedby');
      active = null;
    };
    const hide = () => { restore(); setTip(null); };
    const show = (target: EventTarget | null) => {
      const next = target instanceof Element ? target.closest('.app [title], .app [data-control-help]') : null;
      if (next === active) return;
      const text = next?.getAttribute('data-control-help') ?? next?.getAttribute('title');
      hide();
      if (!next || !text || next.tagName === 'IFRAME') return;
      active = next; nativeTitle = next.getAttribute('title') ?? '';
      if (next.hasAttribute('title')) next.setAttribute('title', '');
      next.setAttribute('aria-describedby', `${next.getAttribute('aria-describedby') ?? ''} ${id}`.trim());
      const rect = next.getBoundingClientRect(), width = Math.min(320, window.innerWidth - 16), above = rect.bottom + 90 > window.innerHeight;
      setTip({ text, width, above, host: next.closest<HTMLElement>('dialog[open]') ?? document.body,
        left: Math.max(8, Math.min(window.innerWidth - width - 8, rect.left + (rect.width - width) / 2)), top: above ? rect.top - 8 : rect.bottom + 8 });
    };
    // Focusing an off-screen toolbar control scrolls it into view. Synthetic
    // pointer enter/leave events from that scroll must not steal its help or
    // aria-describedby binding from keyboard users.
    const over = (event: PointerEvent) => {
      if (active === document.activeElement) return;
      if (event.pointerType !== 'touch') show(event.target);
    };
    const focus = (event: FocusEvent) => show(event.target);
    const leave = (event: PointerEvent | FocusEvent) => { if (event instanceof PointerEvent && active === document.activeElement) return;
      if (active && event.target instanceof Node && active.contains(event.target)
      && !(event.relatedTarget instanceof Node && active.contains(event.relatedTarget))) hide(); };
    const reposition = () => {
      const focused = active && document.activeElement === active ? active : null;
      hide(); if (focused) show(focused);
    };
    const key = (event: KeyboardEvent) => { if (event.key === 'Escape') hide(); };
    document.addEventListener('pointerover', over); document.addEventListener('pointerout', leave);
    document.addEventListener('focusin', focus); document.addEventListener('focusout', leave);
    document.addEventListener('keydown', key); document.addEventListener('pointerdown', hide);
    window.addEventListener('scroll', reposition, true); window.addEventListener('resize', reposition);
    return () => {
      restore(); document.removeEventListener('pointerover', over); document.removeEventListener('pointerout', leave);
      document.removeEventListener('focusin', focus); document.removeEventListener('focusout', leave);
      document.removeEventListener('keydown', key); document.removeEventListener('pointerdown', hide);
      window.removeEventListener('scroll', reposition, true); window.removeEventListener('resize', reposition);
    };
  }, [id]);
  return tip ? createPortal(<div id={id} role="tooltip" className="control-tooltip" style={{ left: tip.left, top: tip.top, width: tip.width, transform: tip.above ? 'translateY(-100%)' : undefined }}>{tip.text}</div>, tip.host) : null;
}
