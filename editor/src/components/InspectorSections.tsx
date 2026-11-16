import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { Icon } from './Icon';

interface Section { element: HTMLElement; title: string; text: string }
function propertyText(element: Element) {
  return [element.textContent, ...[...element.querySelectorAll('[aria-label]')].map(field => field.getAttribute('aria-label'))]
    .join(' ').replace(/\s+/g, ' ').toLocaleLowerCase();
}

/** Keep property forms mounted while finding and revealing their settings. */
export function InspectorSections({ children }: { children: ReactNode }) {
  const content = useRef<HTMLDivElement>(null);
  const search = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState('');
  const [sections, setSections] = useState<Section[]>([]);
  const needle = query.trim().toLocaleLowerCase();
  useLayoutEffect(() => {
    const root = content.current!;
    const collect = () => {
      const next = [...root.querySelectorAll<HTMLElement>(':scope > .property-section')].map(element => ({
        element, title: element.querySelector('h3, h4, summary, legend')?.textContent?.trim() || 'Selection actions',
        text: propertyText(element),
      }));
      setSections(previous => previous.length === next.length && next.every((section, index) =>
        section.element === previous[index]?.element && section.title === previous[index]?.title && section.text === previous[index]?.text) ? previous : next);
    };
    collect();
    const observer = new MutationObserver(collect);
    observer.observe(root, { childList: true, subtree: true, characterData: true });
    return () => observer.disconnect();
  }, []);
  useLayoutEffect(() => {
    const expanded: HTMLDetailsElement[] = [];
    for (const section of sections) {
      section.element.hidden = !!needle && !section.text.includes(needle);
      if (needle && !section.element.hidden) {
        section.element.querySelectorAll('details:not([open])').forEach(details => {
          if (propertyText(details).includes(needle)) {
            (details as HTMLDetailsElement).open = true;
            expanded.push(details as HTMLDetailsElement);
          }
        });
      }
    }
    return () => {
      sections.forEach(section => { section.element.hidden = false; });
      expanded.forEach(details => { details.open = false; });
    };
  }, [sections, needle]);
  useEffect(() => { if (content.current) content.current.scrollTop = 0; }, [needle]);
  const matches = sections.filter(section => !needle || section.text.includes(needle));
  return <>
    <div className="inspector-tools">
      <label className="search-box"><Icon name="search" size={15} /><input ref={search} aria-label="Find a setting" placeholder="Find a setting…" value={query}
        title="Search this selection's properties, including settings inside expandable sections. Escape clears the search."
        onChange={event => setQuery(event.target.value)} onKeyDown={event => {
          if (event.key === 'Escape') { event.stopPropagation(); event.preventDefault(); setQuery(''); }
        }} />{query && <button aria-label="Clear setting search" onClick={() => { setQuery(''); search.current?.focus(); }}><Icon name="close" size={14} /></button>}</label>
      <select aria-label="Jump to properties" value="" onChange={event => {
        const section = sections[Number(event.target.value)];
        if (!section || !content.current) return;
        const root = content.current;
        root.scrollTop += section.element.getBoundingClientRect().top - root.getBoundingClientRect().top;
        const heading = section.element.querySelector<HTMLElement>('h3, h4, summary, legend');
        if (heading?.tagName === 'SUMMARY') heading.closest('details')!.open = true;
        if (heading) { heading.tabIndex = -1; heading.focus({ preventScroll: true }); }
      }}><option value="">Jump to properties…</option>{matches.map(section => <option key={sections.indexOf(section)} value={sections.indexOf(section)}>{section.title}</option>)}</select>
      {needle && <span className="setting-results" role="status">{matches.length ? `${matches.length} matching ${matches.length === 1 ? 'section' : 'sections'}` : 'No settings match. Try heights, textures or lighting.'}</span>}
    </div>
    <div className="inspector-scroll" ref={content}>{children}</div>
  </>;
}
