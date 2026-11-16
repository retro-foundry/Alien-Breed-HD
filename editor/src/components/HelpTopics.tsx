import { Children, isValidElement, useState, type ReactNode } from 'react';
import { Icon } from './Icon';

function searchableText(node: unknown): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(searchableText).join(' ');
  if (isValidElement<{ children?: ReactNode; rows?: unknown }>(node)) return searchableText(node.props.children) + ' ' + searchableText(node.props.rows);
  return '';
}

export function HelpTopics({ children }: { children: ReactNode }) {
  const [query, setQuery] = useState('');
  const introduction: ReactNode[] = [], topics: { title: string; nodes: ReactNode[] }[] = [];
  for (const node of Children.toArray(children)) {
    if (isValidElement(node) && node.type === 'h3') topics.push({ title: searchableText(node), nodes: [node] });
    else if (topics.length) topics.at(-1)!.nodes.push(node);
    else introduction.push(node);
  }
  const matches = topics.filter(topic => searchableText(topic.nodes).toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()));
  return <div className="help-topics">
    {introduction}
    <div className="help-search"><label className="search-box"><Icon name="search" size={16} /><input aria-label="Search editor help" placeholder="Search help: textures, doors, controls…" value={query} onChange={event => setQuery(event.target.value)}
      onKeyDown={event => { if (event.key === 'Escape' && query) { event.preventDefault(); event.stopPropagation(); setQuery(''); } }} />{query && <button aria-label="Clear help search" onClick={() => setQuery('')}><Icon name="close" size={14} /></button>}</label>
      <span role="status">{query ? `${matches.length} matching topics` : 'Browse the guide or search for a task'}</span></div>
    {matches.map(topic => <section key={topic.title} className="help-topic">{topic.nodes}</section>)}
    {!matches.length && <p className="help-no-results">No topics found. Try a shorter word, or clear the search to browse the guide.</p>}
  </div>;
}
