type IconName = 'open' | 'save' | 'export' | 'undo' | 'redo' | 'select' | 'hand' | 'fit' | 'layers' | 'search' | 'help' | 'close' | 'chevron' | 'grid' | 'eye' | 'object' | 'pin' | 'alert';
const paths: Record<IconName, string> = {
  open: 'M3 7h6l2 2h10l-3 11H3V7zm0 0V4h7l2 3h6v2',
  save: 'M4 3h13l4 4v14H3V3h1zm3 0v7h10V3M7 21v-7h10v7',
  export: 'M12 15V3m-5 5 5-5 5 5M4 14v7h16v-7',
  undo: 'M9 5 3 11l6 6M3 11h11a6 6 0 0 1 6 6',
  redo: 'm15 5 6 6-6 6m6-6H10a6 6 0 0 0-6 6',
  select: 'm5 3 14 10-7 1-3 7L5 3z',
  hand: 'M8 12V5a2 2 0 0 1 4 0v6-8a2 2 0 0 1 4 0v8-6a2 2 0 0 1 4 0v10c0 5-3 7-7 7-3 0-5-2-6-4l-4-6a2 2 0 0 1 3-2l2 2z',
  fit: 'M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M8 8h8v8H8z',
  layers: 'm12 3 10 6-10 6L2 9l10-6zm-9 11 9 5 9-5M3 19l9 5 9-5',
  search: 'M10 3a7 7 0 1 0 0 14 7 7 0 0 0 0-14zm5 12 6 6',
  help: 'M9 8a3 3 0 1 1 5 2c-2 1-2 2-2 3m0 4h.01M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z',
  close: 'm6 6 12 12M6 18 18 6', chevron: 'm9 5 7 7-7 7',
  grid: 'M3 3h18v18H3V3zm0 6h18M3 15h18M9 3v18m6-18v18',
  eye: 'M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zm10-3a3 3 0 1 0 0 6 3 3 0 0 0 0-6z',
  object: 'm12 3 9 5v9l-9 5-9-5V8l9-5zm-9 5 9 5 9-5m-9 5v9',
  pin: 'M12 22s7-8 7-13a7 7 0 0 0-14 0c0 5 7 13 7 13zm0-16a3 3 0 1 0 0 6 3 3 0 0 0 0-6z',
  alert: 'm12 3 10 18H2L12 3zm0 6v5m0 3h.01',
};
export function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}
