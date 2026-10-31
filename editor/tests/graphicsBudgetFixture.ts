import { buildGraphicsStreams, importedGraphicsStreams } from '../src/model/graphicsBuild';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';

/** renderer.c case 12 (Backdrop) consumes no payload and is a no-op here.
 * Put the actual shipped draws at the end so a truncated traversal cannot
 * accidentally pass a pixel comparison by drawing the important records first. */
export function graphicsBudgetFixture(count: number, upper = false) {
  const original = openLevel('Original budget reference', originalFiles(1)), streams = importedGraphicsStreams(original), base = streams.get(0)!;
  const records = upper ? base.upper ?? [] : base.lower;
  if (count < records.length) throw new Error('The budget fixture must preserve all original room draws.');
  const generated = [...Array.from({ length: count - records.length }, () => ({ bytes: new Uint8Array([0, 12]) })), ...records];
  streams.set(0, { ...base, ...(upper ? { upper: generated } : { lower: generated }) });
  return openLevel(`Graphics budget ${count} ${upper ? 'upper' : 'lower'}`, {
    ...original.files, graphics: buildGraphicsStreams(original, undefined, undefined, { streams }),
  });
}
