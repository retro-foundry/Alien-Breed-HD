import { describe, expect, it } from 'vitest';
import { findPortalEnvelope } from '../src/model/portalEnvelope';
import { legacySide } from '../src/model/visibilityGeometry';
import type { Position } from '../src/model/geometry';
import { normaliseRoomOutline } from '../src/model/geometry';

const room = (x: number, z: number): Position[] => [{ x, z }, { x, z: z + 128 }, { x: x + 128, z: z + 128 }, { x: x + 128, z }];

describe('source-backed portal tangent envelopes', () => {
  it('finds supporting tangents in every direction independently of gesture winding', () => {
    const source = room(0, 0);
    for (const [x, z] of [[256, 0], [256, 256], [0, 256], [-256, 256], [-256, 0], [-256, -256], [0, -256], [256, -256]]) {
      const target = room(x!, z!), envelope = findPortalEnvelope(source, target)!;
      expect(envelope).not.toBeNull();
      expect(findPortalEnvelope(normaliseRoomOutline(source.toReversed()), normaliseRoomOutline(target.toReversed()))).toEqual(envelope);
      for (const [a, b] of [[envelope.points[0], envelope.points[1]], [envelope.points[3], envelope.points[2]]]) {
        const values = [...source, ...target].map(p => legacySide(a!, b!, p));
        expect(values.every(n => n <= 0) || values.every(n => n >= 0)).toBe(true);
      }
    }
  });
  it('handles a shared portal without an unbounded geometry loop', () => {
    const envelope = findPortalEnvelope(room(0, 0), room(128, 0));
    expect(envelope).not.toBeNull();
    expect(findPortalEnvelope(room(128, 0), room(0, 0))).not.toBeNull();
  });
  it('rejects incomplete boundaries before following tangent indices', () => {
    expect(() => findPortalEnvelope([{ x: 0, z: 0 }], room(512, 0))).toThrow(/three to ten/);
  });
});
