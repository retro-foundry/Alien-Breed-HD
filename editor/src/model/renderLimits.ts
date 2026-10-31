import declaredLimit from '../../../src/renderer_graphics_limit.json' with { type: 'json' };
import type { LevelDocument } from './level';

// Consume the native declaration directly; do not maintain another limit
// table that can drift from renderer.c's draw and inspection passes.
if (!Number.isSafeInteger(declaredLimit) || declaredLimit < 1) throw new Error('The native renderer graphics limit declaration is invalid.');
export const zoneGraphicsRecordLimit = declaredLimit;

export function assertGraphicsRecordBudget(room: string | number, upper: boolean, count: number): void {
  if (count > zoneGraphicsRecordLimit) throw new Error(`Room ${String(room).replace(/^room-/, '')} ${upper ? 'upper' : 'lower'} floor has ${count} graphics commands; the current game engine supports ${zoneGraphicsRecordLimit} per room and floor. Reduce surface detail or divide the room into separate rooms before building. Save project keeps all source data.`);
}

export function assertPlayableGraphics(document: LevelDocument): void {
  for (const zone of document.zones) {
    assertGraphicsRecordBudget(zone.id, false, zone.lower.length);
    assertGraphicsRecordBudget(zone.id, true, zone.upper.length);
  }
}
