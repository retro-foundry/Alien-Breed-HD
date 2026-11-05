import type { LevelDocument } from './level';
import type { SourceRoom, RoomSource } from './roomSource';
import { restoreMechanismIdentities, restoreStaticMechanismPoses } from './staticMechanismPose';
import { mechanismDependencies } from './mechanismDependencies';

/** Retired original records remain immutable templates for independent copies. */
export function activeNativeMechanisms(document: LevelDocument, source: RoomSource = document.roomSource) {
  return document.mechanisms.filter(item => !source.removedNativeMechanisms?.some(retired => retired.kind === item.kind && retired.id === item.id));
}
export function validateRetiredMechanisms(document: LevelDocument): void {
  for (const identity of restoreMechanismIdentities(document.roomSource.removedNativeMechanisms ?? [])) {
    if (!document.mechanisms.some(item => item.kind === identity.kind && item.id === identity.id)) throw new Error('A removed controller has no original native template.');
    const dependencies = mechanismDependencies(document, identity.kind, identity.id);
    for (const zone of new Set([...dependencies.graphicsRooms, ...dependencies.walls.flatMap(wall => wall.collisionRooms)])) {
      const room = document.roomSource.rooms.find(room => room.originZone === zone);
      if (room && !room.staticMechanisms?.some(pose => pose.kind === identity.kind && pose.id === identity.id)) throw new Error('A removed original controller has not released every attached room.');
    }
  }
  for (const room of document.roomSource.rooms) {
    const template = document.zones.find(zone => zone.id === room.recordZone)!;
    for (const pose of restoreStaticMechanismPoses(room.staticMechanisms ?? [], template)) {
      const dependencies = mechanismDependencies(document, pose.kind, pose.id);
      const walls = dependencies.walls.filter(wall => wall.room === room.recordZone);
      const affected = [...dependencies.graphicsRooms, ...dependencies.walls.flatMap(wall => wall.collisionRooms)];
      if (!affected.includes(room.recordZone) || pose.plane !== (dependencies.room === room.recordZone)
        || pose.surface !== (dependencies.surface.room === room.recordZone ? dependencies.surface.graphic : null)
        || walls.length !== pose.walls.length || walls.some(wall => !pose.walls.some(item => item.record === wall.graphic))) throw new Error(`Room ${room.id} has an incomplete or foreign retired controller pose.`);
      if (room.originZone !== null && activeNativeMechanisms(document).some(item => item.kind === pose.kind && item.id === pose.id)
        || document.roomSource.controllerCopies?.some(copy => copy.kind === pose.kind && copy.id === pose.id && (copy.room === room.id || copy.graphicsRooms.some(pair => pair[1] === room.id) || copy.collisionRooms.some(pair => pair[1] === room.id)))) throw new Error(`Room ${room.id} still has a live controller for its static pose.`);
    }
  }
}

/** Resolve wall ownership in the source room, not just its shared template. */
export function nativeWallIsMoving(document: LevelDocument, room: SourceRoom, record: number): boolean {
  if (room.originZone !== null) return activeNativeMechanisms(document).some(item => item.walls.some(wall => wall.graphic === record));
  return (document.roomSource.controllerCopies ?? []).some(copy => copy.graphicsRooms.some(pair => pair[1] === room.id)
    && document.mechanisms.some(item => item.kind === copy.kind && item.id === copy.id && item.walls.some(wall => wall.graphic === record)));
}
