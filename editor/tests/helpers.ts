import { readFileSync } from 'node:fs';
import type { LevelFiles } from '../src/model/level';
import { buildRoomEdges, type RoomOutline } from '../src/model/geometry';
import type { VisibilityRoom } from '../src/model/visibilityGeometry';

export function originalFiles(level: number): LevelFiles {
  const folder = new URL(`../../amiga/data/levels/level_${String.fromCharCode(96 + level)}/`, import.meta.url);
  return { data: new Uint8Array(readFileSync(new URL('twolev.bin', folder))), graphics: new Uint8Array(readFileSync(new URL('twolev.graph.bin', folder))),
    clips: new Uint8Array(readFileSync(new URL('twolev.clips', folder))) };
}

export function roomGeometry(rooms: RoomOutline[]) {
  const points: { x: number; z: number }[] = [], coordinates = new Map<string, number>();
  const edges = buildRoomEdges(rooms);
  const runtime: VisibilityRoom[] = rooms.map(room => {
    const boundary = edges.filter(edge => edge.room === room.id);
    const vertices = boundary.map(edge => {
      const key = `${edge.start.x},${edge.start.z}`;
      let id = coordinates.get(key);
      if (id === undefined) { id = points.length; coordinates.set(key, id); points.push(edge.start); }
      return id;
    });
    return { id: Number(room.id), vertices, neighbours: boundary.map(edge => edge.neighbour === null ? -1 : Number(edge.neighbour)) };
  });
  return { points, rooms: runtime };
}
