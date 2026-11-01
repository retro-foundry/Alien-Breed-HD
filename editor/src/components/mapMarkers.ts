
/** Screen-space drawing and picking share the same visible marker geometry. */
export const cornerHandleHalfSize = 3;
export const objectMarkerRadius = 3.5;
export const selectedObjectRadius = 5;
export const selectedObjectRingRadius = 9;
export const objectMarkerStroke = 1.5;
export const startLabelFont = '10px ui-monospace, monospace';
export const startLabelOffset = { x: 10, y: 4 };

export function startMarkerPoints(x: number, y: number) {
  return [{ x, z: y - 9 }, { x: x + 7, z: y + 6 }, { x: x - 7, z: y + 6 }];
}

export function startMarkerHit(x: number, y: number, center: { x: number; y: number }, label: TextMetrics): boolean {
  const vertices = startMarkerPoints(center.x, center.y);
  if (vertices.every((a, index) => {
    const b = vertices[(index + 1) % vertices.length]!;
    return (b.x - a.x) * (y - a.z) - (b.z - a.z) * (x - a.x) >= 0;
  })) return true;
  // Include the one-pixel visible border (the triangle is stroked at width 2).
  if (vertices.some((a, index) => {
    const b = vertices[(index + 1) % vertices.length]!, dx = b.x - a.x, dy = b.z - a.z;
    const along = Math.max(0, Math.min(1, ((x - a.x) * dx + (y - a.z) * dy) / (dx * dx + dy * dy)));
    return Math.hypot(x - a.x - along * dx, y - a.z - along * dy) <= 1;
  })) return true;
  const left = center.x + startLabelOffset.x, baseline = center.y + startLabelOffset.y;
  return x >= left - label.actualBoundingBoxLeft && x <= left + label.actualBoundingBoxRight
    && y >= baseline - label.actualBoundingBoxAscent && y <= baseline + label.actualBoundingBoxDescent;
}
