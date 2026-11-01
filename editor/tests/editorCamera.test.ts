import { expect, it } from 'vitest';
import { editingViews, cameraFor, cameraBasis, frameCamera, projectPoint, screenRay, rayPlane, rayAxis, rayTriangle, clipCameraPolygon, zoomCamera, add, mul, type Vector3 } from '../src/model/editorCamera';
const point = { x: 128, y: -64, z: 192 };
it.each(editingViews)('$name projection and inverse ray agree in native coordinates', ({ id }) => {
  const camera = { ...cameraFor(id), target: { x: 64, y: -128, z: 64 } }, p = projectPoint(point, camera, 900, 600);
  const result = rayPlane(screenRay(p.x, p.y, camera, 900, 600), point, cameraBasis(camera).back)!;
  for (const axis of ['x', 'y', 'z'] as const) expect(result[axis]).toBeCloseTo(point[axis], 7);
  expect(p.visible).toBe(true);
});
it('top matches map directions and front/side hide the intended axis', () => {
  const origin = { x: 0, y: 0, z: 0 };
  const top = cameraFor('top'), a = projectPoint(origin, top, 800, 600);
  expect(projectPoint({ ...origin, x: 100 }, top, 800, 600).x).toBeGreaterThan(a.x);
  expect(projectPoint({ ...origin, z: 100 }, top, 800, 600).y).toBeLessThan(a.y);
  for (const [view, axis] of [['front', 'z'], ['back', 'z'], ['left', 'x'], ['right', 'x']] as const) {
    const camera = cameraFor(view), a = projectPoint(origin, camera, 800, 600), b = projectPoint({ ...origin, [axis]: 100 }, camera, 800, 600);
    expect(b.x).toBeCloseTo(a.x); expect(b.y).toBeCloseTo(a.y);
    expect(projectPoint({ ...origin, y: -100 }, camera, 800, 600).y).toBeLessThan(a.y);
  }
});
it.each(editingViews)('$name frames every point and keeps cursor-anchored zoom stable', ({ id }) => {
  const points = [{ x: -300, y: -300, z: -500 }, { x: 400, y: 0, z: 300 }];
  const camera = frameCamera(cameraFor(id), points, 900, 600);
  for (const point of points) { const p = projectPoint(point, camera, 900, 600); expect(p.visible).toBe(true); expect(p.x).toBeGreaterThan(0); expect(p.x).toBeLessThan(900); expect(p.y).toBeGreaterThan(0); expect(p.y).toBeLessThan(600); }
  const p = rayPlane(screenRay(350, 220, camera, 900, 600), camera.target, cameraBasis(camera).back)!;
  const next = projectPoint(p, zoomCamera(camera, 350, 220, .7, 900, 600), 900, 600);
  expect(next.x).toBeCloseTo(350); expect(next.y).toBeCloseTo(220);
});
it('axis dragging locks the other dimensions and rejects an edge-on axis', () => {
  const camera = cameraFor('front'), basis = cameraBasis(camera), p = add(camera.target, mul(basis.right, 37));
  const pixel = projectPoint(p, camera, 800, 600), ray = screenRay(pixel.x, pixel.y, camera, 800, 600);
  expect(rayAxis(ray, camera.target, { x: 1, y: 0, z: 0 })).toBeCloseTo(37);
  expect(rayAxis(ray, camera.target, { x: 0, y: 0, z: 1 })).toBeNull();
});
it('clips near-plane crossings and intersects both triangle windings', () => {
  const camera = cameraFor('front');
  const points: Vector3[] = [{ x: -20, y: -20, z: -2000 }, { x: 20, y: -20, z: 0 }, { x: 0, y: 20, z: 0 }];
  const clipped = clipCameraPolygon(points, camera); expect(clipped).toHaveLength(4);
  expect(clipped.every(p => projectPoint(p, camera, 800, 600).visible)).toBe(true);
  const triangle: [Vector3, Vector3, Vector3] = [{ x: -100, y: -100, z: 0 }, { x: 100, y: -100, z: 0 }, { x: 0, y: 100, z: 0 }];
  const ray = screenRay(400, 300, camera, 800, 600);
  expect(rayTriangle(ray, ...triangle)).toBeCloseTo(1024);
  expect(rayTriangle(ray, triangle[2], triangle[1], triangle[0])).toBeCloseTo(1024);
});
