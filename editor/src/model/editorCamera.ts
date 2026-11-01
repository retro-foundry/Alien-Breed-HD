/** Editor cameras use the native coordinates: X/Z horizontal, larger Y down.
 * Top therefore matches the existing map (X right, Z up). */
export interface Vector3 { x: number; y: number; z: number }
export interface Ray { origin: Vector3; direction: Vector3 }
export type EditingView = 'top' | 'bottom' | 'front' | 'back' | 'left' | 'right' | 'orthographic' | 'perspective';
export const editingViews: readonly { id: EditingView; name: string }[] = [
  { id: 'top', name: 'Top (X / Z)' }, { id: 'bottom', name: 'Bottom (X / Z)' },
  { id: 'front', name: 'Front (X / Y)' }, { id: 'back', name: 'Back (X / Y)' },
  { id: 'left', name: 'Left (Z / Y)' }, { id: 'right', name: 'Right (Z / Y)' },
  { id: 'orthographic', name: '3D orthographic' }, { id: 'perspective', name: '3D perspective' },
];
export interface EditorCamera { target: Vector3; distance: number; yaw: number; pitch: number; perspective: boolean }
export const add = (a: Vector3, b: Vector3): Vector3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
export const sub = (a: Vector3, b: Vector3): Vector3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
export const mul = (a: Vector3, n: number): Vector3 => ({ x: a.x * n, y: a.y * n, z: a.z * n });
export const dot = (a: Vector3, b: Vector3) => a.x * b.x + a.y * b.y + a.z * b.z;
export const cross = (a: Vector3, b: Vector3): Vector3 => ({ x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x });
export const unit = (v: Vector3) => mul(v, 1 / Math.hypot(v.x, v.y, v.z));
export function cameraFor(view: EditingView): EditorCamera {
  return { target: { x: 0, y: 0, z: 0 }, distance: 1024, perspective: view === 'perspective',
    yaw: view === 'left' ? -Math.PI / 2 : view === 'right' ? Math.PI / 2 : view === 'back' ? Math.PI : view === 'orthographic' || view === 'perspective' ? -Math.PI / 4 : 0,
    pitch: view === 'top' ? Math.PI / 2 : view === 'bottom' ? -Math.PI / 2 : view === 'orthographic' || view === 'perspective' ? Math.PI / 5 : 0 };
}
export function cameraBasis(camera: EditorCamera) {
  const { yaw, pitch } = camera;
  const back = { x: Math.sin(yaw) * Math.cos(pitch), y: -Math.sin(pitch), z: -Math.cos(yaw) * Math.cos(pitch) };
  const right = { x: Math.cos(yaw), y: 0, z: Math.sin(yaw) };
  return { back, right, up: cross(back, right), eye: add(camera.target, mul(back, camera.distance)) };
}
export function projectPoint(point: Vector3, camera: EditorCamera, width: number, height: number) {
  const basis = cameraBasis(camera), delta = sub(point, camera.target), depth = camera.distance - dot(delta, basis.back);
  const scale = Math.min(width, height) * .9 / (camera.perspective ? depth : camera.distance);
  return { x: width / 2 + dot(delta, basis.right) * scale, y: height / 2 - dot(delta, basis.up) * scale, depth, visible: depth > .1 };
}
export function screenRay(x: number, y: number, camera: EditorCamera, width: number, height: number): Ray {
  const basis = cameraBasis(camera), f = Math.min(width, height) * .9;
  const offset = add(mul(basis.right, (x - width / 2) / f), mul(basis.up, (height / 2 - y) / f));
  return camera.perspective ? { origin: basis.eye, direction: unit(sub(offset, basis.back)) }
    : { origin: add(basis.eye, mul(offset, camera.distance)), direction: mul(basis.back, -1) };
}
export function rayPlane(ray: Ray, origin: Vector3, normal: Vector3): Vector3 | null {
  const denominator = dot(ray.direction, normal);
  if (Math.abs(denominator) < 1e-8) return null;
  const distance = dot(sub(origin, ray.origin), normal) / denominator;
  return distance >= 0 ? add(ray.origin, mul(ray.direction, distance)) : null;
}
/** Closest point on a transform axis; null when looking along that axis. */
export function rayAxis(ray: Ray, origin: Vector3, axis: Vector3): number | null {
  const parallel = dot(ray.direction, axis), denominator = 1 - parallel * parallel;
  if (denominator < 1e-8) return null;
  const delta = sub(ray.origin, origin);
  return (dot(delta, axis) - parallel * dot(delta, ray.direction)) / denominator;
}
export function rayTriangle(ray: Ray, a: Vector3, b: Vector3, c: Vector3): number | null {
  const ab = sub(b, a), ac = sub(c, a), p = cross(ray.direction, ac), determinant = dot(ab, p);
  if (Math.abs(determinant) < 1e-8) return null;
  const from = sub(ray.origin, a), u = dot(from, p) / determinant;
  if (u < -1e-7 || u > 1 + 1e-7) return null;
  const q = cross(from, ab), v = dot(ray.direction, q) / determinant;
  if (v < -1e-7 || u + v > 1 + 1e-7) return null;
  const distance = dot(ac, q) / determinant;
  return distance >= .1 ? distance : null;
}
export function clipCameraPolygon(points: readonly Vector3[], camera: EditorCamera): Vector3[] {
  const { back } = cameraBasis(camera), depth = (p: Vector3) => camera.distance - dot(sub(p, camera.target), back);
  const result: Vector3[] = [];
  points.forEach((b, i) => {
    const a = points[(i + points.length - 1) % points.length]!, da = depth(a), db = depth(b);
    if ((da > .2) !== (db > .2)) result.push(add(a, mul(sub(b, a), (.2 - da) / (db - da))));
    if (db > .2) result.push(b);
  });
  return result;
}
export function zoomCamera(camera: EditorCamera, x: number, y: number, factor: number, width: number, height: number): EditorCamera {
  const normal = cameraBasis(camera).back, before = rayPlane(screenRay(x, y, camera, width, height), camera.target, normal);
  const next = { ...camera, distance: Math.max(5, Math.min(200000, camera.distance * factor)) };
  const after = rayPlane(screenRay(x, y, next, width, height), camera.target, normal);
  return before && after ? { ...next, target: add(next.target, sub(before, after)) } : next;
}
export function frameCamera(camera: EditorCamera, points: readonly Vector3[], width: number, height: number): EditorCamera {
  if (!points.length || width < 1 || height < 1) return camera;
  const bounds = (axis: keyof Vector3) => [Math.min(...points.map(p => p[axis])), Math.max(...points.map(p => p[axis]))];
  const [x0, x1] = bounds('x'), [y0, y1] = bounds('y'), [z0, z1] = bounds('z');
  const target = { x: (x0! + x1!) / 2, y: (y0! + y1!) / 2, z: (z0! + z1!) / 2 }, basis = cameraBasis(camera);
  const f = Math.min(width, height) * .9;
  let distance = 80;
  for (const point of points) {
    const delta = sub(point, target), depth = camera.perspective ? dot(delta, basis.back) : 0;
    distance = Math.max(distance, depth + Math.abs(dot(delta, basis.right)) * f / Math.max(1, width / 2 - 60),
      depth + Math.abs(dot(delta, basis.up)) * f / Math.max(1, height / 2 - 75), dot(delta, basis.back) + 10);
  }
  return { ...camera, target, distance: distance * 1.15 };
}

/** First-person navigation of the editable geometry camera, independent of the
 * game simulation and saved starts. Turning keeps the eye fixed. */
export function lookFromEye(camera: EditorCamera, yaw: number, pitch: number): EditorCamera {
  const eye=cameraBasis(camera).eye, next={...camera,yaw,pitch:Math.max(-1.5,Math.min(1.5,pitch))};
  return {...next,target:sub(eye,mul(cameraBasis(next).back,next.distance))};
}
export function moveEditingCamera(camera: EditorCamera, forward: number, right: number, vertical: number, seconds: number, fast: boolean): EditorCamera {
  const length=Math.hypot(forward,right,vertical);if(!length)return camera;
  const basis=cameraBasis(camera),step=256*(fast?4:1)*Math.min(.05,Math.max(0,seconds))/length;
  const direction={x:-Math.sin(camera.yaw)*forward+basis.right.x*right,y:vertical,z:Math.cos(camera.yaw)*forward+basis.right.z*right};
  return {...camera,target:add(camera.target,mul(direction,step))};
}
