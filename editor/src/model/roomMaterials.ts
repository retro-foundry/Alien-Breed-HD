/** Runtime room identity and material provenance are separate. A newly
 * allocated room may still inherit individually identified native walls. */
export function hasNativeWallMaterials(originZone: number | null, sources: readonly (number | null)[]): boolean {
  return originZone !== null || sources.some(source => source !== null);
}
