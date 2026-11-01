import type { Mechanism } from './level';

export function mechanismNeedsEndpoint(mechanism: Pick<Mechanism, 'kind' | 'direction' | 'conditions'>): boolean {
  // Anims.s/objects.c NotGoBackUp permits an ungated occupied door to open
  // from an intermediate position. Lifts and gated doors use endpoint rules.
  return mechanism.direction === 0 && (mechanism.kind === 'lift' || mechanism.conditions !== 0);
}

export function assertMechanismActivation(mechanism: Pick<Mechanism, 'kind' | 'direction' | 'conditions' | 'current' | 'top' | 'bottom'>): void {
  if (mechanismNeedsEndpoint(mechanism) && mechanism.current !== mechanism.top && mechanism.current !== mechanism.bottom)
    throw new Error(`This stationary ${mechanism.kind} starts at ${mechanism.current}. Keep one destination at this height so its trigger rules can start movement.`);
}
