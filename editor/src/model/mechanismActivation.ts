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

export function assertMechanismRange(mechanism: Pick<Mechanism, 'kind' | 'direction' | 'conditions' | 'current'>, top: number, bottom: number): void {
  for (const [name, value] of [['Upper', top], ['Lower', bottom]] as const)
    if (!Number.isFinite(value) || !Number.isInteger(value * 4) || value < -8192 || value > 8191.75)
      throw new Error(`${name} destination must be between -8192 and 8191.75 in 0.25-unit steps.`);
  if (bottom * 4 === 999) throw new Error('The lower destination 249.75 cannot be stored in this level format. Choose 249.5 or 250 instead.');
  if (top > bottom) throw new Error('The upper destination must be above or equal to the lower destination. Smaller height values are higher.');
  if (mechanism.current < top || mechanism.current > bottom) throw new Error(`The range must include the initial position ${mechanism.current}. No animation ranges were changed.`);
  assertMechanismActivation({ ...mechanism, top, bottom });
}
