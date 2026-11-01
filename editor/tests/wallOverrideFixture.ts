import { readFileSync } from 'node:fs';
import { roomHeightFixture } from './roomHeightFixture';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import { nativeWallSheets } from '../src/model/wallOverrides';
import { readWallTexture } from '../src/model/wallTextures';

export function wallOverrideFixture() {
  const fixture = roomHeightFixture(), choices = roomAppearanceChoices(fixture.document, fixture.from);
  const sheet = nativeWallSheets[choices.lower[0]!.texture!]!;
  const original = readWallTexture(new Uint8Array(readFileSync(new URL(`../../amiga/data/includes/walls/${sheet.name}`, import.meta.url))), sheet).bytes;
  const replacement = original.slice();
  // A deliberately uniform native test material makes any surviving original
  // texture or failed reset visible at every game brightness/distance.
  for (let offset = 0; offset < 2048; offset += 2) { replacement[offset] = 15; replacement[offset + 1] = 0; }
  return { ...fixture, sheet, original, replacement };
}
