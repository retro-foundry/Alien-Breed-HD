import { useState } from 'react';
import type { LevelDocument } from '../model/level';
import type { EditCommand } from '../model/commands';
import { applyCommand } from '../model/commands';
import { buildPlayableLevel } from '../model/playableBuild';
import { captureMechanismSettings, mechanismOwnerKey, mechanismSettingsOwners } from '../model/mechanismSettingsTransfer';
import { linkSwitchMechanism, type SwitchLinkMode } from '../model/switchMechanismLinks';

export const switchTargets = (document: LevelDocument) => [...mechanismSettingsOwners(document, 'door'), ...mechanismSettingsOwners(document, 'lift')];
export interface SwitchTargetDraft { target: string; mode: SwitchLinkMode }

export function SwitchTargetFields({ document, prefix, value, change }: {
  document: LevelDocument; prefix: string; value: SwitchTargetDraft; change: (value: SwitchTargetDraft) => void;
}) {
  const targets = switchTargets(document);
  return <div className="switch-target-fields">
    <label className="select-field"><span>Controls</span><select aria-label={`${prefix} controls`} value={value.target} onChange={event => change({ ...value, target: event.target.value })}>
      <option value="">No door or lift yet</option>{targets.map(item => <option key={mechanismOwnerKey(item.owner)} value={mechanismOwnerKey(item.owner)}>{item.label}</option>)}
    </select></label>
    {!targets.length && <p className="field-note">Create a door or lift first, or place this switch now and connect it later.</p>}
    {value.target && <>
      <label className="select-field"><span>How it controls</span><select aria-label={`${prefix} connection mode`} value={value.mode} onChange={event => change({ ...value, mode: event.target.value as SwitchLinkMode })}>
        <option value="activate">Start opening / raising when on</option><option value="requirement">Unlock the existing trigger</option>
      </select></label>
      <p className="field-note">{value.mode === 'activate' ? 'Uses the timer trigger to start opening or raising once this switch and all other requirements are on.' : 'Keeps the existing opening rule. The player may still need to use or touch the door or lift.'} Its closing/lowering rule and other key/switch requirements stay the same.</p>
    </>}
  </div>;
}

export function SwitchMechanismConnections({ document, slot, edit }: {
  document: LevelDocument; slot: number; edit: (command: EditCommand | (() => EditCommand)) => void;
}) {
  const [draft, setDraft] = useState<SwitchTargetDraft>({ target: '', mode: 'activate' }), [error, setError] = useState('');
  const targets = switchTargets(document), connected = targets.filter(item => (captureMechanismSettings(document, item.owner).conditions & (1 << (slot + 4))) !== 0);
  const apply = (key: string, required: boolean) => {
    try {
      const target = targets.find(item => mechanismOwnerKey(item.owner) === key);
      if (!target) throw new Error('Choose an existing door or lift to connect.');
      const command = linkSwitchMechanism(document, target.owner, slot, required, draft.mode);
      buildPlayableLevel(applyCommand(document, command), -1);
      edit(command); setDraft({ target: '', mode: 'activate' }); setError('');
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  return <div className="switch-connections" role="group" aria-label={`Switch ${slot + 1} connections`}>
    <h4>Connected doors and lifts</h4>
    {connected.length ? <ul>{connected.map(item => <li key={mechanismOwnerKey(item.owner)}><span>{item.label}</span><button type="button" aria-label={`Unlink switch ${slot + 1} from ${item.label}`} onClick={() => apply(mechanismOwnerKey(item.owner), false)}>Unlink</button></li>)}</ul>
      : <p className="field-note">No door or lift requires this switch yet.</p>}
    {!!connected.length && <p className="field-note">Unlink clears this switch requirement and keeps the mechanism's trigger rules.</p>}
    <SwitchTargetFields document={document} prefix={`Switch ${slot + 1}`} value={draft} change={setDraft} />
    <button type="button" className="small-button" disabled={!draft.target} onClick={() => apply(draft.target, true)}>Connect switch {slot + 1}</button>
    {error && <p role="alert" className="field-note">{error}</p>}
  </div>;
}
