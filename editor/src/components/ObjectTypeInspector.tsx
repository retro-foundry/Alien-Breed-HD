import { useEffect, useState } from 'react';
import { objectName, type LevelDocument } from '../model/level';
import { objectTemplates } from '../model/objectAuthoring';
import { loadObjectTypes, changeObjectType, type ObjectTypeChoice } from '../model/objectTypes';
import type { EditCommand } from '../model/commands';

export function ObjectTypeInspector({ document, id, edit }: { document: LevelDocument; id: number; edit: (command: EditCommand | (() => EditCommand)) => void }) {
  const [choices, setChoices] = useState<readonly ObjectTypeChoice[]>(), [error, setError] = useState(''), [retry, setRetry] = useState(0);
  useEffect(() => {
    let active = true; setError('');
    loadObjectTypes().then(value => { if (active) setChoices(value); }).catch(error => { if (active) setError(error instanceof Error ? error.message : String(error)); });
    return () => { active = false; };
  }, [retry]);
  const object = document.objects.find(item => item.id === id)!;
  const enabled = objectTemplates(document).some(item => item.id === id);
  return <><label className="select-field object-type-field"><span>Type</span><select aria-label="Object type" value={object.type} disabled={!enabled || !choices}
    title="Choose an enemy, pickup or scenery type. Applies immediately; Ctrl / Command + Z undoes the change. Position, height, room and floor are kept."
    onChange={event => { const choice = choices!.find(item => item.type === Number(event.target.value)); if (choice) edit(() => changeObjectType(document, id, choice)); }}>
    {!choices?.some(item => item.type === object.type) && <option value={object.type}>{objectName(object.type)}</option>}
    {(['Enemies', 'Pickups', 'Scenery'] as const).map(group => <optgroup key={group} label={group}>{choices?.filter(item => item.group === group).map(item => <option key={item.type} value={item.type}>{item.name}</option>)}</optgroup>)}
  </select></label>{!choices && !error && enabled && <span className="field-note" role="status">Loading object types</span>}
  {error && <div role="alert"><p>{error}</p><button type="button" onClick={() => setRetry(value => value + 1)}>Retry object types</button></div>}</>;
}
