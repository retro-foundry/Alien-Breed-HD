import { useEffect, useState } from 'react';
import { objectName, type LevelDocument } from '../model/level';
import { objectTemplates, replaceObjectSettings } from '../model/objectAuthoring';
import type { EditCommand } from '../model/commands';

export function ObjectTypeInspector({ document, id, edit }: { document: LevelDocument; id: number; edit: (command: EditCommand | (() => EditCommand)) => void }) {
  const [template, setTemplate] = useState('');
  useEffect(() => setTemplate(''), [document, id]);
  const templates = objectTemplates(document);
  if (!templates.some(item => item.id === id)) return null;
  return <form onSubmit={event => { event.preventDefault(); edit(() => replaceObjectSettings(document, id, Number(template))); }}>
    <label className="select-field"><span>Change object type / settings</span><select aria-label="Object type template" value={template}
      title="Choose an existing level object's complete type and settings. Apply keeps this object's position, height, room and floor; replaces artwork, size, health and type-specific settings."
      onChange={event => setTemplate(event.target.value)}>
      <option value="">Choose type and settings</option>
      {templates.filter(item => item.id !== id).map(item => <option key={item.id} value={item.id}>{objectName(item.type)} ? Object {item.id}</option>)}
    </select></label>
    <p className="field-note">Copies the chosen object's type, artwork, size, health and behaviour settings. Keeps this object's position, height, room and floor. Templates currently come from this level. Undo restores all previous settings.</p>
    <button type="submit" className="small-button" disabled={!template} title="Apply the chosen type and complete settings as one undoable edit.">Apply object type</button>
  </form>;
}
