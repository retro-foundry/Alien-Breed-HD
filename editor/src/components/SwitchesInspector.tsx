import type { LevelDocument } from '../model/level';
import type { SourceRoom } from '../model/roomSource';
import type { EditCommand } from '../model/commands';
import { setSwitchAutomaticReset, setCopiedSwitchAutomaticReset } from '../model/switchSettings';

function ResetControl({ enabled, prefix, change }: { enabled: boolean; prefix: string; change: (enabled: boolean) => void }) {
  return <label className="select-field"><span>After activation</span>
    <select aria-label={`${prefix} reset mode`} value={enabled ? 'automatic' : 'manual'}
      title="Choose whether the native timer switches this off automatically or it stays on until used again. Changes apply immediately; Undo restores the previous setting."
      onChange={event => change(event.target.value === 'automatic')}>
      <option value="manual">Use again to reset</option><option value="automatic">Automatically resets</option>
    </select>
  </label>;
}

export function SwitchesInspector({ document, room, zone, edit, frameRoom }: {
  document: LevelDocument; room?: SourceRoom; zone?: number;
  edit: (command: EditCommand | (() => EditCommand)) => void; frameRoom: (room: string) => void;
}) {
  return <>
    {document.switches.filter(item => item.zone >= 0 && item.zone === zone).map(item => <section className="property-section" key={`original-${item.id}`} aria-label={`Switch ${item.id + 1}`}>
      <h3>Switch {item.id + 1}</h3>
      <ResetControl enabled={item.type !== 0} prefix={`switch ${item.id + 1}`} change={enabled => edit(() => setSwitchAutomaticReset(document, item.id, enabled))} />
      <p className="field-note">Stand near the switch and press Space in Build &amp; play. To require it on a door or lift, select Switch {item.id + 1} under Requires all selected.</p>
    </section>)}
    {document.roomSource.switchCopies?.filter(item => item.room === room?.id).map(item => {
      const graphics = document.roomSource.rooms.find(room => room.id === item.graphicsRoom);
      return <section className="property-section" key={`copy-${item.slot}`} aria-label={`Copied switch ${item.slot + 1}`}>
        <h3>Copied switch {item.slot + 1}</h3>
        <ResetControl enabled={item.bytes[2] !== 0} prefix={`copied switch ${item.slot + 1}`} change={enabled => edit(() => setCopiedSwitchAutomaticReset(document, item.slot, enabled))} />
        <p className="field-note">Uses Switch {item.slot + 1} in this level's requirements. To require this copy on a door or lift, select that number under Requires all selected. Stand near it and press Space in Build &amp; play.</p>
        <div className="info-row"><span>Copied from</span><span>Switch {item.id + 1}</span></div>
        <div className="info-row"><span>Activation room</span><span>Room {item.room.replace(/^room-/, '')}</span></div>
        <button className="small-button" disabled={!graphics} title="Frame the room containing this switch's wall graphics while keeping the switch selected."
          onClick={() => { if (graphics) frameRoom(graphics.id); }}>Show switch graphics room</button>
        <p className="field-note">This copy has independent reset settings. Changing the reset mode keeps its native timer state and attachments. Undo restores each edit.</p>
      </section>;
    })}
  </>;
}
