import { RetireMechanism } from './RetireMechanism';
import type { LevelDocument } from '../model/level';
import type { SourceRoom } from '../model/roomSource';
import { editRoomSource, type EditCommand } from '../model/commands';
import { copiedMechanismSettings, setCopiedMechanismRange, setCopiedMechanismRules } from '../model/movingControllerCopies';
import { MovementRangeForm } from './MechanismRangeInspector';
import { MechanismRulesInspector } from './MechanismRulesInspector';
import { CopyMechanismSettings } from './MechanismTransfers';

export function CopiedMechanismInspector({ document, room, edit, onShowRoom }: {
  document: LevelDocument; room: SourceRoom; onShowRoom: (room: string) => void; edit: (command: EditCommand | (() => EditCommand)) => void;
}) {
  return <>{document.roomSource.controllerCopies?.filter(copy => copy.room === room.id).map(copy => {
    const mechanism = copiedMechanismSettings(copy), prefix = `copied ${copy.kind} ${room.id}`;
    return <section className="property-section" key={copy.kind} aria-label={prefix}>
      <h3>Copied {copy.kind === 'door' ? 'door' : 'lift'}</h3>
      <p className="field-note">This room has independent movement settings. Apply range commits both destinations; Escape cancels a field. Rule and requirement changes apply immediately. Undo restores each edit.</p>
      <MovementRangeForm mechanism={mechanism} prefix={prefix} edit={edit}
        createCommand={(top, bottom) => editRoomSource(setCopiedMechanismRange(document, room.id, copy.kind, top, bottom))} />
      <MechanismRulesInspector mechanism={mechanism} prefix={prefix}
        change={(conditions, openRule, closeRule) => edit(() => editRoomSource(setCopiedMechanismRules(document, room.id, copy.kind, conditions, openRule, closeRule)))} />
      <p className="field-note">Key and switch requirements refer to this level's shared conditions. Build &amp; play tests the current settings from the initial position.</p>
      <CopyMechanismSettings document={document} owner={{ type: 'copy', kind: copy.kind, room: room.id }} edit={edit} />
      <RetireMechanism document={document} owner={{ type: 'copy', kind: copy.kind, room: room.id }} edit={edit} onShowRoom={onShowRoom} />
    </section>;
  })}</>;
}
