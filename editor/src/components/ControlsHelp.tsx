export const previewKeys = [['← / →', 'Turn the inspection camera'], ['↑ / ↓', 'Look up / down'], ['Click view', 'Select the zone under the crosshair']] as const;
// Verified against src/input.c and player.c; inspection keys are hosted in runtime/frame.html.
export const playKeys = [['W / S', 'Move forward / backward'], ['A / D', 'Strafe left / right'], ['Mouse', 'Aim after clicking the game view'], ['Left click / Ctrl', 'Fire'], ['Space / right click', 'Use doors and switches'], ['C', 'Crouch'], ['Shift', 'Change run / walk speed'], ['1–6 / mouse wheel', 'Choose an available weapon'], ['Esc', 'Release the mouse / open the game menu'], ['Stop button', 'Return to editing']] as const;

import { zoneGraphicsRecordLimit } from '../model/renderLimits';

function Keys({ rows }: { rows: readonly (readonly [string, string])[] }) {
  return <div className="shortcut-list">{rows.map(([key, action]) => <div key={key}><span>{action}</span><kbd>{key}</kbd></div>)}</div>;
}

export function ControlsHelp() {
  return <><span className="eyebrow">CONTROLS & SHORTCUTS</span><h2>Find your way around</h2>
    <p>Open an original level or your project. Click a zone, object or player start to inspect it. Drag objects and starts to move them; positions must stay inside their zone. Hover over a control or focus it with Tab for a tooltip.</p>
    <h3>Map and editing</h3><Keys rows={[
      ['V', 'Select tool'], ['H / Space + drag', 'Pan the map'], ['Middle / right drag', 'Pan without changing tools'], ['Wheel', 'Zoom around the pointer'],
      ['F', 'Fit the whole map'], ['Alt + drag', 'Bypass grid snapping'], ['/', 'Search map contents'],
      ['Ctrl / ⌘ + Z', 'Undo'], ['Ctrl / ⌘ + Shift + Z', 'Redo (also Ctrl / ⌘ + Y)'], ['Ctrl / ⌘ + S', 'Download a project copy'],
      ['Ctrl / ⌘ + O', 'Open a project or level'], ['Enter', 'Apply a numeric property'], ['Esc', 'Cancel a numeric edit / clear selection / close help'], ['?', 'Open this help'],
    ]} />
    <h3>3D inspection</h3><p>Choose Split view or 3D view. Select a zone in the map or list to visit it. Click inside the 3D view before using its arrow keys.</p><Keys rows={previewKeys} />
    <h3>Room outlines</h3><p>Select a zone as the settings template, then choose Draw room. Click corners in either direction; Enter or clicking the first corner commits the whole outline. Backspace removes the last corner. Escape, Cancel room or changing tools discards unfinished work. Select a room to drag its highlighted corners or open Edit room corners in the inspector. Shared corners move together; crossing and overlapping outlines are rejected. Delete room is undoable. Outlines are saved in project copies, but full game geometry compilation is still in progress.</p>
    <h3>Stairs</h3><p>Choose Build stairs, enter the centre of both ends, width, number of steps, starting and final floor heights, and clearance. Choose wall styles and lighting from an existing material template. Smaller floor values are higher. The first generated room is one step above or below the starting floor; the last reaches the final floor. The labelled plan checks shared corners, overlap, heights and room limits. Increase the step count if a riser exceeds the game's climb allowance. Create stairs applies the flight as one undoable edit; Escape or Cancel stairs discards it. Use Build & play to check standing/crouched clearance and the joins to your existing rooms.</p>
    <h3>Playtesting</h3><p>Choose the target game slot and Level exit in Project. Original slot exit retains the game's built-in completion zone; choose a zone for an authored exit, or No exit to test freely. Choose Build & play, then click the game view to capture the mouse. The first click captures it; subsequent clicks fire. Use Stop to return to editing.</p><Keys rows={playKeys} />
    <h3>Keeping your work</h3><p>Workspace changes are saved in this browser. Save project downloads a portable .ab3d backup, including the target game slot. Choose that slot in Project or the 3D toolbar; it is independent of the project name. Export level puts the game files in the matching levels/level_… directory with installation instructions. Keep project copies to transfer your work or recover it if browser storage is cleared.</p>
    <h3>Engine capacity</h3><p>The current game supports {zoneGraphicsRecordLimit} graphics commands per room and floor. Walls, surfaces and draw controls share this budget. Preview and game export stop with a room-specific error if a stream exceeds it; Save project still preserves your work. This is the current port's limit, not an original Amiga compatibility profile.</p>
  </>;
}
