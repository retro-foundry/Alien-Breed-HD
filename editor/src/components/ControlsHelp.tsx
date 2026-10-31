export const previewKeys = [['← / →', 'Turn the inspection camera'], ['↑ / ↓', 'Look up / down'], ['Click view', 'Select the zone under the crosshair']] as const;
// Verified against src/input.c and player.c; inspection keys are hosted in runtime/frame.html.
export const playKeys = [['W / S', 'Move forward / backward'], ['A / D', 'Strafe left / right'], ['Mouse', 'Aim after clicking the game view'], ['Left click / Ctrl', 'Fire'], ['Space / right click', 'Use doors and switches'], ['C', 'Crouch'], ['Shift', 'Change run / walk speed'], ['1–6 / mouse wheel', 'Choose an available weapon'], ['Esc', 'Release the mouse / open the game menu'], ['Stop button', 'Return to editing']] as const;

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
    <h3>Playtesting</h3><p>Choose Build & play, then click the game view to capture the mouse. The first click captures it; subsequent clicks fire. Use Stop to return to editing.</p><Keys rows={playKeys} />
    <h3>Keeping your work</h3><p>Workspace changes are saved in this browser. Save project downloads a portable .ab3d backup. Export level downloads the three game files in a ZIP. Keep project copies to transfer your work or recover it if browser storage is cleared.</p>
  </>;
}
