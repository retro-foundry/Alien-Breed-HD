export function RoomConnectionHelp() {
  return <details className="room-connection-help"><summary>How to make dotted lines</summary>
    <p>A zone is a room or a section of a room. There is no separate dotted-line zone type: a dotted line is an opening between two adjoining zones. A solid line is a wall.</p>
    <svg viewBox="0 0 240 85" role="img" aria-label="Two adjoining zones with a dotted connection along their shared edge">
      <path d="M120 12H15V62H120M120 12H225V62H120" fill="none" stroke="currentColor" strokeWidth="2" />
      <path d="M120 12V62" fill="none" stroke="#82caff" strokeWidth="2" strokeDasharray="3 4" />
      <text x="65" y="41" textAnchor="middle" fill="currentColor">Zone A</text><text x="175" y="41" textAnchor="middle" fill="currentColor">Zone B</text>
      <text x="120" y="80" textAnchor="middle" fill="#82caff">Shared connection</text>
    </svg>
    <p><strong>Open a wall between existing rooms</strong></p>
    <ol>
      <li>Select either room in the map or room list.</li>
      <li>In its inspector, expand Wall connections (solid / dotted).</li>
      <li>Find the neighbouring room and choose Open edge. Both sides open and the shared line becomes dotted. Undo restores the wall.</li>
    </ol>
    <p>If no neighbour is listed, the rooms do not share an edge yet. Draw an adjoining room using the steps below.</p>
    <p>See-through barrier makes a grille or fence along a shared edge. Its empty texels reveal the next room, but it blocks walking. Orange dash-dot marks a barrier; Open edge removes it.</p>
    <p>Full-height barriers from imported native levels are recognised automatically. Their saved graphics and collision stay intact during other edits; use the same Open/Close controls to change them.</p>
    <p><strong>Draw a new connected room</strong></p>
    <ol>
      <li>In Top view, choose Draw room and draw beside an existing room.</li>
      <li>Put one edge exactly on the other room's edge. Use grid snapping or matching corner coordinates. The edge can cover all or part of the existing edge.</li>
      <li>Press Enter to finish. Starter wall, floor and ceiling textures are assigned automatically, and the shared edge is shown dotted.</li>
      <li>For a level passage, match the two floors and leave enough ceiling height. Use Build &amp; play to check walking through.</li>
    </ol>
    <p>Touching at one corner leaves no passage. Rooms must share an edge without overlapping. Automatic rules preserve solid walls between imported zones; use Open edge to change them. A door or incompatible heights can block a connected boundary. For a hole in the ceiling or a view of the sky, use Storeys and openings instead.</p>
  </details>;
}
