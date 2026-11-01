# Published editor map selection

This targeted patch starts from the published editor source, `956397a`.
Picking now uses the closed primary room outline; secondary collision fences
previously made an ordinary Level 1 Zone 1 click select Zone 0. Player-start
triangles/labels and object markers now share their visible geometry with hit
testing, removing invisible margins that captured clicks in neighbouring zones.

All 121 model tests and all 30 Chromium/Firefox browser cases pass, including
all sixteen original maps, editing, saves, Undo and playtesting. The selection
cases also pass in installed Edge and Brave at device scale factors 1, 1.25
and 2 (18 cases across three viewport sizes).

The four published runtime files retain their original SHA-256 hashes. The
patch retains the published project format and game data.
