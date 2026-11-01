export interface InspectionShortcut {
  key: string; ctrlKey: boolean; metaKey: boolean; altKey: boolean; shiftKey: boolean; repeat: boolean;
}
export function inspectionShortcut(input: unknown): InspectionShortcut | null;
export function inspectionShortcutScope(input: unknown): 'workspace' | 'frame' | null;
