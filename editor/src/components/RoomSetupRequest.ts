export type RoomSetupKind = 'door' | 'lift' | 'switch';
export interface RoomSetupRequest { room: string; kind: RoomSetupKind; sequence: number }
