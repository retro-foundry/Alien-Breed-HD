import { encodeProject } from './model/project';
import type { LevelDocument } from './model/level';

export interface SavedWorkspace { id: string; name: string; updatedAt: number; bytes: Uint8Array }
let database: Promise<IDBDatabase> | undefined;

function openDatabase(): Promise<IDBDatabase> {
  if (!database) database = new Promise((resolve, reject) => {
    const request = indexedDB.open('alien-breed-editor', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('workspaces', { keyPath: 'id' });
    request.onsuccess = () => {
      const db = request.result;
      db.onversionchange = () => { db.close(); database = undefined; };
      resolve(db);
    };
    request.onerror = () => { database = undefined; reject(request.error ?? new Error('Browser storage could not be opened.')); };
    request.onblocked = () => { database = undefined; reject(new Error('Close other editor tabs to finish updating browser storage.')); };
  });
  return database;
}

export async function listWorkspaces(): Promise<SavedWorkspace[]> {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction('workspaces', 'readonly');
    const request = transaction.objectStore('workspaces').getAll();
    request.onsuccess = () => resolve((request.result as SavedWorkspace[]).sort((a, b) => b.updatedAt - a.updatedAt));
    request.onerror = () => reject(request.error ?? new Error('Saved workspaces could not be read.'));
  });
}

export async function saveWorkspace(id: string, document: LevelDocument): Promise<void> {
  const bytes = encodeProject(document);
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction('workspaces', 'readwrite');
    transaction.objectStore('workspaces').put({ id, name: document.name, updatedAt: Date.now(), bytes } satisfies SavedWorkspace);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error('Autosave failed. Download a project copy to keep these edits.'));
    transaction.onabort = () => reject(transaction.error ?? new Error('Autosave was interrupted. Download a project copy to keep these edits.'));
  });
}
