import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./editor/editor.page').then((m) => m.EditorPage),
  },
];
