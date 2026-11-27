/**
 * Stage 22 — the theme (light, dark, or as the system says), the node view and
 * "animations off": remembered per browser, painted as classes on <html>.
 *
 * index.html runs a tiny copy of `loadMode` + `resolveTheme` before the app starts,
 * so the first paint already has the right colours (a test keeps the two in step).
 * No Angular here: the editor holds the signals, this file holds the rules.
 */

export type ThemeMode = 'auto' | 'light' | 'dark';
export type Theme = 'light' | 'dark';
export type NodeView = 'compact' | 'detailed';

export const THEME_KEY = 'mlc.theme';
export const MOTION_KEY = 'mlc.motion';
export const VIEW_KEY = 'mlc.view';
export const DARK_CLASS = 'mlc-dark';
export const NO_MOTION_CLASS = 'no-motion';
export const THEME_ANIM_CLASS = 'theme-anim';
/** How long the colours fade when the theme changes (ms; the CSS uses --dur-3 = 320 ms). */
export const THEME_FADE_MS = 380;
/** The order of the theme switch's buttons. */
export const THEME_MODES: readonly ThemeMode[] = ['light', 'auto', 'dark'];

/** What the editor needs from the browser (tests pass their own). */
export interface ThemeEnv {
  store: { getItem(key: string): string | null; setItem(key: string, value: string): void } | null;
  root: { classList: { add(c: string): void; remove(c: string): void; toggle(c: string, force?: boolean): boolean } };
  systemDark: () => boolean;
  later: (fn: () => void, ms: number) => unknown;
}

export function browserEnv(): ThemeEnv {
  let store: ThemeEnv['store'] = null;
  try {
    store = window.localStorage; // may throw in a private window or with storage blocked
  } catch {
    store = null;
  }
  return {
    store,
    root: document.documentElement,
    systemDark: () => !!window.matchMedia?.('(prefers-color-scheme: dark)').matches,
    later: (fn, ms) => setTimeout(fn, ms),
  };
}

function readPref(env: ThemeEnv, key: string): string | null {
  try {
    return env.store?.getItem(key) ?? null;
  } catch {
    return null;
  }
}

function writePref(env: ThemeEnv, key: string, value: string): void {
  try {
    env.store?.setItem(key, value);
  } catch {
    /* storage full or blocked: the choice just is not remembered */
  }
}

export function parseMode(v: unknown): ThemeMode {
  return v === 'light' || v === 'dark' ? v : 'auto';
}

export function resolveTheme(mode: ThemeMode, systemDark: boolean): Theme {
  return mode === 'auto' ? (systemDark ? 'dark' : 'light') : mode;
}

export function loadMode(env: ThemeEnv): ThemeMode {
  return parseMode(readPref(env, THEME_KEY));
}

/** Paint a theme on <html>; `animate` fades the colours for a moment. */
export function paintTheme(env: ThemeEnv, theme: Theme, animate: boolean): void {
  if (animate) {
    env.root.classList.add(THEME_ANIM_CLASS);
    env.later(() => env.root.classList.remove(THEME_ANIM_CLASS), THEME_FADE_MS);
  }
  env.root.classList.toggle(DARK_CLASS, theme === 'dark');
}

/** Choose a mode: remember it and paint what it means now. */
export function setMode(env: ThemeEnv, mode: ThemeMode, animate = true): Theme {
  writePref(env, THEME_KEY, mode);
  const theme = resolveTheme(mode, env.systemDark());
  paintTheme(env, theme, animate);
  return theme;
}

export function loadMotion(env: ThemeEnv): boolean {
  return readPref(env, MOTION_KEY) !== 'off';
}

export function setMotion(env: ThemeEnv, on: boolean): void {
  writePref(env, MOTION_KEY, on ? 'on' : 'off');
  env.root.classList.toggle(NO_MOTION_CLASS, !on);
}

export function loadView(env: ThemeEnv): NodeView {
  return readPref(env, VIEW_KEY) === 'detailed' ? 'detailed' : 'compact';
}

export function setView(env: ThemeEnv, view: NodeView): void {
  writePref(env, VIEW_KEY, view);
}
