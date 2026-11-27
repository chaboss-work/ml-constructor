/// <reference types="vite/client" />
// Unit tests for the theme, the node view and "animations off" (Stage 22).
// Run alone:  node node_modules/vitest/vitest.mjs run src/app/core/theme.spec.ts --globals --environment node
import indexHtml from '../../index.html?raw';
import {
  DARK_CLASS,
  MOTION_KEY,
  NO_MOTION_CLASS,
  THEME_ANIM_CLASS,
  THEME_KEY,
  ThemeEnv,
  ThemeMode,
  VIEW_KEY,
  loadMode,
  loadMotion,
  loadView,
  parseMode,
  resolveTheme,
  setMode,
  setMotion,
  setView,
} from './theme';

/** A browser stand-in: a key-value store, <html>'s classes, the system's preference, timers. */
function fakeEnv(systemDark = false, store: Map<string, string> | null = new Map()) {
  const classes = new Set<string>();
  const timers: (() => void)[] = [];
  const env: ThemeEnv = {
    store: store && {
      getItem: (k) => store.get(k) ?? null,
      setItem: (k, v) => void store.set(k, v),
    },
    root: {
      classList: {
        add: (c) => void classes.add(c),
        remove: (c) => void classes.delete(c),
        toggle: (c, force) => {
          const on = force ?? !classes.has(c);
          if (on) classes.add(c);
          else classes.delete(c);
          return on;
        },
      },
    },
    systemDark: () => systemDark,
    later: (fn) => timers.push(fn),
  };
  return { env, classes, timers, store };
}

describe('theme modes', () => {
  it('reads only light / dark; anything else means "as the system"', () => {
    expect(parseMode('light')).toBe('light');
    expect(parseMode('dark')).toBe('dark');
    for (const v of ['auto', '', null, undefined, 'DARK', 42]) expect(parseMode(v)).toBe('auto');
  });

  it('"auto" follows the system; light and dark do not', () => {
    expect(resolveTheme('auto', true)).toBe('dark');
    expect(resolveTheme('auto', false)).toBe('light');
    for (const sys of [true, false]) {
      expect(resolveTheme('light', sys)).toBe('light');
      expect(resolveTheme('dark', sys)).toBe('dark');
    }
  });

  it('choosing a mode remembers it, paints <html> and fades the colours for a moment', () => {
    const { env, classes, timers, store } = fakeEnv(false);
    expect(setMode(env, 'dark')).toBe('dark');
    expect(store!.get(THEME_KEY)).toBe('dark');
    expect(classes.has(DARK_CLASS)).toBe(true);
    expect(classes.has(THEME_ANIM_CLASS)).toBe(true); // the fade is on…
    timers.forEach((t) => t());
    expect(classes.has(THEME_ANIM_CLASS)).toBe(false); // …and off again
    expect(setMode(env, 'auto')).toBe('light'); // the system is light
    expect(classes.has(DARK_CLASS)).toBe(false);
    expect(loadMode(env)).toBe('auto');
  });

  it('works without storage (a private window): nothing is remembered, nothing breaks', () => {
    const { env, classes } = fakeEnv(true, null);
    expect(loadMode(env)).toBe('auto');
    expect(setMode(env, 'light', false)).toBe('light');
    expect(classes.has(DARK_CLASS)).toBe(false);
    const throwing: ThemeEnv = {
      ...env,
      store: {
        getItem: () => {
          throw new Error('blocked');
        },
        setItem: () => {
          throw new Error('blocked');
        },
      },
    };
    expect(loadMode(throwing)).toBe('auto');
    expect(() => setMode(throwing, 'dark')).not.toThrow();
    expect(loadMotion(throwing)).toBe(true);
  });

  it('animations and the node view are remembered; the defaults are on and compact', () => {
    const { env, classes, store } = fakeEnv();
    expect(loadMotion(env)).toBe(true);
    expect(loadView(env)).toBe('compact');
    setMotion(env, false);
    expect(store!.get(MOTION_KEY)).toBe('off');
    expect(classes.has(NO_MOTION_CLASS)).toBe(true);
    expect(loadMotion(env)).toBe(false);
    setMotion(env, true);
    expect(classes.has(NO_MOTION_CLASS)).toBe(false);
    setView(env, 'detailed');
    expect(store!.get(VIEW_KEY)).toBe('detailed');
    expect(loadView(env)).toBe('detailed');
  });
});

describe('the first paint (index.html) agrees with the app', () => {
  // the tiny script in index.html runs before the app: the same keys, classes and rule
  const script = /<script>([\s\S]*?)<\/script>/.exec(indexHtml)?.[1] ?? '';

  function runIndexScript(stored: Record<string, string>, systemDark: boolean): Set<string> {
    const classes = new Set<string>();
    const localStorage = { getItem: (k: string) => stored[k] ?? null };
    const window = { matchMedia: (q: string) => ({ matches: systemDark && q === '(prefers-color-scheme: dark)' }) };
    const document = { documentElement: { classList: { add: (...c: string[]) => c.forEach((x) => classes.add(x)) } } };
    new Function('localStorage', 'window', 'document', script)(localStorage, window, document);
    return classes;
  }

  it('has the script', () => {
    expect(script).toContain(THEME_KEY);
    expect(script).toContain(MOTION_KEY);
  });

  it('paints the same theme as resolveTheme for every mode and system setting', () => {
    for (const mode of ['auto', 'light', 'dark', 'nonsense'] as const) {
      for (const sys of [true, false]) {
        const stored = mode === 'auto' ? {} : { [THEME_KEY]: mode };
        const dark = runIndexScript(stored, sys).has(DARK_CLASS);
        expect(dark).toBe(resolveTheme(parseMode(mode) as ThemeMode, sys) === 'dark');
      }
    }
  });

  it('turns animations off before the first paint when they were off', () => {
    expect(runIndexScript({ [MOTION_KEY]: 'off' }, false).has(NO_MOTION_CLASS)).toBe(true);
    expect(runIndexScript({ [MOTION_KEY]: 'on' }, false).has(NO_MOTION_CLASS)).toBe(false);
  });
});
