// "Every screen uses the design system, in both themes" (Stage 22), as tests:
//  - colours are written down only in src/theme/tokens.css: no style or template has its own;
//  - every colour token exists in the light and in the dark theme;
//  - text stays readable (WCAG contrast) on every surface, in both themes;
//  - every variable a style uses is a token (or one the page sets), every icon exists.
// Run alone:  node node_modules/vitest/vitest.mjs run src/app/core/design.spec.ts --globals --environment node
//
// The files are read from disk: vitest hands CSS / SCSS imports over as empty strings.
import FAMILY_DATA from './families.json';

const FS_MODULE = 'node:fs';
const fs = (await import(/* @vite-ignore */ FS_MODULE)) as {
  readFileSync(p: URL, encoding: 'utf-8'): string;
  readdirSync(p: URL, options: { recursive: true }): string[];
};
const SRC = new URL('../../', import.meta.url); // frontend/src/
const read = (rel: string): string => fs.readFileSync(new URL(rel, SRC), 'utf-8');
const APP_FILES = fs.readdirSync(new URL('app/', SRC), { recursive: true }).map((f) => 'app/' + f.replace(/\\/g, '/'));

const tokensCss = read('theme/tokens.css');
const STYLES: Record<string, string> = Object.fromEntries(
  [...APP_FILES.filter((f) => f.endsWith('.scss')), 'styles.scss'].map((f) => [f, read(f)]),
);
const TEMPLATES: Record<string, string> = Object.fromEntries(
  APP_FILES.filter((f) => f.endsWith('.html')).map((f) => [f, read(f)]),
);
const EDITOR_HTML = TEMPLATES['app/editor/editor.page.html'];

// ------------------------------------------------------------------ the tokens
function blocks(css: string, selector: string): Map<string, string> {
  const out = new Map<string, string>();
  const re = /([^{}]+)\{([^}]*)\}/g;
  for (let m = re.exec(css); m; m = re.exec(css)) {
    const sel = m[1].replace(/\/\*[\s\S]*?\*\//g, '').trim();
    if (sel !== selector) continue;
    for (const d of m[2].matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) out.set(d[1], d[2].trim());
  }
  return out;
}
const LIGHT = blocks(tokensCss, ':root');
const DARK_ONLY = blocks(tokensCss, ':root.mlc-dark');
const DARK = new Map([...LIGHT, ...DARK_ONLY]);
const COLOUR = /#[0-9a-f]{3,8}\b|rgba?\(|hsla?\(/i;

/** A token's colour in a theme, as [r, g, b] (0–255). */
function rgb(theme: Map<string, string>, name: string): [number, number, number] {
  const v = theme.get(name);
  const m = v && /^#([0-9a-f]{6})$/i.exec(v);
  if (!m) throw new Error(`${name} is not a #rrggbb colour: ${v}`);
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function luminance([r, g, b]: [number, number, number]): number {
  const lin = (c: number) => {
    const s = c / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}
/** The WCAG contrast ratio of two tokens in a theme (1…21). */
function contrast(theme: Map<string, string>, fg: string, bg: string): number {
  const a = luminance(rgb(theme, fg));
  const b = luminance(rgb(theme, bg));
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

const THEMES: [string, Map<string, string>][] = [
  ['light', LIGHT],
  ['dark', DARK],
];
const SURFACES = ['--bg', '--surface-1', '--surface-2', '--surface-3'];
const FAMILY_KEYS = Object.keys((FAMILY_DATA as { families: Record<string, unknown> }).families);

describe('the tokens', () => {
  it('has both themes', () => {
    expect(LIGHT.size).toBeGreaterThan(40);
    expect(DARK_ONLY.size).toBeGreaterThan(40);
  });

  it('defines every colour in both themes', () => {
    const lightColours = [...LIGHT].filter(([, v]) => COLOUR.test(v)).map(([k]) => k);
    const darkColours = [...DARK_ONLY].filter(([, v]) => COLOUR.test(v)).map(([k]) => k);
    expect(lightColours.length).toBeGreaterThan(40);
    expect(new Set(darkColours)).toEqual(new Set(lightColours));
  });

  it('has a colour for every layer family in both themes', () => {
    for (const key of FAMILY_KEYS) {
      expect(LIGHT.has(`--fam-${key}`), `--fam-${key} (light)`).toBe(true);
      expect(DARK_ONLY.has(`--fam-${key}`), `--fam-${key} (dark)`).toBe(true);
    }
  });
});

describe('text stays readable in both themes (WCAG AA: 4.5 for text, 3 for shapes)', () => {
  for (const [name, t] of THEMES) {
    it(`${name}: body, secondary and hint text on every surface`, () => {
      for (const bg of [...SURFACES, '--surface-4']) {
        expect(contrast(t, '--text', bg), `--text on ${bg}`).toBeGreaterThanOrEqual(7);
        expect(contrast(t, '--text-2', bg), `--text-2 on ${bg}`).toBeGreaterThanOrEqual(4.5);
      }
      for (const bg of SURFACES) expect(contrast(t, '--text-3', bg), `--text-3 on ${bg}`).toBeGreaterThanOrEqual(4.5);
    });

    it(`${name}: accent and status colours read as text on panels and cards`, () => {
      for (const fg of ['--accent', '--success', '--warning', '--danger', '--info']) {
        for (const bg of ['--surface-1', '--surface-2']) {
          expect(contrast(t, fg, bg), `${fg} on ${bg}`).toBeGreaterThanOrEqual(4.5);
        }
      }
      expect(contrast(t, '--on-accent', '--accent')).toBeGreaterThanOrEqual(4.5);
      expect(contrast(t, '--on-grad', '--grad-a')).toBeGreaterThanOrEqual(4.5);
      expect(contrast(t, '--on-grad', '--grad-b')).toBeGreaterThanOrEqual(4.5);
    });

    it(`${name}: family colours read on the cards (they label shapes and parameter counts)`, () => {
      for (const key of FAMILY_KEYS) {
        expect(contrast(t, `--fam-${key}`, '--surface-2'), `--fam-${key}`).toBeGreaterThanOrEqual(4.5);
      }
      for (const g of ['--glow-fwd', '--glow-bwd']) {
        expect(contrast(t, g, '--surface-2'), g).toBeGreaterThanOrEqual(3);
      }
      expect(contrast(t, '--glow-fwd', '--surface-1'), '--glow-fwd as text').toBeGreaterThanOrEqual(4.5);
    });
  }
});

describe('every screen uses the design system', () => {
  const files = Object.entries(STYLES);

  it('finds the styles and the templates — with their contents', () => {
    expect(files.length).toBeGreaterThanOrEqual(10);
    for (const [path, css] of files) {
      if (path !== 'app/app.scss') expect(css.length, path).toBeGreaterThan(1000); // app.scss is empty on purpose
    }
    expect(EDITOR_HTML.length).toBeGreaterThan(20000);
    expect(tokensCss.length).toBeGreaterThan(3000);
  });

  it('writes no colour of its own: every colour comes from a token', () => {
    const bad: string[] = [];
    for (const [path, css] of [...files, ...Object.entries(TEMPLATES)]) {
      const text = css.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
      text.split('\n').forEach((line, i) => {
        if (COLOUR.test(line) || /:\s*(white|black|red|green|blue|gray|grey)\s*[;!]/i.test(line)) {
          bad.push(`${path}:${i + 1}: ${line.trim()}`);
        }
      });
    }
    expect(bad).toEqual([]);
  });

  it("no longer leans on Ionic's palette (the old step colours were light-only)", () => {
    for (const [path, css] of files) expect(css.includes('--ion-'), path).toBe(false);
    expect(EDITOR_HTML.includes('<ion-')).toBe(false);
  });

  it('uses only variables that exist: tokens, or the few the page sets per element', () => {
    // set per element by the page
    const PER_ELEMENT = new Set(['--fam', '--c', '--p', '--pct']);
    const unknown = new Set<string>();
    for (const [path, css] of [...files, ['editor.page.html', EDITOR_HTML] as [string, string]]) {
      for (const m of css.matchAll(/var\(\s*(--[\w-]+)/g)) {
        const v = m[1];
        if (!LIGHT.has(v) && !DARK.has(v) && !PER_ELEMENT.has(v)) unknown.add(`${path}: ${v}`);
      }
    }
    expect([...unknown]).toEqual([]);
  });

  it('has every icon it shows: the toolbar, the families, the library', () => {
    const symbols = new Set([...EDITOR_HTML.matchAll(/<symbol id="i-([\w-]+)"/g)].map((m) => m[1]));
    const used = new Set([...EDITOR_HTML.matchAll(/href="#i-([\w-]+)"/g)].map((m) => m[1]));
    const d = FAMILY_DATA as {
      families: Record<string, { icon: string }>;
      typeIcons: Record<string, string>;
      architectures: Record<string, { icon: string; family: string }>;
    };
    for (const f of Object.values(d.families)) used.add(f.icon);
    for (const i of Object.values(d.typeIcons)) used.add(i);
    for (const a of Object.values(d.architectures)) {
      used.add(a.icon);
      expect(FAMILY_KEYS).toContain(a.family);
    }
    for (const i of ['sun', 'moon', 'auto', 'warn', 'err', 'ok', 'check', 'copy']) used.add(i); // chosen in code
    expect([...used].filter((i) => !symbols.has(i))).toEqual([]);
  });
});
