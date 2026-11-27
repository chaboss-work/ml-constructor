// Unit tests for the layer families, the detailed view's line, the inspector's groups and
// the training wave's order (Stage 22).
// Run alone:  node node_modules/vitest/vitest.mjs run src/app/core/families.spec.ts --globals --environment node
import {
  FAMILIES,
  archLook,
  familyOf,
  fmtNum,
  graphDepths,
  groupByFamily,
  groupParams,
  iconOf,
  isAdvanced,
  keyParams,
  waveOf,
} from './families';
import { ParamDef } from './graph.models';

const P = (name: string, type: ParamDef['type'], dflt: unknown, extra: Partial<ParamDef> = {}): ParamDef => ({
  name,
  type,
  default: dflt,
  ...extra,
});

describe('families', () => {
  it('maps every category the editor knows, and anything else to "other"', () => {
    expect(familyOf('conv').key).toBe('conv');
    expect(familyOf('llm').label).toBe('Language model');
    expect(familyOf('3d').icon).toBe('cube');
    expect(familyOf('no-such-category').key).toBe('other');
    expect(familyOf(undefined).key).toBe('other');
    expect(new Set(FAMILIES.map((f) => f.key)).size).toBe(FAMILIES.length);
  });

  it('gives Input and Output their own icons, the rest their family icon', () => {
    expect(iconOf('Input', 'io')).toBe('input');
    expect(iconOf('Output', 'io')).toBe('output');
    expect(iconOf('Conv2d', 'conv')).toBe('conv');
    expect(iconOf('Mystery', null)).toBe('dot');
  });

  it('gives library entries a look by their category', () => {
    expect(archLook('language')).toEqual({ family: 'llm', icon: 'llm' });
    expect(archLook('audio').icon).toBe('audio');
    expect(archLook('unknown')).toEqual({ family: 'other', icon: 'dot' });
  });

  it('groups layers by family in a fixed order', () => {
    const layers = [
      { type: 'ReLU', category: 'activation' },
      { type: 'Input', category: 'io' },
      { type: 'Conv2d', category: 'conv' },
      { type: 'GELU', category: 'activation' },
    ];
    const groups = groupByFamily(layers);
    expect(groups.map((g) => g.family.key)).toEqual(['io', 'conv', 'activation']);
    expect(groups[2].layers.map((l) => l.type)).toEqual(['ReLU', 'GELU']);
  });
});

describe('the inspector groups', () => {
  it('puts switches under Options and rarely touched settings under Advanced', () => {
    const defs = [
      P('heads', 'int', 8),
      P('kv_heads', 'int', 2),
      P('causal', 'bool', true),
      P('qk_norm', 'bool', false),
      P('rope_theta', 'float', 10000),
      P('eps', 'float', 1e-6),
    ];
    const g = groupParams(defs);
    expect(g.main.map((d) => d.name)).toEqual(['heads', 'kv_heads']);
    expect(g.options.map((d) => d.name)).toEqual(['causal']);
    expect(g.advanced.map((d) => d.name)).toEqual(['qk_norm', 'rope_theta', 'eps']);
    expect(isAdvanced('eps')).toBe(true);
    expect(isAdvanced('out_features')).toBe(false);
  });
});

describe("the detailed view's line of settings", () => {
  const conv = [P('out_channels', 'int', 16), P('kernel_size', 'int', 3), P('stride', 'int', 1), P('padding', 'int', 0)];

  it('shows the key settings short, in the catalogue order', () => {
    expect(keyParams({ out_channels: 32, kernel_size: 3, stride: 1, padding: 1 }, conv)).toBe('ch 32 · k 3 · s 1 · pad 1');
  });

  it('falls back to the default, joins shapes with ×, and stops at the limit', () => {
    expect(keyParams({}, conv, 2)).toBe('ch 16 · k 3');
    expect(keyParams({ shape: [1, 28, 28] }, [P('shape', 'intlist', [784]), P('name', 'str', '')])).toBe('shape 1×28×28');
    expect(keyParams({ repeat: 4 }, [P('repeat', 'int', 1)])).toBe('×4');
  });

  it('names a switch only when it is on, and hides the advanced ones', () => {
    const defs = [P('out_features', 'int', 10), P('bias', 'bool', true), P('per_token', 'bool', false), P('eps', 'float', 1e-5)];
    expect(keyParams({ out_features: 128, bias: true, per_token: false }, defs)).toBe('out 128 · bias');
  });

  it('writes tiny and big numbers readably', () => {
    expect(fmtNum(1e-5)).toBe('1e-5');
    expect(fmtNum(2.5e-4)).toBe('2.5e-4');
    expect(fmtNum(0.001)).toBe('0.001');
    expect(fmtNum(0.1 + 0.2)).toBe('0.3');
    expect(fmtNum(50000)).toBe('50000');
  });
});

describe('the training wave', () => {
  it('orders nodes by the longest path from the inputs', () => {
    // in -> a -> c -> out, in -> b -> c (a diamond with a longer arm)
    const d = graphDepths(
      ['in', 'a', 'b', 'b2', 'c', 'out'],
      [
        ['in', 'a'],
        ['in', 'b'],
        ['b', 'b2'],
        ['a', 'c'],
        ['b2', 'c'],
        ['c', 'out'],
      ],
    );
    expect(Object.fromEntries(d)).toEqual({ in: 0, a: 1, b: 1, b2: 2, c: 3, out: 4 });
    const w = waveOf(d);
    expect(w.get('in')).toBe(0);
    expect(w.get('out')).toBe(1);
    expect(w.get('c')).toBe(0.75);
  });

  it('survives loops, self-links and links to nodes that are gone', () => {
    const d = graphDepths(
      ['x', 'y', 'z'],
      [
        ['x', 'y'],
        ['y', 'x'], // a loop: the backend reports it; the canvas just does not hang
        ['z', 'z'],
        ['z', 'ghost'],
      ],
    );
    expect(d.get('x')).toBe(0);
    expect(d.get('y')).toBe(0);
    expect(d.get('z')).toBe(0);
    expect(Object.fromEntries(waveOf(d))).toEqual({ x: 0, y: 0, z: 0 });
  });
});
