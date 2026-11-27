// Unit tests for live training on the canvas (Stage 22): the stream is read without
// drawing (steps are handed over once per frame), the progress, the training sliders.
// Run alone:  node node_modules/vitest/vitest.mjs run src/app/core/live.spec.ts --globals --environment node
import {
  BATCH_STEPS,
  EPOCH_STEPS,
  FrameBatcher,
  LR_STEPS,
  PASS_MS,
  SAMPLE_STEPS,
  WAVE_TICK_MS,
  downsample,
  ladderPct,
  nearestIndex,
  progressOf,
  rangePct,
  GLOW_MIN,
  glowAt,
  totalSteps,
  wavePhase,
} from './live';

/** A hand-cranked screen: the test decides when a frame happens (a 144 Hz one by default). */
function frames(hz = 144) {
  let t = 0;
  const queue: (() => void)[] = [];
  return {
    schedule: (run: () => void) => void queue.push(run),
    now: () => t,
    tick: () => {
      t += 1000 / hz;
      queue.splice(0).forEach((run) => run());
    },
    pending: () => queue.length,
  };
}

describe('FrameBatcher: reading the stream never waits for drawing', () => {
  it('hands over ten thousand steps in one go, in order, at the next frame', () => {
    const f = frames();
    const got: number[][] = [];
    const b = new FrameBatcher<number>((items) => got.push(items), f.schedule, f.now);
    for (let i = 1; i <= 10_000; i++) b.push(i);
    expect(got.length).toBe(0); // nothing is drawn while the stream is read…
    expect(f.pending()).toBe(1); // …only one frame is asked for
    f.tick();
    expect(got.length).toBe(1);
    expect(b.flushes).toBe(1);
    expect(got[0].length).toBe(10_000);
    expect(got[0][0]).toBe(1);
    expect(got[0][9999]).toBe(10_000);
  });

  it('redraws about 20 times a second, even on a 144 Hz screen — and loses no step', () => {
    const f = frames(144);
    const got: number[] = [];
    const b = new FrameBatcher<number>((items) => got.push(...items), f.schedule, f.now);
    let n = 0;
    for (let frame = 0; frame < 144; frame++) {
      for (let i = 0; i < 25; i++) b.push(n++); // 25 steps arrive between two frames, for one second
      f.tick();
    }
    b.flush(); // the run ends
    expect(b.flushes).toBeGreaterThanOrEqual(17);
    expect(b.flushes).toBeLessThanOrEqual(22); // not 144
    expect(got).toEqual(Array.from({ length: 3600 }, (_, i) => i));
  });

  it('on a slow screen every frame hands over', () => {
    const f = frames(15);
    const b = new FrameBatcher<number>(() => undefined, f.schedule, f.now);
    for (let frame = 0; frame < 15; frame++) {
      b.push(frame);
      f.tick();
    }
    expect(b.flushes).toBe(15);
  });

  it('flush() hands over at once (a run ends, an epoch arrives); reset() forgets', () => {
    const f = frames();
    const got: string[] = [];
    const b = new FrameBatcher<string>((items) => got.push(...items), f.schedule, f.now);
    b.push('a');
    b.push('b');
    b.flush();
    expect(got).toEqual(['a', 'b']);
    f.tick(); // the frame asked for earlier finds nothing left — and ends its chain
    expect(f.pending()).toBe(0);
    expect(b.flushes).toBe(1);
    b.push('c');
    b.reset();
    f.tick();
    expect(got).toEqual(['a', 'b']);
    b.push('d'); // a new chain starts after the old one ended
    for (let i = 0; i < 10; i++) f.tick();
    expect(got).toEqual(['a', 'b', 'd']);
  });

  it('never delivers twice when flush() comes while a frame is awaited', () => {
    const f = frames();
    const got: number[] = [];
    const b = new FrameBatcher<number>((items) => got.push(...items), f.schedule, f.now);
    b.push(1);
    f.tick(); // delivered: 1
    b.push(2); // too soon after the last hand-over: the chain waits…
    f.tick();
    b.flush(); // …an epoch arrives: 2 goes now
    b.push(3); // the waiting chain takes 3 — no second chain
    for (let i = 0; i < 10; i++) f.tick();
    expect(got).toEqual([1, 2, 3]);
    expect(f.pending()).toBe(0);
  });
});

describe('the training wave', () => {
  it('runs along a pass and starts again', () => {
    expect(wavePhase(0)).toBe(0);
    expect(wavePhase(400, 1600)).toBe(0.25);
    expect(wavePhase(1600 + 800, 1600)).toBe(0.5);
    expect(wavePhase(-5)).toBe(0);
    expect(WAVE_TICK_MS).toBeGreaterThanOrEqual(40); // at most 25 moves a second, whatever the screen
    expect(PASS_MS / WAVE_TICK_MS).toBeGreaterThanOrEqual(20); // yet smooth: 20+ moves per pass
  });

  it('lights the inputs first, the deepest node at mid-pass, then runs back up', () => {
    const fwdAt = (phase: number) => [0, 0.5, 1].map((w) => glowAt(phase, w)[0]);
    const bwdAt = (phase: number) => [0, 0.5, 1].map((w) => glowAt(phase, w)[1]);
    expect(fwdAt(0)).toEqual([1, GLOW_MIN, GLOW_MIN]); // the batch enters at the inputs
    expect(fwdAt(0.225)[1]).toBeCloseTo(1); // halfway down
    expect(fwdAt(0.45)[2]).toBeCloseTo(1); // the deepest node
    expect(bwdAt(0.5)[2]).toBeCloseTo(0.9); // the gradients start at the deepest node…
    expect(bwdAt(0.95)[0]).toBeCloseTo(0.9); // …and reach the inputs
    expect(bwdAt(0.2)).toEqual([GLOW_MIN, GLOW_MIN, GLOW_MIN]); // no backward glow during the forward pass
    expect(fwdAt(0.75)).toEqual([GLOW_MIN, GLOW_MIN, GLOW_MIN]); // nor forward during the backward one
  });

  it('never goes fully dark (the browser would redraw the glow at every flash)', () => {
    for (let p = 0; p < 1; p += 0.01) {
      for (const w of [0, 0.3, 0.7, 1]) {
        for (const v of glowAt(p, w)) {
          expect(v).toBeGreaterThanOrEqual(GLOW_MIN);
          expect(v).toBeLessThanOrEqual(1);
        }
      }
    }
  });
});

describe('downsample', () => {
  it('keeps short series as they are and averages long ones into the chart width', () => {
    expect(downsample([1, 2, 3], 220)).toEqual([1, 2, 3]);
    const long = Array.from({ length: 6570 }, (_, i) => i);
    const d = downsample(long, 220);
    expect(d.length).toBe(220);
    expect(d[0]).toBeLessThan(d[219]);
    expect(Math.abs(d.reduce((a, b) => a + b, 0) / 220 - 3284.5)).toBeLessThan(20); // the mean is kept
  });
});

describe('progress', () => {
  it("takes the backend's step count when it sends one", () => {
    expect(totalSteps({ total_steps: 123, train_samples: 5 }, 9, 9)).toBe(123);
  });

  it("works it out like the backend does otherwise: epochs × ⌈rows / batch⌉ (capped)", () => {
    expect(totalSteps({ train_samples: 409 }, 3, 64)).toBe(3 * 7);
    expect(totalSteps({ train_samples: 10 }, 2, 64)).toBe(2); // the batch never exceeds the rows
    expect(totalSteps({ train_samples: 4000 }, 8, 64, 100)).toBe(100);
    expect(totalSteps({}, 3, 64)).toBeNull();
  });

  it('is a share between 0 and 1', () => {
    expect(progressOf(5, 10)).toBe(0.5);
    expect(progressOf(15, 10)).toBe(1);
    expect(progressOf(3, null)).toBe(0);
  });
});

describe('the training sliders', () => {
  it('step along ladders of round values', () => {
    for (const ladder of [EPOCH_STEPS, LR_STEPS, BATCH_STEPS, SAMPLE_STEPS]) {
      expect([...ladder].sort((a, b) => a - b)).toEqual([...ladder]); // increasing
      ladder.forEach((v, i) => expect(nearestIndex(ladder, v)).toBe(i)); // every rung finds itself
    }
  });

  it('find the nearest rung on a log scale', () => {
    expect(LR_STEPS[nearestIndex(LR_STEPS, 0.0007)]).toBe(5e-4);
    expect(LR_STEPS[nearestIndex(LR_STEPS, 0.004)]).toBe(5e-3);
    expect(BATCH_STEPS[nearestIndex(BATCH_STEPS, 100)]).toBe(128);
    expect(SAMPLE_STEPS[nearestIndex(SAMPLE_STEPS, 1e9)]).toBe(60000);
  });

  it('fill their track to where the value is', () => {
    expect(ladderPct(BATCH_STEPS, 1)).toBe('0%');
    expect(ladderPct(BATCH_STEPS, 2048)).toBe('100%');
    expect(rangePct(0, 1, 0.25)).toBe('25%');
    expect(rangePct(0, 1, 7)).toBe('100%');
    expect(rangePct(1, 1, 1)).toBe('0%');
  });
});
