/**
 * Stage 22 — live training on the canvas, without slowing training down, and the
 * ladders the training sliders move along. No Angular here (unit-tested as is).
 *
 * Why batching matters: the backend writes one line per training step and waits when
 * the browser does not read. Reading therefore never draws — events are only collected,
 * and handed to the page once per animation frame (one redraw for many steps).
 */

/**
 * Collects events and hands them over together: at an animation frame, and no more often
 * than every `minIntervalMs` — 20 times a second (a 144 Hz screen would otherwise redraw
 * the page 144 times a second, on the same processor that trains the network).
 */
export class FrameBatcher<T> {
  private pending: T[] = [];
  private scheduled = false;
  private last = -Infinity;
  /** how many hand-overs happened (the tests count them) */
  flushes = 0;

  constructor(
    private readonly onFlush: (items: T[]) => void,
    private readonly schedule: (run: () => void) => void = nextFrame,
    private readonly now: () => number = () => performance.now(),
    private readonly minIntervalMs = 50,
  ) {}

  push(item: T): void {
    this.pending.push(item);
    if (!this.scheduled) {
      this.scheduled = true;
      this.schedule(() => this.onFrame());
    }
  }

  /** A frame came: hand over, unless the last hand-over was too recent — then wait for a later frame. */
  private onFrame(): void {
    if (!this.pending.length) {
      this.scheduled = false; // handed over meanwhile (flush) or forgotten (reset): the chain ends
      return;
    }
    if (this.now() - this.last < this.minIntervalMs) {
      this.schedule(() => this.onFrame());
      return;
    }
    this.scheduled = false;
    this.flush();
  }

  /** Hand over what is waiting now (also called when a run ends, so nothing is left behind). */
  flush(): void {
    if (!this.pending.length) return;
    const items = this.pending;
    this.pending = [];
    this.flushes += 1;
    this.last = this.now();
    this.onFlush(items);
  }

  /** Forget what is waiting (a new run starts). */
  reset(): void {
    this.pending = [];
  }
}

/** The next animation frame; a hidden tab gets no frames, so it falls back to a timer. */
function nextFrame(run: () => void): void {
  const visible = typeof document === 'undefined' || !document.hidden;
  if (visible && typeof requestAnimationFrame === 'function') requestAnimationFrame(() => run());
  else setTimeout(run, 200);
}

/**
 * How many steps a run takes: the backend says so (Stage 22: `total_steps` in its first
 * event); an older backend's is worked out the same way it counts (epochs × batches).
 */
export function totalSteps(
  start: { total_steps?: number | null; train_samples?: number | null },
  epochs: number,
  batch: number,
  stepCap?: number | null,
): number | null {
  if (start.total_steps && start.total_steps > 0) return start.total_steps;
  const n = start.train_samples ?? 0;
  if (n <= 0 || epochs <= 0 || batch <= 0) return null;
  let total = epochs * Math.ceil(n / Math.min(batch, n));
  if (stepCap && stepCap > 0) total = Math.min(total, stepCap);
  return total;
}

/**
 * At most `max` points for a small chart (the mean of each run of values): a 220-px sparkline
 * of 6 000 steps redraws 220 points, not 6 000.
 */
export function downsample(values: readonly number[], max: number): number[] {
  const n = values.length;
  if (n <= max) return values.slice();
  const out: number[] = [];
  for (let b = 0; b < max; b++) {
    const lo = Math.floor((b * n) / max);
    const hi = Math.floor(((b + 1) * n) / max);
    let sum = 0;
    for (let i = lo; i < hi; i++) sum += values[i];
    out.push(sum / (hi - lo));
  }
  return out;
}

/** One forward + backward pass of the training wave on the canvas, and how often it moves. */
export const PASS_MS = 1600;
export const WAVE_TICK_MS = 50;

/** Where the wave is (0…1 along a pass) after `elapsedMs` of training. */
export function wavePhase(elapsedMs: number, passMs = PASS_MS): number {
  if (!(elapsedMs > 0) || !(passMs > 0)) return 0;
  return (elapsedMs % passMs) / passMs;
}

/** The faintest glow: never quite 0, or the browser drops the layer and draws it anew at every flash. */
export const GLOW_MIN = 0.002;

/**
 * How brightly a node glows at a point of the pass: [forward, backward], each GLOW_MIN…1.
 * The front runs down the graph over the first 45 % of a pass (the batch: forward) and back up
 * over 50…95 % (the gradients: backward); a node at depth `wave` (0 at the inputs … 1 at the
 * deepest) glows while the front is within 1/6 of it.
 */
export function glowAt(phase: number, wave: number): [number, number] {
  const fwd = phase / 0.45;
  const bwd = 1 - (phase - 0.5) / 0.45;
  const f = Math.max(GLOW_MIN, 1 - 6 * Math.abs(fwd - wave));
  const b = Math.max(GLOW_MIN, 0.9 - 6 * Math.abs(bwd - wave));
  return [Math.min(1, f), Math.min(1, b)];
}

/** A share done, 0…1. */
export function progressOf(step: number, total: number | null): number {
  if (!total || total <= 0) return 0;
  return Math.max(0, Math.min(1, step / total));
}

// ---- the training sliders: they move along ladders of round values -------------------
export const EPOCH_STEPS: readonly number[] = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 25, 30, 40, 50, 60, 80, 100];
export const LR_STEPS: readonly number[] = [
  1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1,
];
export const BATCH_STEPS: readonly number[] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
export const SAMPLE_STEPS: readonly number[] = [
  16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 16000, 20000, 30000, 40000,
  50000, 60000,
];

/** The ladder rung nearest to a value (compared on a log scale: 3e-4 is nearer 2e-4 than 1e-3). */
export function nearestIndex(steps: readonly number[], v: number): number {
  if (!steps.length) return 0;
  const key = (x: number) => (x > 0 ? Math.log(x) : -1e9);
  const kv = key(v);
  let best = 0;
  for (let i = 1; i < steps.length; i++) {
    if (Math.abs(key(steps[i]) - kv) < Math.abs(key(steps[best]) - kv)) best = i;
  }
  return best;
}

/** How far along its ladder a value is, for the slider's filled track ("37.5%"). */
export function ladderPct(steps: readonly number[], v: number): string {
  if (steps.length < 2) return '0%';
  return `${(nearestIndex(steps, v) / (steps.length - 1)) * 100}%`;
}

/** How far along a [min, max] range a value is ("50%"), for a parameter's slider. */
export function rangePct(min: number, max: number, v: number): string {
  if (!(max > min)) return '0%';
  return `${Math.max(0, Math.min(1, (v - min) / (max - min))) * 100}%`;
}
