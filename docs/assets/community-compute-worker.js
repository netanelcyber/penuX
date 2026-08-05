'use strict';

let wasm = null;

function hex(buffer) {
  return Array.from(new Uint8Array(buffer), b => b.toString(16).padStart(2, '0')).join('');
}

function toBytes(value) {
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  throw new Error('WASM bytes were not supplied to the worker');
}

async function initWasm(input) {
  if (wasm) return wasm;
  const bytes = toBytes(input);
  if (bytes.byteLength < 8) throw new Error(`WASM download is too short (${bytes.byteLength} bytes)`);
  if (!WebAssembly.validate(bytes)) throw new Error(`Downloaded WASM failed validation (${bytes.byteLength} bytes)`);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const loaded = await WebAssembly.instantiate(bytes, {});
  const instance = loaded.instance || loaded;
  const required = ['memory', 'metrics_from_counts', 'coverage', 'descriptives', 'crosstab_2x2'];
  const missing = required.filter(name => !(name in instance.exports));
  if (missing.length) throw new Error(`WASM exports are incomplete: ${missing.join(', ')}`);
  wasm = { instance, hash: hex(digest), byteLength: bytes.byteLength };
  return wasm;
}

function safeNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function ensureMemory(memory, bytes) {
  if (bytes <= memory.buffer.byteLength) return;
  memory.grow(Math.ceil((bytes - memory.buffer.byteLength) / 65536));
}

async function digestResult(value) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(value)));
  return hex(digest);
}

function requireWasm() {
  if (!wasm) throw new Error('Worker was not initialized with the downloaded WASM module');
  return wasm;
}

async function calculateMetrics(payload) {
  const module = requireWasm();
  const { memory, metrics_from_counts: fn } = module.instance.exports;
  const outputOffset = 1024;
  fn(Math.trunc(safeNumber(payload.tp)), Math.trunc(safeNumber(payload.tn)), Math.trunc(safeNumber(payload.fp)), Math.trunc(safeNumber(payload.fn)), outputOffset);
  const values = Array.from(new Float64Array(memory.buffer, outputOffset, 8));
  const result = {
    sensitivity: values[0], specificity: values[1], ppv: values[2], npv: values[3],
    accuracy: values[4], f1: values[5], f2: values[6], balanced_accuracy: values[7],
    wasm_sha256: module.hash
  };
  result.digest = await digestResult({ task_id: payload.task_id || null, ...result });
  return result;
}

async function calculateCoverage(payload) {
  const module = requireWasm();
  const { memory, coverage: fn } = module.instance.exports;
  const flags = Array.isArray(payload.flags) ? payload.flags.map(v => v ? 1 : 0) : [];
  const offset = 4096;
  ensureMemory(memory, offset + flags.length * 4);
  new Int32Array(memory.buffer, offset, flags.length).set(flags);
  const count = fn(flags.length, offset);
  return { count, total: flags.length, coverage: flags.length ? count / flags.length : 0, wasm_sha256: module.hash };
}

async function calculateDescriptives(payload) {
  const module = requireWasm();
  const { memory, descriptives: fn } = module.instance.exports;
  const values = (Array.isArray(payload.values) ? payload.values : []).map(Number).filter(Number.isFinite);
  const inputOffset = 8192;
  const outputOffset = 2048;
  ensureMemory(memory, inputOffset + values.length * 8);
  new Float64Array(memory.buffer, inputOffset, values.length).set(values);
  fn(inputOffset, values.length, outputOffset);
  const out = Array.from(new Float64Array(memory.buffer, outputOffset, 6));
  return { n: out[0], mean: out[1], sd: out[2], minimum: out[3], maximum: out[4], standard_error: out[5], wasm_sha256: module.hash };
}

async function calculateCrosstab(payload) {
  const module = requireWasm();
  const { memory, crosstab_2x2: fn } = module.instance.exports;
  const outputOffset = 3072;
  fn(safeNumber(payload.a), safeNumber(payload.b), safeNumber(payload.c), safeNumber(payload.d), outputOffset);
  const out = Array.from(new Float64Array(memory.buffer, outputOffset, 6));
  return { pearson_chi_square: out[0], p_approx_df1: out[1], phi: out[2], odds_ratio: out[3], relative_risk: out[4], yates_chi_square: out[5], wasm_sha256: module.hash };
}

self.onmessage = async event => {
  const message = event.data || {};
  try {
    if (message.type === 'init') {
      const module = await initWasm(message.payload && message.payload.wasmBytes);
      self.postMessage({ id: message.id, ok: true, type: 'ready', wasm_sha256: module.hash, wasm_bytes: module.byteLength });
      return;
    }
    let result;
    if (message.type === 'metrics') result = await calculateMetrics(message.payload || {});
    else if (message.type === 'coverage') result = await calculateCoverage(message.payload || {});
    else if (message.type === 'descriptives') result = await calculateDescriptives(message.payload || {});
    else if (message.type === 'crosstab') result = await calculateCrosstab(message.payload || {});
    else throw new Error('Unsupported task type');
    self.postMessage({ id: message.id, ok: true, type: message.type, result });
  } catch (error) {
    self.postMessage({ id: message.id, ok: false, error: error instanceof Error ? error.message : String(error) });
  }
};
