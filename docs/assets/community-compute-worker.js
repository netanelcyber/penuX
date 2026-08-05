'use strict';

// Compiled from a small C statistics kernel. No IBM SPSS code is included.
// Exports: metrics_from_counts, coverage, descriptives, crosstab_2x2.
const WASM_BASE64 = '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OQMIIAQgCTkDACAEIAAgB6NEAAAAAAAAAAAgB0QAAAAAAAAAAGQbIAIgAaNEAAAAAAAAAAAgAUQAAAAAAAAAAGQbIgKjRAAAAAAAAAAAIAJEAAAAAAAAAABkGzkDIAuuBgIEfAJ/RAAAAAAAAAAAIQECQCAARAAAAAAAAAAAY0UNAEQAAAAAAAAAQCAAmhCFgICAAKEPC0QAAAAAAADwPyAARIx7PalA99Q/okQAAAAAAADwP6CjIgJELaFVQoT78D+iRDlMAVccQPe/oCACokRX4bpVHL72P6AgAqJEaRQ8zDE10r+gIAKiRH5a7MYgT9A/oCEDAkAgAJogAKIiAEQAAAAAAABJwGMNAAJAAkBEAAAAAAAASUAgACAARAAAAAAAAElAZBsiAZogASABRAAAAAAAAAAAYxsiAETvOfr+Qi7mP6MiBJlEAAAAAAAA4EFjRQ0AIASqIQUMAQtBgICAgHghBQsgACAFt0TvOfr+Qi7mP6KhIgBEAAAAAAAA8D+gIAAgAEQAAAAAAADgP6KiIgSgIABEAAAAAAAACECjIASiIgSgIABEAAAAAAAA0D+iIASiIgSgIABEAAAAAAAAFECjIASiIgSgIABEAAAAAAAAGECjIASiIgSgIABEAAAAAAAAHECjIASiIgSgIABEAAAAAAAAwD+iIASiIgSgIABEAAAAAAAAIkCjIASiIgSgIABEAAAAAAAAJECjIASiIgSgIABEAAAAAAAAJkCjIASiIgSgIABEAAAAAAAAKECjIASiIgSgIABEAAAAAAAAKkCjIASiIgSgIABEAAAAAAAALECjIASiIgSgIABEAAAAAAAALkCjIASiIgSgIABEAAAAAAAAsD+iIASiIgSgIABEAAAAAAAAMUCjIASiIgSgIABEAAAAAAAAMkCjIASioCEERAAAAAAAAPA/IQACQCAFQQFIDQAgBUEHcSEGAkACQCAFQQhPDQBEAAAAAAAA8D8hAAwBCyAFQfj///8HcSEFRAAAAAAAAPA/IQADQCAAIACgIgAgAKAiACAAoCIAIACgIgAgAKAiACAAoCIAIACgIgAgAKAhACAFQXhqIgUNAAsLIAZFDQADQCAAIACgIQAgBkF/aiIGDQALC0QAAAAAAADwPyAEIACiIgCjIAAgAUQAAAAAAAAAAGMbIQELRAAAAAAAAPA/RAAAAAAAAPA/IAIgA6IgAaKhoQsAgQEEbmFtZQAUE3Nwc3MtY29tbXVuaXR5Lndhc20BUAYAE21ldHJpY3NfZnJvbV9jb3VudHMBCGNvdmVyYWdlAgxkZXNjcmlwdGl2ZXMDBWRzcXJ0BAxjcm9zc3RhYl8yeDIFC2VyZmNfYXBwcm94BxIBAA9fX3N0YWNrX3BvaW50ZXIAfwlwcm9kdWNlcnMBDHByb2Nlc3NlZC1ieQEFY2xhbmdfMTcuMC4wIChodHRwczovL2dpdGh1Yi5jb20vc3dpZnRsYW5nL2xsdm0tcHJvamVjdC5naXQgMTA5OTliNmQwMzRmZTMxOGYzZDU2YzgzYmRkYjY1NzI1OTNhOGJiMCkASQ90YXJnZXRfZmVhdHVyZXMEKwptdWx0aXZhbHVlKw9tdXRhYmxlLWdsb2JhbHMrD3JlZmVyZW5jZS10eXBlcysIc2lnbi1leHQ=';

let wasm = null;

function decodeBase64(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function hex(buffer) {
  return Array.from(new Uint8Array(buffer), b => b.toString(16).padStart(2, '0')).join('');
}

async function initWasm() {
  if (wasm) return wasm;
  const bytes = decodeBase64(WASM_BASE64);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const loaded = await WebAssembly.instantiate(bytes, {});
  wasm = { instance: loaded.instance || loaded, hash: hex(digest) };
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

async function calculateMetrics(payload) {
  const module = await initWasm();
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
  const module = await initWasm();
  const { memory, coverage: fn } = module.instance.exports;
  const flags = Array.isArray(payload.flags) ? payload.flags.map(v => v ? 1 : 0) : [];
  const offset = 4096;
  ensureMemory(memory, offset + flags.length * 4);
  new Int32Array(memory.buffer, offset, flags.length).set(flags);
  const count = fn(flags.length, offset);
  return { count, total: flags.length, coverage: flags.length ? count / flags.length : 0, wasm_sha256: module.hash };
}

async function calculateDescriptives(payload) {
  const module = await initWasm();
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
  const module = await initWasm();
  const { memory, crosstab_2x2: fn } = module.instance.exports;
  const outputOffset = 3072;
  fn(safeNumber(payload.a), safeNumber(payload.b), safeNumber(payload.c), safeNumber(payload.d), outputOffset);
  const out = Array.from(new Float64Array(memory.buffer, outputOffset, 6));
  return { pearson_chi_square: out[0], p_approx_df1: out[1], phi: out[2], odds_ratio: out[3], relative_risk: out[4], yates_chi_square: out[5], wasm_sha256: module.hash };
}

self.onmessage = async event => {
  const message = event.data || {};
  try {
    let result;
    if (message.type === 'init') {
      const module = await initWasm();
      self.postMessage({ id: message.id, ok: true, type: 'ready', wasm_sha256: module.hash });
      return;
    }
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
