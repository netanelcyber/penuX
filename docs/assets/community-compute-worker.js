'use strict';

const WASM_BASE64 = 'AGFzbQEAAAABFwNgBX9/f3x/AGAFf39/f38AYAJ/fwF/AwQDAAECBAUBcAEBAQUFAQECgAIGCAF/AUGAiAQLBzYEBm1lbW9yeQIACGV2YWx1YXRlAAATbWV0cmljc19mcm9tX2NvdW50cwABCGNvdmVyYWdlAAIKgwYDywEDBH8BfAF/AkACQCAAQQFODQBBACEFQQAhBkEAIQdBACEIDAELQQAhCEEAIQdBACEGQQAhBQNAIAIrAwAhCQJAAkAgASgCACIKRQ0AIAkgA2ZFDQAgCEEBaiEIDAELAkAgCg0AIAkgA2YNACAHQQFqIQcMAQsCQCAKDQAgCSADZkUNACAFQQFqIQUMAQsgBkEBaiEGCyACQQhqIQIgAUEEaiEBIABBf2oiAA0ACwsgBCAGNgIMIAQgBTYCCCAEIAc2AgQgBCAINgIAC/ACAQt8RAAAAAAAAAAAIQVEAAAAAAAAAAAhBgJAIAG3IgcgArciCKAiCSAAtyIKIAO3IgugIgygIg1EAAAAAAAAAABkRQ0AIAogB6AgDaMhBgsCQCAKIAqgIg0gCKAgC6AiDkQAAAAAAAAAAGRFDQAgDSAOoyEFC0QAAAAAAAAAACENAkAgCkQAAAAAAAAUQKIiDiALRAAAAAAAABBAoqAgCKAiD0QAAAAAAAAAAGRFDQAgDiAPoyENCyAEIA05AzAgBCAFOQMoIAQgBjkDICAEIAcgCaNEAAAAAAAAAAAgCUQAAAAAAAAAAGQbIgk5AwggBCAKIAyjRAAAAAAAAAAAIAxEAAAAAAAAAABkGyIMOQMAIAQgByAHIAugIgujRAAAAAAAAAAAIAtEAAAAAAAAAABkGzkDGCAEIAogCiAIoCIHo0QAAAAAAAAAACAHRAAAAAAAAAAAZBs5AxAgBCAJIAygRAAAAAAAAOA/ojkDOAvBAQEEfwJAIABBAU4NAEEADwsgAEEDcSECAkACQCAAQQRPDQBBACEDQQAhBAwBCyAAQfz///8HcSEFIAEhAEEAIQNBACEEA0AgBCAAKAIAQQBHaiAAQQRqKAIAQQBHaiAAQQhqKAIAQQBHaiAAQQxqKAIAQQBHaiEEIABBEGohACAFIANBBGoiA0cNAAsLAkAgAkUNACABIANBAnRqIQADQCAEIAAoAgBBAEdqIQQgAEEEaiEAIAJBf2oiAg0ACwsgBAsAXgRuYW1lABcWY29tbXVuaXR5X21ldHJpY3Mud2FzbQEqAwAIZXZhbHVhdGUBE21ldHJpY3NfZnJvbV9jb3VudHMCCGNvdmVyYWdlBxIBAA9fX3N0YWNrX3BvaW50ZXIAfwlwcm9kdWNlcnMBDHByb2Nlc3NlZC1ieQEFY2xhbmdfMTcuMC4wIChodHRwczovL2dpdGh1Yi5jb20vc3dpZnRsYW5nL2xsdm0tcHJvamVjdC5naXQgMTA5OTliNmQwMzRmZTMxOGYzZDU2YzgzYmRkYjY1NzI1OTNhOGJiMCkASQ90YXJnZXRfZmVhdHVyZXMEKwptdWx0aXZhbHVlKw9tdXRhYmxlLWdsb2JhbHMrD3JlZmVyZW5jZS10eXBlcysIc2lnbi1leHQ=';

let wasm = null;

function decodeBase64(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function initWasm() {
  if (wasm) return wasm;
  const bytes = decodeBase64(WASM_BASE64);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const hash = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
  const instance = await WebAssembly.instantiate(bytes, {});
  wasm = { instance: instance.instance || instance, hash };
  return wasm;
}

function safeNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

async function calculateMetrics(payload) {
  const module = await initWasm();
  const { memory, metrics_from_counts: metricsFromCounts } = module.instance.exports;
  if (!memory || typeof metricsFromCounts !== 'function') throw new Error('WASM exports are incomplete');

  const outputOffset = 1024;
  metricsFromCounts(
    Math.trunc(safeNumber(payload.tp)),
    Math.trunc(safeNumber(payload.tn)),
    Math.trunc(safeNumber(payload.fp)),
    Math.trunc(safeNumber(payload.fn)),
    outputOffset
  );
  const values = Array.from(new Float64Array(memory.buffer, outputOffset, 8));
  const result = {
    sensitivity: values[0], specificity: values[1], ppv: values[2], npv: values[3],
    accuracy: values[4], f1: values[5], f2: values[6], balanced_accuracy: values[7]
  };
  const canonical = JSON.stringify({ task_id: payload.task_id, ...result });
  const resultDigest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(canonical));
  result.digest = Array.from(new Uint8Array(resultDigest), b => b.toString(16).padStart(2, '0')).join('');
  result.wasm_sha256 = module.hash;
  return result;
}

async function calculateCoverage(payload) {
  const module = await initWasm();
  const { memory, coverage } = module.instance.exports;
  const flags = Array.isArray(payload.flags) ? payload.flags.map(v => v ? 1 : 0) : [];
  const offset = 2048;
  const requiredBytes = offset + flags.length * 4;
  if (requiredBytes > memory.buffer.byteLength) {
    memory.grow(Math.ceil((requiredBytes - memory.buffer.byteLength) / 65536));
  }
  new Int32Array(memory.buffer, offset, flags.length).set(flags);
  const count = coverage(flags.length, offset);
  return { count, total: flags.length, coverage: flags.length ? count / flags.length : 0, wasm_sha256: module.hash };
}

self.onmessage = async event => {
  const message = event.data || {};
  try {
    if (message.type === 'init') {
      const module = await initWasm();
      self.postMessage({ id: message.id, ok: true, type: 'ready', wasm_sha256: module.hash });
      return;
    }
    if (message.type === 'metrics') {
      const result = await calculateMetrics(message.payload || {});
      self.postMessage({ id: message.id, ok: true, type: 'metrics', result });
      return;
    }
    if (message.type === 'coverage') {
      const result = await calculateCoverage(message.payload || {});
      self.postMessage({ id: message.id, ok: true, type: 'coverage', result });
      return;
    }
    throw new Error('Unsupported task type');
  } catch (error) {
    self.postMessage({ id: message.id, ok: false, error: error instanceof Error ? error.message : String(error) });
  }
};
