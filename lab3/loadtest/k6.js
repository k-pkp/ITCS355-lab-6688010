// ITCS355 Lab 3 — load test.
//
//   k6 run -e TARGET=https://<endpoint>/predict -e VUS=10 loadtest/k6.js
//
// Run this at THREE concurrency levels (suggested 1, 10, 50) and record p50, p95, p99,
// throughput, and error rate for each. Commit the results in reports/lab3-load.md.
//
// An uncommitted load test is not evidence.

import http from 'k6/http';
import { check } from 'k6';
import { Trend, Rate } from 'k6/metrics';

const latency = new Trend('predict_latency_ms');
const failures = new Rate('predict_failures');

export const options = {
  vus: Number(__ENV.VUS || 10),
  duration: __ENV.DURATION || '60s',
  thresholds: {
    // TARGET, stated 2026-09-07 before the first measurement was taken.
    //
    // p95 < 100 ms for a single /predict at 10 concurrent users, and fewer than 1% errors.
    //
    // Where 100 ms comes from: this service backs a maintenance triage screen that a
    // technician refreshes while standing at a machine. Anything under about 100 ms reads
    // as instant to a person, so latency below that buys nothing they would notice, and
    // the budget is better spent on a cheaper instance. It is a product decision, not a
    // number copied from a tutorial. The service's own work — one random forest scoring
    // one row — takes single-digit milliseconds, so the remaining budget is for the hop
    // to a managed endpoint, TLS, and the queueing that shows up under load.
    //
    // 10 VUs is the stated concurrency because it is roughly the whole maintenance team
    // hitting the screen at once. Higher levels are measured to find the breaking point,
    // not because the target applies there.
    'predict_latency_ms': ['p(95)<100'],
    'predict_failures': ['rate<0.01'],
  },
};

const payload = JSON.stringify({
  temp_c: 78.4,
  vibration_mm_s: 3.1,
  pressure_kpa: 315.2,
  hours_since_service: 4200,
  load_pct: 68.0,
  ambient_humidity: 55.0,
});

export default function () {
  const res = http.post(__ENV.TARGET, payload, {
    headers: { 'Content-Type': 'application/json' },
  });
  latency.add(res.timings.duration);
  failures.add(res.status !== 200);
  check(res, {
    'status is 200': (r) => r.status === 200,
    'probability present': (r) => r.status === 200 && r.json('probability') !== undefined,
    'version reported': (r) => r.headers['X-Model-Version'] !== undefined,
  });
}
