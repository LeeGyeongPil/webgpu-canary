# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-16

### Added
- `selectRuntime()`: tries (device, dtype) candidates one at a time, with a time limit and `dispose()` of failed runners, and returns the first one whose real output agrees with a baseline.
- Judging for embeddings (cosine) and greedy generation (leading tokens, repeated-token runs) with per-dtype default thresholds and `CanaryConfigError` for dtypes without a threshold.
- Statuses `pass`, `error`, `timeout`, `numeric`, `mismatch`, `skipped` and a JSON report with the evidence per candidate.
- Environment filtering (`no-webgpu`, `no-shader-f16`, dtypes missing from the model) and a decision cache keyed by adapter, browser major version, model, revision and library version, with memory and `localStorage` stores.
- `webgpu-canary/transformers` loader for `@huggingface/transformers` v4, `webgpu-canary/react` hook `useCanaryRuntime`, `webgpu-canary/testing` fake loader with recorded fixtures.
- `webgpu-canary baseline` CLI that creates the cpu fp32 reference output, English and Korean messages (`--lang`).
- Mock and real examples, and a Vite demo page in English and Korean.
