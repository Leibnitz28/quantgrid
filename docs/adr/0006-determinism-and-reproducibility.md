# ADR 0006 — Deterministic engine, hashed specs and reproducibility manifests

**Status:** accepted

## Decision
* A `TaskSpec` contains everything that can change a numerical result (dataset name +
  SHA-256 signature, strategy + version, normalised params, split, fee model, seed,
  workload, fault injection, result schema). `spec_hash` = SHA-256 of its canonical JSON.
* Datasets are recorded (materialised Parquet) and verified against a *logical* content
  hash on every load; synthetic datasets have golden hashes checked in tests.
* Each batch stores a manifest: `code_sha` (hash of every source file of the package),
  git commit/dirty flag, interpreter, platform, the full installed-package list
  (`env_sha`), the `uv.lock` hash and the dataset record. Each attempt records the
  worker's `code_sha`/`env_sha`.
* Cache reuse requires identical `spec_hash`, `code_sha` and `env_sha`.
* `quantgrid reproduce <batch>` re-runs specs from the manifest and compares output
  checksums byte-for-byte, falling back to a documented numerical tolerance when code or
  environment drifted (e.g. different CPU SIMD paths or library versions).

## Consequences
* Bitwise reproducibility is guaranteed only for the same code + environment + CPU
  architecture; elsewhere results are compared within tolerance, never silently replaced.
