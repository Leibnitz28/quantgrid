# ADR 0003 — Content-addressed artifacts with staged commit; publication is a DB row

**Status:** accepted

## Context
Workers write Parquet artifacts. Two attempts of the same task can overlap (zombie +
replacement), and a worker can die between writing a file and recording it.

## Decision
* Artifacts are written to a temp file on the same filesystem, `fsync`ed, hashed, then
  `os.replace`d to `sha256/<aa>/<sha>.parquet` and the directory is `fsync`ed.
* Because the path *is* the content hash, a different result can never overwrite another,
  and identical results (deterministic retries, cache hits) deduplicate naturally.
* The artifact only becomes a **published result** when the single `result_artifacts`
  row (`task_id UNIQUE`) is inserted in the same transaction that moves the task to
  SUCCEEDED under a valid lease. Files without a row are orphans that the verifier reports
  and `gc` may delete; they are never served as results.
* Artifacts embed only spec-derived metadata (no task ids or timestamps), so identical
  specs produce byte-identical files across batches and machines with the same environment.

## Consequences
* Readers trust only DB-referenced artifacts, re-verifying the checksum when needed.
* An S3-compatible store fits the same protocol (PUT to the content key is atomic per
  object); only the local store is implemented in the MVP.
