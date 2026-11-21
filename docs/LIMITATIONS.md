# Limitations, assumptions and out-of-scope

## This is research infrastructure, not a trading system
* **No live trading.** No broker/exchange credentials, endpoints or order placement exist anywhere.
* **Simulated results are not evidence of profitability.** The synthetic datasets are generated with a
  *known* mean-reverting structure, so strategies find it by construction. `pairs-rw-control-v1`
  (independent random walks) is the negative control: a sane strategy should not look good there.

## Financial-data assumptions (synthetic data)
* Prices: two-asset log-price model; one asset a random walk with drift, the spread an
  Ornstein-Uhlenbeck process. No dividends, splits, corporate actions or survivorship issues exist
  because there is no real universe; real data would need all of these documented.
* Timestamps: int64 nanoseconds, **UTC**, strictly increasing. Gaps are allowed (`pairs-ou-gappy-v1`);
  bars are treated sequentially, with no calendar or session model, no holidays, no time zones.
* Costs: linear commission + slippage in bps on gross notional, both legs. No market impact, borrow
  cost, financing, short-sale constraints, partial fills or latency.
* Execution: signal at the close of bar *i*, position held over bar *i+1* (no look-ahead; tested against
  a pure-Python reference and with perturbation tests).
* Hedge ratio is fit once on the train segment (OLS on log prices) and **not** re-estimated in the test
  segment. Train/test split is chronological with optional embargo.
* Statistics reported (Sharpe, bootstrap CI) are descriptive. No multiple-testing correction is applied
  across a parameter grid - a grid search over 144 configurations will find lucky winners.

## Engineering scope
* Local-first: native dev services or Docker Compose. **No Kubernetes**, no managed services.
* Authentication is static API keys; quotas are one deployment-wide policy, not per-tenant billing.
* Artifact store is local filesystem only (the `ArtifactStore` protocol leaves room for S3; not built).
* Single PostgreSQL, single Redis, no replication or failover. Redis can be flushed without data loss
  (work is re-dispatched from PostgreSQL); PostgreSQL loss is loss of the system.
* The optional external-backtester adapter is **not implemented**; the standalone engine is the only
  engine. `domain.specs.register_strategy` and the engine registry are the extension points.
* Ray was not added: Celery + PostgreSQL leases were sufficient for the measured workloads.
* The AI planner is optional, read-only and advisory. Its output is validated by the same
  deterministic policy as a human request, and nothing it says is ever a source of numbers.

## Production hardening deliberately omitted
mTLS and key rotation, per-tenant quotas and billing, audit-log shipping, backup/restore runbooks,
SLO alerting, multi-region, secrets management, an S3 artifact store with lifecycle rules,
dataset licensing/approval workflow, and a UI.
