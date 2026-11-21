# Limitations

## Scope

QuantGrid is research infrastructure. It has no broker or exchange connectivity and places no orders. Results from the synthetic datasets say nothing about profitability: the data is generated with a known mean-reverting structure, so strategies find it by construction. `pairs-rw-control-v1`, two independent random walks, is the negative control.

## Modelling assumptions

- Prices come from a two-asset log-price model: one asset is a random walk with drift, the spread is an Ornstein-Uhlenbeck process. There are no dividends, splits, corporate actions or survivorship effects because there is no real universe.
- Timestamps are int64 nanoseconds in UTC and strictly increasing. Gaps are allowed and bars are treated as a sequence. There is no trading calendar, session model or time-zone handling.
- Costs are linear commission plus slippage in basis points on gross traded notional, both legs. There is no market impact, borrow cost, financing, short-sale constraint, partial fill or latency.
- A signal is computed at the close of bar `i` and held over bar `i+1`.
- The hedge ratio is fitted once on the training segment and not re-estimated in the test segment. The split is chronological with an optional embargo.
- Reported statistics are descriptive. There is no multiple-testing correction across a parameter grid; a grid of 144 configurations will contain lucky winners.

## System

- Authentication is static API keys. Quotas are one deployment-wide policy.
- Traffic is HTTP, and PostgreSQL and Redis are unauthenticated in the local setup. See [SECURITY.md](../SECURITY.md).
- The artifact store is a local filesystem. An S3-compatible store is not implemented.
- There is one PostgreSQL and one Redis, with no replication. Losing PostgreSQL loses the system; losing Redis loses nothing.
- Only the built-in strategy `pairs_zscore` exists. `register_strategy` and the engine registry are the extension points for others.
- The Compose stack and the Docker image have no published release; build them from source.

## Not built

mTLS and key rotation, per-tenant billing, audit-log shipping, backup and restore tooling, alerting rules, multi-region deployment, secrets management, dataset licensing workflow, a web UI. See [ROADMAP.md](../ROADMAP.md).
