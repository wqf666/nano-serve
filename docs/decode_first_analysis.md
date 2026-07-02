# Decode-first Scheduler Analysis

## Strategy

Each scheduling round:
1. Schedule ALL running decode requests first
2. If no decode work, schedule waiting prefill requests
3. No chunked prefill in this phase

## Trade-off vs FCFS

| Metric | Expected | Actual (mixed, 64 req, rate=4) |
|--------|----------|-------------------------------|
| TTFT | Increase (prefill waits behind decode) | P50: 55ms → 2.32s (+42x) |
| ITL P50 | Decrease | 4.6ms → 6.1ms (+33%) |
| ITL P95 | Decrease | 6.2ms → 8.2ms (+32%) |
| E2E P95 | Similar | 2.68s → 5.69s (+112%) |
| Throughput | Similar | 989 → 938 tok/s (-5%) |

## Observations

1. TTFT degradation is expected and significant. When decode requests exist,
   new prefill requests must wait until all decode work is done in that step.
   Under concurrent load, prefill requests queue up behind continuous decode
   batches, causing multi-second TTFT.

2. ITL did NOT improve as expected. Possible reasons:
   - The strict separation means newly-prefilled sequences (transitioned
     from prefill to running) must wait one additional step before their
     first decode, adding latency to their ITL stream.
   - With FCFS, prefill and decode share the same step, allowing the model
     to process prefill tokens while decode sequences are waiting, leading
     to better overall step utilization.

3. Throughput decreased ~5%, likely because the decode-first ordering
   leaves token budget unused when there are fewer decode sequences than
   max_num_batched_tokens.

## Conclusion

For this workload configuration (mixed, 64 requests, rate=4, concurrency=16),
decode-first does not improve ITL and significantly degrades TTFT. The
strict separation of decode and prefill phases creates inefficiency compared
to the mixed scheduling of FCFS.

Future phases (Chunked Prefill, SLO-aware) will address this by allowing
partial prefill within the same step as decode, reducing TTFT while
maintaining decode stability.

## Next Steps

- Phase 5 (Chunked Prefill): Allow prefill to be split across steps,
  interleaving prefill chunks with decode in the same step.
- Phase 7 (SLO-aware): Dynamically adjust decode/prefill priority based
  on SLO violation risk.
