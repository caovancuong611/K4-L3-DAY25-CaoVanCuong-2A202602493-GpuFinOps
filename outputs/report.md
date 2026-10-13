# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $15,004  
**Projected savings:** $12,129  (**45%**)

## Savings by lever

| Lever | Savings (USD) | % of baseline |
|---|---|---|
| Inference (cascade/cache/batch) | $1,200 | 4.4% |
| Purchasing (spot/reserved) | $9,789 | 36.1% |
| Right-size (MBU/roofline fit) | $540 | 2.0% |
| Kill idle GPUs | $600 | 2.2% |

## Sustainability

- Energy per query: 0.91 Wh
- Carbon per query: 0.344 gCO2e
- Cleanest region (best for carbon): europe-north1
- Cheapest electricity region: us-east-wa
- Balanced (50/50 cost + carbon): us-east-wa

## Unit economics ($/1M-token)

|  | Baseline | Optimized |
|---|---|---|
| $/day (inference API) | $48.87 | $8.88 |
| $/1M-token | $6.488 | $1.179 |

Inference savings 81.8%, applied in order on 7,533,027 tokens/day:

| Inference lever | $/day | $/month | % of inference baseline |
|---|---|---|---|
| cascade | $37.40 | $1,122 | 76.5% |
| cache | $0.80 | $24 | 1.6% |
| batch | $1.80 | $54 | 3.7% |

Cascade dominates because 80% of requests are easy enough for the small model, which is 15x cheaper on input and 37.5x on output; caching and batch then discount an already-small bill. Measured per GPU-hour none of this is visible: the same GPUs serve the traffic, but each 1M tokens costs 5.5x less.

## Why GPU-Util lies

| GPU | Type | GPU-Util | MFU | MBU | Billed $/month | Lost vs healthy MFU |
|---|---|---|---|---|---|---|
| gpu-h100-4 | H100 | 98.2% | 0.194 | 0.207 | $1,800 | $955 |
| gpu-a10g-1 | A10G | 96.9% | 0.268 | 0.302 | $720 | $253 |

`nvidia-smi` GPU-Util is the share of the sample window in which *at least one kernel* was running on the device. It says nothing about how many SMs were busy or whether tensor cores were fed. A single small kernel, a kernel waiting on HBM, or a stream of tiny kernels separated by launch overhead all read as ~100% busy.

`gpu-h100-4` is the clear case: 98% util but MFU 0.194 against 0.41 on its healthy H100 peers. Its MBU is also low, so it is neither compute- nor bandwidth-saturated: the GPU is mostly *waiting* (input pipeline / dataloader stalls, host-device copies, synchronisation, small per-GPU batch, kernel-launch overhead). Relative to its peers, about $1,208/month of billed GPU time across the flagged GPUs buys no useful FLOPs. The first fix is software (profile with Nsight/DCGM `sm_active`/`tensor_active`, larger batches, CUDA graphs, async data loading), which recovers the FLOPs without changing hardware; right-sizing (below) is the fallback.

## Recommended actions (ordered by ROI = savings / effort)

| # | Action | $/month | Effort | Why this order |
|---|---|---|---|---|
| 1 | Turn on cascade routing (small model first) | $1,200 | Low: router + eval set; no infra change | Largest $/1M-token win, fully reversible |
| 2 | Move interruptible jobs to spot + commit 3yr for 24/7 inference | $9,789 | Medium: checkpointing + a commitment decision | Largest $ lever; commitment only after usage is stable |
| 3 | Kill idle GPUs (auto-shutdown after training) | $600 | Low: idle-timeout policy | Pure waste, zero risk |
| 4 | Fix gpu-h100-4 software / right-size by MBU | $540 | Medium-High: profiling, migration tests | Needs perf validation before switching GPUs |

Inference and idle-GPU fixes go first despite a smaller $ than purchasing because they are cheap, immediate and reversible. Purchasing is the biggest lever but a reservation locks money in, so it is sized after cascade/right-sizing have shrunk the steady fleet; otherwise you commit to capacity you are about to remove.

## Extension 1: cost-based `recommend_tier()`

New policy in `finops/pricing.py::recommend_tier_detailed`: per-GPU spot interruption rate (H100 3%/h ... A10G 10%/h), reserved billed 24h/day (you pay for the commitment even when idle), 1yr vs 3yr amortised over the job's lifetime; cheapest option wins.

| Job | GPU | Old tier | New tier | On-demand | Old cost | New cost |
|---|---|---|---|---|---|---|
| job-train-llm | H100 | spot | spot | $12,000 | $7,596 | $7,524 |
| job-train-embed | A100 | spot | spot | $2,148 | $1,393 | $1,393 |
| job-finetune | H100 | spot | spot | $900 | $570 | $564 |
| job-infer-chat | A10G | reserved | reserved-3yr | $4,320 | $2,592 | $2,592 |
| job-infer-rag | A100 | reserved | reserved-3yr | $3,866 | $2,160 | $2,160 |
| job-infer-search | L4 | reserved | reserved-3yr | $1,728 | $972 | $1,296 |
| job-dev-sandbox | A10G | spot | spot | $480 | $203 | $208 |
| job-batch-eval | H100 | spot | spot | $225 | $142 | $141 |

Savings: old policy claimed 39.1%, new policy 38.1% (on-demand $25,667 -> $15,878). The old number was optimistic: it billed reserved GPUs only for hours used (L4 search at 18h/day) and assumed a 3yr term for everything. Sensitivity to the always-on service lifetime: 365 days -> 29.7%, 730 days -> 29.7%, 1095 days -> 38.1%. Below ~2.1-2.25 years (r3/r1 x 3 years, by GPU) the 3yr deal loses to 1yr (unused commitment years are sunk), so NimbusAI should only sign 3yr for services it is confident will still exist. Run `python missions/m3_purchasing.py` for the full GPU x duty x interruptible matrix.

## Extension 2: right-sizing by MBU

A candidate must hold the peak VRAM used, serve the observed bandwidth at <= 60% MBU and the observed FLOPs at <= 40% MFU; the cheapest $/GPU-hr that fits wins.

| GPU | Current | Proposed | MBU now | MBU new | Save | $/month | Why not the cheapest GPU |
|---|---|---|---|---|---|---|---|
| gpu-h100-4 | H100 | MI300X | 0.207 | 0.131 | 22.0% | $396 | L4 HBM 24GB < 67GB; L4 BW 0.3TB/s too low (MBU would be 230.7%); L4 FP16 too low |
| gpu-a100-0 | A100 | A100 | 0.276 | 0.276 | 0.0% | $0 | L4 HBM 24GB < 68GB; L4 BW 0.3TB/s too low (MBU would be 184.2%); L4 FP16 too low |
| gpu-a100-1 | A100 | A100 | 0.247 | 0.247 | 0.0% | $0 | L4 HBM 24GB < 67GB; L4 BW 0.3TB/s too low (MBU would be 164.9%); L4 FP16 too low |
| gpu-a10g-0 | A10G | L4 | 0.235 | 0.47 | 20.0% | $144 | - |
| gpu-a10g-1 | A10G | A10G | 0.302 | 0.302 | 0.0% | $0 | L4 BW 0.3TB/s too low (MBU would be 60.4%) |
| gpu-l4-0 | L4 | L4 | 0.328 | 0.328 | 0.0% | $0 | - |

Total: $540/month. Unit prices: best $/TB/s is MI300X ($0.368), worst L4 ($2.667); the cheapest GPU per hour is the most expensive per unit of bandwidth. Decode speed scales with TB/s and the model + KV cache must fit in HBM, so picking by $/GPU-hr would either OOM or push MBU past 100% (slower tokens, SLO miss). The A100 inference GPUs cannot shrink (they hold ~68 GB) - their lever is consolidation/batching, not a smaller card. MI300X for gpu-h100-4 needs a ROCm validation first.

## Extension 3: `cache_is_worth_it()`

Model: one prefix write per team x tier x hour (1.25x input price) + 1h storage (small $1.00, large $4.50 per 1M tokens/hour); each read costs 0.1x.

| Tier | Break-even reads | Actual avg reads | Groups worth caching |
|---|---|---|---|
| large | 1.94 | 4.2 | 86/95 |
| small | 5.83 | 18.8 | 95/96 |

Naive cache saving $1.20/day becomes $0.80/day net of $0.37/day write+storage. The small tier needs ~3x more reads to break even because storage is priced per token while its input is 15x cheaper. The dataset clears both thresholds on average, but the low-traffic hours do not, and those groups are now billed uncached.

## Extension 4: reasoning budget

|  | Reasoning | Plain |
|---|---|---|
| Share of requests | 8.4% | 91.6% |
| $/day | $1.40 | $7.11 |
| Wh/day | 29,788 | 1,888 |
| Wh/query | 148.2 | 0.86 |

Reasoning is 8.4% of traffic, 16.4% of $ and 94.0% of energy. It costs ~80x more energy per query because the model decodes a long hidden chain of thought (6x the output tokens here), decode is memory-bound so every token re-reads the weights and a growing KV cache, and the 80x deck factor is applied on top (an upper bound; it partly double-counts the longer output).

Routing rule: *use reasoning only when the complexity router already sends the task to the large tier*. That downgrades 162 requests and saves $0.11/day and 23,904 Wh/day. Caps: 10% of traffic -> $0.00/day, 0 Wh/day; 5% of traffic -> $0.07/day, 15,443 Wh/day; 2% of traffic -> $0.10/day, 23,437 Wh/day. A 10% cap changes nothing (traffic is already below it). In this dataset reasoning is an energy problem far more than a $ problem, because eval traffic is batch-discounted and mostly small-tier.

## Extension 5: carbon-aware scheduling

Interruptible jobs use 4,312 kWh/month (GPU-hours x TDP x 0.85 x PUE 1.2).

| Region | $/kWh | gCO2/kWh | RTT ms | Energy $/month | kgCO2/month | CO2 vs us-east-1 |
|---|---|---|---|---|---|---|
| us-east-1 | 0.12 | 380 | 10 | $517 | 1,638 | 0% |
| us-west-2 | 0.07 | 120 | 70 | $302 | 517 | 68% |
| europe-north1 | 0.09 | 30 | 110 | $388 | 129 | 92% |
| europe-central2 | 0.18 | 660 | 105 | $776 | 2,846 | -74% |
| us-east-wa | 0.055 | 90 | 65 | $237 | 388 | 76% |

Moving all interruptible jobs from us-east-1 to europe-north1 saves 1,509 kgCO2e/month (92%) and $129/month of electricity. Cheapest: us-east-wa; cleanest: europe-north1; balanced: us-east-wa (76% less CO2 and $280 cheaper than us-east-1). The 'optimal' region depends on the company's priority: a net-zero target picks europe-north1, a cost target picks us-east-wa. Both are 65-110 ms from US users, which is fine for training/eval/sandbox but not for chat inference, so only interruptible work moves.

## Sustainability commentary

- The median request (3,017 tokens) uses ~0.91 Wh; reasoning requests average 148 Wh. Energy is dominated by a few reasoning calls, not by volume.
- Carbon and cost move together: the clean hydro/nordic regions are also cheaper per kWh than us-east-1, so carbon-aware placement of batch work is a cost saving, not a tax. The only dirty *and* expensive region is europe-central2 - never schedule there.
- Every $ lever above also cuts energy: cascade serves tokens on a smaller model, killing idle GPUs removes ~15% TDP idle draw, and right-sizing moves work to GPUs with lower watts.

## Assumptions & limitations

- Monthly = 30 x the one-day token log / telemetry sample; the telemetry fleet is assumed to be the same GPUs that run `workloads.csv`, so right-size and idle savings come out of that spend.
- Always-on services are assumed to live 3 years when sizing commitments (see Ext 1 sensitivity).
- Interruption rates, cache storage prices, latency and grid figures are illustrative 2026 values.

_Figures are June-2026 as-of snapshots; re-baseline before acting._