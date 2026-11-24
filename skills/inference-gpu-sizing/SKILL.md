---
name: inference-gpu-sizing
description: Estimate LLM inference GPU counts for a fixed dense peak compute target, model memory fit, or a benchmarked service load. Use for accelerator selection, tensor parallelism, KV-cache planning, replica sizing, or explaining why theoretical PFLOP/s does not determine inference capacity. Not for training or procurement.
---

# Inference GPU sizing

Separate a theoretical compute target, a model-memory fit check, and capacity validated against an SLA. Ask which outcome matters only when it cannot be inferred. Treat missing data as an explicit assumption or an unresolved requirement, not a vendor specification.

## Workflow

1. **Peak compute:** Confirm precision, dense versus sparse, exact card SKU and form factor, sourced per-card throughput in TFLOP/s, and GPUs per node. The peak count is target PFLOP/s × 1,000 ÷ card TFLOP/s, rounded up. Round again for full nodes if required and report actual peak on that hardware. Do not add efficiency factors unless asked for a separate scenario. Do not infer inference throughput or latency from this result.
2. **Memory fit:** Confirm total resident parameters, weight format, max live context including output, simultaneous sequences per replica, and the model's KV layout. Weight bytes = parameters × weight bits ÷ 8. Standard MHA/GQA/MQA KV bytes = 2 × layers × KV heads × head dimension × live tokens × simultaneous sequences × KV bits ÷ 8. Do not substitute total attention heads for KV heads. For MLA, hybrid or unusual layouts, provide runtime-derived bytes per cached token; do not reuse the standard formula. Reserve memory for runtime, quantization metadata, fragmentation, and transient allocations. Do not use MoE active parameters as resident weights.
3. **TP and replicas:** Evaluate only TP widths the runtime/model actually supports. A memory fit under perfectly even sharding is a lower bound; account for replicated KV heads when TP exceeds KV-head count, and verify real allocations and GPU topology. Additional replicas improve aggregate capacity; they do not make the model fit into one replica. Treat a per-replica benchmark as reusable only when model, precision, TP, context and output distributions, concurrency, software, and hardware match the deployment. Use only achieved capacity that also satisfies stated TTFT, TPOT and error-rate limits. Validate scale-out under production-like load before committing.
4. **Compare candidates:** If no card is specified, compare up to three compatible candidates using official SKU specs or clearly labeled hypothetical inputs. Compare memory, supported weight/KV formats, fabric and TP placement, runtime maturity, benchmarked throughput and latency, and any geographic or procurement requirements the user supplied. Do not invent a catalog ranking or guarantee compatibility. Decode can be memory-bandwidth-bound, but confirm with measured behavior; long-context prefill, cache movement, and TP communication may change the bottleneck.

## Calculator

Run `python3 scripts/size.py < request.json` from this folder (Python 3.9+, standard library). Provide a JSON object with one or more of:

- `peak`: `target_pflops`, `per_gpu_tflops`, `gpus_per_node`; both rates must describe the same precision/sparsity.
- `memory`: `parameters`, `weight_bits`, `live_tokens`, `sequences`, `gpu_vram_gib`, `reserve_fraction` (fraction to reserve on **each** GPU), and `tp_candidates` (user-verified widths). Supply `layers`, `kv_heads`, `head_dim`, `kv_bits` for standard attention, or supply `kv_bytes_per_token` instead. Optional `kv_sharding` is `replicated` by default (conservative) or `ideal` (even-sharding lower bound).
- `service`: `peak_requests_per_second`, `qualified_requests_per_second_per_replica`, `cards_per_replica`, optional `capacity_fraction` (default 1). Provide only benchmark results that meet the SLA; the calculator does not test the SLA. Run separate estimates or replay a mixed benchmark when traffic classes differ.

Use the calculator once per candidate. Combine results only after validating that the same deployment configuration meets each independent constraint. Return assumptions and provenance, calculation units, card/node and TP/replica counts when supported, the likely bottleneck, and the next measurement that could change the recommendation.

## Reproduce the theoretical case

```sh
printf '%s\n' '{"peak":{"target_pflops":500,"per_gpu_tflops":989.5,"gpus_per_node":8}}' | python3 scripts/size.py
python3 -m unittest discover -s scripts -v
```

At the illustrative dense FP16 input of 989.5 TFLOP/s per card: 506 minimum cards, or 64 full eight-card nodes (512 cards), for 506.624 PFLOP/s theoretical peak. This is neither a measured workload result nor a complete deployment or procurement plan.
