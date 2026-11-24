# Which GPU should I try first?

I wrote this Skill for people who get asked to size an LLM service before they have a benchmark. It starts with the model, the longest requests you expect, and the latency you can tolerate. It then separates three answers: how much nominal compute you install, what fits in one replica, and how many replicas you can justify with a load test.

[Open SKILL.md](./SKILL.md) for the workflow and calculator input fields. The Skill uses public specs and sample inputs; the Python calculator runs offline and doesn’t need a GPU, API key, or access to hospital data.

## Try the 70B example

From this directory, with Python 3.9 or later:

```sh
python3 scripts/size.py < examples/h100-memory.json
python3 scripts/size.py < examples/h200-memory.json
python3 -m unittest discover -s scripts -v
```

Both inputs use a 70B FP16 model with 16 active 8,192-token sequences. The calculator estimates 130.4 GiB of weights and 40 GiB of KV cache per replica. With the sample 15% reserve and ideal sharding, the first widths worth testing are **TP4 on H100 SXM** and **TP2 on H200 SXM**. The nominal 80 GB and 141 GB capacities in the examples are converted to GiB and rounded; check the memory actually available to your runtime. The examples model ideal KV sharding explicitly. Check whether your model and serving engine really allocate it that way.

A memory fit is enough to start a test. For a card count, replay the intended request mix on both candidates and measure requests/s that meets your TTFT, TPOT and error-rate targets. Use that result in the calculator’s `service` input. There are no measured service results in this repository.

Related: [500 PFLOP/s healthcare AI compute center](../../case-studies/healthcare-ai-compute-center.md) · [LLM latency troubleshooting lab](https://github.com/C1806xixi-lin/llm-latency-troubleshooting)
