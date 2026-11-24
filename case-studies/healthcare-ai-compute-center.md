# A 500 PFLOP/s compute center for healthcare AI

This is a reference design built from a sample brief and public hardware specs. It isn’t a hospital deployment, a procurement quote, or a benchmark. All workloads and acceptance targets below are illustrative; there are no patient records in this case.

## The brief

Several hospital teams need a shared cluster for clinical-document model evaluation, imaging research, and fine-tuning. A smaller pool has to keep interactive LLM requests responsive while researchers submit batch jobs. I’d start by asking for a peak-hour request trace with token lengths and timestamps but no content, the largest model and context they intend to serve, and the rack power available after a power feed goes down.

I treat **500 PFLOP/s as installed dense FP16 Tensor Core peak**. I use a separate serving test: under the agreed peak traffic mix, target end-to-end p95 time to first token (TTFT) ≤ 2 s and p95 time per output token (TPOT) ≤ 100 ms. The test also counts timeouts and rejected requests, so dropping difficult requests can’t make the latency look better.

## GPUs and rack power

| Design decision | First-pass sizing |
| --- | --- |
| Reference compute node | 8 × H100 SXM 80 GB with NVSwitch, using DGX H100 as the reference chassis. Keeping tensor parallelism inside a node avoids cross-node communication on each decode step. |
| Installed capacity | NVIDIA publishes 1,979 TFLOP/s FP16 with structured sparsity. The equivalent dense peak is about 989.5 TFLOP/s per GPU. The arithmetic minimum is 64 nodes: `ceil(500 / (8 × 0.9895))`. That gives **512 GPUs and 506.6 PFLOP/s** installed. |
| Maintenance allowance | With one node unavailable the remaining 63 nodes are rated at **498.7 PFLOP/s**. If 500 must remain installed and available during single-node maintenance, fit 65 nodes. A two-node rack outage would require a different reserve, e.g. 66 nodes to leave 64. |
| Facility check | Two 10.2 kW reference nodes draw up to 20.4 kW per compute rack. At an assumed 24 kW usable after a feed loss, 64 nodes occupy **32 compute racks** and have a **652.8 kW** combined node maximum. That excludes network, storage, and cooling. I’d check rack depth, weight, PDU circuits, actual chassis redundancy, and cooling with the facility team before locking the layout. |

[H100 datasheet](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306?download=1) · [DGX H100/H200 chassis specifications](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html)

**Where I’d trial H200.** H200 SXM has the same advertised FP16 peak as H100 SXM, but 141 GB of HBM at 4.8 TB/s instead of 80 GB at 3.35 TB/s. That matters for long-context serving. Take a 70B dense FP16 model with 80 layers, 8 KV heads of width 128, and 16 active sequences at 8,192 total tokens each. Its weights are about 130 GiB and the FP16 KV cache about 40 GiB per replica. Even with ideal sharding across two cards, that’s about 85 GiB per card before runtime buffers, so two 80 GB H100s won’t fit. Two H200s are worth testing against four H100s. I’d compare SLO-qualified throughput per GPU, p95 latency and wall power at the same model, traffic mix and output quality. I would try quantization only with a separate quality check on an approved evaluation set. There’s no serving GPU count in the 500 PFLOP/s figure.

[H200 specs](https://www.nvidia.com/en-eu/data-center/h200/) · [HGX memory and interconnect comparison](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)

## Compute, storage, and management networks

I’d keep GPU collectives on 400 Gb/s NDR InfiniBand, storage and service traffic on redundant Ethernet, and BMC management on an isolated out-of-band network. An inference replica that fits inside one server doesn’t need the full compute fabric to handle ordinary HTTP traffic.

For the mixed training and serving layout, the eight 400 Gb/s host links per node give **512 logical host links**, grouped into eight rails. In each rail, connect 32 hosts to each of two 64-port leaves. Each leaf uses its other 32 ports for uplinks, 16 to each of two spines. This gives **1:1 downlink-to-uplink bandwidth per leaf** in normal operation, 16 leaves and 16 spines across the cluster, and 512 logical leaf-to-spine links. A spine outage halves the affected leaf uplink bandwidth in that rail. A leaf outage cuts that rail’s host links for the attached nodes; I would test whether jobs continue, need a restart, or should be kept inside a healthy pod. These are logical counts: the [QM9700 switch](https://networking-docs.nvidia.com/qm97x0hw/introduction) has 64 logical NDR ports in 32 OSFP cages. Transceivers, paired links, fibre paths and reach need a separate cable schedule.

If the local team already operates RoCE, I’d compare it with the same cross-rack collective workload and failure test before deciding. That comparison includes ECN/DCQCN, the site’s priority and PFC policy, and congestion behaviour during an incast, not just switch port speed.

## Storage that can keep up with the jobs

I’d use local NVMe for model caches and scratch, a shared all-flash parallel file tier for hot datasets and checkpoints, and a cheaper object tier for retained datasets, model versions and recovery copies. Local scratch doesn’t count as durable capacity. Jobs read approved copies in the AI zone; the cluster is not the system of record for hospital imaging or clinical records.

| Tier | Sample sizing input | What I’d check |
| --- | --- | --- |
| Shared flash | 160 TB of hot datasets, 40 TB of staging and eight jobs with 3 retained 2 TB checkpoints each: 248 TB occupied. A 30% free-space policy calls for **~355 TB provisioned usable**, before converting to raw drive capacity. | Eight jobs each writing 2 TB in 300 s needs **53.3 GB/s aggregate payload writes**, plus bandwidth for concurrent readers. Run the test during a storage-node failure or rebuild; record checkpoint tail times and resume a job from the result. |
| Object / capacity | Example: 500 TB retained datasets + 100 TB model/archive copies = 600 TB of payload. An 8+3 erasure code, if supported and laid out across failure domains, adds 1.375× data overhead. With 30% free space that is **~1.18 PB raw** for the primary copy before metadata and other vendor-specific overheads. | Independent backup copies are additional. Test both large-image reads and small-file/metadata behaviour, recovery speed after a node loss, and how long it takes to rebuild the indexes the workload needs. |

For a dual-fabric storage network, 2 × 200 GbE per storage node is one starting point, not a performance claim. If one path fails, surviving paths and storage controllers still have to carry the agreed read/write mix. I’d check ports, client throughput and protocol overhead together.

## Scheduling and operations

I’d start with Kubernetes, NVIDIA GPU Operator and DCGM monitoring, a versioned model registry, and vLLM for the text-serving pool. Production inference gets a reserved GPU pool with at least two warm replicas on different nodes and racks. Pin tensor-parallel workers to a suitable NVSwitch domain, drain before driver upgrades, and roll out new images and model revisions to a small slice of traffic first. Set context limits, tenant quotas and admission limits at the gateway. Keep serving capacity out of the research queue: an idle replica may be needed for the next burst or a node failure.

For batch work, use Kueue resource flavors and per-team quotas so a small evaluation can’t reserve the entire cluster. Permit spare capacity borrowing within the batch pool and use topology-aware placement for distributed jobs. In the operator view I’d put TTFT, TPOT, queue time, cache pressure, GPU memory and hardware errors next to each other. GPU utilization alone won’t tell an on-call engineer whether to add replicas, change batching, or investigate a slow storage read.

[GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/) · [Kueue quota and admission](https://kueue.sigs.k8s.io/docs/concepts/cluster_queue/) · [vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics/)

## Security and disaster recovery

- Separate clinical source systems, controlled import, research, serving, storage, and management into distinct zones. Apply per-project service identities, least-privilege dataset access, encryption in transit and at rest, managed keys, and default-deny egress. Don’t send raw prompts, responses or identifiers to shared telemetry. Keep access and model-deployment audit records separately with restricted access.
- Version the manifests, images, model weights and indexes required to stand up a service. Back up control-plane state and critical metadata, and keep encrypted immutable recovery copies in a separate failure domain under separate credentials. Test recovery with the primary site and its identity/control plane assumed unavailable. A snapshot on the primary array is not a site backup.
- For the sample brief, propose **RPO ≤ 1 h / RTO ≤ 4 h** for configuration and registry metadata. Set separate data-class targets with application owners. A four-hour full restore of even 100 TB needs about **6.9 GB/s** of payload throughput before other recovery work, so a four-hour service target would need a pre-staged footprint and standby GPUs, not a last-minute copy of the archive.

## What I’d ask to see before signing off

1. **Serving:** Replay the agreed arrival pattern with mixed prompt lengths; report end-to-end p95 TTFT and TPOT, completed requests/s, timeouts and rejects. Repeat with a drained node and with a cold model start.
2. **Fabric:** Compare NCCL throughput within a node, across racks and across the two pods. Verify transport and placement, then repeat after a link or spine failure while running a representative job.
3. **Storage and scheduling:** Run the checkpoint and restore test during a batch burst. Confirm live inference remains within its agreed latency budget while the research queue grows.
4. **Recovery:** Restore the registry, credentials and serving manifests to the recovery environment, verify model hashes, and send a smoke-test request. Record observed data loss and elapsed time; only sign off on RPO/RTO after the drill.

The [GPU sizing skill](../skills/inference-gpu-sizing/) is for the separate model-serving decision: which cards to test, how many per replica, and what evidence is still missing.
