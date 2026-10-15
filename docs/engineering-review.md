# ReTA engineering review and implementation decision

Review date: 2026-10-15. Scope: the supplied Desktop source snapshot, the linked public repository, and the supplied paper. Recommendations are inferred from public framework documentation and inspected code, not knowledge of OpenAI's or Anthropic's private infrastructure.

## Decision

Retain native PyTorch + PyTorch Geometric (PyG), exact dense template retrieval, and the paper's REINFORCE objective. Fix correctness, recovery, batching and experiment provenance before adding distributed training. Use one GPU per independent seed initially; this makes better use of the available hardware without coupling trajectory state across workers. This is a mainstream software foundation for a specialized research algorithm. ReTA's clinical augmentation policy itself is a research method, not an established industry standard.

### Three implementation options

| Option | What it owns | Advantages for ReTA | Costs and limits | Decision |
|---|---|---|---|---|
| Native PyTorch + PyG | Modules, autograd, Adam, graph batches, checkpoint state dictionaries; project owns its two-stage loop | Smallest change, transparent reward/state semantics, existing checkpoints and tests remain useful | Project must explicitly manage recovery, logging, device placement and any future DDP synchronization | Implement now |
| Lightning Fabric + PyG | Device/distributed wrapping, precision, backward, checkpoint machinery; project retains loop | Good fit for custom RL loops; official RL example; reduces distributed infrastructure ownership | Requires deliberate wrapping of multiple models/optimizers and handling conditional unused parameters, recurrent state and global statistics; migration needs parity tests | Preferred future abstraction if single-run multi-GPU training is justified |
| Hugging Face Accelerate + PyG | Model/optimizer preparation, backward, mixed precision and state checkpointing | Familiar lightweight integration; save/load support includes RNG state and custom registered objects | PyG variable graphs, trajectory sharding, utility EMA and policy baseline still need application-specific handling; introduces another runtime dependency | Reasonable if the surrounding team already uses Accelerate |

Sources: [PyG batching](https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html), [Fabric methods](https://lightning.ai/docs/fabric/latest/api/fabric_methods), [Fabric RL example](https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/fabric/reinforcement_learning/README.md), [Accelerate checkpointing](https://huggingface.co/docs/accelerate/usage_guides/checkpoint).

PPO, a contextual bandit or a supervised selector would be algorithmic alternatives, not infrastructure refactors. Replacing REINFORCE with them changes the paper experiment and requires matched-budget retraining and evaluation. There is no reason to introduce PPO merely because a framework offers it.

## Current method, precisely

1. Normalize diagnosis codes, map to CCS and form patient-disjoint visit trajectories.
2. Build an offline grounded knowledge pool using distillation responses, ontology evidence, ClinicalBERT embeddings and agglomerative clustering.
3. Retrieve exact Top-K templates using maximum code-to-template cosine similarity blended with recurrent history similarity.
4. Stage 1 trains the semantic/structural encoder and next-visit predictor with random augmentation exposure.
5. Stage 2 uses a GRU state and categorical policy over 2K+1 actions: Soft, Hard, or Skip. REINFORCE uses discounted returns, a running baseline and optional entropy regularization. Supervised BCE continues to refine the live encoder.
6. A frozen Stage-1 encoder supplies uncertainty. Raw-versus-edited reward must use the same live encoder, memories and disabled dropout.
7. Inference performs sequential diagnosis prediction. Published numeric JSON files are reference records, not evidence that this checkout reproduced those results.

The existing implementation already uses PyG GATConv, PyG Batch, torch.nn.MultiheadAttention, GRU and torch.optim.Adam. These are appropriate reuse, not reinvention. The domain-specific graph grafting, token contracts, grounding rules and reward are legitimate project logic.

## Findings and changes

### Correctness: reward attribution

Before this review, rl_train.py compared frozen Stage-1 raw BCE with live-encoder augmented BCE. That difference mixes augmentation effects with encoder improvement. Paper §2.3.2 specifies both inputs under the current encoder; §2.5 freezes Stage 1 for uncertainty. The update uses the already-computed current raw loss for reward and keeps the frozen network for uncertainty. Historical checkpoints can still be evaluated, but historical training results are not retroactively corrected.

### Recovery: weights are not training state

Existing checkpoints omitted optimizer momentum, Python/NumPy/Torch/CUDA RNG state, training position and the running policy-return baseline. They were inference exports, not resumable training checkpoints. The update uses native state_dict APIs and epoch/iteration-boundary saves with atomic replacement. Stage 2 also restores the frozen models, policy state encoder and template utilities. Configuration drift is rejected except extending iteration/epoch budgets and changing output directories. Legacy checkpoints remain usable for inference but cannot be resumed exactly.

Recovery is at completed boundaries, not mid-trajectory. Matching CUDA RNG state is necessary but does not guarantee bitwise results across devices or nondeterministic scatter reductions. Resume validation requires matching visible CUDA device counts. No claim of cross-version numerical identity is made.

[PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) documents saving optimizer state and training progress alongside weights.

### Utilization: gradient accumulation is not batching

Stage 1's batch_size previously meant a number of serial transitions per optimizer update. Stage 2 also forwarded one graph at a time. The update batches Stage-2 supervised replay graphs using PyG, retaining each graph's temporal memory. It replaces handwritten readout loops with global_add_pool and removes a dense-to-node Python loop. It explicitly masks Soft Import after its biased projection so zero placeholders do not alter Hard or Skip graphs.

Stage-1 and policy rollouts remain sequential to preserve causal history. Real batching across independent patients is the next performance step if profiling supports it. Do not flatten visits into a shuffled loader: that destroys the temporal contract. Mixed-action batched and separate outputs/gradients are tested with dropout disabled; training dropout draws need not match the previous serial schedule.

### Metrics: rely on the library

A broad exception handler around sklearn average_precision_score silently selected a handwritten rank implementation. Tied predictions could then change scores with input order, and dependency errors could disappear. Evaluation now requires scikit-learn's implementation; a tied-score regression covers the behavior. Average precision is the reported metric implementation; it should not be confused with trapezoidal integration of a PR curve.

### CUDA compatibility

The original torch==2.1.2 environment is a legacy reproduction target, not a suitable Blackwell target. Official Blackwell/CUDA 12.8 support first appears in [PyTorch 2.7 release documentation](https://pytorch.org/blog/pytorch-2-7/). A newer compatible build must be tested with the actual driver. An installed CUDA toolkit or a positive nvidia-smi result alone does not prove that GAT forward/backward works.

The validation environment and actual results are recorded separately. No dependency-version bump is represented as proof that full ClinicalBERT pool generation or all clinical experiments work.

## What an infrastructure review would push back on

- **Do not build a graph framework.** Use PyG graph collation, GAT kernels and pooling. Keep clinical topology decisions in small domain functions.
- **Do not build a second metric library.** Use scikit-learn and preserve exact definitions/scales in outputs.
- **Do not build a distributed scheduler.** On front1 use Slurm allocations and job arrays; do not run GPU workloads on its login node. On ampere inspect capacity first and isolate output paths.
- **Do not build a vector service for roughly a thousand templates.** Exact matrix scoring is simple, inspectable and preserves the composite retrieval equation. FAISS becomes an option after measurements show retrieval dominates at a larger pool size. Plain IndexFlatIP over one averaged query does not reproduce max-over-code plus history scoring. If using approximate candidates, rerank with the original score and measure candidate recall. See [FAISS metric semantics](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances).
- **Do not add FSDP/DeepSpeed without a memory problem.** This is a modest graph/GRU model, not an LLM training workload. DDP is useful when genuine per-GPU batch work can amortize communication. Conditional Soft Import and absent-history branches can leave parameters unused; naive DDP wrapping can hang or diverge unless that and global statistics are handled. See [DDP documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).
- **Do not enable every acceleration flag together.** Establish FP32 correctness first, profile, then evaluate AMP for supervised updates separately from reward subtraction. Small BCE differences may be precision sensitive. Measure wall-clock throughput and quality, not just GPU allocation. See [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

## Hardware plan

Observed ampere: three RTX A6000 48 GB GPUs, each mostly occupied during inspection. Observed front1: Slurm login node; PRO 6000 resources on gpu-pro6000 and zyao partitions. A short allocation on zyao exposed one RTX PRO 6000 Blackwell Server Edition with approximately 96 GB total memory. These are observations at review time, not reservations or guarantees of later availability.

Use one GPU for correctness and profiling. Then use a Slurm seed array with unique config/output directories, allowing at most two concurrent tasks on the two PRO 6000s when scheduling policy permits. The same single-device jobs can use free A6000s with a compatible runtime. Do not pool the two different hosts into a single distributed job without evidence that networking and scaling justify it.

[Slurm arrays](https://slurm.schedmd.com/job_array.html) and [GPU resource management](https://slurm.schedmd.com/gres.html) supply the orchestration; the project should not reimplement them.

## Unresolved evidence, not fixable by synthetic CUDA tests

The supplied snapshot contains a one-template example and pool specifications, not the paper's full frozen pool. It also lacks the credentialed processed datasets, experimental checkpoints and original run traces. The paper's frozen-pool release claim therefore exceeds the inspected release contents. Do not fabricate these artifacts or treat stored paper numbers as freshly measured results.

Full MIMIC reproduction, ClinicalBERT pool construction on complete inputs, multi-seed quality/latency measurements, validation-based checkpoint selection, mortality/readmission task reproduction and broad ablation coverage remain unverified. The new synthetic tests establish execution and recovery properties, not clinical performance. Additional GPUs cannot fill missing data and provenance.

Before declaring paper reproduction complete: obtain the authorized input paths and actual frozen pool; verify patient splits and pool hashes; run the two stages with independent seeds; choose checkpoints on validation data only; evaluate the held-out test split once per selected model; record environment, inputs, seeds, training state and metric definitions. Use the supplied experiment artifacts only when their provenance is established.
