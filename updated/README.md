# FP32 on NVIDIA Tesla T4 and Dynamic INT8 on CPU: Reproducible ML Benchmark

**Question and budget.** Can a small classifier retain accuracy and reduce its checkpoint size after dynamic INT8 quantization, and what can be observed about deployment latency when GPU and CPU execution paths differ? On one NVIDIA Tesla T4 in Google Colab (and its host CPU), train a 64–128–64–10 neural network on the publicly available scikit-learn handwritten-digits dataset for 30 epochs across three seeds; measure 30 warmups and 120 timed single-item inferences per variant and seed. **Decision:** assess whether the small model's INT8 checkpoint savings justify further same-device deployment testing; this is not evidence that INT8 is faster on a GPU.

## Verified Colab run — 2026-10-28 (observed, not estimated)

- **GPU:** Tesla T4. **CPU:** Intel Xeon CPU @ 2.00GHz (Colab host, 2 logical CPUs). `device_requested=cuda`, `cuda_available=true`.
- **Software:** Python 3.13.16, PyTorch 2.10.0+cu128, scikit-learn 1.8.0, NumPy 2.3.5; full machine log in `results/hardware.json`.
- **Actual execution:** FP32 trained and timed on the **GPU**; dynamic INT8 evaluated and timed on the **CPU**. Do not compute a precision speedup using those two latency columns.
- **Dataset:** 1,797 real digits; fixed split 1,347 training / 450 evaluation; seeds 11, 23, 37. Accuracy is out of 450 for each seed.

| Variant / actual inference device | Accuracy mean ± sample SD | p50 latency mean ± sample SD | p95 latency mean ± sample SD | Serialized model |
|---|---:|---:|---:|---:|
| FP32 / **Tesla T4 GPU** | 96.37% ± 0.68 percentage points | 0.242 ± 0.046 ms | 0.352 ± 0.087 ms | 71,637 bytes |
| Dynamic INT8 / **Colab CPU** | 96.30% ± 0.79 percentage points | 0.262 ± 0.060 ms | 0.395 ± 0.061 ms | 22,591 bytes |

The INT8 checkpoint is **68.5% smaller**; observed average accuracy differs by about **0.07 percentage points**. Each timing is an observed model-forward time, not an end-to-end API latency. **Uncertainty is between-seed sample SD, not a confidence interval.** The code saves 720 individual timing trials; p50/p95 are computed within seeds and summarized across seeds. Checkpoint size is not peak RAM or GPU VRAM.

### Individual runs (actual uploaded Colab measurements)

| Seed | FP32 T4 accuracy | INT8 CPU accuracy | FP32 T4 p50 (ms) | INT8 CPU p50 (ms) |
|---|---:|---:|---:|---:|
| 11 | 96.22% | 96.22% | 0.2950 | 0.3277 |
| 23 | 97.11% | 97.11% | 0.2089 | 0.2083 |
| 37 | 95.78% | 95.56% | 0.2215 | 0.2486 |

### Preserved original CPU-only baseline

The **original independent CPU-only run** is preserved in `results/cpu_baseline/`, including its raw 720 trial timings, aggregate CSV, and `hardware.json`. It used an **Intel Xeon Platinum 8573C** and PyTorch **2.10.0+cpu**, with both FP32 and INT8 running on CPU. In that separate CPU-only run, FP32 and INT8 mean accuracies were 96.37% and 96.30%; mean p50 latencies were approximately 0.0199 ms and 0.0692 ms, respectively. Different machine and runtime: never combine the two sets of latency numbers into a device performance claim.

## Intended use and out-of-scope use

Intended: educational reproducibility demonstration and deployment engineering decision support for a **small fully connected model on a CPU**. Out of scope: clinical, financial, employment, or legal decision automation; handwritten-digit OCR product deployment; legal-text accuracy claims; GPU inference conclusions; formal noninferiority or safety claims.

## Dataset provenance and license

- Public **Optical Recognition of Handwritten Digits**, original creators E. Alpaydin and C. Kaynak (1998), UCI Machine Learning Repository; DOI: https://doi.org/10.24432/C50P49
- Data are loaded from **scikit-learn 1.8.0** `sklearn.datasets.load_digits()` (1,797 samples, 8×8 grayscale pixel-count features, ten digit classes, pixels 0–16); no downloads, manufactured records, or private/customer data.
- This packaged subset is the UCI dataset's original *test* portion. Our 1,347/450 train/evaluation split is a **new within-subset split**, not a claim to use the original UCI train/test partition; original test-subset provenance is a distribution caveat.
- Original UCI dataset license: **CC BY 4.0**, citation/attribution required. Dataset page: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits ; documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html . No dataset is repackaged in this repository; the dependency supplies the data.

## Model, versions, and reproducibility

- Architecture: PyTorch multilayer perceptron, `Linear(64,128)`, ReLU, `Linear(128,64)`, ReLU, `Linear(64,10)`.
- Training: Adam optimizer, learning rate 0.004, 30 epochs, minibatch 64, seeds 11/23/37; 75/25 stratified split fixed by `random_state=2026`; inputs divided by 16; no selection or tuning on held-out test scores.
- FP32 evaluated on GPU against post-training CPU `torch.ao.quantization.quantize_dynamic(..., {nn.Linear}, dtype=torch.qint8)`; dynamic INT8 uses weight-only quantization and does not imply all activations use integer representation.
- Python and package versions recorded in `requirements.txt` and `results/hardware.json`. Latest PyTorch quantization APIs may differ: `torch.ao.quantization` is deprecated in the benchmarked version. **Treat this as an archival reproducible baseline, not a forward-looking library recommendation.**

### From clean clone to reproduced figure

```bash
git clone YOUR_REPOSITORY_URL
cd single_gpu_portfolio
python -m venv .venv
source .venv/bin/activate             # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/experiment.py --mode plot
```

The `--mode plot` command regenerates `figures/quantization_results.png` **from committed `results/measurements.csv` without retraining or editing the chart**. It works without a GPU and is the quickest audit path. The included PNG must agree with the CSV.

To **rerun a CPU-only baseline** (results may vary by machine and will overwrite the main results unless backed up):

```bash
python src/experiment.py --mode benchmark --device cpu
```

To reproduce the **Colab GPU FP32 plus CPU INT8** execution:

```bash
# CUDA-enabled PyTorch installation is required; consult https://pytorch.org/get-started/locally/
python src/experiment.py --mode benchmark --device cuda
```

**Important:** CUDA mode measures FP32 *on GPU* and INT8 *on CPU*. They are **not comparable speedups**. It overwrites main CSVs, so use a fresh clone. Archived CPU results in `results/cpu_baseline/` remain separate. A true GPU precision ablation requires a supported GPU backend and equal-device baselines. The main committed results are the observed T4 GPU FP32 and CPU INT8 run; original CPU-only results are archived under `results/cpu_baseline/`.

### Repository structure

```text
README.md                      Methods, intended use, limitations, interpretation
requirements.txt               Pinned direct Python dependencies
src/experiment.py              Train, benchmark, log hardware, write raw CSV, regenerate figure
results/measurements.csv       Per-seed accuracy, p50/p95, serialized byte sizes
results/latency_trials.csv     720 observed per-inference latencies (3 seeds × 2 variants × 120)
results/hardware.json          Tesla T4 Colab runtime / split / seed / hardware metadata
results/cpu_baseline/           Original independent CPU-only benchmark: hardware, trials and summary
figures/quantization_results.png   Derived only from measurements.csv
docs/walkthrough_outline.md    5–8-minute unscripted screen-share guide
docs/portfolio_summary.pdf     One-page reviewer summary
```

## Exact evaluation protocol

1. Load scikit-learn bundled digits dataset; perform one stratified 75/25 split (`random_state=2026`) yielding 1,347 training and 450 evaluation examples.
2. Train separate FP32 models on Tesla T4 for seeds 11, 23, 37, fixed architecture and hyperparameters; host CPU is restricted to one PyTorch thread.
3. Convert each GPU-trained FP32 model to dynamic INT8 on CPU after training; FP32 is evaluated on GPU and INT8 on CPU using the same 450 evaluation inputs.
4. Record exact correct predictions / 450; save serialized `state_dict` size in bytes using `torch.save` to an in-memory stream (checkpoint size **not peak RAM or VRAM**).
5. For *each model*, perform 30 warmup forwards and then 120 sequential single-image forwards, measuring `time.perf_counter_ns()` around forward only, with device synchronization on CUDA. Each latency is retained in `results/latency_trials.csv`; p50 and p95 are computed within each seed.
6. Report mean and **sample SD over 3 seeds** for accuracy and latency; no confidence-interval, statistical significance, throughput, or production tail-latency claim.
7. Plot exclusively from `results/measurements.csv`. Hardware and runtime are written to `results/hardware.json`.

**Compute budget (not a measured energy bill):** 3 seeds × 30 epochs × ceil(1,347/64)=22 batches = 1,980 optimizer steps. Timing: 3 × 2 × (30 warmups + 120 timed requests)=900 forwards, of which 720 are timed. MLP has 17,226 learned parameters (including biases); a rough FP32 parameter-byte lower bound is 68,904 bytes, not total allocated memory. This describes a small, single-device experiment, **not proof of a GPU INT8 speed advantage**.

## Failure modes, auditability, and limitations

- Most important: **FP32 inference and model training used a Tesla T4 GPU; dynamic INT8 inference used a CPU.** This is a genuine one-GPU run but NOT a controlled same-device GPU quantization comparison. No GPU-INT8 latency claim is supported.
- The digits dataset is not legal text, and this toy MLP says nothing about Thomson Reuters model or product accuracy. This is benchmarking-skills evidence only.
- Three seeds estimate run variation coarsely. The same evaluation examples are reused across seeds, so those accuracies are correlated.
- CPU cloud-host scheduling jitter and small single-request kernels strongly affect microsecond-scale timing. Repeated timing samples are not independent datasets; p95 is noisy. Benchmarks are latency for Python+framework forward calls, not end-to-end service latency.
- Dataset was originally the UCI test subset, resplit for this learning exercise; no truly independent new-source validation, writer-group split, robustness slices, confidence calibration, or drift testing.
- Serialized checkpoint bytes ≠ runtime memory, peak RAM, or VRAM. Quantized operations are CPU-specific and may be slower when models are tiny.
- Input preprocessing not included in timed forward path; test data are resident in memory. Accuracy is on only 450 examples/seed.
- PyTorch deprecated this eager-mode quantization API; migration to torchao merits a separate versioned replication.
- GPU training can be nondeterministic across devices and library versions even with fixed seeds. The benchmark uses seeded initialization and shuffling but makes **no bitwise cross-hardware equivalence guarantee**.

## Why this matters here: Thomson Reuters / ML Engineer, AI Evaluation

Thomson Reuters announced its next-generation CoCounsel Legal on **August 20, 2026** and describes legal workflows emphasizing verified authority, traceability, and review. Source: https://www.thomsonreuters.com/en/press-releases/2026/august/thomson-reuters-launches-next-generation-of-cocounsel-legal-the-ai-ecosystem-built-for-legal-professionals . This artifact tests the habit of *checking performance claims instead of presuming them*. The analogy is evaluation rigor, not that handwriting accuracy transfers to law. A model evaluation engineer would require domain-specific legal benchmarks, groundedness/citation checks, sensitivity/privacy reviews, and independent reproducibility before making a product decision.

## Skills demonstrated, mapped to evidence

| Competency | Verifiable evidence |
|---|---|
| Single-machine training / deterministic controls | `src/experiment.py`: seeds, fixed hyperparameters, bounded epochs |
| Honest benchmarking / tail latency | `results/latency_trials.csv`: 720 individual timings + p50/p95 calculations |
| Accuracy and uncertainty | `results/measurements.csv`: three separate runs and 450 held-out labels each |
| Memory/size budgeting | Bytes per saved state dict, with explicit caveat that bytes are NOT RAM |
| Quantization trade-off analysis | GPU FP32 vs CPU dynamic INT8 accuracy and size; heterogeneous-device latency must not be interpreted as precision speedup |
| Reproduction and artifacts | Clean-clone commands, pinned direct dependencies, single plot-generating entry point |
| Governance/model documentation | Source/license, intended/out-of-scope uses, failure modes, hardware footnote |

## References

1. Alpaydin, E., & Kaynak, C. (1998). *Optical Recognition of Handwritten Digits* [Dataset]. UCI ML Repository. https://doi.org/10.24432/C50P49
2. scikit-learn. *load_digits* reference. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
3. PyTorch. *quantize_dynamic* API documentation. https://docs.pytorch.org/docs/main/generated/torch.ao.quantization.quantize_dynamic.html
4. Thomson Reuters. (August 20, 2026). *Thomson Reuters Launches Next Generation of CoCounsel Legal*. https://www.thomsonreuters.com/en/press-releases/2026/august/thomson-reuters-launches-next-generation-of-cocounsel-legal-the-ai-ecosystem-built-for-legal-professionals
