# Video walkthrough — 5 to 8 minutes (practice notes; deliver unscripted)

**Open before recording:** GitHub repo README, `src/experiment.py`, `results/hardware.json`, `results/measurements.csv`, `figures/quantization_results.png`, and your successful Google Colab notebook. Screen-share with camera if required.

**0:00–1:00 — GitHub repo homepage**. Introduce the narrow question: accuracy and checkpoint size of dynamic INT8; performance depends on hardware. Industry target: ML evaluation at Thomson Reuters, a regulated/traceability-sensitive environment.

**1:00–2:00 — README dataset/protocol; src/experiment.py**. Real scikit-learn digits (1,797), 1,347/450 stratified split, 3 seeds, 30 epochs, 30 warmups + 120 measurements for both variants. Explain fixed budget and reproducibility.

**2:00–3:00 — results/hardware.json and Colab**. Show `device_requested: cuda`, `cuda_available: true`, `gpu_name: Tesla T4`, PyTorch `2.10.0+cu128`. Point out that FP32 actually uses GPU, but dynamic INT8 uses Colab CPU because of operator support.

**3:00–4:30 — figures/quantization_results.png and results/measurements.csv**. Read mean FP32 GPU accuracy 96.37%, INT8 CPU 96.30%, checkpoint size 71,637 vs 22,591 bytes (68.5% smaller). GPU FP32 p50 mean 0.242 ms; CPU INT8 p50 mean 0.262 ms. IMPORTANT: Do NOT call that a speedup or slowdown attributable to quantization; different devices. Show seeds and uncertainty; p50 mean ± sample SD 0.242 ± 0.046 ms on GPU, 0.262 ± 0.060 ms on CPU.

**4:30–5:30 — results/cpu_baseline/**. Show original CPU-only data in a separate directory and explain that the two hardware experiments are not combined. Show plot regeneration command `python src/experiment.py --mode plot` (can be described if unable to run live).

**5:30–6:30 — README limitations, intended use, licensing**. Digit recognition is not legal text; three seeds, tiny MLP, no end-to-end latency or peak VRAM; raw timing subject to cloud scheduling jitter.

**6:30–7:00 — README career & skills section**. Say how measured evidence, pinned environment, provenance and auditability translate to responsible ML engineering. More GPUs: evaluate larger models, extra seeds, and a properly supported same-GPU FP32/FP16 or GPU-INT8 comparison.

**Recording note:** Use these as prompts; course requires an unscripted walkthrough. Do not say GPU INT8 ran — it did not.
