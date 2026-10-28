"""Reproducible real-data CPU INT8 baseline; optional CUDA FP32 training.
Usage: python src/experiment.py --mode plot | benchmark --device cpu | cuda
"""
import argparse, csv, json, os, platform, random, statistics, time
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
SEEDS = [11, 23, 37]
EPOCHS = 30
BATCH_SIZE = 64
WARMUPS = 30
TRIALS = 120

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    def forward(self, x):
        return self.net(x)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_data():
    ds = load_digits()
    x = (ds.data.astype('float32') / 16.0)
    y = ds.target.astype('int64')
    ix_tr, ix_te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=2026, stratify=y)
    return torch.from_numpy(x[ix_tr]), torch.from_numpy(y[ix_tr]), torch.from_numpy(x[ix_te]), torch.from_numpy(y[ix_te]), ix_tr, ix_te

def train(seed, device, xtr, ytr):
    seed_all(seed)
    model = Net().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.004)
    # deterministic epoch permutations generated on the CPU
    g = torch.Generator().manual_seed(seed)
    xtr, ytr = xtr.to(device), ytr.to(device)
    model.train()
    for epoch in range(EPOCHS):
        order = torch.randperm(len(ytr), generator=g)
        for ids in order.split(BATCH_SIZE):
            optim.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xtr[ids]), ytr[ids])
            loss.backward(); optim.step()
    return model.cpu().eval()

def sync(device):
    if device == 'cuda': torch.cuda.synchronize()

def latency(model, sample, device, warmups=WARMUPS, trials=TRIALS):
    model = model.to(device).eval()
    sample = sample.to(device)
    durations=[]
    with torch.inference_mode():
        for _ in range(warmups): model(sample)
        sync(device)
        for _ in range(trials):
            sync(device)
            t0 = time.perf_counter_ns(); model(sample); sync(device)
            durations.append((time.perf_counter_ns()-t0)/1e6)
    return np.asarray(durations)

def serialized_bytes(model):
    import io
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return len(stream.getvalue())

def benchmark(device):
    if device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested, but no CUDA device is available; no GPU results can be claimed.')
    torch.set_num_threads(1)
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    xtr,ytr,xte,yte,itr,ite = get_data()
    meta = dict(timestamp_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        device_requested=device, torch_version=torch.__version__, sklearn_version=sklearn.__version__,
        numpy_version=np.__version__, python=platform.python_version(), platform=platform.platform(),
        cpu=(next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), platform.processor() or 'not reported') if Path('/proc/cpuinfo').exists() else (platform.processor() or 'not reported')), cpu_count=os.cpu_count(),
        cuda_available=torch.cuda.is_available(), gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        threads=torch.get_num_threads(), n_train=len(itr), n_test=len(ite), dataset='sklearn.load_digits',
        test_fraction=0.25, split_seed=2026, seeds=SEEDS, epochs=EPOCHS,
        warmups=WARMUPS, timed_trials=TRIALS, batch_size=1, label='OBSERVED; no estimated measurements')
    records=[]; raw=[]
    for seed in SEEDS:
        print(f'training seed {seed}', flush=True)
        fp=train(seed,device,xtr,ytr)
        # Dynamic quantization currently executed on CPU; CUDA not supported for this operator.
        q=torch.ao.quantization.quantize_dynamic(fp, {nn.Linear}, dtype=torch.qint8).eval()
        for name,model,measure_device in [('fp32_cpu' if device=='cpu' else 'fp32_cuda',fp,device),('dynamic_int8_cpu',q,'cpu')]:
            with torch.inference_mode():
                xx=xte.to(measure_device)
                pred=model.to(measure_device)(xx).argmax(1).cpu().numpy()
                correct=int(np.sum(pred==yte.numpy()))
            durations=latency(model,xte[:1],measure_device)
            for i,d in enumerate(durations): raw.append(dict(seed=seed,variant=name,trial=i,latency_ms=f'{d:.9f}'))
            records.append(dict(seed=seed,variant=name,n_test=len(yte),correct=correct,
                accuracy=correct/len(yte),p50_ms=float(np.percentile(durations,50)),
                p95_ms=float(np.percentile(durations,95)),serialized_bytes=serialized_bytes(model.cpu())))
    for file,items in [('measurements.csv',records),('latency_trials.csv',raw)]:
        with (RESULTS/file).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=items[0].keys()); writer.writeheader();writer.writerows(items)
    (RESULTS/'hardware.json').write_text(json.dumps(meta,indent=2)+'\n')
    plot()

def plot():
    FIGURES.mkdir(exist_ok=True)
    with (RESULTS/'measurements.csv').open() as f: rows=list(csv.DictReader(f))
    variants=list(dict.fromkeys(r['variant'] for r in rows))
    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    for ax, field, label in zip(axs,['accuracy','p50_ms'],['Test accuracy','Single-item p50 latency (ms)']):
        means=[];spread=[]
        for variant in variants:
            a=np.array([float(r[field]) for r in rows if r['variant']==variant]);means.append(a.mean());spread.append(a.std(ddof=1) if len(a)>1 else 0)
        bars=ax.bar(range(len(variants)),means,yerr=spread,capsize=5)
        ax.set_xticks(range(len(variants)),variants,rotation=15,ha='right');ax.set_ylabel(label)
        ax.set_title('Observed mean ± SD across 3 training seeds')
        for bar,n in zip(bars,means): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(means)*.035,f'{n:.3f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(means)*1.22)
    fig.suptitle('Digits MLP: quantization tradeoff (committed measurements)')
    fig.tight_layout()
    fig.savefig(FIGURES/'quantization_results.png',dpi=160,bbox_inches='tight')
    plt.close(fig)
    print('Rendered figures/quantization_results.png from results/measurements.csv')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['plot','benchmark'],default='plot');ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=ap.parse_args()
    if args.mode=='plot': plot()
    else: benchmark(args.device)
