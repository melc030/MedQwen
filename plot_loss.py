"""
Parse training.log and generate a loss curve plot.
Saves loss_curve.png in the project root.
"""
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG_FILE = "training.log"
OUT_FILE = "loss_curve.png"

train_steps, train_losses = [], []
eval_steps,  eval_losses  = [], []

epoch_boundaries = []  # step numbers where epochs start
current_epoch = 0

with open(LOG_FILE, "r") as f:
    for line in f:
        # training step: "step 10/11358 (0.1%) | loss 2.8357 | ..."
        m = re.search(r"step (\d+)/(\d+).*\| loss ([\d.]+)", line)
        if m:
            step      = int(m.group(1))
            total     = int(m.group(2))
            loss      = float(m.group(3))
            train_steps.append(step)
            train_losses.append(loss)

        # eval loss: "eval loss: 2.1178"
        m = re.search(r"eval loss: ([\d.]+)", line)
        if m and train_steps:
            eval_steps.append(train_steps[-1])
            eval_losses.append(float(m.group(1)))

        # epoch boundary
        m = re.search(r"=== Epoch (\d+)/", line)
        if m:
            epoch_num = int(m.group(1))
            if epoch_num > 1 and train_steps:
                epoch_boundaries.append(train_steps[-1])

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

# smoothed train loss (rolling average)
window = 20
smoothed = []
for i in range(len(train_losses)):
    start = max(0, i - window + 1)
    smoothed.append(sum(train_losses[start:i+1]) / (i - start + 1))

ax.plot(train_steps, train_losses, color='#b0c4de', linewidth=0.6,
        alpha=0.5, label='Train loss (raw)')
ax.plot(train_steps, smoothed, color='#2166ac', linewidth=1.8,
        label=f'Train loss (smoothed, window={window})')
ax.plot(eval_steps, eval_losses, 'o-', color='#d73027', linewidth=2,
        markersize=5, label='Eval loss')

# epoch lines
for i, step in enumerate(epoch_boundaries):
    ax.axvline(x=step, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(step + 50, ax.get_ylim()[1] * 0.97,
            f'Epoch {i+2}', fontsize=8, color='gray')

ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Loss', fontsize=11)
ax.set_title('MedQwen (Qwen2.5-1.5B) — Training & Eval Loss', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(True, alpha=0.3)
fig.tight_layout()

plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
print(f"saved to {OUT_FILE}")
