# RoboDuet state-capture throughput result

The training environment now requests a full runtime state only when it has a
consumer. Ordinary training still reads joints before each 5 ms physics step,
advances physics and scene timestamps four times, and refreshes the complete
state at the 20 ms policy boundary. Evaluation's substep callback still receives
every 5 ms state. `step_control()` keeps `capture_state=True` by default for
existing direct callers. Observations, rewards, controllers and PPO are unchanged.

## Complete-update comparison, 2026-09-13

The device operator ran two separate processes from the same full checkpoint
at iteration 3200, using 4096 environments, 24 transitions per update, the
original Stage 1 networks, five PPO epochs and four minibatches. Each process
performed five warmup updates and ten measured updates. The original long
training process continued running during both probes; the probes ran
sequentially. Both exited 0 and reached iteration 3215.

| Measured full update | Capture every substep | Capture at policy boundary |
| --- | ---: | ---: |
| Mean time | 6.30638 s | 4.12311 s |
| Median time | 6.25435 s | 4.16516 s |
| Transitions/s | 15,588.02 | 23,842.21 |

This pair measured **1.52952 times the throughput**, or 34.6200% less mean
update time. Synchronization brackets the complete `run_iteration`; checkpoint
writes are outside timing. This is one pair under concurrent training load,
not a guarantee for other hardware, workloads or Stage 2.

All 15 updates' non-timing metrics, final network and optimizer states,
learning rates, RNG states, task/curriculum buffers and arm observation cache
were exactly equal. Dog PPO and adaptation optimizer counters advanced from
64000 to 64300; Stage 1 did not update the arm optimizers.

One cached contact-force element differed: `[3369, 15, 1]` was
-1257.4554443359375 N versus -1257.45556640625 N. The absolute difference,
0.0001220703125 N, is one float32 ULP at this magnitude. It did not change the
recorded metrics or final model/optimizer state in this pair. All other compared
numeric fields matched and were finite. Independent processes do not restore
PhysX's internal contact caches, and this pair does not distinguish numerical
variation from readout-schedule effects. The preceding 32-environment
fixed-action comparison matched all recorded physical/contact/reward/termination
fields exactly. Neither comparison proves all future trajectories equal or
successful learning.

## Reproduce the bounded comparison

After preparing the [official Lab environment and RoboDuet asset](roboduet-training.md),
run from the checkout using a complete checkpoint and two new output directories.
Use the same checkpoint, environment, physics asset and device for both commands:

```bash
python scripts/probe_roboduet_training_throughput.py \
  --visualizer none --device cuda:0 --checkpoint /path/to/model_3200.pt \
  --mode legacy --warmup 5 --measured 10 --output runs/throughput_legacy
python scripts/probe_roboduet_training_throughput.py \
  --visualizer none --device cuda:0 --checkpoint /path/to/model_3200.pt \
  --mode boundary --warmup 5 --measured 10 --output runs/throughput_boundary
```

Each probe saves its input identities, per-update records and final full
checkpoint. `scripts/probe_roboduet_state_capture.py` is the smaller physical
state comparison. The CPU tests in `test_roboduet_state_capture.py` cover the
unchanged physics-call sequence and evaluation callback behavior; they are not
a substitute for the measured complete-update pair.

The original local evidence is under
`outputs/roboduet-ppo-throughput-20260912/`: `report.md`, `comparison.json`,
`compare.py`, `source.json`, and the `legacy/` and `boundary/` records.
The preceding physical probe is under `outputs/roboduet-state-capture-20260912/`.
These local run artifacts are not included in the source checkout or wheel.
Source installation and export consumption also do not establish that a running
training service has loaded a new implementation; each continuation records its
own source and checkpoint identity.
