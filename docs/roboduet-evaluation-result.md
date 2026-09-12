# RoboDuet early fixed-policy physical result

Checkpoints 0 and 1600 both completed the frozen physical evaluation runner
with exit 0. Model 0 completes 2/9 full 20-second cases; model 1600 completes
0/9. This early comparison does not establish learning improvement. The
50000-update training experiment and the required 10001/50000 evaluations
remain separate unfinished work.

The [fixed protocol](roboduet-evaluation.md) uses seeds 2027/2028/2029, three
leg/arm cases, nominal plane, the original LPY/RPY/T_traj semantics, and all five
`official_play` modules. Both checkpoints are evaluated with the arm released
in Stage 2 and both have zero arm PPO updates; this comparison does not isolate
a causal effect of Stage 1 training.

| Measurement | Checkpoint 0 | Checkpoint 1600 |
| --- | --- | --- |
| Full 20 s cases | 2/9 | 0/9 |
| Early termination | 7/9, at 0.14–0.18 s | 9/9, at 0.16–0.30 s |
| Early termination cause | Original body-height condition | Original body-height condition |
| Minimum 5 ms world-up dot | 0.9942 | 0.9136 |
| Inversion / reverse termination in recorded prefixes | None | None |
| Actual-input source/export graph maximum error | 0 | 0 |

Times exclude the separately saved 0.02 s zero-action initialization. Terminal
state is copied before automatic reset, and every 5 ms state/control/contact
sample through that terminal interval is retained. The sent effort command is
not a measured solver joint force; implicit spring/damper contributions are
not included in that command field. Head-specific contact is unavailable in
the collapsed body list and is not fabricated as zero.

On each pair's common first prefix, model 1600 has higher planar linear-velocity
and yaw-rate error in all nine cases. Initial saved root states and joint
positions match exactly for the paired seeds. EE position and orientation
changes are mixed. The two locomotion cases that complete 20 s at model 0
terminate at 0.24 and 0.30 s at model 1600. These results do not support learned
coordination.

All model 1600 cases and all standing/simultaneous model 0 cases terminate
before the first 2.5 s target change. Their prefixes exercise only the first
command; the dynamic sequence is specified but not successfully covered.
The retained near-upright height failures do not reproduce or explain any
inversion observed elsewhere in randomized training.

The local `outputs/roboduet-fixed-evaluation-20260912/model0-vs1600-report.md`
contains the per-case table, paired errors, contacts, saturation, raw-artifact
paths and evidence limits. In the same directory,
`model0-vs1600-seed2027-simultaneous-v2.mp4`
shows recorded body positions at 0.25x speed, including both early-ended
prefixes. It is a CPU skeleton replay, not another simulation or a 20-second
success video. Full decode and first/middle/final image inspection verified the
1280×720, 25 fps, 51-frame artifact.
These local raw artifacts and video are not bundled in the source checkout or
wheel; the portable protocol and executable evaluator are included.
