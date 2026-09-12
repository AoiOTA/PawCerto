# DeepWBC terrain

`pawcerto.methods.deepwbc.terrain` implements the public DeepWBC terrain on CPU. It does not import Gym, start Isaac Sim, or allocate a GPU tensor. The nominal mesh is preserved at full resolution. After repairing the xyzw-to-USD quaternion conversion, the complete terrain was exercised by fresh 16-environment/21-update training and a 500-step evaluation: all 500 samples were within the correct terrain coverage and had body contact above 1 N. The earlier zero-contact run remains disqualified and retained below. These results establish the bounded physical execution path, not learned whole-body control.

## Source and license

The Perlin class and its original `generate_perlin_noise_2d` and `generate_fractal_noise_2d` methods come from `legged_gym/legged_gym/utils/terrain.py` in DeepWBC revision `8159e4ed8695b2d3f62a40d2ab8d88205ac5021a`. Its BSD-3-Clause NVIDIA/ETH notice is retained in the module. The only functional adaptation to the noise routines is an instance-local NumPy `RandomState(seed)` instead of global `np.random`, retaining MT19937 and draw order for a fixed terrain seed. Constructor diagnostics are omitted; the float-to-int16 operation and its warning are retained.

DeepWBC imports the separately distributed Isaac Gym mesh converter. A local original helper exists in the pinned UMI reference, revision `f10aadf938a0b095919ad0df33c33a0b2a1d936d`, at `mani-centric-wbc/legged_gym/env/isaacgym/terrain_utils.py`. That helper has a restrictive NVIDIA notice and is **not copied into the implementation**. The converter used here is the BSD-3-Clause Isaac Lab `convert_height_field_to_mesh` from revision `412fb31b30ee605b4ffec4327436fc0fe53281d8`, `source/isaaclab/isaaclab/terrains/height_field/utils.py`. Independent tests execute the original helper's AST without importing Gym and compare exact vertices and triangle indices, including slope correction. This establishes tested converter parity, not proof of which exact Gym package upstream ran.

## Lab consumer contract

```python
from pawcerto.methods.deepwbc.terrain import build_terrain, make_terrain_cfg

terrain = build_terrain(seed=1, cfg=config['terrain'])  # CPU; may run before AppLauncher
terrain_cfg, contact_paths = make_terrain_cfg(terrain)  # after AppLauncher
# Go1WidowXIsaac(terrain_cfg=terrain_cfg, terrain_contact_paths=contact_paths, ...)
```

| Value | Nominal contract |
| --- | --- |
| `heightsamples` | `(600, 10000)` int16, first axis x, second axis y |
| `heightsamples_float` | Original float64 pre-quantization samples, including the original offset |
| `vertices` | `(6000000, 3)` float32, local xyz, x-major C flattening |
| `triangles` | `(11978802, 3)` uint32, original diagonal and winding |
| Horizontal / vertical scale | 0.025 m / 0.00001 m |
| Mesh world transform | Translation `(-7.5, -125, 0)` applied once; Lab orientation is xyzw and converted to USD real/imaginary order |
| Collision/contact prims | `/World/DeepWBCTerrain/tile_000` through `tile_018`, shared static meshes, collision group -1 |
| Collision approximation | `none`; original triangles, no hull or decimation |
| Material | Static friction 1, dynamic friction 1, restitution 0 |
| Environment origin bounds | x uniform `[-3.75, -3]`, y uniform `[-115, 115]`, z 0 |

The environment owner samples origins with the original Torch random path and handles reset perturbations. Origins are already world coordinates; an additional clone grid offset would change placement. `environment_origin_bounds()` exposes the original bounds. The original environment also reshapes height samples to `(tot_rows, tot_cols)` for a tensor; that storage view must not be confused with the `(tot_cols, tot_rows)` collision geometry orientation.

The USD callback uses supported public USD APIs and ordinary static mesh collision. It captures NumPy arrays in a closure so configuration deep copies do not duplicate the full terrain. The physics material binds directly to every collision tile. All returned child paths are passed to the runtime contact sensors, whose force aggregation sums the terrain filters together with the box and other robot bodies. Default implementation requires no modified PhysX or physics-engine build.

## Original overflow behavior

The upstream constructor adds `100000` to rows starting at `tot_cols // 2 - 100` and then computes `(heightsamples_float * (1 / vertical_scale)).astype(np.int16)`. These values exceed int16 and int32 range. With the tested NumPy 2.5.1 runtime, the affected nominal rows 200 onward become zero and emit `RuntimeWarning: invalid value encountered in cast`. This exact operation is preserved, and independent upstream source execution gives identical arrays. It is not replaced by an intentional flat-plane fallback. A separate full nominal run of the original source under the existing reference NumPy 1.23.4 environment produced exactly the same 6,000,000 int16 values (zero mismatches), including all-zero rows 200 onward. Its height-array SHA256 also matched `8ea3581b4f5357d6165d1c6dfcead7033d543cc44b520af17839106d7c83a36c`. Out-of-range float-to-integer conversion is runtime dependent; matching a different historical NumPy/platform requires checking that runtime rather than assuming bitwise equivalence. No claim of parity with an unspecified historical binary is made.

## Observed validation

On 2026-09-12, the actual nominal seed-1 generation ran in a dedicated systemd user service with `MemoryMax=4G`, `MemorySwapMax=0`, `TasksMax=16`, `CPUQuota=100%`, and `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`. Interpreter: `/home/lyb/miniconda3/envs/pawcerto-lab-sim610/bin/python`, NumPy 2.5.1. It exited successfully after 2.59 seconds including array saves; measured process peak RSS was 1,078,244 KiB (about 1.03 GiB). The transient unit was garbage-collected after exit; later `systemctl show` defaults are not its runtime resource limits.

The saved evidence is under gitignored `outputs/deepwbc-terrain/`:

- `nominal_probe.py`: actual nominal validation command body.
- `nominal.log`: original overflow warning and complete result.
- `nominal-result.json`: shapes, bounds, timing, RSS, thread settings, array hashes.
- `reference_numpy_probe.py`, `reference-numpy.log`, `reference-numpy-result.json`: independent original-source full-height comparison in NumPy 1.23.4, skipping duplicate mesh construction. Actual cgroup limits were captured as 4,294,967,296 bytes memory, zero swap, one CPU quota, and 16 tasks; peak RSS 1,091,296 KiB. This run exited successfully with zero height mismatches.
- `heightsamples.npy`, `vertices.npy`, `triangles.npy`: full mesh arrays, about 217 MiB total.

The probe checks all vertex heights against the quantized source height field, finite coordinates, all index bounds, first/last face topology, exact nominal counts, origin bounds, and the nonzero original Perlin region. Raw height range was `[0, 16855]`, or `[0, 0.16855]` metres. Local bounds were approximately `[0, 0, 0]` to `[14.975, 249.975, 0.16855]`.

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/lyb/miniconda3/envs/pawcerto-lab-sim610/bin/python -m unittest discover -s tests -p test_deepwbc_terrain.py` passes 5 tests: two fixed-seed independent Perlin/mesh comparisons, slope/triangle parity, origin/RNG isolation, exact tiled-face/boundary preservation, and actual USD callback authoring on an in-memory CPU stage (the first comparison is one test with two seeds). The USD test inspects three tiles including a partial final tile: points, indices, static collision, triangle approximation, inherited transform, material binding, friction, restitution, and exact equality of returned filter paths to all authored colliders. Original-source parity tests explicitly skip when the pinned reference checkouts are absent; the USD test skips if USD libraries are unavailable.

This evidence proves CPU generation and small-mesh USD authoring. It does not prove nominal mesh cooking, contact filtering, simulation stability, training, or learned control; those require the owning Lab runtime execution.

## Repair after actual PhysX cooking failure

The first full-resolution single-mesh runtime attempt emitted `Too many child nodes` and `BV4 tree failed to build` in `outputs/deepwbc-native-wrench-execution-20260912/train-attempt2.log` at lines 24–33. PhysX could not create `/World/DeepWBCTerrain`. This failure is retained; nominal CPU generation alone did not establish a usable collider.

The spawn callback now authors a translated parent Xform with 19 separate static triangle meshes, each spanning at most 32 original cell rows (330,000 vertices and 639,936 triangles maximum; the last tile is smaller). `iter_mesh_tiles` slices original vertices and faces without regenerating the height field or mesh. It rebases only face indices. Adjacent tiles duplicate their shared vertex row but partition the original faces into disjoint consecutive ranges; no face is dropped or duplicated. The exact world coordinates, triangle winding, original int16 overflow, full 15 m by 250 m grid extent, and sample resolution remain unchanged.

`outputs/deepwbc-terrain/tile-result.json` records full nominal validation against the saved original arrays: all 11,978,802 original faces are retained exactly once, every tile vertex equals its original vertex, and all adjacent edges match. That bounded CPU service used a 4 GiB memory limit and one CPU quota, with peak RSS 275,224 KiB. `tile_probe.py` and `tiles.log` retain the executable check and result. This reused the already validated nominal geometry rather than generating another terrain.

The runtime owner confirmed it consumes every returned tile path. The GPU owner continues the original 16-environment, 21-iteration command unchanged to check actual cooking and contact behavior. At that point, 32-cell-row tiles were a proposed repair supported by exact geometry evidence; subsequent actual GPU results are recorded below. Separate collision shapes can change contact generation and solving at tile boundaries; exact geometry does not imply bitwise identical physics.

## Combined-scene allocation investigation

The subsequent reconstructed-wrench training attempt failed before learning: `outputs/deepwbc-reconstructed-execution-20260912/train-attempt1.log` requested 579,166,759,168 bytes of pinned host memory from `PxgGeometryManager`, followed by a corrupted scene and secondary material-readback/close errors. The recorded host limit was not reported as a cgroup OOM. This is distinct from the earlier single-mesh BV4 cooking failure.

The GPU owner then ran `outputs/deepwbc-terrain/terrain_gpu_probe.py`, keeping all original 19 tiles, 6,000,000 vertices, and 11,978,802 triangles. The official GPU scene reset and one step completed, process exit 0, peak RSS 6,108,360 KiB. The owner inspected the complete log and found no current-run PhysX allocation, cooking, or corrupted-scene error (old crash-report metadata was present). Evidence: `outputs/deepwbc-reconstructed-execution-20260912/terrain-only.log`, `terrain-only-probe1/result.json`, and the operator's saved status/source snapshot.

That probe and the failed training use identical terrain/config source hashes and the same public-fresh seed-1 construction. The probe loads the saved, independently validated seed-1 arrays. The failing training did not save its in-memory arrays, so this is deterministic construction equivalence rather than a captured bytewise comparison of its process memory.

This actual GPU result establishes that the full tile terrain can initialize in an isolated scene. It does not establish success of the combined training scene. No terrain resolution or tiling change is justified by the isolated success alone; robot/material/cloning and combined-scene paths are being distinguished with the owning runtime and GPU operators before returning to the unfinished 16-environment, 21-update run.

The one-robot combined probe then reproduced the allocation failure at 36,875,815,168 bytes (`combined-one.log`). The one-robot and 16-robot requests satisfy the exact relation `723085568 + 50 * num_envs * 723054592` bytes. This points to repeated terrain-sized accounting tied to the 50 collision shapes per robot, but does not itself identify the native trigger. `outputs/deepwbc-terrain/allocation-scaling.json` preserves that calculation. An empty-dynamic-body terrain scene may not traverse every upload branch of an articulated scene, so its earlier success must not be used to exclude terrain registration interactions.

Batching the final USD pair-filter writes while preserving their targets did **not** resolve the one-robot failure: `combined-batched.log` requested exactly the same 36,875,815,168 bytes. This is negative evidence for the proposed per-write-notice explanation as a sufficient cause/fix. The original terrain arrays and 19-tile representation remain unchanged during this investigation.

The next one-robot diagnostic passed an empty terrain list to the material-channel helper. This omitted **two** operations together: terrain-to-self-collider pair relationships and rebinding the terrain to the shared ordinary material (also coefficient 1 with average combining). It retained all terrain triangles and 50 robot collision shapes, their original terrain material, and the other filters. This diagnostic completed runtime creation and one step, exit 0, peak RSS 6,263,412 KiB (`no-terrain-pairs.log` and `combined-no-terrain-pairs/result.json`). It does **not** isolate the pair-filter relationships from material rebinding as the trigger. The script/result label saying “only” terrain-to-self filters were omitted is an incomplete description of the actual helper call; those original artifacts are retained, with this correction. Omitting those filters changes physical behavior and is **not** accepted for training. The runtime owner then ran a single-variable comparison keeping every pair relationship and omitting only the redundant terrain material rebind; its result is recorded below.

The single-variable no-rebind probe also failed with the same 36,875,815,168-byte request and corrupted-scene error (`no-terrain-bind.log`, lines 105–111; frozen diagnostic change recorded in `diagnostic-no-terrain-bind.source.json`). Thus redundant rebinding is not required for the failure. Compared with the successful diagnostic omitting both operations, this supports the final terrain pair-filter representation as the reproduced trigger when the original material binding is retained. It does not establish an exact internal native call sequence. The runtime owner is implementing an equivalent single-membership collision-group representation and validating its collision semantics; no terrain geometry or tiling change is part of that repair. The one-environment grouped-filter validation and status of the original 16-environment, 21-update training are recorded below.

On 2026-09-13 (local time), the GPU owner ran the same complete terrain plus one real Go1/WidowX robot with the runtime owner’s supported single-membership collision groups, normal terrain material binding, and reconstructed sensor signal. Runtime creation and one control step passed, exit 0, peak RSS 6,194,008 KiB (5.91 GiB). The runtime read back 50 robot shapes, four collision groups, and 70 single-membership shapes including all 19 terrain tiles and the box. Evidence: `combined-groups.log`, `combined-one-groups/result.json`, and `combined-groups-source.json` under `outputs/deepwbc-reconstructed-execution-20260912/`. The terrain implementation hash remained `97ead2de076f44e8ec7da1fe4bb9a839a62b0e4985b771d6c780588ed35b762f`; only the filtering owner’s implementation changed. This established construction and step return only; the quaternion bug found later means this result did not establish correct terrain interaction. The original 16-environment, 21-update command then ran as `train-attempt2` with a 12 GiB host limit and 1 GiB swap limit; its completed result is recorded below.

## Superseded 21-update execution: incorrect terrain placement

The original 16-environment, seed-1 public-fresh run completed all 21 updates and exited successfully on 2026-09-13. It used the unchanged complete 19-tile terrain, the runtime owner’s 49 single-membership collision groups, normal material binding, and the explicitly approximate reconstructed-wrench signal. The runtime reported 835 grouped shapes (16 × 50 robot shapes, 16 boxes, and 19 terrain tiles). The preserved int16-cast warning still appears; the run did not report the earlier BV4, pinned-memory-allocation, or corrupted-scene failures.

Evidence under `outputs/deepwbc-reconstructed-execution-20260912/` is `train-attempt2-command.json`, `train-attempt2-source.json`, `train-attempt2.log`, and `train-attempt2-status.txt` (`Result=success`, `ExecMainStatus=0`, inactive). The output directory `outputs/isaac/deepwbc-reconstructed-21updates-20260912/` contains 21 metric records indexed 0–20, the final DAgger update at index 20, `model_21.pt`, and `evidence.json` reporting `iterations_completed: 21`. Terrain source SHA256 is unchanged throughout the failed and successful combined tests: `97ead2de076f44e8ec7da1fe4bb9a839a62b0e4985b771d6c780588ed35b762f`.

These finite updates and the saved checkpoint established optimizer execution, but the subsequent quaternion finding withdrew their qualification as a valid original-terrain training run. They do not establish learned whole-body control, stability over longer training, original Gym force-sensor parity, or bitwise-equivalent contact solving at tile boundaries. The reconstruction approximation and collision-channel validation belong to the runtime’s separate evidence; no physics-engine patch or reduced terrain was used to obtain this result.

## Quaternion correction after the fixed 500-step audit

The first fixed evaluation recorded zero contact force for every frame and every body, with 33 base-height terminations after approximately 0.3 seconds per episode. The terrain callback incorrectly interpreted current Isaac Lab’s `(x, y, z, w)` orientation as `(w, x, y, z)`. `AssetBaseCfg.InitialStateCfg.rot` is `(0, 0, 0, 1)`, so the old conversion authored a 180-degree rotation about z instead of identity. Translation alone had been checked in the earlier USD test; supplying an old-convention identity in that test masked the error.

The smallest repair converts xyzw to `Gf.Quatf(quat[3], Gf.Vec3f(*quat[:3]))` and uses the xyzw identity when no orientation is passed. Height generation, all original vertices/faces, the 19-tile partition, materials and filtering are unchanged. The regression now passes the actual installed Lab default, checks transformed mesh points (not only translation), checks a nonidentity 90-degree rotation, and compares the source default to the test contract. This test failed against the old code and all five terrain tests passed after repair.

`outputs/deepwbc-terrain/quaternion-repair-20260913/world-result.json` records a full nominal CPU USD stage using the repaired callback: world bounds are `[-7.5, -125, 0]` to approximately `[7.475, 124.975, 0.16855]`. The old source transform instead placed the mesh at x `[-22.475, -7.5]`, y `[-374.975, -125]`. The actual prior 500 recorded root positions have x `[-3.70843, -2.81069]`, y `[-95.77306, -94.77486]`: all 500 were outside the old terrain and all 500 lie inside the corrected coverage. Original height samples under those corrected XY positions range from 0.0419 to 0.14801 m. This explains a concrete coverage failure; the fresh contact-physics result is recorded below.

The full USD check completed with peak RSS 634,704 KiB under a 4 GiB memory cap and explicit single-thread USD execution. Its first attempt hit the task thread limit and crashed when USD tried to create extra threads; `world.log` and `world-first-status.txt` retain that failure. The successful repeat explicitly set `PXR_WORK_THREAD_LIMIT=1` and `Work.SetConcurrencyLimit(1)`; see `world-fixed.log`. No GPU or simulator was started by this CPU check.

The earlier frozen source, 21-update checkpoint, export and zero-contact evaluation remain retained as invalid physical-acceptance artifacts. The GPU owner used the corrected source for fresh original 16-environment/21-update seed-1 training, export, and the same fixed 500-step seed-2027 evaluation. No old checkpoint was reused as the corrected training result. Corrected terrain source SHA256: `a763e806432e56d71b74e4912d969e7c78ce56fe268573799f28ef3175031c2c`.

## Final corrected physical execution

The fresh run in `outputs/isaac/deepwbc-reconstructed-21updates-terrainfixed-20260913/` completed 21 updates and exited 0, with the original 16 environments, seed 1, public-fresh recipe, complete terrain and explicitly approximate reconstructed-wrench signal. `model_21.pt`, metrics and `evidence.json` retain this corrected training identity. The exact source/command and logs are `train-terrainfixed-source.json`, `train-terrainfixed-command.json`, `train-terrainfixed.log` and `train-terrainfixed-journal.txt` in `outputs/deepwbc-reconstructed-execution-20260912/`.

The exported corrected model then completed the original-play fixed 500-step evaluation at seed 2027, exit 0, in `outputs/isaac/deepwbc-terrainfixed-model21-fixed500-seed2027-20260913/`. Its `trajectory-audit.json` records all 500 finite samples inside the corrected terrain XY coverage, with at least one body contact above 1 N in every sample. Maximum body-contact magnitude was 258.992 N; median summed vertical contact force was 149.973 N. One `goal_signed_roll` termination occurred at step 44, followed by a 455-sample unfinished episode. There were no base-height terminations like the earlier free-fall run. The source exporter’s actions matched the policy on all 500 actual observations; separate packaged-consumer evidence belongs to source delivery.

This closes the terrain owner’s corrected bounded execution outcome: the full original grid and all triangle faces are present, placed correctly, and participate in real contact during the authorized training/evaluation path. It does not establish learned locomotion or end-effector control: the audit reports mean forward velocity approximately 0.00122 m/s against a 0.5 m/s command, and median end-effector position error approximately 0.486 m. No original-Gym six-axis parity, long-run stability, or bitwise contact equivalence is claimed. The old wrong-placement training, export, evaluation, and source freeze are retained with their qualification explicitly withdrawn.
