# DeepWBC training integration

PPO, rollout storage, and trajectory utilities are copied from the official
Deep-Whole-Body-Control source at commit
`8159e4ed8695b2d3f62a40d2ab8d88205ac5021a`, under the retained BSD-3-Clause
notices and LICENSE. The PPO/storage mathematical path is unchanged; imports
resolve within PawCerto. Runner orchestration follows the same rollout order.

Use `OnPolicyRunner(env, default_config(), device="cpu")`. The full config has
`env`, `algorithm`, `policy`, and `runner` sections. Environment attributes are
`num_envs`, `num_obs=860`, `num_privileged_obs=None` (or 860), and `num_actions=18`.
Provide `reset()`, `get_observations()`, `get_privileged_observations()`,
`update_command_curriculum()`, and `step(actions)` returning
`(obs, privileged_obs, leg_rewards, arm_rewards, dones, infos)`.
Observation is [76 proprio, 24 privileged, 10 x 76 history]; separate critic
observations, if provided, must have the same layout. Rewards, dones and optional
`infos["time_outs"]` are length-N tensors. The environment owns history/reset,
commands/curricula and rewards. No physics is implemented here.

Optional original torque supervision additionally requires explicit six-element
`default_arm_p_gains`, `default_arm_d_gains`, `default_arm_dof_pos` on the env,
and N x 6 `target_arm_torques`, `current_arm_dof_pos`, `current_arm_dof_vel` in
step infos. It is disabled in the public configuration and not covered by the
CPU parity evidence. Recurrent policies are outside this integration; the
upstream recurrent storage iterator does not match the dual-head PPO tuple.

Iteration 0 and every 20th iteration collect history-policy rollouts and update
only the history encoder. Other iterations use privileged encoding and original
dual-advantage PPO. Both increment the same schedule counter. `learn(n)` returns
iteration/stage/loss records. Logging backends are not imported.

Checkpoint saves retain original model/optimizer/iter/infos keys and add history
Adam, schedule counter, effective learning rate, CPU RNG, config and step count.
`iter` is the next iteration, including periodic checkpoints (upstream periodic
saves retained a stale loop-start iteration). Loading original checkpoints is
supported but emits a warning for absent history Adam; their missing state and
possibly stale iteration cannot be reconstructed. Environment state is not
checkpointed. Exact continuation is demonstrated only for the deterministic
CPU interface fixture, not across simulator reset or GPU execution.

`get_inference_policy` retains the original callable semantics: pass
`hist_encoding=True` for deployment without privileged inputs. `export(path)`
traces the original history-conditioned actor, input N x 860, output N x 18.
This intentionally supplies an executable full-input export instead of the
upstream save_jit script's mismatched hard-coded 20-DoF dimensions. TorchScript
is deprecated by the installed PyTorch but is retained for this export contract.

CPU tests compare returns, both losses, every parameter, and both Adam states
bit-for-bit with the original source; actual updates exercise stages 0 through
20, resumed PPO and loaded TorchScript inference. This establishes component
and interface parity, not learned control, simulator fidelity or task success.
