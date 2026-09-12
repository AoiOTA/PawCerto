# Explicit upstream license sources

The fixed Learning Force Control README (Improbable-AI/learning-compliance,
`c760e1d74ad165d3c069d4f57ab5d066f6a41eb6`, lines 34–39) explicitly identifies
leggedrobotics/legged_gym and leggedrobotics/rsl_rl as its environment and
training-code origins. Its file headers reference missing `LICENSES` paths.
These files are the original texts from those explicitly identified upstreams;
they are **not** claimed to be recovered copies of LFC's omitted directories.

- `legged_gym/LICENSE`: https://raw.githubusercontent.com/leggedrobotics/legged_gym/ae614c029977157123225f538ecdd3f873e54bd4/LICENSE
- `rsl_rl/LICENSE`: https://raw.githubusercontent.com/leggedrobotics/rsl_rl/ff9e971c979a1cf42cd984cf6e0f50922229cb9c/LICENSE
- LFC source declaration: https://github.com/Improbable-AI/learning-compliance/blob/c760e1d74ad165d3c069d4f57ab5d066f6a41eb6/README.md#L34-L39

The rsl_rl initial `rsl_rl/utils/utils.py` has the NVIDIA BSD-3-Clause header.
Its `unpad_trajectories` matches LFC's AST exactly; the split helper differs only
by one character in the docstring, with identical executable statements.
The legged_gym initial `legged_gym/envs/base/legged_robot.py` also carries that
file-level BSD header. LFC has substantial environment modifications and retains
its own root MIT notice; both original notices and source references are kept.
The unrelated assets/dependencies references in the upstream license texts are
retained verbatim, not interpreted as licensing every LFC asset/dependency.
