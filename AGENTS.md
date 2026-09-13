# Agent notes

Load skills before working here: `sglang-sm121-nvfp4`, `inference-release-engineering`,
`gb10-cluster-host-ops`, `dsv4-sm121-native-serve-qualify`.

Cluster: 4-node CRS812 (see ~/projects/crs812-cluster/VERDICT.md). N1 head
192.168.68.59/f100.1, N2 192.168.68.51/f100.2, N3 192.168.68.78/f100.3,
N4 192.168.68.56/f100.4. SSH: `~/.ssh/id_ed25519_crs812` +
`known_hosts_crs812_fabric`.

Rules:
- Nothing mutating on N1 while the qwen3.8 bench runs (user gate).
- Long processes under tmux, one durable owner.
- All code in this repo is written from scratch. Reference implementations
  (LMSYS blog/cookbook, community Spark recipes) are credited in the plan's
  References section only — never copy their code.
- Engram design: local per-node NVMe store, no collectives in the lookup path.
