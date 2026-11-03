# Config library

Analysis settings as TOML files, passed with `--config` (run from the repo
root):

```bash
fast -d <dataset dir> --config configs/example.toml
fastplus -d <dir> --config configs/fastplus.example.toml
fast-preview <movie> --config configs/taka.toml
```

Every section and key is optional: keys you set override the built-in
defaults, explicit CLI flags override the file, and several `--config` files
layer left to right (e.g. a lab-wide base plus a per-experiment file). Needs
Python 3.11+ (`tomllib`).

| file | used by | what it is |
|---|---|---|
| [`example.toml`](example.toml) | `fast`, `fast-batch`, `fast-preview` | **Reference**: every `fast` section and key, commented, with defaults |
| [`fastplus.example.toml`](fastplus.example.toml) | `fastplus` | **Reference**: every FASTplus (directional, two-colour) setting |
| [`perturb.example.toml`](perturb.example.toml) | `fastplus` (sidecar) | **Template** for a per-movie LED switch schedule — copy next to a movie as `<movie>.perturb.toml`; not passed with `--config` |
| [`taka.toml`](taka.toml) | `fast`, `fast-batch`, `fast-preview` | Taka's low-SNR movies (04302026): in-pipeline preprocessing + ridge detector with the SNR filter |

## Adding a config

Name it after the experiment or setup (`<lab-or-dataset>.toml`), start the file
with a comment saying what data it is for, when/why it was tuned and the command
to run it, and set only the keys that differ from the defaults — copy them from
`example.toml`. `tests/test_example_library.py` loads every file here, so a typo
in a section or key name fails the test suite.
