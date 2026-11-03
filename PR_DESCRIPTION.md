# Example library: move configs and dataset manifests into `configs/` and `datasets/`

## Summary

The config files and `fast-batch` manifests lived loose in the repo root under
`config.*.toml` / `datasets.*.csv` / `mylist.csv` names. They now live in two
folders that form a library of examples, each with a README index, and a test
keeps every file in them loadable.

```
configs/                    analysis settings, passed with --config
  README.md                 index: what each file is for, how to add one
  example.toml              <- config.example.toml         (reference: every fast setting)
  fastplus.example.toml     <- config.plus.example.toml    (reference: every FASTplus setting)
  perturb.example.toml      <- config.perturb.example.toml (template: per-movie LED sidecar)
  taka.toml                 <- config.taka.toml
datasets/                   fast-batch manifests
  README.md                 index + the path rule
  example.csv               <- datasets.example.csv        (reference: the column format)
  taka.csv                  <- datasets.taka.csv
  unloaded.csv              <- mylist.csv
```

```bash
fast -d <dataset> --config configs/example.toml
fast-batch datasets/taka.csv
fast-preview <movie> --config configs/taka.toml
```

## What changed

- **Moves** with `git mv` (history follows the files). The folder now says what
  a file is, so the `config.` / `datasets.` prefixes are dropped; `mylist.csv`
  became `datasets/unloaded.csv` (it lists the unloaded-motility example).
- **Manifest paths.** `fast-batch` resolves relative paths against the
  manifest's own folder, so the manifests now point at `../examples/...` and
  `../configs/...`. They resolve to exactly the same directories and configs as
  before, from any working directory.
- **READMEs** in both folders: an index table (file, which command uses it,
  what it is — *reference*, *template* or a specific experiment), the
  layering/override rules for configs, the path rule for manifests, and how to
  add a new entry.
- **`tests/test_example_library.py`**: loads every config in `configs/` through
  `Settings.from_toml` (the `--config` code path, which rejects unknown
  sections/keys — so a typo fails the suite), parses the perturbation template
  as a sidecar, parses every manifest in `datasets/` and checks each referenced
  config exists in `configs/`, and fails if a config or manifest is added back at
  the repo root. The movie data (`examples/`, not in git) is not required.
- **References updated**: README (config section + an "Example library"
  paragraph), `docs/fastplus.md`, `docs/PR_fastplus_directional.md`,
  `docs/preview.md`, `hpc/README.md` (plus a note on copying a library manifest
  into a Sherlock run dir), comments inside the moved files, CLAUDE.md,
  CHANGELOG. Journals keep the old names — they record what was run at the time.

## Verification

- Full suite: **183 passed, 5 skipped** (cloud; +9 library tests).
- `fast-batch datasets/taka.csv --preflight-only` run from another directory
  resolves all four slides to `<repo>/examples/taka/04302026 Motility
  Experiment/slide_0N` with `<repo>/configs/taka.toml`.
- A config with a misspelled key or section is rejected by
  `Settings.from_toml` (checked), so the library test catches typos.

## Notes for users

- Commands and scripts that name the old files need the new paths, e.g.
  `fast-batch datasets.taka.csv` → `fast-batch datasets/taka.csv`,
  `--config config.taka.toml` → `--config configs/taka.toml`.
- Sherlock run dirs that hold their own copy of a manifest + config are
  unaffected. To run a library manifest there, use it in place (with the
  `examples` symlink at the repo root), or copy it together with its config and
  fix the relative paths.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01GHpL1AmJemrX4ED1LFE5fB
