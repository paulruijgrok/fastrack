# 2026-10-19 — Example library: configs/ and datasets/

Mirror of the Notion session log: *2026/10/19 Example library (configs/, datasets/)*
(hub: **Fastrack** — <https://app.notion.com/p/39edaf1f293b81179192ea41a1110d6b>).

## TL;DR

- Config TOMLs and fast-batch manifests moved from the repo root into
  `configs/` and `datasets/`, each with a README index, on branch
  `example-library` (from main `aba20f0`, i.e. after PR #15 / preview mode).
- Manifest paths rewritten for their new folder (`../examples/...`,
  `../configs/...`); they resolve to the same places as before.
- New `tests/test_example_library.py` (9 tests) keeps the library loadable;
  suite 183 passed, 5 skipped.

## Detailed narrative

Renames (`git mv`): `config.example.toml` → `configs/example.toml`,
`config.plus.example.toml` → `configs/fastplus.example.toml`,
`config.perturb.example.toml` → `configs/perturb.example.toml`,
`config.taka.toml` → `configs/taka.toml`, `datasets.example.csv` →
`datasets/example.csv`, `datasets.taka.csv` → `datasets/taka.csv`,
`mylist.csv` → `datasets/unloaded.csv`. Prefixes dropped since the folder names
say what the files are.

`read_manifest` resolves relative paths against the manifest's directory, so the
manifests now use `../`. fast-batch logs / resume state go to
`fastrack_batch_logs/` in the current working directory (unchanged by the move).
`hpc/` scripts take any manifest and use the same reader; added a note to
`hpc/README.md` about copying a library manifest into a Sherlock run dir.

Library test: every non-sidecar TOML through `Settings.from_toml` (verified it
rejects unknown keys and sections), the perturb template through
`perturbation.from_sidecar`, every manifest through `read_manifest` with its
configs required to exist under `configs/`, and no `config*.toml` /
`datasets*.csv` allowed back at the root.

Journals were left as they are (historical commands).

## Commands

```bash
fast -d <dataset> --config configs/example.toml
fast-batch datasets/taka.csv --preflight-only
python -m pytest -q tests/test_example_library.py
```

## Files / commits

| file | change |
|---|---|
| `configs/` (4 TOML + README) | moved / new index |
| `datasets/` (3 CSV + README) | moved, paths rewritten / new index |
| `tests/test_example_library.py` | new |
| README, `docs/fastplus.md`, `docs/PR_fastplus_directional.md`, `docs/preview.md`, `hpc/README.md`, CLAUDE.md, CHANGELOG, PR_DESCRIPTION.md | references |

## Open items / next steps

- On the Mac: `fast-batch datasets/taka.csv --preflight-only` should report 4/4.
- Sherlock: update any launch commands that named `datasets.taka.csv` /
  `config.taka.toml` in the repo checkout.
- Grow the library: one config + manifest per experiment/setup, indexed in the
  READMEs.
