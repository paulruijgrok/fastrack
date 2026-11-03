# Dataset manifests

Lists of datasets for `fast-batch` (and the HPC array jobs in `hpc/`), one row
per dataset (run from the repo root):

```bash
fast-batch datasets/taka.csv
fast-batch datasets/taka.csv --preflight-only     # check every row first
```

Columns (header row, case-insensitive): `base_dir` (the directory passed to
`fast -d`, required), `config` (a TOML from [`../configs/`](../configs/), blank =
defaults) and `name` (label for logs and resume state; defaults to the folder
name). Lines starting with `#` are comments. `.xlsx` manifests work too.

**Relative paths are resolved against the manifest's own folder** — so from
here the data and configs are `../examples/...` and `../configs/...`. That keeps
a manifest valid wherever you run `fast-batch` from.

| file | datasets | config |
|---|---|---|
| [`example.csv`](example.csv) | **Reference**: the column format, with two example rows | `example.toml` / defaults |
| [`taka.csv`](taka.csv) | Taka's 04302026 Motility Experiment, slides 01–04 | `taka.toml` |
| [`unloaded.csv`](unloaded.csv) | the unloaded-motility example movie | `example.toml` |

The movie data itself (`examples/`) is not in git; these manifests point at
where it lives in a local checkout. `tests/test_example_library.py` parses every
manifest here and checks that each referenced config exists.
