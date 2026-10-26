# Figures and plot themes

Every figure FASTrack writes takes its look from **one master stylesheet**, a
TOML *theme*. Run configs and the command line can override any part of it,
for all figures or for one.

```
built-in theme  <  your theme file  <  [plotting.*] in the run config  <  --plot key=value
```

```bash
fast -d DIR                                           # default theme
fast -d DIR --theme legacy                            # the original FAST look
fast -d DIR --theme my_lab.toml                       # your own theme
fast -d DIR --plot paths_2D.colormap=blue-red --plot rc.font.size=12
fast -d DIR --plot-formats png                        # summary figures as PNG only
fast -d DIR --skip-plot paths_2D                      # don't write a figure
```

The same four options (`--theme`, `--plot`, `--plot-formats`, `--skip-plot`)
exist on `fast`, `fastplus`, `lima`, `fast-preview` and `fast-batch`.

## Themes

Built-in themes live in `src/fastrack/viz/themes/`:

| theme | what it is |
|---|---|
| `default` | the master stylesheet; every other theme starts from it |
| `legacy` | the original FAST look: original paths figure, top labels, 1500 nm/s velocity axis, PNG only |
| `paper` | starting point for publication figures: PDF + SVG only, units on the length-velocity labels |

A theme file has these sections:

| section | contents |
|---|---|
| `extends` | theme to start from (`"default"`, another built-in, or a relative path) |
| `[tokens]` | backend-neutral design values: `font_family`, `font_scale` (multiplies every explicit font size), `background`, `ink` |
| `[colors]` | named colours the figures use (`moving`, `maxvel`, `stuck`, `good` / `marginal` / `poor`, polarity classes ...) |
| `[colormaps.<name>]` | colour scales as `stops`, plus the `ink`, `stuck` and `background` colours that go with them and `centred` (diverging around the median) |
| `[rc]` | matplotlib rcParams, applied **only while a FASTrack figure is drawn** |
| `[bokeh]` | Bokeh-only settings for interactive output (prepared; see below) |
| `[output]` | default formats: `summary = ["png", "pdf", "svg"]`, `diagnostic = ["png"]` |
| `[figures.<name>]` | per-figure options (table below), plus `formats`, `enabled`, `rc` (merged over `[rc]`) and `rc_base = "matplotlib"` (start from matplotlib's defaults) |

Your own theme only needs what differs:

```toml
# my_lab.toml
extends = "default"

[tokens]
font_family = ["Helvetica", "Arial"]

[colors]
maxvel = "#d62728"

[figures.paths_2D]
colormap = "blue-red"
```

## In a run config

```toml
[plotting]
theme = "default"                 # or "legacy", "paper", "path/to/theme.toml"
formats = ["png", "pdf"]          # summary figures (default: from the theme)
skip = ["filaments_frame"]        # figures not to write
xmax = 10000                      # longest filament (nm) in the statistics -- not a style

[plotting.paths_2D]               # any [figures.paths_2D] option
colormap = "blue-red"

[plotting.rc]                     # any matplotlib rcParam, for every figure
"font.size" = 12

[plotting.colors]                 # also [plotting.tokens], [plotting.colormaps], [plotting.output]
stuck = "0.5"
```

The flat keys from before (`ymax`, `label_position`, `maxvel_color`,
`paths_style`, `paths_labels`, `paths_colormap`, `paths_points`,
`scale_bar_um`) still work, with a deprecation warning; the matching CLI
shortcuts (`-ymax`, `-cl`) set the same options.

`--plot` values are parsed like TOML: `--plot paths_2D.points=false`,
`--plot 'length_velocity.formats=["png"]'`, `--plot colors.stuck=0.5`.

## Figure catalogue

*Summary* figures are written as PNG + PDF + SVG by default (PDF/SVG keep
the text editable); *diagnostic* figures as PNG. Every option below can be set
in a theme (`[figures.<name>]`), a config (`[plotting.<name>]`) or with
`--plot <name>.<option>=value`.

| figure | pipeline | kind | files | options (defaults) |
|---|---|---|---|---|
| `length_velocity` | fast | summary | `<movie>_length_velocity.*, combined/<dataset>_length_velocity.*` | dpi 200; `ymax = 0`, `label_position = "bottom"`, `label_units = false`, `text_font_size = 30`, `pad_inches = 0.3` |
| `paths_2D` | fast | summary | `<movie>_2D.*, <movie>/paths_2D.*` | dpi 300; `style = "velocity"`, `colormap = "blue"`, `points = true`, `labels = false`, `scale_bar_um = 10.0`, `width_in = 8.0`, `line_width = 1.2`, `legacy_dpi = 400`, `pad_inches = 0.05` |
| `correlation_length` | fast | summary | `correlation_length.*` | dpi 200 |
| `filaments_frame` | fast | diagnostic | `filaments_<frame>.*` | dpi 200 |
| `preview_dashboard` | fast-preview | diagnostic | `<movie>_preview.*` | dpi 80 |
| `polarity_montage` | fastplus | diagnostic | `qc_overlay.*` | dpi 110 |
| `frame_average` | fastplus | summary | `frame_average.*` | dpi 150 |
| `lima_load` | lima | summary | `<A>_vs_<B>_stop_model*.*` | dpi 100 |
| `lima_length` | lima | summary | `<A>_vs_<B>_length*.*` | dpi 200 |

Each output folder also gets a **`figures.json`** index: one entry per figure
written (`figure`, `title`, `pipeline`, `kind`, `theme`, `files`). It is what a
PDF report or an interactive viewer can be built from without knowing the
pipelines' file names.

Not themed (they are movies, not figures): the overlay movies (`-m`,
`--overlay-movie`, the fastplus `qc_overlay.mp4`). Their frames do use the
theme's `[rc]`, and the `-m` movie draws its track background in the
paths colour scheme.

## For developers: drawing a figure

```python
from fastrack.viz.theme import as_theme

def plot_something(data, out_dir, theme=None):
    with as_theme(theme).figure("my_figure", some_option=None) as fx:   # rc applied here only
        if not fx.enabled:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(data.x, data.y, color=fx.color("moving"))
        ax.set_title("...", fontsize=fx.fs(12))                       # scaled by font_scale
        cmap = fx.scheme(fx.opt["colormap"])["cmap"]                    # theme colour scale
        fx.save(fig, out_dir + "/my_figure")                            # formats + figures.json
        plt.close(fig)
```

1. Register the figure in `viz/figures.py` (`register_figure(FigureSpec(...))`:
   name, title, pipeline, `summary` / `diagnostic`, file pattern, option
   defaults) and add a `[figures.my_figure]` section to `default.toml` with
   the same defaults (a test checks they agree).
2. Never hard-code colours, dpi or formats: take them from `fx`.
3. Keep the data step (arrays / tables) separate from the drawing, as
   `Motility.paths_2d_data` / `_render_paths_2d_mpl` do. A second backend
   then only needs a new renderer.
4. Keyword arguments of the public plot function default to `None` and are
   passed to `figure(...)`, so explicit arguments override the theme.

Importing FASTrack no longer changes matplotlib's global settings (the old
`viz/plotparams.py` set FAST's 40 pt fonts for the whole process); the
`[rc]` of the default theme holds those values and applies them per figure.
The fastplus figures and the preview dashboard start from matplotlib's
defaults (`rc_base = "matplotlib"`), as they were designed for.

## Interactive figures (prepared, not built yet)

The theme and the catalogue are ready for a second, interactive backend
(Bokeh, standalone HTML):

* `[tokens]` / `[colors]` / `[colormaps]` are backend-neutral;
  `Theme.bokeh_theme_json()` turns them into a Bokeh theme and
  `Theme.palette(name, n)` turns a colour scale into a Bokeh palette.
* `register_renderer(figure, "bokeh", fn)` attaches a renderer
  `fn(data, fx, path)` to a figure; `html` in `formats` is then written by it.
  Until a figure has one, `html` is skipped with a warning.
* `pip install 'fastrack[interactive]'` installs Bokeh; nothing imports it yet.

Planned uses: linked length-velocity / paths views (select points, see the
tracks), hover details per track, an HTML preview dashboard, a per-dataset
viewer; and a PDF report assembled from `figures.json`.
