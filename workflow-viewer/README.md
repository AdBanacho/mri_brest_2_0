# Workflow viewer

Interactive diagram of the `mribreast` 2.0 pipeline: data preparation → controller → train → validate → summarize.

**Open it:** double-click `index.html`. You don't need a server, a build step or npm.

## Files

| File | Role |
| --- | --- |
| `workflow.json` | **Source of truth.** Stages (purpose, inputs/outputs, parameters, source file:line, uncertain points), edges, play order. Written from an analysis of the code and `docs/`. |
| `workflow-data.js` | Generated copy of `workflow.json` (`window.WORKFLOW = …`), so the page works from `file://`. Never edit it by hand. |
| `config_space.py` | Builds the configuration space using the project's own code: `fields.tsv`, `schema.validate_combinations` (validity), `cli.required_inputs` (input files) and the shipped `experiment.conf` / `configs/*.conf`. |
| `config-data.js` | Generated configuration space. Never edit it by hand. |
| `sync_data.py` | Regenerates both `.js` data files and checks the stage references. `--check` exits 1 if either file is stale. Re-run it after changing `workflow.json`, `fields.tsv`, `schema.py` or any `.conf`. |
| `index.html`, `styles.css`, `app.js`, `configs.js` | The app (plain HTML/CSS/JS, no dependencies). |

After you edit `workflow.json`:

```bash
python sync_data.py           # regenerate the embedded copy
python sync_data.py --check   # e.g. in CI / pre-commit
```

When the page is served over HTTP (for example `python -m http.server`), it also loads `workflow.json` and shows a banner if the embedded copy differs from it.

## Configurations view

Switch with the **Configurations** tab (or `V`).

- **All valid configurations:** every combination of the 13 settings that change the pipeline's shape (branches, profile, label scheme, selection split, image dimension/source, subtraction, mask provider/mode, 2D slice selection, box mode, box erasing, feature selector), checked by the real validator. 54,456 combinations were checked and 3,250 are valid. A setting with no effect in a combination is shown as `n/a`, so identical pipelines are not counted twice. Filters show how many configurations each choice leaves and grey out impossible ones. Open a configuration to see which stages are skipped or behave differently, the input files it needs, how many variants the independent options add (≈ 103 billion in total), and the CLI / `experiment.conf` lines to run it. **Show in pipeline** fades the unused stages on the diagram, and Play skips them.
- **Every option:** all 107 keys of `fields.tsv` with type, default, allowed values, whether the key is part of `train_id`, and the stage each one configures. Filter by search, group, stage or "shape only".
- **Shipped config files:** `experiment.conf` and `configs/*.conf` expanded the way the controller expands them, with each row validated and linked to its configuration.

Free numeric settings (epochs, learning rate, sizes and so on, 51 fields) can take any value in range and are listed but not counted.

## Using it

- Click a stage to see a plain-language explanation, inputs and outputs, connections and a placeholder example. **Technical details** (parameters, libraries, how it works, source references) are in an expandable section.
- **Play / Next / Back / Start** steps through the pipeline in order. The speed slider controls auto-play. The active stage is highlighted, and a dot moves along each incoming data arrow.
- A **?** badge means part of that stage is unverified or uncertain according to the code and `docs/acceptance.md`.
- Deep links: `#stage=fusion`, `#step=17`, `#theme=light`, `#view=configs` (`&sub=options|shipped`, `&config=1406`), `#overlay=1387` (configuration shown on the diagram); combine them with `&`.
- Keys: arrows move between stages, Enter opens one, `N`/`B` next/back, `P` play, `[`/`]` speed, `+`/`-`/`0` zoom, `D` details, `O` overview, `V` switch view, `T` theme, `?` help.

All examples are labelled placeholders. No patient data was read or embedded.
