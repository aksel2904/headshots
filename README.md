# tennis-wire-headshots

Two scripts that together produce a starter player roster with pictures.

`roster.py` queries Wikidata for tennis players ranked by how many Wikipedia
language versions they have — a decent proxy for public recognition, which
surfaces both current tour players and retired legends without needing an
editorial list. It writes a CSV with names in English and Russian, IOC country
code, birth date, tour identifiers and the Commons filename of the player's
image.

`headshots.py` takes that CSV, crops a square around each player's head and
shoulders and removes the background, producing a transparent PNG plus the same
frame with its background intact, and a manifest carrying author and licence per
file.

Headshots exist here because freely licensed ones do not: the posed shots belong
to ATP, WTA, World Tennis and the agencies. What reaches Commons is stand and
press-room photography, from which a headshot has to be cropped.

This is a stopgap. Once a proper headshot subscription is in place the repo goes
away — the attribution obligations do not. See [Licensing](#licensing).

## Usage

```bash
uv sync

uv run roster.py --check          # verify the Wikidata ids first
uv run roster.py -n 400
uv run headshots.py out/players.csv
```

The first headshot run downloads the cutout model (~900 MB, cached in
`~/.rembg/models`) and the YuNet face detector (~232 KB, cached under `out/`).

### roster.py

| Flag | Purpose |
|---|---|
| `--check` | resolve every Wikidata id this script uses, print what each one actually is, exit |
| `--tour` | require an `atp` id, a `wta` id, or either (default: `both`) |
| `-n` | how many players to keep, ranked by Wikipedia reach (default: 400) |
| `--min-sitelinks` | drop players below this many Wikipedia versions (default: 5) |

Run `--check` before trusting a roster. Wikidata property numbers are easy to
misremember — `P536` is the ATP player id while `P2642` is the Billie Jean King
Cup, and `P321` is country-for-sport while `P1626` is the IOC code. The check
resolves all of them against live labels and reports mismatches.

Two outputs land in `out/`: `roster.csv` with every column, and `players.csv`
with just `qid,name` for the next step.

### headshots.py

Accepts either `players.csv` from `roster.py` or a plain file with one name per
line. With a `qid` column present the Wikidata name search is skipped, which
removes the same-name failure mode entirely — prefer the CSV.

| Flag | Purpose |
|---|---|
| `--model` | cutout model. `birefnet-general` beat `birefnet-portrait` on match photography |
| `--compare` | run several models over the same crops and write `out/compare.html` |
| `--face-scale` | crop width in face widths. Lower is tighter, leaving less background to misread |
| `--min-face` | reject sources whose face is narrower than this, in px |
| `--top-pad` | headroom above the face |
| `--force` | re-download cached sources |

`--help` lists the rest.

## Output

```
out/roster.csv                full roster with names, country, tour ids
out/players.csv               qid,name — input for headshots.py
out/headshots/<slug>.png      cutout with alpha
out/headshots/<slug>.jpg      same frame, background intact
out/headshots/manifest.json   author, licence, modifications
out/review.html               visual check against a checkerboard
out/raw/                      Commons originals
out/yunet.onnx                face detector weights
```

`out/` is gitignored.

Each run ends with counts: how many headshots were produced and how many came
out without warnings. The second number is the honest one — it excludes cutouts
where the alpha channel swallowed almost everything or failed to separate the
background at all. Coverage thins out sharply below the top tiers, so a fallback
avatar built from initials and a flag is the normal state for much of a roster,
not an error path.

## Licensing

The photos are free of charge, not free of obligations.

**CC BY** requires the author's name, a link to the licence, and a statement
that the work was modified. Cropping and background removal both count; the
manifest records them as `modifications: ["cropped", "background_removed"]`.

**CC BY-SA** additionally requires the cutout itself to stay under BY-SA. The
manifest flags this as `share_alike`. It applies to the image, not to
surrounding page content.

Attribution is stored as structured fields rather than a rendered string,
because the consuming service is multilingual. Author names and licence
identifiers are not translated; only the surrounding wording is.

Personality rights are a separate layer and are unaffected by the licence. The
manifest's `restrictions` field occasionally carries a Commons note about them.

### Do not lose the manifest

`manifest.json` lives under the gitignored `out/`. On import it must be written
into `player_photo` as rows, not copied as a folder. Otherwise the PNGs outlive
their provenance and become unattributable — exactly the problem that rules out
crowd-sourced artwork sources in the first place.

The scripts are disposable; the table is not. Moving to a paid source changes
the `source` value and the licence fields, nothing else.

## Deliberately absent

No CI, no Renovate. These run a handful of times a year and a weekly dependency
chore for them would not pay for itself. Run `uv run ruff check .` by hand.