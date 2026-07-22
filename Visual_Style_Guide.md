# Dissertation Visual Style Guide

**Dissertation:** Beyond the Visibility Score: Auditing Source Prominence, Citation Accuracy, and Metric Stability Across Five Generative Engines

**Author:** Ganenthra Ravindran | MSc Business Analytics | Trinity College Dublin | BU7170

**Purpose:** This document governs every figure, table, and chart produced for the dissertation. All visuals must conform to these specifications without exception.

---

## 1. Engine Colour Palette

Each engine has a single assigned colour used consistently across every visual in the dissertation. No engine colour may be reassigned, swapped, or approximated.

| Engine | Model | Hex Code | RGB | Usage Label |
|---|---|---|---|---|
| ChatGPT | GPT-5.5 | `#2CA058` | (44, 160, 88) | `chatgpt` |
| Claude | Sonnet 4.6 | `#D97706` | (217, 119, 6) | `claude` |
| Gemini | Flash 3.5 | `#2563EB` | (37, 99, 235) | `gemini` |
| Kimi | K2.6 | `#7C3AED` | (124, 58, 237) | `kimi` |
| Perplexity | Sonar | `#0891B2` | (8, 145, 178) | `perplexity` |

**Palette rationale:** Five hues separated by a minimum of 40 degrees in hue space. Tested for deuteranopia and protanopia distinguishability. All five remain separable when printed in greyscale due to luminance differentiation.

### Supplementary Colours

| Role | Hex Code | RGB | When to Use |
|---|---|---|---|
| Primary accent | `#1F4E79` | (31, 78, 121) | Single-series bars, histograms, and query-composition charts (primary series at 100% opacity) |
| Neutral / secondary | `#6B7280` | (107, 114, 128) | De-emphasised elements only: "other/undefined" categories, comparison baselines, and cross-engine aggregate indicators. Not for primary single-series bars. |
| Reference line / threshold highlight | `#DC2626` | (220, 38, 38) | Reference lines, mechanical ceilings, and critical thresholds exclusively. Also the highlight colour for a single bar crossing a threshold in a `#1F4E79` bar chart. Never used as a fill across a series or as a sequential ramp. |
| Background grid | `#E5E7EB` | (229, 231, 235) | Axis gridlines only; never solid background fills |
| Confidence band | Engine colour at 20% opacity | — | Shaded regions around means or fitted lines |
| Heatmap no-data | `#D1D5DB` | (209, 213, 219) | NaN cells in heatmaps only; always annotated "—". This is the only sanctioned use of grey inside a heatmap. |

### Python Implementation

```python
ENGINE_COLOURS = {
    "chatgpt":    "#2CA058",
    "claude":     "#D97706",
    "gemini":     "#2563EB",
    "kimi":       "#7C3AED",
    "perplexity": "#0891B2",
}

ENGINE_ORDER = ["chatgpt", "claude", "gemini", "kimi", "perplexity"]

COLOUR_PRIMARY   = "#1F4E79"   # single-series primary bars and histograms
COLOUR_NEUTRAL   = "#6B7280"   # secondary / de-emphasised elements only
COLOUR_REFERENCE = "#DC2626"   # reference lines, ceilings, thresholds; threshold-highlight bars
COLOUR_GRID      = "#E5E7EB"
```

Engine order is alphabetical and fixed. Every grouped bar chart, legend, and faceted panel follows this order left-to-right or top-to-bottom.

---

## 2. Typography

A single font family is used across all visuals. No exceptions.

| Element | Font | Weight | Size (pt) |
|---|---|---|---|
| Figure title | Calibri | Bold | 14 |
| Axis titles | Calibri | Semi-Bold | 12 |
| Axis tick labels | Calibri | Regular | 11 |
| Legend labels | Calibri | Regular | 11 |
| Annotation text | Calibri | Regular | 10 |
| Statistical annotations (p-values, ρ) | Calibri | Italic | 10 |
| Panel labels (A, B, C) | Calibri | Bold | 14 |

**Fallback stack (matplotlib):** If Calibri is unavailable on the rendering system, fall back to `DejaVu Sans` then `Arial`. Never fall back to a serif font.

```python
import matplotlib as mpl
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Calibri", "DejaVu Sans", "Arial"]
```

**Rules:**

- All text in visuals is sentence case. Never use FULL CAPITALS for axis labels or titles.
- Metric names in axis labels use their abbreviation followed by the full name in parentheses on first appearance per figure (see §11 for canonical forms). Subsequent axes in the same figure may use the abbreviation alone.
- Engine names on axes are always capitalised as shown in the palette table: ChatGPT, Claude, Gemini, Kimi, Perplexity. Never lowercase, never abbreviated.

---

## 3. Figure and Table Numbering

Per Trinity BU7170 guidelines, all figures and tables are numbered by chapter and sequence within that chapter.

| Element | Format | Example |
|---|---|---|
| Figure | Figure [chapter].[sequence] | Figure 3.2 |
| Table | Table [chapter].[sequence] | Table 4.1 |

Every figure and table must have a descriptive caption that makes it interpretable without reading the surrounding text.

### Working-Label Rule

Internal cell codes (E1, E13, etc.) and working figure codes must never appear in rendered figure titles. Figure titles in the final document are descriptive only. Sequential "Figure [chapter].[n]" numbering is applied via the caption, not baked into the plotted title.

---

## 4. Layout and Dimensions

### Single-panel figures

| Property | Value |
|---|---|
| Width | 7 inches (17.8 cm) |
| Height | 4.5 inches (11.4 cm) |
| DPI (export) | 300 |
| Export format | PNG (for Word insertion) and PDF (for archival) |

### Multi-panel figures (faceted)

| Property | Value |
|---|---|
| Panel width | 3.3 inches per panel |
| Panel height | 3.3 inches per panel |
| Maximum panels per figure | 6 (2 columns × 3 rows or 3 columns × 2 rows) |
| Panel label position | Top-left corner, inside the plot area, bold 14pt |

### Margins and spacing

| Property | Value |
|---|---|
| Left margin (y-axis label clearance) | 0.85 inches |
| Bottom margin (x-axis label clearance) | 0.75 inches |
| Inter-panel gap | 0.4 inches horizontal, 0.5 inches vertical |

---

## 5. Axes, Gridlines, and Spines

| Property | Specification |
|---|---|
| Spine visibility | Left and bottom only. Top and right spines removed. |
| Spine colour | `#374151` (dark grey, not black) |
| Spine weight | 0.8pt |
| Gridlines | Horizontal only (y-axis). Colour `#E5E7EB`, weight 0.5pt, style dashed. |
| Tick direction | Outward |
| Tick length | 4pt |
| Axis label padding | 10pt from axis |

```python
def apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#374151")
    ax.spines["bottom"].set_color("#374151")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4, labelsize=11)
```

---

## 6. Legends

| Property | Specification |
|---|---|
| Position | Outside the plot area, top-right or below the figure. Never overlapping data. |
| Frame | No border, no background fill. |
| Orientation | Horizontal when five or fewer items; vertical otherwise. |
| Order | Alphabetical engine order as defined in Section 1. |
| Marker size in legend | Minimum 8pt, matching the marker used in the plot. |

---

## 7. Chart-Specific Conventions

### Single-series bar charts, histograms, and query-composition charts

- Primary series colour: `COLOUR_PRIMARY` (`#1F4E79`) at 100% opacity.
- Secondary or de-emphasised categories (e.g. "undefined", "other"): `COLOUR_NEUTRAL` (`#6B7280`).
- When a single bar needs to signal that a threshold has been crossed, colour that bar `COLOUR_REFERENCE` (`#DC2626`); all other bars remain `#1F4E79`.
- Bar width: 0.6 (relative units).
- Value labels placed above each bar in Calibri Regular 10pt when the number of bars is five or fewer.
- Horizontal orientation preferred when category labels are long (avoids rotated text).

### Engine-grouped bar charts

- Bars coloured by engine using `ENGINE_COLOURS` as defined in Section 1.
- Bar width: 0.6 (relative units).
- Engine order follows `ENGINE_ORDER` left-to-right.

### Box plots

- Median line: engine colour, 2pt weight.
- Box fill: engine colour at 30% opacity.
- Whiskers: engine colour, 1pt weight, cap width 0.15.
- Outliers: engine colour, marker `o`, size 4pt, 50% opacity.

### Scatter plots

- Marker: filled circle (`o`), size 30pt.
- Marker colour: engine colour at 70% opacity.
- Edge colour: engine colour at 100% opacity, 0.5pt.
- When overlapping is expected, reduce opacity to 40% and add a kernel density contour or marginal histograms.

### Heatmaps

- Colour scale: `CMAP_SEQUENTIAL` — a sequential single-hue teal-blue ramp anchored `#EAF0F7` (near-white) → `#5B8AC4` (mid-blue) → `#1F4E79` (dark teal). See §8 for the full implementation.
- NaN / no-data cells: rendered in `#D1D5DB` (mid-grey) via `CMAP_SEQUENTIAL.set_bad("#D1D5DB")`, always annotated with the string "—". This mid-grey is the **only** sanctioned use of grey inside a heatmap.
- Cell annotations: Calibri Regular 10pt, white text on dark cells (`v > 0.5` on the normalised scale), `#1F2937` dark text on light cells.
- **Prohibited colour scales for heatmaps:** (a) `Reds` or any red sequential ramp — red is reserved for reference lines and must not be used as a magnitude scale; (b) any engine-palette colour as a ramp — that would falsely imply an engine identity; (c) rainbow or jet colormaps.

### Line charts and bump charts

- Line weight: 2pt.
- Line colour: engine colour.
- Marker at each data point: filled circle, 5pt.
- When lines cross, ensure the topmost line is drawn last (z-order matches the engine with the highest value at that point).

---

## 8. Colour Scales

### Sequential scale — `CMAP_SEQUENTIAL`

Used for all magnitude-only data encoded by colour: heatmaps, choropleth cells, and any single-metric colour gradient. Anchored in the brand teal-blue family for visual harmony with the dissertation document.

| Anchor | Role | Hex |
|---|---|---|
| Low | Near-white | `#EAF0F7` |
| Mid | Mid-blue | `#5B8AC4` |
| High | Dark teal | `#1F4E79` |

```python
from matplotlib.colors import LinearSegmentedColormap

CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "teal_sequential",
    ["#EAF0F7", "#5B8AC4", "#1F4E79"]
)
CMAP_SEQUENTIAL.set_bad("#D1D5DB")  # NaN cells: mid-grey, annotate "—"
```

**Rationale:** Perceptually near-uniform in lightness from low to high; colourblind-safe (blue channel dominant); prints legibly in greyscale; harmonises with the `#1F4E79` document heading colour.

### Divergent scale — `CMAP_DIVERGENT`

Used only for signed quantities that have a meaningful zero centre (e.g. difference-from-mean, raw-vs-cleaned deltas). Must not be used for magnitude-only data.

| Anchor | Role | Hex |
|---|---|---|
| Negative | Amber-brown | `#B45309` |
| Centre (zero) | Near-white | `#F3F4F6` |
| Positive | Dark teal | `#1F4E79` |

```python
CMAP_DIVERGENT = LinearSegmentedColormap.from_list(
    "teal_divergent",
    ["#B45309", "#F3F4F6", "#1F4E79"]
)
```

**Rationale:** Negative and positive poles are visually distinguishable in both colour and greyscale (amber is warm-dark; teal is cool-dark); the near-white centre clearly marks zero.

---

## 9. Accessibility

- All figures must be interpretable in greyscale. Use distinct marker shapes (circle, square, triangle, diamond, cross) as a secondary channel alongside colour when five engines appear in the same scatter plot or line chart.
- Minimum contrast ratio between any data element and the white background: 3:1 (WCAG AA for non-text elements).
- Never use colour alone to convey meaning. Pair colour with position, label, or shape.

### Greyscale marker assignments (secondary channel)

| Engine | Marker |
|---|---|
| ChatGPT | `o` (circle) |
| Claude | `s` (square) |
| Gemini | `^` (triangle up) |
| Kimi | `D` (diamond) |
| Perplexity | `X` (cross) |

---

## 10. Statistical Annotations

- Report p-values as: *p* < 0.001, *p* < 0.01, *p* < 0.05, or exact value to three decimal places when *p* ≥ 0.05.
- Correlation coefficients annotated inside the panel as: ρ = 0.XX (Spearman) or r = 0.XX (Pearson), italic, 10pt.
- Significance brackets above bar charts: thin line (0.5pt, `#374151`), with the p-value or asterisk notation centred above.
- Asterisk convention when used: *** *p* < 0.001, ** *p* < 0.01, * *p* < 0.05, ns = not significant.

---

## 11. Canonical Metric Names and Axis Labels

The following verbatim forms are mandatory on every axis label and figure caption. Invented expansions or informal synonyms are not permitted.

| Metric | Full axis label (first appearance per figure) | Abbreviation-only (subsequent axes in same figure) |
|---|---|---|
| PAWC | `PAWC (position-weighted supported-content mass)` | `PAWC` |
| PAWC_norm | `PAWC_norm (normalised source prominence)` | `PAWC_norm` |
| AIS | `AIS (Attributable-to-Identified-Sources rate)` | `AIS` |
| CV | `CV (coefficient of variation)` | `CV` |

**Prohibited expansions:**
- "attribution integrity score" — do not use anywhere (neither in axis labels nor in figure captions nor in analysis text).
- Any other expansion of AIS not listed above.

**Abbreviation rule:** The abbreviation alone may be used on secondary axes within a multi-panel figure once the full form has appeared at least once in that figure (e.g. in the first panel's y-axis label or in the figure-level title).

---

## 12. Export and File Naming

| Property | Specification |
|---|---|
| Format | PNG at 300 DPI (primary for Word insertion), PDF (archival copy) |
| Filename pattern | `fig_[chapter]_[sequence]_[short_description].png` |
| Example | `fig_3_02_pawc_norm_by_engine.png` |
| Colour profile | sRGB |
| Background | Transparent (PNG) or white (PDF) |

---

## 13. Matplotlib Global Configuration Block

Apply this at the top of every analysis script before generating any figure.

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# --- Font ---
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Calibri", "DejaVu Sans", "Arial"]
mpl.rcParams["font.size"] = 11

# --- Axes ---
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.edgecolor"] = "#374151"
mpl.rcParams["axes.linewidth"] = 0.8
mpl.rcParams["axes.labelpad"] = 10
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.labelweight"] = "medium"

# --- Grid ---
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.grid.axis"] = "y"
mpl.rcParams["grid.color"] = "#E5E7EB"
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["grid.linestyle"] = "--"
mpl.rcParams["axes.axisbelow"] = True

# --- Ticks ---
mpl.rcParams["xtick.direction"] = "out"
mpl.rcParams["ytick.direction"] = "out"
mpl.rcParams["xtick.major.size"] = 4
mpl.rcParams["ytick.major.size"] = 4
mpl.rcParams["xtick.labelsize"] = 11
mpl.rcParams["ytick.labelsize"] = 11

# --- Legend ---
mpl.rcParams["legend.frameon"] = False
mpl.rcParams["legend.fontsize"] = 11

# --- Figure ---
mpl.rcParams["figure.figsize"] = (7, 4.5)
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0.15

# --- Engine palette ---
ENGINE_COLOURS = {
    "chatgpt":    "#2CA058",
    "claude":     "#D97706",
    "gemini":     "#2563EB",
    "kimi":       "#7C3AED",
    "perplexity": "#0891B2",
}

ENGINE_ORDER = ["chatgpt", "claude", "gemini", "kimi", "perplexity"]

ENGINE_LABELS = {
    "chatgpt":    "ChatGPT",
    "claude":     "Claude",
    "gemini":     "Gemini",
    "kimi":       "Kimi",
    "perplexity": "Perplexity",
}

ENGINE_MARKERS = {
    "chatgpt":    "o",
    "claude":     "s",
    "gemini":     "^",
    "kimi":       "D",
    "perplexity": "X",
}

# --- Supplementary colours ---
COLOUR_PRIMARY   = "#1F4E79"   # single-series primary bars and histograms
COLOUR_NEUTRAL   = "#6B7280"   # secondary / de-emphasised elements only
COLOUR_REFERENCE = "#DC2626"   # reference lines, ceilings, thresholds; threshold-highlight bars
COLOUR_GRID      = "#E5E7EB"

# --- Sequential heatmap ramp (teal-blue) ---
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "teal_sequential",
    ["#EAF0F7", "#5B8AC4", "#1F4E79"]
)
CMAP_SEQUENTIAL.set_bad("#D1D5DB")  # NaN cells rendered mid-grey; annotate with "—"

# --- Divergent ramp (amber-brown → white → teal) for signed quantities ---
CMAP_DIVERGENT = LinearSegmentedColormap.from_list(
    "teal_divergent",
    ["#B45309", "#F3F4F6", "#1F4E79"]
)
```

---

*Last updated: 26 June 2026*
