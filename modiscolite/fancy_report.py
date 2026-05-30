"""Self-contained, interactive TF-MoDISco report generation.

This module implements the ``modisco report-fancy`` workflow. It generates
trimmed MoDISco logos, annotates patterns with TOMTOM-lite, embeds motif logos
as base64 images, and writes a standalone HTML report plus a TSV summary.
"""

import base64
import html as html_lib
import inspect
import io
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


CANONICAL_GROUPS = ("pos_patterns", "neg_patterns")
BASES = ("A", "C", "G", "T")

REQUIRED_LOGO_FILES = (
    "trimmed_cwm_fwd_logo.png",
    "trimmed_cwm_rev_logo.png",
)

LOGO_FILES = {
    "cwm_fwd": "trimmed_cwm_fwd_logo.png",
    "cwm_rev": "trimmed_cwm_rev_logo.png",
    "ic_ppm": "ic_ppm_logo.png",
}

FLAT_LOGO_NAME_MAP = {
    "cwm.fwd.png": "trimmed_cwm_fwd_logo.png",
    "cwm.rev.png": "trimmed_cwm_rev_logo.png",
    "trimmed_cwm_fwd_logo.png": "trimmed_cwm_fwd_logo.png",
    "trimmed_cwm_rev_logo.png": "trimmed_cwm_rev_logo.png",
    "cwm_logo.png": "cwm_logo.png",
    "hcwm_logo.png": "hcwm_logo.png",
    "ic_ppm_logo.png": "ic_ppm_logo.png",
}

FLAT_LOGO_FILES = {
    "cwm_fwd": "cwm.fwd.png",
    "cwm_rev": "cwm.rev.png",
}

IUPAC_MAP = {
    frozenset(["A"]): "A",
    frozenset(["C"]): "C",
    frozenset(["G"]): "G",
    frozenset(["T"]): "T",
    frozenset(["A", "G"]): "R",
    frozenset(["C", "T"]): "Y",
    frozenset(["G", "C"]): "S",
    frozenset(["A", "T"]): "W",
    frozenset(["G", "T"]): "K",
    frozenset(["A", "C"]): "M",
    frozenset(["C", "G", "T"]): "B",
    frozenset(["A", "G", "T"]): "D",
    frozenset(["A", "C", "T"]): "H",
    frozenset(["A", "C", "G"]): "V",
    frozenset(["A", "C", "G", "T"]): "N",
}

FAMILY_COLORS = {
    "SP/KLF": "#2196F3",
    "CTCF/CTCFL": "#F44336",
    "NF-Y": "#4CAF50",
    "FOX": "#FF9800",
    "ETS": "#9C27B0",
    "ZIC/ZNF": "#795548",
    "NRF/EGR": "#00BCD4",
    "HNF4/NR": "#E91E63",
    "AP-1/bZIP": "#FF5722",
    "TEAD": "#607D8B",
    "YY1/YY2": "#009688",
    "CEBP": "#8BC34A",
    "HNF1": "#FFC107",
    "THAP11": "#673AB7",
    "Other/ZNF": "#9E9E9E",
}

EMBEDDED_CSS = """
*, *::before, *::after { box-sizing: border-box; }
html { font-size: 14px; scroll-behavior: smooth; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  background: #f5f6f7;
  color: #222;
}
#sidebar {
  width: 230px;
  min-width: 230px;
  background: #1a1a2e;
  color: #ccc;
  position: fixed;
  top: 0;
  left: 0;
  bottom: 0;
  overflow-y: auto;
  z-index: 100;
  padding-bottom: 40px;
}
#sidebar-header {
  background: #007559;
  color: #fff;
  padding: 14px 12px 10px;
  font-weight: 700;
  line-height: 1.3;
  border-bottom: 2px solid #005a44;
}
#sidebar-header small {
  display: block;
  font-weight: 400;
  font-size: 0.8em;
  opacity: 0.85;
  margin-top: 3px;
}
#search-box {
  width: calc(100% - 16px);
  margin: 8px;
  padding: 6px 8px;
  border: 0;
  border-radius: 4px;
  background: #2a2a4a;
  color: #eee;
  font-size: 0.85em;
  outline: 0;
}
#search-box::placeholder { color: #888; }
.sidebar-section-label {
  padding: 8px 12px 4px;
  font-size: 0.72em;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #888;
  font-weight: 600;
  margin-top: 6px;
}
.sidebar-link {
  display: flex;
  align-items: center;
  padding: 5px 10px 5px 9px;
  text-decoration: none;
  color: #ccc;
  font-size: 0.8em;
  transition: background 0.15s;
  gap: 6px;
}
.sidebar-link:hover { background: #2a2a4a; color: #fff; }
.sl-id { font-weight: 700; min-width: 48px; color: #eee; }
.sl-tf { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #aaa; }
.sl-n { font-size: 0.85em; color: #777; white-space: nowrap; }
#main {
  margin-left: 230px;
  min-height: 100vh;
}
#page-header {
  background: linear-gradient(135deg, #007559 0%, #005a44 100%);
  color: #fff;
  padding: 28px 36px 22px;
}
#page-header h1 {
  font-size: 1.6em;
  font-weight: 700;
  margin: 0 0 6px;
}
#page-header .subtitle {
  font-size: 0.9em;
  opacity: 0.86;
  margin-bottom: 14px;
}
.header-stats {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
  margin-top: 10px;
}
.hstat {
  background: rgba(255, 255, 255, 0.15);
  border-radius: 6px;
  padding: 6px 14px;
  font-size: 0.85em;
}
.hstat strong { display: block; font-size: 1.3em; }
.family-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 12px 36px;
  background: #fff;
  border-bottom: 1px solid #e0e0e0;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 0.8em;
  color: #444;
}
.legend-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  flex-shrink: 0;
}
#tab-bar {
  display: flex;
  gap: 0;
  background: #fff;
  border-bottom: 2px solid #007559;
  padding: 0 36px;
  position: sticky;
  top: 0;
  z-index: 50;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
}
.tab-btn {
  padding: 10px 24px;
  border: 0;
  background: none;
  font-size: 0.95em;
  font-weight: 600;
  color: #666;
  cursor: pointer;
  border-bottom: 3px solid transparent;
  margin-bottom: -2px;
  transition: all 0.15s;
}
.tab-btn.active {
  color: #007559;
  border-bottom-color: #007559;
}
.tab-btn:hover { color: #007559; }
.tab-controls {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}
.ctrl-btn {
  padding: 5px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #fff;
  font-size: 0.8em;
  cursor: pointer;
  color: #555;
  transition: all 0.15s;
}
.ctrl-btn:hover { background: #007559; color: #fff; border-color: #007559; }
.section-content { padding: 20px 36px 40px; }
.section-content.hidden { display: none; }
.section-heading {
  font-size: 1.2em;
  font-weight: 700;
  color: #007559;
  margin-bottom: 14px;
  padding-bottom: 6px;
  border-bottom: 2px solid #007559;
}
.section-subheading {
  font-size: 0.85em;
  color: #666;
  margin-bottom: 18px;
}
.summary-table-wrap {
  overflow-x: auto;
  margin-bottom: 28px;
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
}
.summary-table {
  width: 100%;
  border-collapse: collapse;
  background: #fff;
  font-size: 0.82em;
}
.summary-table th {
  background: #007559;
  color: #fff;
  padding: 8px 10px;
  text-align: left;
  font-weight: 600;
  white-space: nowrap;
  cursor: pointer;
  user-select: none;
}
.summary-table th:hover { background: #005a44; }
.summary-table th::after { content: " ^v"; opacity: 0.45; font-size: 0.8em; }
.summary-table td {
  padding: 6px 10px;
  border-bottom: 1px solid #f0f0f0;
  vertical-align: middle;
}
.summary-table tr:hover td { background: #f9fff9; }
.summary-match-cell { min-width: 140px; line-height: 1.4; }
.pattern-link { text-decoration: none; font-family: monospace; }
.pattern-link:hover { text-decoration: underline; }
.num-col { text-align: right; white-space: nowrap; }
.pattern-card {
  background: #fff;
  border-radius: 8px;
  margin-bottom: 10px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
  overflow: hidden;
  transition: box-shadow 0.15s;
}
.pattern-card:hover { box-shadow: 0 2px 10px rgba(0, 0, 0, 0.13); }
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px 10px 11px;
  cursor: pointer;
  background: #fafafa;
  transition: background 0.15s;
  gap: 12px;
}
.card-header:hover { background: #f0f8f5; }
.card-title {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.pattern-id {
  font-family: monospace;
  font-size: 1.05em;
  font-weight: 700;
  color: #222;
}
.top-tf-hint { font-size: 0.85em; color: #666; }
.card-meta {
  display: flex;
  align-items: center;
  gap: 14px;
  font-size: 0.82em;
  color: #777;
  white-space: nowrap;
}
.seqlet-count { font-weight: 600; color: #444; }
.toggle-icon { font-size: 0.9em; transition: transform 0.2s; }
.toggle-icon.open { transform: rotate(180deg); }
.card-body {
  padding: 16px 20px 20px;
  border-top: 1px solid #f0f0f0;
}
.logo-row {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.logo-box {
  flex: 1;
  min-width: 180px;
  background: #f8f8f8;
  border-radius: 6px;
  padding: 8px 10px;
  text-align: center;
}
.logo-label {
  font-size: 0.75em;
  color: #888;
  margin-bottom: 6px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.pattern-logo {
  max-width: 100%;
  max-height: 80px;
  object-fit: contain;
}
.no-logo { color: #999; font-size: 0.85em; }
.stats-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.stat-box {
  background: #f5f5f5;
  border-radius: 6px;
  padding: 8px 14px;
  min-width: 90px;
}
.consensus-box { flex: 2; min-width: 200px; }
.stat-label {
  font-size: 0.7em;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 3px;
}
.stat-value {
  font-size: 1.05em;
  font-weight: 600;
  color: #333;
}
.consensus-seq {
  font-family: monospace;
  font-size: 0.85em;
  word-break: break-all;
  letter-spacing: 0.05em;
}
.matches-title {
  font-size: 0.8em;
  font-weight: 700;
  color: #007559;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 8px;
}
.match-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82em;
}
.match-table th {
  background: #e8f5f0;
  color: #007559;
  padding: 6px 10px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #007559;
}
.match-table td {
  padding: 6px 10px;
  border-bottom: 1px solid #f0f0f0;
  vertical-align: middle;
}
.match-table tr:hover td { background: #f9fff9; }
.rank-col { width: 28px; text-align: center; color: #aaa; font-weight: 700; }
.jaspar-id-col { width: 110px; }
.tf-name-col { width: 130px; }
.family-col { width: 130px; }
.pval-col { width: 100px; text-align: right; font-family: monospace; }
.logo-col { min-width: 120px; }
.jaspar-logo {
  max-height: 36px;
  max-width: 160px;
  object-fit: contain;
  vertical-align: middle;
}
.family-badge { display: inline-block; }
.show-more-btn {
  background: none;
  border: 1px solid #ccc;
  border-radius: 4px;
  padding: 3px 10px;
  font-size: 0.8em;
  cursor: pointer;
  color: #666;
  margin: 4px 0;
}
.show-more-btn:hover { background: #007559; color: #fff; border-color: #007559; }
.no-matches { color: #888; font-style: italic; font-size: 0.85em; }
.inner-extra { margin: 0; }
.inner-extra td { background: #fff; }
#sidebar::-webkit-scrollbar { width: 5px; }
#sidebar::-webkit-scrollbar-track { background: #1a1a2e; }
#sidebar::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
@media (max-width: 768px) {
  #sidebar { display: none; }
  #main { margin-left: 0; }
  #page-header { padding: 22px 18px 18px; }
  #tab-bar { padding: 0 12px; overflow-x: auto; }
  .section-content { padding: 12px 16px 30px; }
  .tab-controls { display: none; }
  .card-header { align-items: flex-start; flex-direction: column; }
}
"""

EMBEDDED_JS = """
function switchTab(tab) {
  document.getElementById('section-pos').classList.toggle('hidden', tab !== 'pos');
  document.getElementById('section-neg').classList.toggle('hidden', tab !== 'neg');
  document.getElementById('tab-pos').classList.toggle('active', tab === 'pos');
  document.getElementById('tab-neg').classList.toggle('active', tab === 'neg');
}
function toggleCard(anchor) {
  const body = document.getElementById('body-' + anchor);
  const icon = document.getElementById('icon-' + anchor);
  if (!body || !icon) return;
  const isOpen = body.style.display !== 'none';
  body.style.display = isOpen ? 'none' : 'block';
  icon.classList.toggle('open', !isOpen);
}
function expandAll() {
  const activeSection = document.querySelector('.section-content:not(.hidden)');
  if (!activeSection) return;
  activeSection.querySelectorAll('.card-body').forEach(b => b.style.display = 'block');
  activeSection.querySelectorAll('.toggle-icon').forEach(i => i.classList.add('open'));
}
function collapseAll() {
  const activeSection = document.querySelector('.section-content:not(.hidden)');
  if (!activeSection) return;
  activeSection.querySelectorAll('.card-body').forEach(b => b.style.display = 'none');
  activeSection.querySelectorAll('.toggle-icon').forEach(i => i.classList.remove('open'));
}
function toggleExtra(anchor) {
  const el = document.getElementById('extra-' + anchor);
  if (!el) return;
  const isHidden = el.style.display === 'none';
  el.style.display = isHidden ? 'table-row' : 'none';
  const btn = document.getElementById('more-' + anchor);
  if (btn) btn.textContent = isHidden ? 'Show fewer' : btn.dataset.moreText;
}
function filterSidebar(query) {
  const q = query.toLowerCase();
  document.querySelectorAll('.sidebar-link').forEach(link => {
    const text = link.textContent.toLowerCase();
    link.style.display = (!q || text.includes(q)) ? 'flex' : 'none';
  });
}
function sortTable(tableId, col) {
  const table = document.getElementById(tableId);
  if (!table) return;
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const asc = table.dataset.sortCol == col && table.dataset.sortDir === 'asc';
  table.dataset.sortCol = col;
  table.dataset.sortDir = asc ? 'desc' : 'asc';
  rows.sort((a, b) => {
    const av = a.cells[col] ? a.cells[col].textContent.trim() : '';
    const bv = b.cells[col] ? b.cells[col].textContent.trim() : '';
    const an = parseFloat(av.replace(/,/g, '').replace('%', ''));
    const bn = parseFloat(bv.replace(/,/g, '').replace('%', ''));
    if (!isNaN(an) && !isNaN(bn)) return asc ? bn - an : an - bn;
    return asc ? bv.localeCompare(av) : av.localeCompare(bv);
  });
  rows.forEach(r => tbody.appendChild(r));
}
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', function() {
    const anchor = this.getAttribute('href').slice(1);
    const card = document.getElementById(anchor);
    if (card && card.dataset.group) switchTab(card.dataset.group);
    const body = document.getElementById('body-' + anchor);
    if (body && body.style.display === 'none') toggleCard(anchor);
  });
});
"""


def esc(value) -> str:
    """HTML-escape a scalar value."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return html_lib.escape(str(value), quote=True)


def default_logos_dir(output_path: os.PathLike) -> Path:
    """Return the default logo directory for an output HTML path."""
    output = Path(output_path)
    return Path(str(output.with_suffix("")) + "_logos")


def safe_rmtree(path: Path) -> None:
    """Remove a logo directory while refusing filesystem roots and files."""
    resolved = path.resolve()
    if not resolved.exists():
        return
    if not resolved.is_dir():
        raise RuntimeError("Cannot remove non-directory logo path: {}".format(resolved))
    if resolved.parent == resolved:
        raise RuntimeError("Refusing to remove filesystem root: {}".format(resolved))
    shutil.rmtree(str(resolved))


def call_with_supported_kwargs(fn, kwargs: Dict):
    """Call a helper while filtering kwargs for older or newer APIs."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if not accepts_kwargs:
        kwargs = {
            key: value for key, value in kwargs.items()
            if key in signature.parameters
        }
    return fn(**kwargs)


def pattern_sort_key(name: str) -> Tuple[int, object]:
    """Sort pattern names by trailing numeric index when present."""
    try:
        return 0, int(str(name).split("_")[-1])
    except ValueError:
        return 1, str(name)


def ppm_to_iupac(seq_matrix: np.ndarray, threshold: float = 0.3) -> str:
    """Convert an ``(L, 4)`` PPM/one-hot matrix to an IUPAC consensus."""
    result = []
    arr = np.asarray(seq_matrix)
    if arr.ndim != 2:
        return ""
    if arr.shape[0] == 4 and arr.shape[1] != 4:
        arr = arr.T
    if arr.shape[1] != 4:
        return ""
    for pos in arr:
        max_val = float(np.max(pos))
        if max_val < 0.05:
            result.append("N")
            continue
        present = frozenset(
            BASES[i] for i in range(4) if float(pos[i]) >= threshold * max_val
        )
        result.append(IUPAC_MAP.get(present, "N"))
    return "".join(result)


def load_h5_patterns(h5_path: os.PathLike) -> List[Dict]:
    """Read all positive and negative patterns from a TF-MoDISco HDF5 file."""
    patterns = []
    with h5py.File(h5_path, "r") as h5:
        for group_name in CANONICAL_GROUPS:
            if group_name not in h5:
                continue
            group = h5[group_name]
            for pattern_name in sorted(group.keys(), key=pattern_sort_key):
                pattern = group[pattern_name]
                contrib = np.asarray(pattern["contrib_scores"][:])
                sequence = np.asarray(pattern["sequence"][:])
                seqlets = pattern["seqlets"]
                n_seqlets = int(np.asarray(seqlets["n_seqlets"][:]).ravel()[0])
                is_revcomp = (
                    np.asarray(seqlets["is_revcomp"][:])
                    if "is_revcomp" in seqlets else np.array([])
                )
                pct_revcomp = float(is_revcomp.mean() * 100.0) if is_revcomp.size else 0.0
                patterns.append({
                    "tag": "{}.{}".format(group_name, pattern_name),
                    "group": group_name,
                    "pattern": pattern_name,
                    "n_seqlets": n_seqlets,
                    "pct_revcomp": pct_revcomp,
                    "mean_abs_contrib": float(np.abs(contrib).mean()),
                    "iupac": ppm_to_iupac(sequence),
                    "length": int(contrib.shape[0]),
                    "sequence": sequence,
                })
    return patterns


def normalize_logo_layout(logos_dir: os.PathLike, pattern_tags: Iterable[str]) -> int:
    """Move flat logo files into per-pattern directories used by the report."""
    logos_path = Path(logos_dir)
    n_normalized = 0
    for tag in pattern_tags:
        pattern_dir = logos_path / tag
        pattern_dir.mkdir(parents=True, exist_ok=True)
        for flat_name, report_name in FLAT_LOGO_NAME_MAP.items():
            src = logos_path / "{}.{}".format(tag, flat_name)
            if not src.is_file():
                continue
            dst = pattern_dir / report_name
            src.replace(dst)
            n_normalized += 1
    return n_normalized


def summarize_logo_layout(logos_dir: os.PathLike, pattern_tags: Iterable[str]) -> Dict:
    """Return a small diagnostic summary for a generated logo directory."""
    logos_path = Path(logos_dir)
    tags = list(pattern_tags)
    if not logos_path.exists():
        return {
            "pattern_dirs": [],
            "extra_dirs": [],
            "top_level_pngs": [],
            "total_files": 0,
            "missing_required": {tag: list(REQUIRED_LOGO_FILES) for tag in tags},
        }
    all_entries = sorted(path.name for path in logos_path.iterdir())
    all_dirs = [
        entry for entry in all_entries
        if (logos_path / entry).is_dir()
    ]
    top_level_pngs = [
        entry for entry in all_entries
        if entry.lower().endswith(".png") and (logos_path / entry).is_file()
    ]
    expected_tags = set(tags)
    pattern_dirs = [tag for tag in tags if tag in all_dirs]
    extra_dirs = [entry for entry in all_dirs if entry not in expected_tags]
    total_files = 0
    missing_required = {}
    for tag in tags:
        pattern_dir = logos_path / tag
        if not pattern_dir.is_dir():
            missing_required[tag] = list(REQUIRED_LOGO_FILES)
            continue
        present = {
            path.name for path in pattern_dir.iterdir()
            if path.is_file()
        }
        total_files += len(present)
        missing = [name for name in REQUIRED_LOGO_FILES if name not in present]
        if missing:
            missing_required[tag] = missing
    return {
        "pattern_dirs": pattern_dirs,
        "extra_dirs": extra_dirs,
        "top_level_pngs": top_level_pngs,
        "total_files": total_files,
        "missing_required": missing_required,
    }


def generate_logos(
    h5_path: os.PathLike,
    logos_dir: os.PathLike,
    groups: List[str],
    pattern_tags: List[str],
    trim_threshold: float,
    force: bool = False,
) -> Dict:
    """Generate MoDISco logos and normalize them for the fancy report."""
    logos_path = Path(logos_dir)
    if force:
        safe_rmtree(logos_path)
    elif logos_path.exists() and not logos_path.is_dir():
        raise RuntimeError("Logo path exists but is not a directory: {}".format(logos_path))
    elif logos_path.exists() and any(logos_path.iterdir()):
        print("  NOTE: using existing non-empty logos directory: {}".format(logos_path))
        print("        Pass --force-logos to delete it before regeneration.")

    logos_path.mkdir(parents=True, exist_ok=True)

    import modiscolite.report as report_mod

    print("[2/5] Generating MoDISco logos...")
    print("  Logos dir : {}".format(logos_path))
    print("  Groups    : {}".format(groups))
    print("  Trim thr  : {}".format(trim_threshold))

    call_with_supported_kwargs(
        report_mod.create_modisco_logos,
        {
            "modisco_h5py": h5_path,
            "modisco_logo_dir": str(logos_path),
            "trim_threshold": trim_threshold,
            "pattern_groups": groups,
        },
    )
    n_normalized = normalize_logo_layout(logos_path, pattern_tags)
    summary = summarize_logo_layout(logos_path, pattern_tags)
    print("  Normalized flat files: {}".format(n_normalized))
    print("  Pattern directories  : {}".format(len(summary["pattern_dirs"])))
    print("  Total logo files     : {}".format(summary["total_files"]))
    if summary["missing_required"]:
        sample_tag = next(iter(summary["missing_required"]))
        missing = ", ".join(summary["missing_required"][sample_tag])
        print(
            "  WARNING: {} patterns are missing required logos; example {}: {}".format(
                len(summary["missing_required"]), sample_tag, missing
            )
        )
    if summary["top_level_pngs"]:
        shown = ", ".join(summary["top_level_pngs"][:5])
        suffix = "..." if len(summary["top_level_pngs"]) > 5 else ""
        print(
            "  WARNING: {} top-level PNG files were not normalized: {}{}".format(
                len(summary["top_level_pngs"]), shown, suffix
            )
        )
    return summary


def run_tomtom_lite(
    h5_path: os.PathLike,
    meme_db: os.PathLike,
    work_dir: os.PathLike,
    top_n: int,
    trim_threshold: float,
    trim_min_length: int,
    groups: List[str],
) -> pd.DataFrame:
    """Run the bundled TOMTOM-lite implementation and return annotations."""
    import modiscolite.report as report_mod

    print("  Running TOMTOM-lite (top_n={}, trim_threshold={})...".format(
        top_n, trim_threshold
    ))
    return call_with_supported_kwargs(
        report_mod.tomtomlite_dataframe,
        {
            "modisco_h5py": h5_path,
            "output_dir": work_dir,
            "meme_motif_db": meme_db,
            "pattern_groups": groups,
            "top_n_matches": top_n,
            "trim_threshold": trim_threshold,
            "trim_min_length": trim_min_length,
        },
    )


def annotate_patterns_with_tomtom(
    h5_path: os.PathLike,
    meme_db: os.PathLike,
    patterns: List[Dict],
    groups: List[str],
    top_n: int,
    trim_threshold: float,
    trim_min_length: int,
) -> pd.DataFrame:
    """Run TOMTOM-lite and add pattern tags after validating row order."""
    print("[3/5] Running TOMTOM-lite annotation...")
    with tempfile.TemporaryDirectory() as work_dir:
        tomtom_df = run_tomtom_lite(
            h5_path=h5_path,
            meme_db=meme_db,
            work_dir=work_dir,
            top_n=top_n,
            trim_threshold=trim_threshold,
            trim_min_length=trim_min_length,
            groups=groups,
        )
    if tomtom_df is None:
        raise RuntimeError(
            "TOMTOM row count does not match loaded patterns: "
            "tomtomlite_dataframe returned None for {} loaded patterns.".format(
                len(patterns)
            )
        )
    tomtom_df = tomtom_df.reset_index(drop=True)
    if len(tomtom_df) != len(patterns):
        raise RuntimeError(
            "TOMTOM row count does not match loaded patterns: returned {} rows "
            "for {} loaded patterns. Refusing to assign pattern tags.".format(
                len(tomtom_df), len(patterns)
            )
        )
    tomtom_df["tag"] = [p["tag"] for p in patterns]
    print("  Annotated: {} patterns".format(len(tomtom_df)))
    return tomtom_df


def b64_encode_file(path: os.PathLike) -> str:
    """Base64-encode a file; return an empty string if the file is missing."""
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def data_uri_from_b64(b64: str) -> str:
    """Return a PNG data URI for base64 payloads."""
    return "data:image/png;base64,{}".format(b64) if b64 else ""


def render_weights_b64(weights: np.ndarray, figsize: Tuple[float, float],
                       clamp: bool = True) -> str:
    """Render a sequence-logo weight matrix to base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import logomaker
    except Exception as exc:
        print("  WARNING: could not import logo rendering dependencies: {}".format(exc))
        return ""

    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 2:
        return ""
    if arr.shape[0] == 4 and arr.shape[1] != 4:
        arr = arr.T
    if arr.shape[1] != 4:
        return ""

    fig, ax = plt.subplots(figsize=figsize)
    df = pd.DataFrame(arr, columns=BASES)
    logomaker.Logo(df, ax=ax, color_scheme="classic")
    if clamp:
        row_sums = df.sum(axis=1)
        ymin = min(float(row_sums.min()), 0.0)
        ymax = max(float(row_sums.max()), 0.1)
        ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.04)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, transparent=True)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def normalize_ppm(ppm: np.ndarray) -> Optional[np.ndarray]:
    """Normalize a motif array to an ``(L, 4)`` PPM."""
    try:
        arr = np.asarray(ppm, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2:
        return None

    row_error = np.inf
    col_error = np.inf
    if arr.shape[1] == 4:
        row_error = float(np.mean(np.abs(arr.sum(axis=1) - 1.0)))
    if arr.shape[0] == 4:
        col_error = float(np.mean(np.abs(arr.sum(axis=0) - 1.0)))

    if arr.shape[0] == 4 and (arr.shape[1] != 4 or col_error < row_error):
        arr = arr.T
    if arr.shape[1] != 4:
        return None
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return arr / row_sums


def ppm_ic_weights(ppm: np.ndarray) -> Optional[np.ndarray]:
    """Return information-content scaled PPM weights."""
    arr = normalize_ppm(ppm)
    if arr is None:
        return None
    eps = 1e-9
    ic = 2.0 + np.sum(arr * np.log2(arr + eps), axis=1)
    return arr * ic[:, None]


def render_ppm_logo_b64(ppm: np.ndarray) -> str:
    """Render an H5 sequence/PPM matrix as an IC-scaled logo."""
    weights = ppm_ic_weights(ppm)
    if weights is None:
        return ""
    return render_weights_b64(
        weights,
        figsize=(max(2.4, weights.shape[0] * 0.12), 1.0),
        clamp=False,
    )


def load_logos(logos_dir: os.PathLike, patterns: List[Dict]) -> Dict[str, Dict[str, str]]:
    """Base64-encode pattern logos, rendering IC PPM logos when absent."""
    logos_path = Path(logos_dir)
    logos = {}
    missing = 0
    for pattern in patterns:
        tag = pattern["tag"]
        pattern_dir = logos_path / tag
        logos[tag] = {}
        for key, filename in LOGO_FILES.items():
            candidates = [pattern_dir / filename, logos_path / "{}.{}".format(tag, filename)]
            flat_filename = FLAT_LOGO_FILES.get(key)
            if flat_filename:
                candidates.append(logos_path / "{}.{}".format(tag, flat_filename))
            encoded = ""
            for candidate in candidates:
                encoded = b64_encode_file(candidate)
                if encoded:
                    break
            if not encoded and key == "ic_ppm":
                encoded = render_ppm_logo_b64(pattern.get("sequence"))
            logos[tag][key] = encoded
            if not encoded:
                missing += 1
    total = len(patterns) * len(LOGO_FILES)
    print("  Logos encoded: {}/{} ({} missing)".format(total - missing, total, missing))
    return logos


def write_embedded_logo_files(
    logos_dir: os.PathLike,
    patterns: List[Dict],
    logos: Dict[str, Dict[str, str]],
) -> int:
    """Write any embedded-only pattern logos into the report logo directory."""
    logos_path = Path(logos_dir)
    n_written = 0
    for pattern in patterns:
        tag = pattern["tag"]
        pattern_dir = logos_path / tag
        pattern_dir.mkdir(parents=True, exist_ok=True)
        for key, filename in LOGO_FILES.items():
            output_path = pattern_dir / filename
            if output_path.exists():
                continue
            encoded = logos.get(tag, {}).get(key, "")
            if not encoded:
                continue
            if encoded.startswith("data:image"):
                encoded = encoded.split(",", 1)[-1]
            try:
                output_path.write_bytes(base64.b64decode(encoded))
            except (OSError, ValueError):
                continue
            n_written += 1
    return n_written


def normalize_meme_motifs(raw_motifs) -> Dict[str, np.ndarray]:
    """Normalize parser-specific MEME motif objects to name -> ``(L, 4)`` PPM."""
    normalized = {}
    items = raw_motifs.items() if hasattr(raw_motifs, "items") else raw_motifs
    for item in items:
        if isinstance(item, tuple) and len(item) >= 2:
            name, pwm = item[0], item[1]
        else:
            name = getattr(item, "name", None) or getattr(item, "id", None)
            pwm = item
        if not name:
            continue
        candidates = [
            pwm,
            getattr(pwm, "ppm", None),
            getattr(pwm, "pwm", None),
            getattr(pwm, "probabilities", None),
        ]
        for candidate in candidates:
            arr = normalize_ppm(candidate)
            if arr is not None:
                normalized[str(name).strip()] = arr
                break
    return normalized


def parse_meme_motifs_direct(meme_db: os.PathLike) -> Dict[str, np.ndarray]:
    """Small MEME parser fallback for letter-probability matrices."""
    motifs = {}
    current_name = None
    rows = []
    reading_matrix = False

    def flush() -> None:
        if current_name and rows:
            arr = normalize_ppm(np.asarray(rows, dtype=float))
            if arr is not None:
                motifs[current_name] = arr

    try:
        with open(meme_db, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("MOTIF "):
                    flush()
                    current_name = stripped.split(None, 1)[1].strip()
                    rows = []
                    reading_matrix = False
                    continue
                if stripped.startswith("letter-probability matrix"):
                    rows = []
                    reading_matrix = True
                    continue
                if reading_matrix:
                    parts = stripped.split()
                    if len(parts) < 4:
                        flush()
                        rows = []
                        reading_matrix = False
                        continue
                    try:
                        rows.append([float(value) for value in parts[:4]])
                    except ValueError:
                        flush()
                        rows = []
                        reading_matrix = False
            flush()
    except OSError:
        return {}
    return motifs


def load_meme_motifs(meme_db: os.PathLike) -> Dict[str, np.ndarray]:
    """Read motifs from a MEME file, returning name -> ``(L, 4)`` PPM arrays."""
    parser_warnings = []
    try:
        from memelite.io import read_meme
    except ImportError as exc:
        parser_warnings.append("memelite.io.read_meme unavailable: {}".format(exc))
    else:
        try:
            normalized = normalize_meme_motifs(read_meme(meme_db))
            if normalized:
                return normalized
            parser_warnings.append("memelite parser returned no usable motifs")
        except Exception as exc:
            parser_warnings.append("memelite.io.read_meme failed: {}".format(exc))

    normalized = parse_meme_motifs_direct(meme_db)
    if normalized and parser_warnings:
        print("  WARNING: falling back to direct MEME parser after parser issues:")
        for warning in parser_warnings:
            print("    - {}".format(warning))
    if not normalized:
        print("  WARNING: could not parse MEME motifs; match logos will be omitted.")
        for warning in parser_warnings:
            print("    - {}".format(warning))
    return normalized


def meme_lookup_keys(key: str) -> List[str]:
    """Return robust lookup keys for MEME motif names with inconsistent spacing."""
    if key is None:
        return []
    exact = str(key).strip()
    if not exact:
        return []
    normalized = " ".join(exact.split())
    keys = [exact]
    if normalized not in keys:
        keys.append(normalized)
    motif_id = normalized.split(maxsplit=1)[0]
    if motif_id and motif_id not in keys:
        keys.append(motif_id)
    return keys


def find_meme_motif(match_str: str, motifs: Dict[str, np.ndarray]):
    """Find a MEME motif by full TOMTOM match string or motif ID prefix."""
    if match_str is None:
        return None, None
    try:
        if pd.isna(match_str):
            return None, None
    except (TypeError, ValueError):
        pass
    lookup = {}
    for key, ppm in motifs.items():
        for lookup_key in meme_lookup_keys(key):
            lookup.setdefault(lookup_key, (key, ppm))
    for query_key in meme_lookup_keys(str(match_str).strip()):
        if query_key in lookup:
            return lookup[query_key]
    return None, None


def render_motif_logo(match_str: str, motifs: Dict[str, np.ndarray]) -> str:
    """Render a matched database motif to a PNG data URI."""
    key, ppm = find_meme_motif(match_str, motifs)
    if ppm is None:
        return ""
    weights = ppm_ic_weights(ppm)
    if weights is None:
        return ""
    try:
        b64 = render_weights_b64(weights, figsize=(2.35, 1.02), clamp=False)
        return data_uri_from_b64(b64)
    except Exception as exc:
        print("  WARNING: could not render motif logo for {}: {}".format(key, exc))
        return ""


def load_match_logos(
    meme_db: os.PathLike,
    tomtom_df: pd.DataFrame,
    top_n: int,
) -> Dict[str, str]:
    """Render and cache TOMTOM match logos as data URIs."""
    if tomtom_df is None or len(tomtom_df) == 0:
        return {}
    motifs = load_meme_motifs(meme_db)
    if not motifs:
        return {}
    matches = []
    for _, row in tomtom_df.iterrows():
        for i in range(top_n):
            match = row.get("match{}".format(i), "")
            if not pd.isna(match) and str(match).strip():
                matches.append(str(match).strip())
    match_logos = {
        match: render_motif_logo(match, motifs)
        for match in sorted(set(matches))
    }
    n_rendered = sum(1 for logo in match_logos.values() if logo)
    print("  Match logos encoded: {}/{}".format(n_rendered, len(match_logos)))
    return match_logos


def parse_match_str(match_str: str) -> Tuple[str, str]:
    """Parse a TOMTOM match string into ``(motif_id, tf_name)``."""
    if match_str is None:
        return "", ""
    try:
        if pd.isna(match_str):
            return "", ""
    except (TypeError, ValueError):
        pass
    parts = str(match_str).strip().split(maxsplit=1)
    if not parts:
        return "", ""
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], parts[0]


def parse_tf_name(match_str: str) -> str:
    """Extract a human-readable TF name from a TOMTOM match string."""
    _, tf_name = parse_match_str(match_str)
    if not tf_name:
        return ""
    base = re.sub(r"_\d+$", "", tf_name.strip())
    if "-" in base:
        parts = base.split("-")
        last = parts[-1]
        if any(token in last.upper() for token in [
            "CTCF", "KLF", "SP", "ZNF", "ZIC", "ETS", "FOX", "NFY",
            "EGR", "E2F", "MBD", "THAP", "SETBP", "GLIS", "ZBTB",
            "ZSCAN", "HNF", "JUN", "FOS", "TEAD", "CEBP", "YY",
        ]):
            return last
    return base


def assign_family(tf_name: str) -> str:
    """Map a TF name or full motif label to a broad family label."""
    if not tf_name:
        return "Other/ZNF"
    name = tf_name.upper()
    if "CTCF" in name or "CTCFL" in name:
        return "CTCF/CTCFL"
    if any(token in name for token in ["SP", "KLF"]):
        return "SP/KLF"
    if any(token in name for token in ["NFY", "NF-Y"]):
        return "NF-Y"
    if "FOX" in name:
        return "FOX"
    if any(token in name for token in ["ETS", "ETV", "ERG", "ELF", "ERF", "FLI", "GABPA", "ELK"]):
        return "ETS"
    if any(token in name for token in ["ZIC", "ZNF", "ZFP", "ZBT", "ZBTB", "ZSCAN", "ZKSCAN", "PRDM"]):
        return "ZIC/ZNF"
    if any(token in name for token in ["NRF", "EGR", "RREB", "WT1"]):
        return "NRF/EGR"
    if any(token in name for token in ["HNF4", "NR", "PPAR", "RXR", "ESR", "CREB3L", "RARA", "VDR"]):
        return "HNF4/NR"
    if any(token in name for token in ["JUN", "FOS", "BATF", "ATF", "MAF", "NFE2", "BACH"]):
        return "AP-1/bZIP"
    if "TEAD" in name:
        return "TEAD"
    if "YY1" in name or "YY2" in name:
        return "YY1/YY2"
    if "CEBP" in name or "CEBPA" in name or "CEBPB" in name or "CEBPG" in name:
        return "CEBP"
    if "HNF1" in name:
        return "HNF1"
    if "THAP" in name:
        return "THAP11"
    return "Other/ZNF"


def fmt_pval(value) -> str:
    """Format a p-value in compact scientific notation for HTML display."""
    try:
        fval = float(value)
        if np.isnan(fval):
            return "&mdash;"
        if fval <= 0:
            return "0"
        exp = int(np.floor(np.log10(abs(fval))))
        mantissa = fval / (10 ** exp)
        if abs(mantissa - 1.0) < 0.05:
            return "10<sup>{}</sup>".format(exp)
        return "{:.1f}&times;10<sup>{}</sup>".format(mantissa, exp)
    except (TypeError, ValueError):
        return esc(value) if value else "&mdash;"


def motif_db_label(meme_db: os.PathLike) -> str:
    """Infer a compact display label from a motif database filename."""
    if not meme_db:
        return "motif database"
    base = os.path.basename(str(meme_db))
    jaspar = re.search(r"JASPAR[_-]?(\d{4})", base, flags=re.IGNORECASE)
    label = "JASPAR {}".format(jaspar.group(1)) if jaspar else os.path.splitext(base)[0]
    bits = []
    if "CORE" in base.upper():
        bits.append("CORE")
    if "vertebrates" in base.lower():
        bits.append("vertebrates")
    if bits and jaspar:
        label = "{} {}".format(label, " ".join(bits))
    return label


def count_meme_motifs(meme_db: os.PathLike) -> Optional[int]:
    """Count MOTIF records in a MEME file without depending on a parser."""
    try:
        with open(meme_db, "r", encoding="utf-8", errors="replace") as handle:
            return sum(1 for line in handle if line.startswith("MOTIF "))
    except OSError:
        return None


def pattern_index(pattern_name: str) -> Optional[int]:
    match = re.search(r"(\d+)$", str(pattern_name))
    return int(match.group(1)) if match else None


def group_short(group: str) -> str:
    return "pos" if group == "pos_patterns" else "neg"


def pattern_label(pattern: Dict) -> str:
    idx = pattern_index(pattern.get("pattern", ""))
    prefix = group_short(pattern.get("group", ""))
    if idx is not None:
        return "{}_{}".format(prefix, idx)
    return "{}_{}".format(prefix, pattern.get("pattern", ""))


def anchor_id(tag: str) -> str:
    return str(tag).replace(".", "-")


def get_tomtom_row(tomtom_df: pd.DataFrame, tag: str):
    if tomtom_df is None or len(tomtom_df) == 0 or "tag" not in tomtom_df:
        return None
    rows = tomtom_df[tomtom_df["tag"] == tag]
    return rows.iloc[0] if len(rows) else None


def family_badge(family: str, color: str) -> str:
    return (
        '<span class="family-badge" style="background:{};color:#fff;'
        'padding:2px 7px;border-radius:10px;font-size:0.75em;'
        'font-weight:600;">{}</span>'
    ).format(esc(color), esc(family))


def logo_or_placeholder(b64: str, alt: str, klass: str, style: str = "") -> str:
    if not b64:
        return '<span class="no-logo">not available</span>'
    style_attr = ' style="{}"'.format(esc(style)) if style else ""
    return '<img src="{}" alt="{}" class="{}"{}>'.format(
        data_uri_from_b64(b64), esc(alt), esc(klass), style_attr
    )


def get_pattern_matches(row, top_n: int, match_logos: Dict[str, str]) -> List[Dict]:
    if row is None:
        return []
    matches = []
    for i in range(top_n):
        match_value = row.get("match{}".format(i), "")
        if pd.isna(match_value) or not str(match_value).strip():
            continue
        match_str = str(match_value).strip()
        motif_id, raw_tf = parse_match_str(match_str)
        tf_name = parse_tf_name(match_str) or raw_tf or motif_id
        if not motif_id and not tf_name:
            continue
        family = assign_family(tf_name or raw_tf or match_str)
        matches.append({
            "rank": i + 1,
            "match": match_str,
            "motif_id": motif_id,
            "tf_name": tf_name,
            "family": family,
            "color": FAMILY_COLORS.get(family, "#9E9E9E"),
            "pval": row.get("pval{}".format(i), ""),
            "logo": match_logos.get(match_str, ""),
        })
    return matches


def build_pattern_records(
    patterns: List[Dict],
    logos: Dict[str, Dict[str, str]],
    tomtom_df: pd.DataFrame,
    top_n: int,
    match_logos: Dict[str, str],
) -> List[Dict]:
    """Combine H5 pattern metrics, rendered logos, and TOMTOM matches."""
    records = []
    for pattern in patterns:
        row = get_tomtom_row(tomtom_df, pattern["tag"])
        matches = get_pattern_matches(row, top_n, match_logos)
        top_match = matches[0] if matches else None
        family = top_match["family"] if top_match else "Other/ZNF"
        record = dict(pattern)
        record.update({
            "anchor": anchor_id(pattern["tag"]),
            "label": pattern_label(pattern),
            "group_short": group_short(pattern["group"]),
            "logos": logos.get(pattern["tag"], {}),
            "matches": matches,
            "top_tf": top_match["tf_name"] if top_match else "",
            "family": family,
            "color": FAMILY_COLORS.get(family, "#9E9E9E"),
        })
        records.append(record)
    return records


def build_page_header(
    title: str,
    records: List[Dict],
    meme_db: Optional[os.PathLike] = None,
    subtitle: Optional[str] = None,
    n_footprints: Optional[int] = None,
) -> str:
    pos = [record for record in records if record["group_short"] == "pos"]
    neg = [record for record in records if record["group_short"] == "neg"]
    pos_seqlets = sum(record["n_seqlets"] for record in pos)
    neg_seqlets = sum(record["n_seqlets"] for record in neg)
    motif_count = count_meme_motifs(meme_db) if meme_db else None
    meme_label = motif_db_label(meme_db) if meme_db else "motif database"
    if subtitle:
        subtitle_html = esc(subtitle)
    else:
        subtitle_html = (
            "{} &middot; {} annotation (TOMTOM-lite)".format(
                esc(title), esc(meme_label)
            )
        )
    first_stat = (
        ("{:,}".format(n_footprints), "footprints")
        if n_footprints is not None else
        ("{:,}".format(len(records)), "patterns")
    )
    last_stat = (
        ("{:,}".format(motif_count), "{} motifs".format(meme_label))
        if motif_count is not None else
        ("{:,}".format(pos_seqlets + neg_seqlets), "total seqlets")
    )
    stats = [
        first_stat,
        ("{:,}".format(len(pos)), "positive patterns"),
        ("{:,}".format(len(neg)), "negative patterns"),
        ("{:,}".format(pos_seqlets), "pos seqlets"),
        ("{:,}".format(neg_seqlets), "neg seqlets"),
        last_stat,
    ]
    stat_html = "\n".join(
        '    <div class="hstat"><strong>{}</strong>{}</div>'.format(
            esc(value), esc(label)
        )
        for value, label in stats
    )
    return """
<div id="page-header">
  <h1>{}</h1>
  <div class="subtitle">{}</div>
  <div class="header-stats">
{}
  </div>
</div>""".format(esc(title), subtitle_html, stat_html)


def build_family_legend() -> str:
    legend = "\n".join(
        '<span class="legend-item"><span class="legend-dot" '
        'style="background:{}"></span>{}</span>'.format(color, esc(family))
        for family, color in FAMILY_COLORS.items()
    )
    return '<div class="family-legend">\n{}\n</div>'.format(legend)


def build_sidebar(pos_records: List[Dict], neg_records: List[Dict],
                  sidebar_subtitle: str) -> str:
    def section(label: str, records: List[Dict]) -> str:
        links = []
        for record in records:
            tf_name = record["top_tf"] or "unannotated"
            links.append(
                '  <a href="#{}" class="sidebar-link" '
                'style="border-left:3px solid {};" data-family="{}">'
                '<span class="sl-id">{}</span>'
                '<span class="sl-tf">{}</span>'
                '<span class="sl-n">{:,}</span></a>'.format(
                    esc(record["anchor"]),
                    esc(record["color"]),
                    esc(record["family"]),
                    esc(record["label"]),
                    esc(tf_name),
                    record["n_seqlets"],
                )
            )
        return '<div class="sidebar-section-label">{}</div>\n{}'.format(
            esc(label), "\n".join(links)
        )

    return """
<nav id="sidebar">
  <div id="sidebar-header">
    TF-MoDISco Report
    <small>{}</small>
  </div>
  <input type="text" id="search-box" placeholder="Search TF or pattern..." oninput="filterSidebar(this.value)">
  {}
  {}
</nav>""".format(
        esc(sidebar_subtitle),
        section("Positive Patterns", pos_records),
        section("Negative Patterns", neg_records),
    )


def build_tab_bar(pos_count: int, neg_count: int) -> str:
    return """
<div id="tab-bar">
  <button id="tab-pos" class="tab-btn active" onclick="switchTab('pos')">Positive Patterns ({})</button>
  <button id="tab-neg" class="tab-btn" onclick="switchTab('neg')">Negative Patterns ({})</button>
  <div class="tab-controls">
    <button class="ctrl-btn" onclick="expandAll()">Expand all</button>
    <button class="ctrl-btn" onclick="collapseAll()">Collapse all</button>
  </div>
</div>""".format(pos_count, neg_count)


def build_summary_match_cell(match: Optional[Dict]) -> str:
    if not match:
        return '<td class="summary-match-cell">&mdash;</td>'
    return """
      <td class="summary-match-cell">
        {}<br>
        <strong>{}</strong><br>
        <code style="font-size:0.75em">{}</code><br>
        <span style="font-size:0.8em;color:#555">{}</span>
      </td>""".format(
        family_badge(match["family"], match["color"]),
        esc(match["tf_name"]),
        esc(match["motif_id"]),
        fmt_pval(match["pval"]),
    )


def build_summary_table(records: List[Dict], table_id: str) -> str:
    match_headers = "\n".join(
        '        <th onclick="sortTable(\'{}\',{})">Match {}</th>'.format(
            esc(table_id), 4 + i, i + 1
        )
        for i in range(3)
    )
    rows = []
    for record in records:
        cwm_fwd = logo_or_placeholder(
            record["logos"].get("cwm_fwd", ""),
            "CWM fwd",
            "pattern-logo",
            "max-height:40px;max-width:120px;",
        )
        match_cells = "".join(
            build_summary_match_cell(
                record["matches"][i] if i < len(record["matches"]) else None
            )
            for i in range(3)
        )
        rows.append("""
    <tr class="summary-row" data-family="{}" data-group="{}">
      <td><a href="#{}" class="pattern-link" style="color:{};font-weight:bold;">{}</a></td>
      <td class="num-col">{:,}</td>
      <td class="num-col">{:.0f}%</td>
      <td class="logo-col">{}</td>
{}
    </tr>""".format(
            esc(record["family"]),
            esc(record["group_short"]),
            esc(record["anchor"]),
            esc(record["color"]),
            esc(record["label"]),
            record["n_seqlets"],
            record["pct_revcomp"],
            cwm_fwd,
            match_cells,
        ))

    return """
<div class="summary-table-wrap">
  <table class="summary-table" id="{}">
    <thead>
      <tr>
        <th onclick="sortTable('{}',0)">Pattern</th>
        <th onclick="sortTable('{}',1)">Seqlets</th>
        <th onclick="sortTable('{}',2)">Rev-comp%</th>
        <th onclick="sortTable('{}',3)">CWM fwd</th>
{}
      </tr>
    </thead>
    <tbody>
{}
    </tbody>
  </table>
</div>""".format(
        esc(table_id), esc(table_id), esc(table_id), esc(table_id),
        esc(table_id), match_headers, "\n".join(rows)
    )


def build_match_row(match: Dict) -> str:
    if match.get("logo"):
        logo = '<img src="{}" alt="{} logo" class="jaspar-logo">'.format(
            esc(match["logo"]), esc(match["tf_name"])
        )
    else:
        logo = '<span class="no-logo">not available</span>'
    return """
      <tr class="match-row">
        <td class="rank-col">{}</td>
        <td class="jaspar-id-col"><code>{}</code></td>
        <td class="tf-name-col"><strong>{}</strong></td>
        <td class="family-col">{}</td>
        <td class="pval-col">{}</td>
        <td class="logo-col">{}</td>
      </tr>""".format(
        match["rank"],
        esc(match["motif_id"]),
        esc(match["tf_name"]),
        family_badge(match["family"], match["color"]),
        fmt_pval(match["pval"]),
        logo,
    )


def build_match_table(record: Dict) -> str:
    matches = record["matches"]
    if not matches:
        return '<p class="no-matches">No TOMTOM annotation available.</p>'
    visible = matches[:3]
    extra = matches[3:]
    rows = "".join(build_match_row(match) for match in visible)
    if extra:
        extra_rows = "".join(build_match_row(match) for match in extra)
        more_text = "Show more ({} more)".format(len(extra))
        rows += """
      <tr class="extra-matches" id="extra-{}" style="display:none;">
        <td colspan="6">
          <table class="match-table inner-extra"><tbody>
{}
          </tbody></table>
        </td>
      </tr>
      <tr>
        <td colspan="6"><button id="more-{}" class="show-more-btn" data-more-text="{}" onclick="toggleExtra('{}')">{}</button></td>
      </tr>""".format(
            esc(record["anchor"]), extra_rows, esc(record["anchor"]),
            esc(more_text), esc(record["anchor"]), esc(more_text)
        )
    return """
<table class="match-table">
  <thead>
    <tr>
      <th class="rank-col">#</th>
      <th class="jaspar-id-col">Motif ID</th>
      <th class="tf-name-col">TF name</th>
      <th class="family-col">Family</th>
      <th class="pval-col">p-value</th>
      <th class="logo-col">Logo</th>
    </tr>
  </thead>
  <tbody>
{}
  </tbody>
</table>""".format(rows)


def build_logo_box(label: str, b64: str, alt: str) -> str:
    return """
      <div class="logo-box">
        <div class="logo-label">{}</div>
        {}
      </div>""".format(
        esc(label), logo_or_placeholder(b64, alt, "pattern-logo")
    )


def build_pattern_card(record: Dict) -> str:
    logos = record["logos"]
    top_tf_hint = (
        '<span class="top-tf-hint">=&gt; {}</span>'.format(esc(record["top_tf"]))
        if record["top_tf"] else ""
    )
    return """
<div class="pattern-card" id="{}" data-family="{}" data-group="{}">
  <div class="card-header" onclick="toggleCard('{}')" style="border-left:5px solid {};">
    <div class="card-title">
      <span class="pattern-id">{}</span>
      {}
      {}
    </div>
    <div class="card-meta">
      <span class="seqlet-count">{:,} seqlets</span>
      <span class="rc-pct">{:.0f}% rev-comp</span>
      <span class="toggle-icon" id="icon-{}">&#9660;</span>
    </div>
  </div>
  <div class="card-body" id="body-{}" style="display:none;">
    <div class="logo-row">
{}
{}
{}
    </div>
    <div class="stats-row">
      <div class="stat-box"><div class="stat-label">Seqlets</div><div class="stat-value">{:,}</div></div>
      <div class="stat-box"><div class="stat-label">Rev-comp</div><div class="stat-value">{:.0f}%</div></div>
      <div class="stat-box"><div class="stat-label">Mean |contrib|</div><div class="stat-value">{:.4f}</div></div>
      <div class="stat-box"><div class="stat-label">Length</div><div class="stat-value">{} bp</div></div>
      <div class="stat-box consensus-box"><div class="stat-label">IUPAC consensus</div><div class="stat-value consensus-seq">{}</div></div>
    </div>
    <div class="matches-section">
      <div class="matches-title">Top TOMTOM matches</div>
      {}
    </div>
  </div>
</div>""".format(
        esc(record["anchor"]),
        esc(record["family"]),
        esc(record["group_short"]),
        esc(record["anchor"]),
        esc(record["color"]),
        esc(record["label"]),
        family_badge(record["family"], record["color"]),
        top_tf_hint,
        record["n_seqlets"],
        record["pct_revcomp"],
        esc(record["anchor"]),
        esc(record["anchor"]),
        build_logo_box("CWM fwd (trimmed)", logos.get("cwm_fwd", ""), "CWM forward logo"),
        build_logo_box("CWM rev (trimmed)", logos.get("cwm_rev", ""), "CWM reverse logo"),
        build_logo_box("IC PPM", logos.get("ic_ppm", ""), "Information content PPM logo"),
        record["n_seqlets"],
        record["pct_revcomp"],
        record["mean_abs_contrib"],
        record["length"],
        esc(record["iupac"]),
        build_match_table(record),
    )


def build_html(
    title: str,
    patterns: List[Dict],
    logos: Dict[str, Dict[str, str]],
    tomtom_df: pd.DataFrame,
    top_n: int,
    match_logos: Optional[Dict[str, str]] = None,
    meme_db: Optional[os.PathLike] = None,
    subtitle: Optional[str] = None,
    n_footprints: Optional[int] = None,
) -> str:
    """Assemble the complete self-contained HTML document."""
    records = build_pattern_records(patterns, logos, tomtom_df, top_n, match_logos or {})
    pos_records = sorted(
        [record for record in records if record["group_short"] == "pos"],
        key=lambda record: record["n_seqlets"],
        reverse=True,
    )
    neg_records = sorted(
        [record for record in records if record["group_short"] == "neg"],
        key=lambda record: record["n_seqlets"],
        reverse=True,
    )
    sorted_records = pos_records + neg_records
    sidebar_subtitle = title.replace("TF-MoDISco Report", "").strip(" -:") or title

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{}</title>
<style>
{}
</style>
</head>
<body>
{}
<div id="main">
  {}
  {}
  {}

  <div id="section-pos" class="section-content">
    <div class="section-heading">Positive Patterns</div>
    <div class="section-subheading">Sequence features with positive contribution scores. Sorted by seqlet count.</div>
    {}
    <div id="pos-cards">
{}
    </div>
  </div>

  <div id="section-neg" class="section-content hidden">
    <div class="section-heading">Negative Patterns</div>
    <div class="section-subheading">Sequence features with negative contribution scores. Sorted by seqlet count.</div>
    {}
    <div id="neg-cards">
{}
    </div>
  </div>
</div>
<script>
{}
</script>
</body>
</html>""".format(
        esc(title),
        EMBEDDED_CSS,
        build_sidebar(pos_records, neg_records, sidebar_subtitle),
        build_page_header(title, sorted_records, meme_db=meme_db,
                          subtitle=subtitle, n_footprints=n_footprints),
        build_family_legend(),
        build_tab_bar(len(pos_records), len(neg_records)),
        build_summary_table(pos_records, "pos-summary-table"),
        "".join(build_pattern_card(record) for record in pos_records),
        build_summary_table(neg_records, "neg-summary-table"),
        "".join(build_pattern_card(record) for record in neg_records),
        EMBEDDED_JS,
    )


def build_tsv(patterns: List[Dict], tomtom_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Build the per-pattern annotation DataFrame for TSV export."""
    rows = []
    for pattern in patterns:
        tag = pattern["tag"]
        row = {
            "tag": tag,
            "group": pattern["group"],
            "pattern": pattern["pattern"],
            "n_seqlets": pattern["n_seqlets"],
            "pct_revcomp": round(pattern["pct_revcomp"], 2),
            "mean_abs_contrib": pattern["mean_abs_contrib"],
            "length": pattern["length"],
            "iupac": pattern["iupac"],
        }
        if tomtom_df is not None and len(tomtom_df) > 0:
            tomtom_row = get_tomtom_row(tomtom_df, tag)
            if tomtom_row is not None:
                for i in range(top_n):
                    match_value = tomtom_row.get("match{}".format(i), "")
                    motif_id, tf_name = parse_match_str(match_value)
                    row["match{}".format(i)] = match_value
                    row["pval{}".format(i)] = tomtom_row.get("pval{}".format(i), "")
                    row["motif_id{}".format(i)] = motif_id
                    row["tf_name{}".format(i)] = tf_name
                    row["family{}".format(i)] = assign_family(tf_name)
        rows.append(row)
    return pd.DataFrame(rows)


def validate_inputs(h5_path: os.PathLike, meme_db: os.PathLike) -> None:
    """Validate required report inputs."""
    if not os.path.isfile(h5_path):
        raise FileNotFoundError("H5 not found: {}".format(h5_path))
    if not os.path.isfile(meme_db):
        raise FileNotFoundError("meme-db not found: {}".format(meme_db))


def generate_fancy_report(
    h5_path: os.PathLike,
    meme_db: os.PathLike,
    output: os.PathLike,
    logos_dir: Optional[os.PathLike] = None,
    force_logos: bool = False,
    top_n_matches: int = 5,
    trim_threshold: float = 0.3,
    trim_min_length: int = 3,
    title: str = "TF-MoDISco Report",
    subtitle: Optional[str] = None,
    n_footprints: Optional[int] = None,
) -> Dict[str, object]:
    """Generate the fancy HTML report and its TSV companion file."""
    validate_inputs(h5_path, meme_db)
    output_path = Path(output)
    logo_path = Path(logos_dir) if logos_dir else default_logos_dir(output_path)

    print("=== Fancy TF-MoDISco Report Workflow ===")
    print("H5        : {}".format(h5_path))
    print("MEME DB   : {}".format(meme_db))
    print("Output    : {}".format(output_path))
    print("Logos dir : {}".format(logo_path))
    print("Top-N     : {}".format(top_n_matches))
    print()

    print("[1/5] Reading H5 patterns...")
    patterns = load_h5_patterns(h5_path)
    groups = [
        group for group in CANONICAL_GROUPS
        if any(pattern["group"] == group for pattern in patterns)
    ]
    if not groups:
        raise RuntimeError("H5 file contains neither 'pos_patterns' nor 'neg_patterns'.")
    pattern_tags = [pattern["tag"] for pattern in patterns]
    pos_count = sum(1 for pattern in patterns if pattern["group"] == "pos_patterns")
    neg_count = sum(1 for pattern in patterns if pattern["group"] == "neg_patterns")
    print("  Total: {} ({} pos, {} neg)".format(len(patterns), pos_count, neg_count))

    generate_logos(
        h5_path=h5_path,
        logos_dir=logo_path,
        groups=groups,
        pattern_tags=pattern_tags,
        trim_threshold=trim_threshold,
        force=force_logos,
    )
    tomtom_df = annotate_patterns_with_tomtom(
        h5_path=h5_path,
        meme_db=meme_db,
        patterns=patterns,
        groups=groups,
        top_n=top_n_matches,
        trim_threshold=trim_threshold,
        trim_min_length=trim_min_length,
    )

    print("[4/5] Loading logos and rendering motif logos...")
    logos = load_logos(logo_path, patterns)
    n_written_embedded_logos = write_embedded_logo_files(logo_path, patterns, logos)
    if n_written_embedded_logos:
        print("  Wrote embedded-only logo files: {}".format(n_written_embedded_logos))
    match_logos = load_match_logos(meme_db, tomtom_df, top_n_matches)

    print("[5/5] Building HTML and TSV report...")
    html = build_html(
        title=title,
        patterns=patterns,
        logos=logos,
        tomtom_df=tomtom_df,
        top_n=top_n_matches,
        match_logos=match_logos,
        meme_db=meme_db,
        subtitle=subtitle,
        n_footprints=n_footprints,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    size_mb = output_path.stat().st_size / 1e6

    tsv_path = output_path.with_suffix(".tsv")
    tsv_df = build_tsv(patterns, tomtom_df, top_n_matches)
    tsv_df.to_csv(tsv_path, sep="\t", index=False)
    final_logo_summary = summarize_logo_layout(logo_path, pattern_tags)

    n_cards = html.count('class="pattern-card"')
    n_b64 = html.count("data:image/png;base64,")
    n_links = html.count('class="sidebar-link"')
    print("  HTML: {} ({:.1f} MB)".format(output_path, size_mb))
    print("  TSV : {} ({} rows x {} cols)".format(
        tsv_path, len(tsv_df), len(tsv_df.columns)
    ))
    print("  Logos: {} ({} pattern dirs, {} files)".format(
        logo_path,
        len(final_logo_summary["pattern_dirs"]),
        final_logo_summary["total_files"],
    ))
    print()
    print("Sanity check:")
    print("  Pattern cards : {} (expected {})".format(n_cards, len(patterns)))
    print("  Base64 images : {}".format(n_b64))
    print("  Sidebar links : {} (expected {})".format(n_links, len(patterns)))
    print()
    print("Done.")

    return {
        "html": output_path,
        "tsv": tsv_path,
        "logos_dir": logo_path,
        "logo_files": final_logo_summary["total_files"],
        "patterns": len(patterns),
        "cards": n_cards,
        "base64_images": n_b64,
        "sidebar_links": n_links,
    }
