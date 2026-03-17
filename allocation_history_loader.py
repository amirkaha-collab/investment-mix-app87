# -*- coding: utf-8 -*-
"""
allocation_history_loader.py  –  v2
────────────────────────────────────
Loads and normalises investment-allocation history data from Google Sheets.

Bug fixed in v2:
  - "סוג התאריך" column was matched as the date column (substring "תאריך")
    instead of the actual "תאריך" column.  Now uses exact-match priority.
  - Filters Year-type rows; keeps only Month-type rows for the time series.
  - Sheet gid discovery: HTML parser improved + probe-gids-0..11 fallback.
  - Detects HTML login-redirect responses (auth required) and reports clearly.

Normalised output schema
────────────────────────
manager         : str
track           : str
date            : datetime64[ns]  – first day of the relevant month
year            : int
month           : int
allocation_name : str
allocation_value: float           – percent value, e.g. 38.5
source_sheet    : str
"""

from __future__ import annotations

import re
import io
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# ─── Hebrew month name → number ──────────────────────────────────────────────
_HEB_MONTHS = {
    "ינואר": 1, "פברואר": 2, "מרץ": 3, "מרס": 3,
    "אפריל": 4, "מאי": 5, "יוני": 6,
    "יולי": 7, "אוגוסט": 8, "ספטמבר": 9,
    "אוקטובר": 10, "נובמבר": 11, "דצמבר": 12,
}

# ─── Sheet-name → (manager, track) lookup ────────────────────────────────────
_SHEET_META: dict[str, dict] = {
    "הראל כללי":   {"manager": "הראל", "track": "כללי"},
    "הראל מנייתי": {"manager": "הראל", "track": "מנייתי"},
    # ← add more entries here as new sheets arrive
}

_MANAGER_PATTERNS = [
    "הראל", "מגדל", "כלל", "מנורה", "הפניקס", "אנליסט", "מיטב",
    "ילין", "פסגות", "אלטשולר", "ברקת", "אלומות",
]
_TRACK_PATTERNS = {
    "כלל": "כללי", "כללי": "כללי",
    "מנייתי": "מנייתי", "מניות": "מנייתי",
}


def _infer_meta(sheet_name: str) -> dict:
    s = sheet_name.strip()
    for key, meta in _SHEET_META.items():
        if key in s:
            return meta
    manager = next((m for m in _MANAGER_PATTERNS if m in s), s)
    track = "כללי"
    for pat, val in _TRACK_PATTERNS.items():
        if pat in s:
            track = val
            break
    return {"manager": manager, "track": track}


def _extract_sheet_id(url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        raise ValueError(f"לא ניתן לחלץ Sheet ID מהכתובת: {url}")
    return m.group(1)


def _csv_export_url(sheet_id: str, gid: int = 0) -> str:
    return (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/export?format=csv&gid={gid}"
    )


# ─── Sheet discovery ──────────────────────────────────────────────────────────

def _discover_sheet_gids(sheet_id: str, max_probe: int = 12) -> list[tuple[str, int]]:
    """
    Return list of (sheet_name, gid).

    Strategy:
      1. Parse HTML of the edit page (several regex patterns).
      2. Fallback: probe gids 0..max_probe-1 via CSV export headers.
    """
    found: list[tuple[str, int]] = []

    # ── Attempt 1: HTML parse ─────────────────────────────────────────
    try:
        r = requests.get(
            f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit",
            timeout=15,
        )
        html = r.text
        # Pattern A: "sheetId":NNN … "title":"Name"
        for gid_str, title in re.findall(
            r'"sheetId":(\d+)[^}]{0,200}?"title":"([^"]+)"', html
        ):
            found.append((title, int(gid_str)))
        if not found:
            # Pattern B (reversed order)
            for title, gid_str in re.findall(
                r'"title":"([^"]+)"[^}]{0,200}?"sheetId":(\d+)', html
            ):
                found.append((title, int(gid_str)))
        if not found:
            # Pattern C: older format ["Tab name",null,gid,…]
            for m in re.finditer(r'\["([^"]{1,80})",null,(\d+)', html):
                found.append((m.group(1), int(m.group(2))))
    except Exception as e:
        logger.warning(f"HTML discovery failed: {e}")

    if found:
        logger.info(f"Discovered {len(found)} sheets via HTML")
        return found

    # ── Attempt 2: probe gids 0..N ────────────────────────────────────
    logger.info("HTML discovery failed – probing gids 0..%d", max_probe - 1)
    for gid in range(max_probe):
        try:
            rr = requests.get(_csv_export_url(sheet_id, gid), timeout=15)
            ct = rr.headers.get("Content-Type", "")
            if rr.status_code != 200:
                break  # non-existent gid → stop
            if "html" in ct.lower():
                # auth redirect – all gids will fail the same way
                logger.warning("CSV export returned HTML for gid=%d (auth redirect?)", gid)
                break
            if len(rr.text.strip()) > 20:
                found.append((f"גליון_{gid}", gid))
        except Exception:
            break

    return found if found else [("גליון_0", 0)]


# ─── Date parsing ─────────────────────────────────────────────────────────────

def _parse_date_value(val) -> Optional[datetime]:
    """
    Parse various date formats to the first day of the relevant month.
    Returns None if the value cannot be parsed.
    """
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (datetime, pd.Timestamp)):
        return pd.Timestamp(val).replace(day=1).to_pydatetime()

    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return None

    # Hebrew month name + year: "ינואר 2024"
    for heb, month_num in _HEB_MONTHS.items():
        if heb in s:
            y = re.search(r"(\d{4})", s)
            if y:
                return datetime(int(y.group(1)), month_num, 1)

    # Ordered format attempts
    for fmt in (
        "%Y-%m-%d", "%d/%m/%Y", "%m/%Y", "%Y-%m",
        "%b-%Y", "%B %Y", "%b %Y", "%Y/%m/%d", "%d-%m-%Y",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(day=1)
        except ValueError:
            pass

    # Last resort – pandas
    try:
        dt = pd.to_datetime(s, dayfirst=True)
        return dt.replace(day=1).to_pydatetime()
    except Exception:
        return None


# ─── Percent parsing ──────────────────────────────────────────────────────────

def _parse_percent(val) -> Optional[float]:
    """
    Parse a percent value to float (0–100 scale).
    Handles "38.5%", 0.385 (auto-scales), "38,5", NaN, None.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and np.isnan(val):
            return None
        # Fraction stored as 0–1
        if abs(val) <= 1.5:
            return round(float(val) * 100, 4)
        return round(float(val), 4)
    s = str(val).strip().replace(",", ".").replace("%", "").strip()
    if not s:
        return None
    try:
        f = float(s)
        if abs(f) <= 1.5:
            return round(f * 100, 4)
        return round(f, 4)
    except ValueError:
        return None


# ─── Core normaliser ──────────────────────────────────────────────────────────

# Keywords that indicate a "row-type" / "period-type" column (not a date or alloc)
_ROW_TYPE_KEYWORDS = ["סוג", "type", "kind", "period", "תקופה"]

# Column names whose VALUES should be treated as the period type
_PERIOD_TYPE_MONTH_VALUES = {"month", "חודשי", "חודש", "monthly"}


def _find_date_col(columns: list[str]) -> Optional[str]:
    """
    Identify the date column.  Priority:
      1. Exact match: "תאריך" / "date" / "חודש"
      2. Ends-with match
      3. Contains match  (but NOT if the column also matches a row-type keyword)
    """
    exact_kws = {"תאריך", "date", "חודש", "month", "time", "תאריך_"}
    cols_lower = {c: str(c).strip().lower() for c in columns}

    # 1. Exact
    for c, cl in cols_lower.items():
        if cl in exact_kws:
            return c

    # 2. Ends-with
    for c, cl in cols_lower.items():
        if any(cl.endswith(kw) for kw in exact_kws):
            # Reject if it also looks like a row-type column
            if not any(rk in cl for rk in _ROW_TYPE_KEYWORDS):
                return c

    # 3. Contains – skip row-type columns
    for c, cl in cols_lower.items():
        if any(rk in cl for rk in _ROW_TYPE_KEYWORDS):
            continue
        if any(kw in cl for kw in exact_kws):
            return c

    return None


def _find_type_col(columns: list[str]) -> Optional[str]:
    """Find the optional 'סוג התאריך' / 'type' column."""
    for c in columns:
        cl = str(c).strip().lower()
        if any(rk in cl for rk in _ROW_TYPE_KEYWORDS):
            return c
    return None


def _normalise_sheet_df(raw: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    Convert a raw wide-format single-sheet DataFrame to the normalised long format.

    Handles the specific structure observed in the sheet:
        [empty col] | סוג התאריך | תאריך | alloc_1 | alloc_2 | …
        (empty)     | Year       | 2014  |  val    |  val    | …
        (empty)     | Month      | 2025-02 | val   |  val    | …

    - Uses "תאריך" as date column (exact match takes priority).
    - Filters to Month-type rows when a type column is present.
    - Year-type rows are available but ignored for the main time series.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    meta = _infer_meta(sheet_name)

    # ── Identify columns ──────────────────────────────────────────────
    date_col = _find_date_col(list(raw.columns))
    type_col = _find_type_col(list(raw.columns))

    if date_col is None:
        logger.warning(f"No date column found in sheet '{sheet_name}'. Columns: {list(raw.columns)}")
        return pd.DataFrame()

    # ── Filter: keep only Month-type rows ────────────────────────────
    working = raw.copy()
    if type_col is not None:
        month_mask = working[type_col].astype(str).str.strip().str.lower().isin(
            _PERIOD_TYPE_MONTH_VALUES
        )
        if month_mask.any():
            working = working[month_mask].copy()
        # If no "Month" rows found, fall back to all rows (don't discard data)

    # ── Allocation columns ────────────────────────────────────────────
    skip_cols = {date_col}
    if type_col:
        skip_cols.add(type_col)
    # Also skip unnamed / index-like columns (empty header, "Unnamed: N")
    alloc_cols = [
        c for c in working.columns
        if c not in skip_cols
        and not str(c).strip().startswith("Unnamed")
        and str(c).strip() not in ("", "nan")
    ]

    if not alloc_cols:
        logger.warning(f"No allocation columns found in sheet '{sheet_name}'")
        return pd.DataFrame()

    # ── Build long-format rows ────────────────────────────────────────
    rows = []
    for _, row in working.iterrows():
        dt = _parse_date_value(row[date_col])
        if dt is None:
            continue
        for col in alloc_cols:
            val = _parse_percent(row[col])
            if val is None:
                continue
            rows.append({
                "manager":          meta["manager"],
                "track":            meta["track"],
                "date":             pd.Timestamp(dt),
                "year":             dt.year,
                "month":            dt.month,
                "allocation_name":  str(col).strip(),
                "allocation_value": val,
                "source_sheet":     sheet_name,
            })

    if not rows:
        logger.warning(f"Sheet '{sheet_name}': all rows failed to parse")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ─── CSV transport ────────────────────────────────────────────────────────────

def _load_sheet_via_csv(
    sheet_id: str, gid: int, sheet_name: str
) -> tuple[pd.DataFrame, Optional[str]]:
    """
    Fetch one sheet as CSV and normalise.
    Returns (df, error_message_or_None).
    """
    url = _csv_export_url(sheet_id, gid)
    try:
        r = requests.get(url, timeout=25, allow_redirects=True)
        ct = r.headers.get("Content-Type", "")

        # Detect login-page redirect
        if "html" in ct.lower() or r.text.strip().startswith("<!"):
            return pd.DataFrame(), (
                f"גליון '{sheet_name}' (gid={gid}): הגיליון דורש התחברות. "
                "יש לפרסם אותו ב-File → Share → Publish to web → CSV, "
                "או להגדיר service account בסודות."
            )

        if r.status_code != 200:
            return pd.DataFrame(), f"גליון '{sheet_name}': HTTP {r.status_code}"

        raw = pd.read_csv(io.StringIO(r.text), header=0)
        raw = raw.dropna(how="all").reset_index(drop=True)
        df = _normalise_sheet_df(raw, sheet_name)

        if df.empty:
            return pd.DataFrame(), (
                f"גליון '{sheet_name}': נטען אך לא נמצאו נתונים תקינים "
                f"(עמודות שנמצאו: {list(raw.columns)[:6]})"
            )
        return df, None

    except Exception as e:
        return pd.DataFrame(), f"גליון '{sheet_name}' (gid={gid}): {e}"


# ─── gspread transport ────────────────────────────────────────────────────────

def _load_via_gspread(sheet_url: str) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        creds_dict = dict(st.secrets["gcp_service_account"])
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_url(sheet_url)

        frames = []
        for ws in sh.worksheets():
            try:
                data = ws.get_all_values()
                if not data or len(data) < 2:
                    continue
                raw = pd.DataFrame(data[1:], columns=data[0])
                raw = raw.dropna(how="all").reset_index(drop=True)
                norm = _normalise_sheet_df(raw, ws.title)
                if not norm.empty:
                    frames.append(norm)
                else:
                    errors.append(f"gspread: גליון '{ws.title}' — אין נתונים")
            except Exception as e:
                errors.append(f"gspread: גליון '{ws.title}' — {e}")

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return df, errors
    except Exception as e:
        return pd.DataFrame(), [f"gspread נכשל: {e}"]


# ─── Main public API ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_allocation_history(sheet_url: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Load and normalise all sheets from a Google Sheets URL.

    Returns
    -------
    df     : normalised DataFrame (empty on full failure)
    errors : list of warning / error strings for display
    """
    errors: list[str] = []

    if not sheet_url or not sheet_url.strip():
        return pd.DataFrame(), ["לא הוגדר קישור ל-Google Sheets"]

    # ── gspread first (if service account configured) ─────────────────
    has_sa = hasattr(st, "secrets") and "gcp_service_account" in st.secrets
    if has_sa:
        df, errs = _load_via_gspread(sheet_url)
        errors.extend(errs)
        if not df.empty:
            return df, errors
        errors.append("gspread נכשל — עובר ל-CSV ציבורי")

    # ── CSV export (public / anyone-with-link) ────────────────────────
    try:
        sheet_id = _extract_sheet_id(sheet_url)
    except ValueError as e:
        return pd.DataFrame(), errors + [str(e)]

    sheets = _discover_sheet_gids(sheet_id)
    frames: list[pd.DataFrame] = []

    for name, gid in sheets:
        df_sheet, err = _load_sheet_via_csv(sheet_id, gid, name)
        if err:
            errors.append(err)
        if not df_sheet.empty:
            frames.append(df_sheet)

    if not frames:
        return pd.DataFrame(), errors + ["לא נטענו נתונים מאף גליון"]

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["manager", "track", "allocation_name", "date"]).reset_index(drop=True)
    return df, errors
