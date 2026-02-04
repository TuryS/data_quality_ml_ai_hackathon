"""
Pre-processing with aggressive carrier master mapping.

Features:
- Loads CSV and normalizes datetime fields
- Computes transit and delay metrics + basic flags
- Applies carrier master mapping using:
    1) Exact normalized LUT match
    2) Brand keyword match
    3) Token-set overlap scoring
    4) Fuzzy string match
    5) Context-based category fallback (Vessel / port/airport patterns / forwarder hints)

Usage:
    enriched = run_pipeline("shipments.csv",
                            carrier_col="Carrier",
                            lut_path="carrier_master_lookup.json")
    enriched.to_csv("shipments_with_master_carrier.csv", index=False)
"""

import json
import re
import difflib
from typing import Dict, Any, Tuple, Optional

import pandas as pd


# =========================
# Configuration defaults
# =========================
DEFAULT_CARRIER_COL = "Carrier"
DEFAULT_LUT_PATH = "data/carrier_master_lookup.json"

# Regex and normalization helpers
_MULTI_SPACE = re.compile(r"\s+")
# Drop commas and odd punctuation, keep &, /, (), spaces
_STRIP_PUNCT = re.compile(r"[^\w\s&/()]")

# Conservative company suffix removal
_COMPANY_SUFFIX_PAT = re.compile(
    r"""
    \b(pty\.?\s*ltd|co\.?\s*ltd|s\.?a\.?(\.?|s)?|s\.?r\.?l\.?u?|
       gmbh|b\.?v\.?|nv|bvba|sarl|spa|plc|ltd|inc\.?|corp\.?|
       limited|company|ag|a\/s|saog|slu|bv|nv|
       branch|office|offices|department|division
    )\b
    """,
    re.IGNORECASE | re.VERBOSE
)

# Canonical replacements
_CANONICAL = {
    "&amp;": "&",
    "&": " and ",
}

# Forwarder hints (extend as needed)
_FORWARDER_HINTS = {
    "vanguard", "ecu", "shipco", "globelink", "saco", "ssc consolidation",
    "teamglobal", "transglory", "ifs international", "carotrans", "freight systems",
    "pace", "network airline services", "mercury air cargo", "cargo marketing services",
    "goodrich maritime", "cnan italia", "tarros", "macs", "uasc", "rohlig",
    "agency", "agent", "handler", "gsa"
}

# Brand keyword dictionary
# Add or adjust tokens to fit your lane. Longer keys checked first.
_BRAND_KEYWORDS = {
    # Ocean
    "mediterranean shipping": ("MSC", "Ocean"),
    "hamburg sud":           ("Maersk", "Ocean"),
    "one ocean network":     ("ONE", "Ocean"),
    "atlantic container line": ("RORO", "Ocean"),
    "wallenius":             ("RORO", "Ocean"),
    "grimaldi":              ("RORO", "Ocean"),
    "hoegh":                 ("RORO", "Ocean"),
    "hapag lloyd":           ("Hapag-Lloyd", "Ocean"),
    "k line":                ("ONE", "Ocean"),
    "nyk":                   ("ONE", "Ocean"),
    "mol":                   ("ONE", "Ocean"),
    "sm line":               ("SM Line", "Ocean"),
    "ts lines":              ("Intra-Asia Feeder", "Ocean"),
    "kmtc":                  ("Intra-Asia Feeder", "Ocean"),
    "sinokor":               ("Intra-Asia Feeder", "Ocean"),
    "yang ming":             ("Yang Ming", "Ocean"),
    "evergreen":             ("Evergreen", "Ocean"),
    "maersk":                ("Maersk", "Ocean"),
    "sealand":               ("Maersk", "Ocean"),
    "msc":                   ("MSC", "Ocean"),
    "oocl":                  ("ONE", "Ocean"),
    "cosco":                 ("COSCO", "Ocean"),
    "pil":                   ("PIL", "Ocean"),
    "zim":                   ("ZIM", "Ocean"),
    "hmm":                   ("HMM", "Ocean"),
    "wan hai":               ("Wan Hai", "Ocean"),

    # Air
    "air france":            ("Air France-KLM Cargo", "Air"),
    "klm":                   ("Air France-KLM Cargo", "Air"),
    "lufthansa":             ("Lufthansa Cargo", "Air"),
    "swiss":                 ("Swiss WorldCargo", "Air"),
    "british airways":       ("IAG Cargo", "Air"),
    "iberia":                ("IAG Cargo", "Air"),
    "american airlines":     ("American Airlines Cargo", "Air"),
    "united":                ("United Cargo", "Air"),
    "delta":                 ("Delta Cargo", "Air"),
    "emirates":              ("Emirates SkyCargo", "Air"),
    "qatar airways":         ("Qatar Airways Cargo", "Air"),
    "etihad":                ("Etihad Cargo", "Air"),
    "air china":             ("Air China Cargo", "Air"),
    "china eastern":         ("China Eastern Cargo", "Air"),
    "china southern":        ("China Southern Cargo", "Air"),
    "singapore airlines":    ("SIA Cargo", "Air"),
    "malaysia airlines":     ("MASkargo", "Air"),
    "thai airways":          ("Thai Cargo", "Air"),
    "eva air":               ("EVA Air Cargo", "Air"),
    "japan airlines":        ("JAL Cargo", "Air"),
    "jal":                   ("JAL Cargo", "Air"),
    "korean air":            ("Korean Air Cargo", "Air"),
    "turkish airlines":      ("Turkish Cargo", "Air"),
    "air new zealand":       ("Air New Zealand Cargo", "Air"),
    "latam":                 ("LATAM Cargo", "Air"),
    "cargolux":              ("Cargolux", "Air"),
    "atlas air":             ("Atlas Air", "Air"),
    "polar air":             ("Polar Air Cargo", "Air"),
    "airbridgecargo":        ("AirBridgeCargo", "Air"),
    "etihad airways":        ("Etihad Cargo", "Air"),
}


# =========================
# Normalization
# =========================
def _normalize(value: Any) -> str:
    """Normalize carrier strings for robust matching."""
    if pd.isna(value):
        return ""
    s = str(value).lower().strip()

    for k, v in _CANONICAL.items():
        s = s.replace(k, v)

    # normalize separators early
    s = s.replace("-", " ")   # key step: "hapag-lloyd" -> "hapag lloyd"
    s = s.replace("/", " ")
    s = s.replace(".", " ")

    # remove unwanted punctuation
    s = _STRIP_PUNCT.sub(" ", s)

    # remove generic legal/company suffixes
    s = _COMPANY_SUFFIX_PAT.sub(" ", s)

    # collapse spaces
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s


# =========================
# Lookup helpers
# =========================
def load_lookup_normalized(path: str) -> Dict[str, Dict[str, str]]:
    """Load LUT and normalize keys with the same function used for inputs."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    norm = {}
    for k, v in raw.items():
        nk = _normalize(k)
        norm[nk] = {
            "master_carrier": v.get("master_carrier", "Unmapped"),
            "category": v.get("category", "Other"),
        }
    return norm


def _brand_keyword_match(key: str) -> Optional[Tuple[str, str]]:
    # Match longest keywords first
    for kw in sorted(_BRAND_KEYWORDS.keys(), key=len, reverse=True):
        if kw in key:
            return _BRAND_KEYWORDS[kw]
    return None



def _token_overlap_best(
    key: str,
    candidates: Dict[str, Dict[str, str]],
    min_score: float = 0.5
) -> Optional[Tuple[str, str]]:
    """Pick best by Jaccard-like token overlap."""
    kset = set(key.split())
    if not kset:
        return None

    best_score = 0.0
    best_meta: Optional[Dict[str, str]] = None

    for ck, meta in candidates.items():
        cset = set(ck.split())
        if not cset:
            continue
        common = len(kset & cset)
        union = len(kset | cset)
        score = common / union if union else 0.0
        if score > best_score:
            best_score = score
            best_meta = meta

    if best_meta is not None and best_score >= min_score:
        return best_meta["master_carrier"], best_meta["category"]
    return None


def _fuzzy_best(
    key: str,
    candidates: Dict[str, Dict[str, str]],
    cutoff: float = 0.9
) -> Optional[Tuple[str, str]]:
    """Fuzzy match by difflib as a last resort."""
    names = list(candidates.keys())
    matches = difflib.get_close_matches(key, names, n=1, cutoff=cutoff)
    if matches:
        meta = candidates[matches[0]]
        return meta["master_carrier"], meta["category"]
    return None



def _category_from_context(row: pd.Series) -> str:
    """Guess category from per-row context if still unmapped."""
    # Ensure safe string conversions
    vessel = str(row.get("Vessel", "") or "").strip()
    pol = str(row.get("POL", "") or "")
    pod = str(row.get("POD", "") or "")

    # UN/LOCODEs are 5 letters. IATA airport codes are 3 letters.
    looks_ocean = bool(vessel) or (len(pol) == 5 and len(pod) == 5)
    looks_air = (len(pol) == 3 and len(pod) == 3 and not vessel)

    ckey = str(row.get("carrier_key", "") or "")
    is_forwarder = any(h in ckey for h in _FORWARDER_HINTS)

    if is_forwarder:
        return "Forwarder"
    if looks_ocean and not looks_air:
        return "Ocean"
    if looks_air and not looks_ocean:
        return "Air"
    return "Other"


def aggressive_match_row(
    ckey: str,
    lut_norm: Dict[str, Dict[str, str]],
    row: Optional[pd.Series] = None,
    token_overlap_min: float = 0.5,
    fuzzy_cutoff: float = 0.9
) -> Tuple[str, str]:
    """Apply the full matching cascade for a single row."""
    # 1) exact normalized LUT hit
    if ckey in lut_norm:
        meta = lut_norm[ckey]
        return meta["master_carrier"], meta["category"]

    # 2) brand keyword match
    hit = _brand_keyword_match(ckey)
    if hit:
        return hit

    # 3) token overlap
    hit = _token_overlap_best(ckey, lut_norm, min_score=token_overlap_min)
    if hit:
        return hit

    # 4) fuzzy
    hit = _fuzzy_best(ckey, lut_norm, cutoff=fuzzy_cutoff)
    if hit:
        return hit

    # 5) context category fallback
    safe_row = row if row is not None else pd.Series()
    cat = _category_from_context(safe_row)
    return "Unmapped", cat


# =========================
# Your original data prep
# =========================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize date field names to expected schema
    rename_map = {
        "ETD": "etd",
        "ATD": "atd",
        "ETA": "eta",
        "ATA": "ata"
    }
    df = df.rename(columns=rename_map)

    # Delivery field aliases
    if "etd" in df.columns and "estimated_delivery" not in df.columns:
        df["estimated_delivery"] = df["etd"]
    if "atd" in df.columns and "actual_delivery" not in df.columns:
        df["actual_delivery"] = df["atd"]

    # Parse timestamps
    date_cols = ["eta", "ata", "etd", "atd", "estimated_delivery", "actual_delivery"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Transit time (arrival minus departure)
    if {"atd", "ata"}.issubset(df.columns):
        df["transit_time_days"] = (df["ata"] - df["atd"]).dt.days

    # Delay (actual vs estimated)
    if {"estimated_delivery", "actual_delivery"}.issubset(df.columns):
        df["delay_days"] = (df["actual_delivery"] - df["estimated_delivery"]).dt.days

    return df


def basic_validation_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "transit_time_days" in df.columns:
        df["flag_negative_transit"] = df["transit_time_days"] < 0
    else:
        df["flag_negative_transit"] = False

    if "delay_days" in df.columns:
        df["flag_negative_delay"] = df["delay_days"] < -5
    else:
        df["flag_negative_delay"] = False

    return df


# =========================
# Carrier mapping entry
# =========================
def apply_carrier_lookup_aggressive(
    df: pd.DataFrame,
    lut_norm: Dict[str, Dict[str, str]],
    carrier_col: str = DEFAULT_CARRIER_COL,
    token_overlap_min: float = 0.5,
    fuzzy_cutoff: float = 0.9
) -> pd.DataFrame:
    df = df.copy()

    if carrier_col not in df.columns:
        raise KeyError(f"Carrier column '{carrier_col}' not found")

    # Normalize input carrier strings
    df["carrier_key"] = df[carrier_col].apply(_normalize)

    # Per-row cascade matching
    results = df.apply(
        lambda r: aggressive_match_row(
            ckey=str(r["carrier_key"]),
            lut_norm=lut_norm,
            row=r,                            # always a Series here
            token_overlap_min=token_overlap_min,
            fuzzy_cutoff=fuzzy_cutoff
        ),
        axis=1
    )

    df["MasterCarrier"] = [mc for mc, _ in results]
    df["CarrierCategory"] = [cat for _, cat in results]
    return df


def carrier_qc(df: pd.DataFrame, carrier_col: str = DEFAULT_CARRIER_COL) -> pd.DataFrame:
    coverage = float((df["MasterCarrier"] != "Unmapped").mean()) if len(df) else 0.0
    print(f"Carrier mapping coverage: {coverage:.1%}")

    sample_unmapped = (
        df.loc[df["MasterCarrier"] == "Unmapped", carrier_col]
          .value_counts()
          .head(25)
          .reset_index()
          .rename(columns={"index": carrier_col, carrier_col: "count"})
    )
    return sample_unmapped


# =========================
# Full pipeline
# =========================
def run_pipeline(
    csv_path: str,
    carrier_col: str = DEFAULT_CARRIER_COL,
    lut_path: str = DEFAULT_LUT_PATH,
    token_overlap_min: float = 0.5,
    fuzzy_cutoff: float = 0.9
) -> pd.DataFrame:
    """
    Full pipeline:
      - load data
      - add derived fields and flags
      - load LUT and apply aggressive mapping
      - print coverage and show top unmapped
    """
    df = load_data(csv_path)
    df = add_derived_fields(df)
    df = basic_validation_flags(df)

    lut_norm = load_lookup_normalized(lut_path)
    df = apply_carrier_lookup_aggressive(
        df,
        lut_norm,
        carrier_col=carrier_col,
        token_overlap_min=token_overlap_min,
        fuzzy_cutoff=fuzzy_cutoff
    )

    unmapped = carrier_qc(df, carrier_col=carrier_col)
    if not unmapped.empty:
        print("\nTop Unmapped (sample):")
        print(unmapped.to_string(index=False))

    df = df[df["MasterCarrier"] != "Unmapped"]

    return df


if __name__ == "__main__":
    # Example:
    enriched = run_pipeline("data/processed_shipments_data.csv")
    enriched.to_csv("data/shipments_with_master_carrier.csv", index=False)