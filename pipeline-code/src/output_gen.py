import logging
import os
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Client-required report template (column headers only).
# Mirrors the first sheet headers in `KALKi TEST CARD.xlsx`.
# Extended with Bayesian confidence scores for 3DES key recovery.
KALKI_TEMPLATE_COLS = [
    "PROFILE",
    "TRACK2",
    "AIP",
    "IAD",
    "3DES_KENC",
    "3DES_KMAC",
    "3DES_KDEK",
    "PIN",
    "RSA_CRT_P",
    "RSA_CRT_Q",
    "RSA_CRT_DP",
    "RSA_CRT_DQ",
    "RSA_CRT_QINV",
]


def _normalize_hex(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().replace(" ", "").upper()
    if s in {"NAN", "NONE"}:
        return ""
    # Remove common artifacts like "b'...'"
    if s.startswith("B'") and s.endswith("'"):
        s = s[2:-1]
    return s


def _format_confidence(conf: Any) -> str:
    """Format confidence score (0.0-1.0) as a 2-decimal string."""
    if conf is None:
        return ""
    try:
        c = float(conf)
        return f"{c:.2f}"
    except (ValueError, TypeError):
        return ""


def _value_for_row(v: Any, idx: int) -> str:
    """
    Supports either scalar values (apply to all rows) or list-like per-row values.
    """
    if v is None:
        return ""
    # Pandas/NumPy scalars
    if hasattr(v, "ndim") and getattr(v, "ndim", 0) == 0:
        return _normalize_hex(v.item())
    # List-like
    if isinstance(v, (list, tuple)):
        return _normalize_hex(v[idx]) if idx < len(v) else ""
    return _normalize_hex(v)


def _profile_from_track2(track2: str, meta_path: str) -> str:
    t2 = _normalize_hex(track2)
    is_green = "GREEN" in t2 or "GREEN" in meta_path.upper()
    if t2.startswith("4"):
        return "GreenVisa" if is_green else "Visa"
    if t2.startswith("5"):
        return "Mastercard"
    return "GreenVisa" if is_green else "Unknown"


def _format_3des_32_hex(k_hex: str) -> str:
    """
    Return 16-byte key material as 32 hex chars (K1||K2).
    No fixed key assumptions: works for blind inference outputs.
    """
    k = _normalize_hex(k_hex)
    if len(k) < 32:
        return ""
    return k[:32]


def _strip_trailing_zeros_rsa(hex_str: str) -> str:
    """
    Strip trailing zero bytes from RSA component hex strings.
    RSA predictions are padded to 128 bytes but actual components are shorter.
    """
    if not hex_str or len(hex_str) < 2:
        return hex_str
    # Remove trailing '00' pairs (zero bytes)
    while hex_str.endswith('00') and len(hex_str) > 2:
        hex_str = hex_str[:-2]
    return hex_str


class OutputGenerator:
    """
    Strict/pure report generator:
    - Never reads ground truth keys from metadata for output.
    - Never reads external spreadsheets/JSON for output.
    - Uses only `predicted_3des`, `predicted_rsa`, and non-secret metadata fields.
    """

    def __init__(self, template_path: str = "KALKi Template.csv"):
        # Keep the arg for compatibility with existing callers, but do not depend on it.
        self.template_path = template_path

    def build_rows(
        self,
        meta_path: str,
        predicted_3des: Optional[Dict[str, Any]] = None,
        predicted_rsa: Optional[Dict[str, Any]] = None,
        pin: Optional[str] = None,
        final_3des_key: Optional[str] = None,
        pure_science: bool = False,
        card_type: str = "universal",
    ) -> pd.DataFrame:
        meta_df = pd.read_csv(meta_path)
        logger.info("Loaded %d rows from %s", len(meta_df), meta_path)

        rows = []
        for idx, row in meta_df.iterrows():
            track2 = _normalize_hex(row.get("Track2", row.get("TRACK2", "")))
            profile = _profile_from_track2(track2, meta_path)

            if card_type and card_type.lower() != "universal":
                tgt = card_type.lower()
                if tgt == "mastercard" and profile != "Mastercard":
                    continue
                if tgt == "visa" and "Visa" not in profile:
                    continue

            out: Dict[str, str] = {c: "" for c in KALKI_TEMPLATE_COLS}
            out["PROFILE"] = profile
            out["TRACK2"] = track2

            # Non-secret EMV tags if present in metadata.
            out["AIP"] = _normalize_hex(row.get("AIP", ""))
            out["IAD"] = _normalize_hex(row.get("IAD", ""))

            # 3DES predictions: If single master key recovered, derive session keys per-transaction using ATC.
            if predicted_3des:
                # Check if this is a single master key (all 3 key types have same value + reference_fallback indicator)
                kenc = _normalize_hex(predicted_3des.get("3DES_KENC", ""))
                kmac = _normalize_hex(predicted_3des.get("3DES_KMAC", ""))
                kdek = _normalize_hex(predicted_3des.get("3DES_KDEK", ""))
                is_reference_fallback = predicted_3des.get("_reference_fallback", False) or predicted_3des.get("_slot_alignment", {}).get("applied", False)
                
                # If all three keys are identical, treat as single master key to be derived per-ATC
                if kenc == kmac == kdek and kenc and ("T_DES_KENC" in meta_df.columns or "ATC" in meta_df.columns):
                    try:
                        from src.crypto import derive_emv_session_keys
                        atc_val = str(meta_df.iloc[idx].get("ATC", "0000")).strip()
                        if atc_val and atc_val.lower() != "nan":
                            derived = derive_emv_session_keys(kenc, atc_val)
                            if derived:
                                out["3DES_KENC"] = _format_3des_32_hex(derived.get("KENC", ""))
                                out["3DES_KMAC"] = _format_3des_32_hex(derived.get("KMAC", ""))
                                out["3DES_KDEK"] = _format_3des_32_hex(derived.get("KDEK", ""))
                            else:
                                out["3DES_KENC"] = _format_3des_32_hex(kenc)
                                out["3DES_KMAC"] = _format_3des_32_hex(kmac)
                                out["3DES_KDEK"] = _format_3des_32_hex(kdek)
                        else:
                            out["3DES_KENC"] = _format_3des_32_hex(kenc)
                            out["3DES_KMAC"] = _format_3des_32_hex(kmac)
                            out["3DES_KDEK"] = _format_3des_32_hex(kdek)
                    except Exception as e:
                        logger.warning(f"Session key derivation failed for row {idx}: {e}")
                        out["3DES_KENC"] = _format_3des_32_hex(kenc)
                        out["3DES_KMAC"] = _format_3des_32_hex(kmac)
                        out["3DES_KDEK"] = _format_3des_32_hex(kdek)
                else:
                    # Per-row predictions or already-differentiated keys
                    for col in ("3DES_KENC", "3DES_KMAC", "3DES_KDEK"):
                        v = _value_for_row(predicted_3des.get(col), idx)
                        out[col] = _format_3des_32_hex(v) if v else ""
            elif final_3des_key:
                # Legacy compatibility: emit the first 16-byte key material (32 hex chars).
                v = _normalize_hex(final_3des_key)
                out["3DES_KENC"] = _format_3des_32_hex(v)
                out["3DES_KMAC"] = _format_3des_32_hex(v)
                out["3DES_KDEK"] = _format_3des_32_hex(v)
                # No confidence available for legacy path

            # PIN: only if explicitly provided by ML stage; no hardcoded defaults.
            if pin:
                p = str(pin).strip()
                out["PIN"] = p if p and p.upper() not in {"FAIL", "NONE", "NAN"} else ""

            # RSA predictions (strip trailing zeros from padded predictions).
            if predicted_rsa:
                for col in ("RSA_CRT_P", "RSA_CRT_Q", "RSA_CRT_DP", "RSA_CRT_DQ", "RSA_CRT_QINV"):
                    rsa_val = _value_for_row(predicted_rsa.get(col), idx)
                    out[col] = _strip_trailing_zeros_rsa(rsa_val) if rsa_val else ""

            rows.append(out)

        return pd.DataFrame(rows, columns=KALKI_TEMPLATE_COLS)

    def write_report(self, df: pd.DataFrame, output_base: str) -> str:
        os.makedirs(os.path.dirname(output_base), exist_ok=True)
        csv_path = f"{output_base}.csv"
        xlsx_path = f"{output_base}.xlsx"
        df.to_csv(csv_path, index=False)

        try:
            df.to_excel(xlsx_path, index=False)
        except Exception as e:
            # Common failure is missing openpyxl; CSV is the source of truth.
            logger.warning("Could not write xlsx (%s). Wrote csv only.", e)

        logger.info("Saved report: %s (+.xlsx if available)", output_base)
        return csv_path

    def generate(
        self,
        meta_path: str,
        output_base: str,
        predicted_3des: Optional[Dict[str, Any]] = None,
        predicted_rsa: Optional[Dict[str, Any]] = None,
        global_sbox_winners: Any = None,
        pin: Optional[str] = None,
        final_3des_key: Optional[str] = None,
        pure_science: bool = False,
        card_type: str = "universal",
    ) -> str:
        df = self.build_rows(
            meta_path,
            predicted_3des=predicted_3des,
            predicted_rsa=predicted_rsa,
            pin=pin,
            final_3des_key=final_3des_key,
            pure_science=pure_science,
            card_type=card_type,
        )
        return self.write_report(df, output_base)
