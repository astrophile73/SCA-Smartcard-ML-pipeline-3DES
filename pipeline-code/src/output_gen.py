import logging
import os
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Client-required report template (column headers only).
# Mirrors the first sheet headers in `KALKi TEST CARD.xlsx`.
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


def _format_3des_48_hex(k16_hex: str) -> str:
    """
    Input is expected to be 16-byte (32 hex) 2-key TDEA: K1||K2.
    Output is 24-byte (48 hex): K1||K2||K1 for tool compatibility.
    """
    k = _normalize_hex(k16_hex)
    if len(k) < 32:
        return ""
    k = k[:32]
    k1 = k[:16]
    k2 = k[16:32]
    return f"{k1}{k2}{k1}"


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

            # 3DES predictions (prefer explicit predicted_3des).
            # Support either new-style fields (3DES_KENC/...) or legacy final_3des_key (single MK).
            if predicted_3des:
                for col in ("3DES_KENC", "3DES_KMAC", "3DES_KDEK"):
                    v = _value_for_row(predicted_3des.get(col), idx)
                    out[col] = _format_3des_48_hex(v) if v else ""
            elif final_3des_key:
                # Legacy compatibility: treat final_3des_key as 16-byte and repeat as 2-key TDEA.
                v = _normalize_hex(final_3des_key)
                out["3DES_KENC"] = _format_3des_48_hex(v)
                out["3DES_KMAC"] = _format_3des_48_hex(v)
                out["3DES_KDEK"] = _format_3des_48_hex(v)

            # PIN: only if explicitly provided by ML stage; no hardcoded defaults.
            if pin:
                p = str(pin).strip()
                out["PIN"] = p if p and p.upper() not in {"FAIL", "NONE", "NAN"} else ""

            # RSA predictions (exact strings; no rstrip/padding that changes value).
            if predicted_rsa:
                for col in ("RSA_CRT_P", "RSA_CRT_Q", "RSA_CRT_DP", "RSA_CRT_DQ", "RSA_CRT_QINV"):
                    out[col] = _value_for_row(predicted_rsa.get(col), idx)

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
