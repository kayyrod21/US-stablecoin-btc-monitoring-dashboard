from typing import Any, Dict, List


ALLOWED_STABLES = {"USDT", "USDC", "USD1", "USAT"}


def get_btc_stablecoin_ws_pairs(assetpairs: Dict[str, Any]) -> List[str]:
    """
    Return a sorted list of Kraken wsname strings for BTC vs stablecoin pairs.

    Filters:
    - Requires a wsname (e.g. "XBT/USDT")
    - One side must be BTC (XBT/XXBT or wsname containing XBT)
    - Other side must be one of the allowed stablecoins
    - Prefers status == "online", but includes others after
    """
    if not assetpairs:
        return []

    online: List[str] = []
    others: List[str] = []

    for meta in assetpairs.values():
        if not isinstance(meta, dict):
            continue

        wsname = meta.get("wsname")
        if not isinstance(wsname, str):
            continue

        ws_upper = wsname.upper()
        if "/" in wsname:
            base_ws, quote_ws = [
                part.strip().upper() for part in wsname.split("/", 1)
            ]
        else:
            base_ws, quote_ws = ws_upper, ""

        base_id = str(meta.get("base", "")).upper()
        quote_id = str(meta.get("quote", "")).upper()

        def is_btc(sym: str) -> bool:
            return sym in {"XBT", "XXBT", "BTC"}

        def is_stable(sym: str) -> bool:
            return sym in ALLOWED_STABLES

        has_btc = (
            is_btc(base_id)
            or is_btc(quote_id)
            or "XBT" in base_ws
            or "XBT" in quote_ws
        )

        has_stable = (
            is_stable(base_id)
            or is_stable(quote_id)
            or any(
                ws_upper.endswith(f"/{stable}") or ws_upper.startswith(f"{stable}/")
                for stable in ALLOWED_STABLES
            )
        )

        if not (has_btc and has_stable):
            continue

        status = str(meta.get("status", "")).lower()
        target = online if status == "online" else others
        if wsname not in target:
            target.append(wsname)

    return sorted(online) + sorted(others)

