import logging
from typing import Any, Dict, Optional

import requests


BASE_API_URL = "https://api.kraken.com/0/public/"


def get_asset_pairs(
    pair: Optional[str] = None,
    aclass_base: Optional[str] = None,
    info: Optional[str] = None,
    country_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch available asset pairs from Kraken's public API.

    Parameters loosely mirror Kraken's AssetPairs endpoint:
    - pair: optional comma-separated list of pairs (e.g. "XBTUSD,ETHUSD")
    - aclass_base: base asset class filter (e.g. "currency", "tokenized_asset")
    - info: "info", "leverage", "fees", or "margin"
    - country_code: optional 2-letter country code filter

    Returns a dictionary mapping pair names (e.g. "XBTUSD") to
    their metadata. Returns an empty dict if the request fails
    or the response is not in the expected format.
    """
    url = f"{BASE_API_URL}AssetPairs"

    params: Dict[str, Any] = {}
    if pair:
        params["pair"] = pair
    if aclass_base:
        params["aclass_base"] = aclass_base
    if info:
        params["info"] = info
    if country_code:
        params["country_code"] = country_code

    try:
        response = requests.get(url, params=params or None, timeout=10)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        logging.warning("Failed to fetch asset pairs from Kraken: %s", exc)
        return {}

    # Kraken wraps results in {"error": [...], "result": {...}}
    if not isinstance(data, dict):
        return {}

    if data.get("error"):
        logging.warning("Kraken API returned errors: %s", data["error"])
        return {}

    result = data.get("result")
    return result if isinstance(result, dict) else {}

