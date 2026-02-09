def get_ticker_aliases(ticker: str):
    """
    Expand a trading ticker into common Reddit-friendly aliases.
    """
    base = ticker.upper().strip()

    aliases = set()

    # Dollar Index (DXY) — Reddit/search uses "dollar index", "DXY"
    if base in ("DXY", "DOLLARINDEX", "DOLLAR-INDEX"):
        return ["dollar index", "DXY", "DXY index", "dollar index DXY", "US dollar index", "USD index"]

    # Handle crypto pairs like SOL-USD, BTC-USD
    if "-" in base:
        asset = base.split("-")[0]
        quote = base.split("-")[1] if len(base.split("-")) > 1 else ""
        aliases.update({
            asset,
            f"${asset}",
            base,
            base.replace("-", "/"),
            f"{asset} {quote}",   # e.g. "btc usd"
            f"{asset}{quote}",    # e.g. "btcusd"
        })

        # Hard-coded common crypto/commodity names (extendable)
        crypto_names = {
            "BTC": "Bitcoin",
            "ETH": "Ethereum",
            "SOL": "Solana",
            "ADA": "Cardano",
            "XRP": "Ripple",
            "DOGE": "Dogecoin",
            "AVAX": "Avalanche",
            "XAU": "Gold",
            "XAG": "Silver",
        }
        if asset in crypto_names:
            aliases.add(crypto_names[asset])

        # Forex: common names so news (e.g. FXStreet) matches "Euro", "Pound", "Yen"
        forex_names = {
            "EUR": "Euro",
            "GBP": "Pound",
            "JPY": "Yen",
            "AUD": "Aussie",
            "CHF": "Franc",
            "CAD": "Loonie",
            "NZD": "Kiwi",
            "USD": "dollar",
        }
        if asset in forex_names:
            aliases.add(forex_names[asset])
        if quote == "USD" and quote in forex_names:
            aliases.add(forex_names[quote])

    else:
        aliases.update({
            base,
            f"${base}"
        })

    return list(aliases)
