# US Stablecoin & BTC Market Structure Monitoring Dashboard

This project is a Streamlit-based monitoring dashboard for **U.S.-quoted stablecoins** and **BTC market structure** using **Kraken public REST API** data.

It is designed to distinguish between two very different market structures. For **stablecoins**, the dashboard evaluates **peg stability**, **near-peg liquidity**, and **structural signs of depeg risk**. For **BTC pairs**, the dashboard removes peg logic entirely and instead evaluates **order book quality**, **spread conditions**, **depth near mid-price**, **imbalance**, and **market regime**. The result is a monitoring framework that treats stablecoins as dollar-linked instruments requiring structural defense around par, while treating BTC as a freely floating market priced through liquidity and two-sided depth.

## Repository

**GitHub Repository:**  
[https://github.com/kayyrod21/US-stablecoin-btc-monitoring-dashboard

## Live App

**Live Dashboard URL:**  
[https://us-stablecoin-btc-monitoring-dashboard.streamlit.app/

## Overview

The dashboard dynamically adapts based on the selected **Asset Group**.

In **Stablecoins** mode, the system monitors instruments quoted against the U.S. dollar and evaluates whether they are trading close to par with sufficient liquidity support near the mid-price. In this mode, the dashboard emphasizes **peg deviation**, **near-peg depth**, **spread compression or widening**, **depeg event detection**, and **structural market quality**.

In **BTC / Majors** mode, the system does not apply peg logic. Instead, it evaluates whether the selected BTC pair is trading in a structurally healthy market with strong depth, tight spreads, and low execution friction. The focus shifts from peg defense to **market quality**, **liquidity resilience**, and **order book structure**.

## Stablecoin Market Structure Analysis

Stablecoins are structurally different from BTC because they are intended to maintain parity with the U.S. dollar. In normal market conditions, price should remain tightly clustered around $1.00, with only minor microstructure noise around par. Because of this, the dashboard treats stablecoin monitoring as a question of **defense strength**: how much visible liquidity exists near the peg, how balanced the order book remains, and whether price begins to drift only after liquidity has already deteriorated.

Across the stablecoin pairs observed in this dashboard, the strongest regimes were characterized by **very tight spreads**, **high depth within ±10bp and ±25bp**, and **minimal peg deviation**. In those conditions, price behavior remained stable and the order book showed credible two-sided support near the mid-price. This is exactly what a healthy stablecoin market should look like: not simply a printed price close to $1.00, but an actively defended structure around the peg.

One of the clearest patterns observed was that **liquidity deterioration often appeared before meaningful price instability**. In weaker stablecoin pairs, the first warning sign was not always immediate deviation from par. Instead, the near-mid order book became thin, especially within ±10bp, while spreads widened modestly and imbalance skewed toward one side. That pattern matters because it suggests the market was becoming easier to push before the price itself had fully reflected the stress. In other words, **depth compression can act as an early warning signal before visible depeg behavior emerges**.

The dashboard’s **combined depth** and **near-peg heat map** were especially useful in making that distinction. Pairs like **USDT/USD** and **USDC/USD** consistently showed much deeper near-mid liquidity than thinner stablecoin pairs, which supports the interpretation that their market structure is more resilient. By contrast, pairs such as **USD1/USD** and **USAT/USD** frequently showed materially shallower depth near the mid-price. Even when they were not fully depegged, they were structurally less defended. This was visible not only in the absolute ±10bp and ±25bp numbers, but also in the relative contrast shown across the cross-pair depth view.

Another pattern we observed was that **spread widening alone was not enough to interpret as structural failure**, but spread widening combined with shallow near-mid depth was far more concerning. A pair might temporarily show mild imbalance or modest movement without being unstable. But once shallow depth, wider spread, and recurring depeg events appeared together, the market quality clearly deteriorated. This is why the dashboard separates **market regime** and **monitoring state** from the raw price itself. A stablecoin can appear near par while still trading in a thinner, weaker market structure.

The **depeg alerts** are intentionally threshold-based rather than noise-sensitive. Small oscillations around $1.00 are common and should not be overinterpreted. The dashboard therefore attempts to distinguish between ordinary microstructure movement and true structural instability. This makes the alerts more useful from a monitoring perspective because they are meant to surface deterioration in peg quality, not harmless fluctuations.

Taken together, the stablecoin analysis suggests a clear ordering of stress. In healthy conditions, spreads remain near zero, depth close to the peg is strong, and the book remains relatively balanced. In weaker conditions, depth thins first, spreads widen next, and only then do visible depeg events become more likely. That sequence is one of the most important structural observations surfaced by this dashboard.

## BTC Market Structure Analysis

BTC pairs are evaluated differently because BTC has no peg target. For BTC/USD and BTC/USDT, the relevant question is not whether price remains close to a fixed anchor, but whether the market is trading with **tight spread**, **deep liquidity**, **credible near-mid support**, and **order book resilience**.

In the BTC pairs observed through this dashboard, the order books showed substantially deeper near-mid liquidity than the thinner stablecoin pairs. The liquidity wall and liquidity summary made it clear that BTC/USD in particular could support much larger notional movement near mid-price before slippage increased materially. This is consistent with a healthier execution environment and stronger two-sided participation.

The dashboard’s **DEEP / STABLE** regime classification for BTC corresponded to tight bid-ask spreads, substantial depth within ±10bp and ±25bp, and relatively low stress readings. In this context, fluctuations in imbalance are more about short-term positioning and directional flow than about systemic structural weakness. BTC liquidity is not defending a fixed peg. It is adapting to price discovery. That makes the interpretation fundamentally different from stablecoin monitoring.

This distinction is central to the project. Stablecoin liquidity is best understood as **defensive liquidity around par**, while BTC liquidity is best understood as **adaptive market structure supporting execution quality and price discovery**. The dashboard is built around that difference rather than forcing both asset types into the same monitoring framework.

## Dashboard Architecture

- **`app.py`**  
  Handles layout, asset-group branching, KPI ribbon logic, tab rendering, alerts, monitoring state labels, and data orchestration.

- **`src/ui_components.py`**  
  Renders reusable visual components including the order book depth profile, heat map, and supporting chart elements.

- **Stablecoin mode**  
  Includes peg-aware monitoring such as peg deviation, near-peg depth, combined depth, and depeg event tracking.

- **BTC / Majors mode**  
  Removes peg-specific logic and focuses on spread, depth, order book imbalance, stress, and liquidity regime.

- **Pair selection logic**  
  Driven by Kraken AssetPairs metadata rather than hardcoded static lists where possible, while still preserving BTC-focused majors presentation in the UI.

## Tabs

### Stablecoins

- Overview
- Peg & Stability
- Liquidity
- Surveillance
- Audit Log

### BTC / Majors

- Overview
- Liquidity
- Surveillance
- Audit Log

## Alerts & Thresholds

The alerts section behaves differently depending on asset group.

In **Stablecoins** mode, thresholds are used to monitor peg deterioration and liquidity deterioration near the peg. This includes depeg event tracking and near-mid liquidity warnings.

In **BTC / Majors** mode, alerts focus on liquidity and market-quality conditions rather than peg behavior. Since BTC is not a dollar-pegged instrument, there is no depeg logic in this branch of the dashboard.

## Data Source

This dashboard uses **Kraken Public REST API** endpoints.

Endpoints currently used include:

- `/0/public/AssetPairs`
- `/0/public/Depth`
- `/0/public/Ticker`
- `/0/public/OHLC`

Kraken API documentation:  
https://docs.kraken.com/api/docs/rest-api/get-tradable-asset-pairs/

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

