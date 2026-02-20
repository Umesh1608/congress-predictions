# Competitive Performance Assessment: Our ML System vs. the Market

*February 2026*

## Our System's Verified Performance

| Model | Accuracy | AUC | Notes |
|-------|----------|-----|-------|
| **CatBoost (best)** | 69.96% ± 8.16% | 0.733 ± 0.087 | 5-fold temporal CV, Optuna-tuned |
| AutoGluon | 68.50% ± 7.56% | 0.729 ± 0.088 | Multi-model stacking, 300s |
| FT-Transformer | 68.01% ± 7.27% | 0.719 ± 0.090 | GNN embeddings sparse |
| LightGBM | 62.20% ± 8.55% | 0.658 ± 0.089 | Baseline |
| Ensemble (meta) | 62.16% ± 8.51% | 0.658 ± 0.089 | LogReg stacking |

- **Task:** Binary classification — will trade profit >2% at 180d horizon
- **Validation:** Walk-forward temporal CV, 45d purge gap + 14d embargo
- **Dataset:** 10,184 samples (from 11,367 trades after ETF/options filter)
- **Features:** 78 surviving features across 6 groups + 7 interaction terms
- **Fold variance:** Fold 2 hit ~75% (trending market), Folds 3-4 dropped to ~63-65% (regime shift) — **19-point swing between best and worst folds**

**Top 5 feature importances (LightGBM):** volatility_21d, volume_ratio_5d, ticker_purchase_ratio, price_change_5d, macd_signal. Market technicals dominate; interaction features (dir_x_momentum_21d, dir_x_52w_pos) rank 8th and 10th.

---

## Competitor Landscape

### Tier 1: Major Platforms (data aggregation — NO ML)

| Platform | What They Do | Performance Claims | Cost | Uses ML? |
|----------|-------------|-------------------|------|----------|
| **Unusual Whales** | Trade aggregation + NANC/KRUZ ETFs | 2024: Dems avg 31%, Reps 26%. 2025: only 32% beat market | $48/mo | **No** |
| **Capitol Trades** | Trade browser + performance calc | 2025: 140 members, 14,451 trades, $720M total. Top: Tuberville +179% | Free | **No** |
| **Quiver Quantitative** | Alt data + congressional backtester | Pelosi strategy: +744% since 2014 (backtested). 21% CAGR | $25/mo | **No** |
| **Autopilot** | Auto copy-trading via broker API | Pelosi tracker: +42% in 2024. $750M AUM, 132K accounts | $29/qtr | **No** |

### Tier 2: ETF Products (NANC vs. KRUZ vs. SPY)

| ETF | 2024 Return | 12mo Sharpe | Expense Ratio | AUM | SPY Correlation |
|-----|-----------|-------------|--------------|-----|----------------|
| **NANC** (Democrat) | +26.83% | 0.99 | 0.75% | $264.8M | 0.95 |
| **KRUZ** (Republican) | +14.45% | 1.02 | 0.83% | $69.4M | — |
| **SPY** (benchmark) | +24.9% | 0.96 | 0.09% | ~$600B | 1.00 |

**Key insight:** NANC beat SPY by ~1.9% gross in 2024, but only ~1.15% after fees. The outperformance is almost entirely explained by NVDA/Big Tech concentration (Pelosi effect), not stock-picking alpha. KRUZ lagged SPY by 10.5 points due to energy/industrial tilt. In 2025, only 32% of Congress beat the market — the "edge" is not consistent.

---

## Academic Benchmarks

### Studies Finding Alpha

| Study | Period | Finding |
|-------|--------|---------|
| Ziobrowski 2004 (JFQA) | 1993-1998 | Senate: +85 bps/month (~12% annualized) |
| Ziobrowski 2011 | 1985-2001 | House: +6% annually |
| **Wei & Zhou 2025 (NBER w34524)** | 1995-2021 | **Leaders who ascend to leadership: +47% annual alpha**. Two channels: political influence (shape policy to benefit holdings) + corporate access (predict corporate news via donors) |
| 2024 (Review of Intl Econ) | 100K+ trades | Positive returns linked to policy uncertainty |

### Studies Finding No Alpha

| Study | Period | Finding |
|-------|--------|---------|
| **Eggers & Hainmueller 2013** (J. Politics) | 2004-2008 | No evidence of informed trading. Avg member underperformed passive index by 2-3%/yr |
| **Belmont & Sacerdote 2020** (NBER) | 2012-2020 | "Senators as feckless as the rest of us." Purchases underperformed by 26 bps/6mo |
| 2022 (J. Public Economics) | Post-STOCK Act | Significant reduction in informed trading after STOCK Act |

### The Critical Alpha Decay Finding

**Ozlen & Batumoglu (2025, SSRN):** **70-80% of total alpha dissipates between the transaction date and the next trading day**, well before any public disclosure. By filing date, the remaining signal is "largely confirmatory rather than informative." This means copy-trading platforms (Autopilot, NANC) capture at best 20-30% of the original signal.

---

## Head-to-Head Comparison

| Capability | Unusual Whales | Capitol Trades | Quiver | Autopilot | **Our System** |
|-----------|---------------|---------------|--------|-----------|---------------|
| Trade aggregation | Yes | Yes | Yes | Yes | Yes (multi-source fallback) |
| Backward performance | Yes | Yes | Yes | Yes | Yes |
| **Forward ML prediction** | No | No | No | No | **Yes — 70% acc, 0.73 AUC** |
| **Network graph** | No | No | No | No | **Neo4j (6 node types, 11 rels)** |
| **NLP sentiment** | No | No | No | No | **FinBERT + spaCy NER** |
| **Trade-legislation timing** | No | No | No | No | **Suspicion scoring** |
| **Multi-signal fusion** | No | No | No | No | **4 signal types, composite** |
| **Position sizing** | No | No | No | No | **Screener with allocation** |
| Copy-trade execution | No | No | No | Yes ($750M AUM) | No |
| ETF product | NANC/KRUZ ($334M) | No | No | No | No |
| Backtesting | No | No | Yes ($25/mo) | No | Built into pipeline |

---

## Is 70% Accuracy / 0.73 AUC Good?

**vs. baselines:**
- Random (50%): +20 points — meaningful statistical edge
- Majority class (~54%): +16 points
- The fold variance (±8%) means in favorable regimes we hit 75%, in tough regimes we drop to 63%

**vs. competitor approaches:**
- No competitor makes forward predictions at all — they all show what *already happened*
- NANC ETF Sharpe of 0.99 is achieved with zero ML, purely through Big Tech concentration
- Autopilot's $750M AUM proves market demand, but it executes at disclosure date (30-45d lag)

**vs. academic findings:**
- Wei & Zhou show leadership trades carry massive alpha (+47%/yr) — our model captures this indirectly through `member_avg_return` and `member_win_rate`
- Ozlen's 70-80% alpha decay validates our dual-dating architecture, but also means actionable signals (post-disclosure) retain only 20-30% of the original edge

---

## Honest Weaknesses

1. **High fold variance (±8%)** — some market regimes are much harder; 19-point swing between best/worst folds
2. **Market technicals dominate** — top 5 features are all market-based (volatility, volume, momentum), meaning the model may be partly a market timing tool rather than a congressional-alpha extractor
3. **Sentiment features are dead weight** — 126 media items total; sentiment contributes near-zero signal for 95%+ of trades
4. **Network features limited** — only 4% of campaign contributions matched to tickers; GNN embeddings 115/128 columns zero-variance (Neo4j graph too sparse)
5. **No live trading validation** — all metrics from cross-validation; no paper trading track record. Autopilot has $750M AUM with real returns
6. **Disclosure lag problem** — per Ozlen, 70-80% of alpha decays before disclosure. Our screener acts at disclosure time
7. **No risk-adjusted metrics** — we measure accuracy/AUC, not Sharpe ratio or max drawdown

---

## Bottom Line

### What We Do Better Than Everyone

**We are the only system that makes forward-looking predictions.** Every competitor is reactive — they show what Congress *already did*. We predict whether those trades *will be profitable* at 70% accuracy. The network graph, legislative timing analysis, and multi-signal fusion are capabilities no competitor offers at any price.

### What We Need to Be Honest About

1. **No live track record.** Autopilot has $750M AUM and real user returns. We have CV metrics only.
2. **The alpha decay problem is real.** Ozlen shows 70-80% of alpha gone by filing date — our actionable edge is smaller than raw accuracy suggests.
3. **NANC has beaten the market with zero ML** — just by following Pelosi's Big Tech bets at 0.75% fees. Simple may beat sophisticated.
4. **Accuracy ≠ profitability.** 70% directional accuracy doesn't account for win/loss magnitude, position sizing, or transaction costs.
5. **Market technicals are doing a lot of heavy lifting** — the "congressional" alpha features (legislative timing, network, sentiment) contribute less than basic RSI/MACD/volatility.

### Strategic Opportunities

1. **Paper trade for 3-6 months** — build a live track record with the screener vs. NANC/SPY to prove (or disprove) the edge
2. **Target leadership trades specifically** — Wei & Zhou show +47%/yr for leaders. Add congressional leadership role as a high-weight feature
3. **Enrich sparse features** — media data (126 items) and campaign-to-ticker matching (4%) are low-hanging fruit
4. **Compute risk-adjusted metrics** — Sharpe ratio, max drawdown, and information ratio for screener recommendations
5. **Pre-disclosure prediction** — use legislative signals and committee activity to predict *which members will trade*, not just whether past trades profit. This captures alpha before it decays

---

## Sources

- [Unusual Whales 2024 Report](https://unusualwhales.com/congress-trading-report-2024) / [2025 Report](https://unusualwhales.com/congress-trading-report-2025)
- [Capitol Trades 2024 Year in Review](https://www.capitoltrades.com/articles/capitol-trades-2024-year-in-review-a-journey-through-market-moves-politics-breakthroughs-2025-01-10)
- [Quiver Quantitative - Pelosi Strategy](https://www.quiverquant.com/strategies/s/Nancy%20Pelosi/)
- [Autopilot $750M AUM - InvestmentNews](https://www.investmentnews.com/fintech/autopilot-surges-to-750m-aum-touts-ria-growth-as-users-copy-pelosi-buffett-trades/260729)
- [NANC vs KRUZ - etf.com](https://www.etf.com/sections/etf-basics/nanc-vs-kruz-battle-congress-stock-trackers)
- [NANC ETF - PortfoliosLab](https://portfolioslab.com/symbol/NANC)
- [Wei & Zhou 2025 - NBER w34524](https://www.nber.org/papers/w34524)
- [Eggers & Hainmueller 2013](https://andy.egge.rs/papers/EggMueller_capitol_losses_jop_2013.pdf)
- [Ozlen & Batumoglu 2025 - SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5966834)
- [UCSD/PNAS 2025 - Congressional trading erodes public trust](https://rady.ucsd.edu/why/news/2025/05-20-congressional-stock-trading-severely-undermines-public-trust-and-compliance-with-the-law.html)
