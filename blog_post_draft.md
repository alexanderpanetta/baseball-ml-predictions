# Play Ball! What Machine Learning Predicts This Baseball Season

*Opening Day 2026*

Every spring, the baseball world buzzes with predictions. Who'll lead the league in homers? Which pitchers will dominate? This year, I asked two machine learning systems to make their calls — and the results reveal as much about AI as they do about baseball.

I trained two independent ML systems on a decade of player statistics from the SABR Lahman Database (2015–2025) and asked each to predict the 2026 season. One uses scikit-learn's Gradient Boosting algorithm — a well-understood, transparent model that any data scientist can reproduce on a laptop. The other uses Google's Vertex AI AutoML, a black-box system that automatically tests hundreds of model architectures and picks the best one.

The full predictions for every qualified MLB player are in the linked spreadsheets below. Here are the highlights.

## The Headline Predictions

### Home Run Leaders (scikit-learn)

| Player | Predicted HR | 2025 Actual |
|--------|-------------|-------------|
| Kyle Schwarber | 42 | 56 |
| Aaron Judge | 41 | 53 |
| Shohei Ohtani | 40 | 55 |
| Cal Raleigh | 39 | 60 |
| Junior Caminero | 35 | 45 |
| Pete Alonso | 33 | 38 |

The model predicts **regression toward the mean** for last year's power hitters — a well-documented statistical phenomenon. Cal Raleigh's breakout 60-homer season is projected to drop significantly. The model has learned that outlier seasons rarely repeat: the features that matter most are the *weighted 3-year average* and *career average*, not just last year's number.

### Batting Average Leaders (scikit-learn)

| Player | Predicted AVG | 2025 Actual |
|--------|--------------|-------------|
| Bobby Witt Jr. | .288 | .295 |
| Aaron Judge | .288 | .331 |
| Juan Soto | .281 | .263 |
| Shohei Ohtani | .273 | .282 |
| Vladimir Guerrero Jr. | .275 | .292 |

**A note on what this means — and doesn't mean.** The model isn't predicting that the 2026 batting champion will hit .288. It's saying that .288 is the highest *expected value* for any individual player, given their track record. But whoever actually wins the batting title will, by definition, be having an unusually good year — an outcome the model can't foresee for a specific player. Think of it like this: if you could replay the 2026 season 1,000 times, Judge would average around .288. But in some of those simulations he'd hit .320, and in others .260. The model gives you the center of that distribution, not the upside. Someone will get hot and hit .315+. The model just can't tell you who.

### Top Pitchers by ERA (scikit-learn)

| Pitcher | Predicted ERA | 2025 Actual |
|---------|--------------|-------------|
| Garrett Crochet | 2.91 | 2.59 |
| Jeremiah Estrada | 2.78 | 3.45 |
| Devin Williams | 2.80 | 4.79 |
| Tarik Skubal | 3.01 | 2.21 |
| Paul Skenes | 3.05 | 1.97 |

Paul Skenes' extraordinary 1.97 ERA in 2025 is predicted to regress to 3.05 — still excellent, but the model knows that sub-2.00 ERAs almost never repeat in consecutive seasons.

## How the Models Work

Both models use the same input features — this is key for a fair comparison:

1. **Previous season stats** — last year's numbers
2. **Weighted 3-year rolling average** — recent seasons weighted higher (3x weight for most recent, 2x for year before, 1x for two years ago)
3. **Career averages** — lifetime performance baseline
4. **Year-over-year trend** — is the player improving or declining?
5. **Age and age²** — captures the well-known aging curve (players peak around 27-29, then gradually decline)
6. **Experience** — number of qualified MLB seasons

The scikit-learn model is a **Gradient Boosting Regressor** with fixed hyperparameters and a random seed of 42 — meaning anyone who runs the code will get the exact same predictions.

## How Accurate Are These Predictions?

I held out the entire 2025 season as a test set — the models never saw 2025 data during training. Here's how they performed:

| Stat | Mean Absolute Error | What That Means |
|------|-------------------|-----------------|
| AVG | .020 | Off by ~20 points of batting average |
| HR | 6.1 | Off by ~6 home runs |
| RBI | 16.4 | Off by ~16 RBIs |
| ERA | 0.85 | Off by ~0.85 ERA points |
| SO (pitching) | 26.9 | Off by ~27 strikeouts |

**The honest truth:** Baseball is inherently unpredictable. Injuries, trades, slumps, and hot streaks create variance that no model can capture. R² scores ranged from 0.13 to 0.58 — meaning our models explain 13-58% of the variance in next-year performance, depending on the stat. Stolen bases (R²=0.54) and saves (R²=0.58) were the most predictable; slugging percentage (R²=0.19) and ERA (R²=0.14) were the hardest.

This is consistent with academic research on baseball prediction — season-level forecasting is fundamentally limited by the randomness inherent in the sport.

## scikit-learn vs. Google AutoML

[TODO: Fill in once AutoML results are ready]

How did the $40 black-box compare to the free, transparent model? For batting average and home runs:

- AutoML predicted AVG of X vs. scikit-learn's Y
- AutoML predicted HR of X vs. scikit-learn's Y
- The correlation between the two models was X

[Analysis of where they agreed and diverged]

## What the Models Can't See

These predictions assume every player stays healthy and plays a full season. They don't account for:

- **Injuries** — the biggest source of prediction error
- **Trades and team changes** — a hitter moving to Coors Field vs. Oracle Park matters
- **Rule changes** — any new MLB rule changes for 2026
- **Rookies with <2 seasons** — limited data means wider prediction intervals
- **The 2020 COVID season** — that 60-game season is in the training data and may skew counting stats

## Reproduce This Yourself

All code and data are freely available. You need Python 3.10+ and a few packages:

```bash
pip install pandas requests pyreadr scikit-learn openpyxl
python3 01_pull_data.py      # Download Lahman database
python3 02_sklearn_models.py  # Train models, generate predictions
python3 03_create_spreadsheets.py  # Create spreadsheet
```

The data comes from the SABR Lahman Baseball Database (Creative Commons licensed), and every model uses `random_state=42` for exact reproducibility.

## The Spreadsheets

- **[Batting Predictions](link)** — All 348 qualified batters with predicted AVG, HR, RBI, R, H, 2B, SB, BB, OBP, SLG
- **[Pitching Predictions](link)** — All 339 qualified pitchers with predicted ERA, W, L, SO, WHIP, IP, SV
- **[Methodology](link)** — Full documentation of features, model parameters, and evaluation metrics

---

*Alex Panetta covers AI governance and data science. This analysis was built with assistance from Claude (Anthropic) for code development and scikit-learn/Google Vertex AI for machine learning.*
