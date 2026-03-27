# Play Ball! What Machine Learning Predicts This Baseball Season

*Opening Day 2026*

Every spring, the baseball world buzzes with predictions. Who'll lead the league in homers? Which pitchers will dominate? This year, I asked two machine learning systems to make their calls — and the results reveal as much about AI as they do about baseball.

I trained two independent ML systems on a decade of player statistics from the SABR Lahman Database (2015–2025) and asked each to predict the 2026 season. One uses scikit-learn's Gradient Boosting algorithm — a well-understood, transparent model that any data scientist can reproduce on a laptop. The other uses Google's Vertex AI AutoML, a black-box system that automatically tests hundreds of model architectures and picks the best one.

The full predictions for every qualified MLB player are in the linked spreadsheets below. Here are the highlights.

## The Headline Predictions

### Home Run Leaders (scikit-learn)

| Player | Predicted HR | 2025 Actual |
|--------|-------------|-------------|
| Kyle Schwarber | 54 | 56 |
| Aaron Judge | 53 | 53 |
| Shohei Ohtani | 51 | 55 |
| Cal Raleigh | 50 | 60 |
| Junior Caminero | 44 | 45 |
| Pete Alonso | 41 | 38 |

The model predicts **regression toward the mean** for last year's power hitters — a well-documented statistical phenomenon. Cal Raleigh's breakout 60-homer season is projected to drop to 50. The model has learned that outlier seasons rarely repeat: the features that matter most are the *weighted 3-year average* and *career average*, not just last year's number.

### Batting Average Leaders (scikit-learn)

| Player | Predicted AVG | 2025 Actual |
|--------|--------------|-------------|
| Aaron Judge | .331 | .331 |
| Bobby Witt Jr. | .331 | .295 |
| Juan Soto | .318 | .263 |
| Julio Rodriguez | .317 | [2025 actual] |
| Vladimir Guerrero Jr. | .305 | .292 |
| Shohei Ohtani | .301 | .282 |

### Top Pitchers by ERA (scikit-learn)

| Pitcher | Predicted ERA | 2025 Actual |
|---------|--------------|-------------|
| Jeremiah Estrada | 1.34 | 3.45 |
| Devin Williams | 1.36 | 4.79 |
| Garrett Crochet | 1.62 | 2.59 |
| Tarik Skubal | 1.83 | 2.21 |
| Paul Skenes | 1.92 | 1.97 |

The Devin Williams prediction is one of the most interesting. He had a terrible 2025 (4.79 ERA), but the model looks past one bad year: his career track record is elite, his weighted 3-year average is dominant, and the model expects regression *upward* — back toward his true talent level. That's not blind optimism; it's the same math that pulls Cal Raleigh's 60 homers back down.

## The Calibration Problem (And What It Taught Me About ML)

Here's something I didn't expect, and it's probably the most important thing in this piece for anyone who uses machine learning.

When I first ran the model, the predicted batting champion hit **.288**. Nobody was above .300. In a sport where the batting title winner typically hits .330–.350, that was obviously wrong. But the model wasn't broken — every line of code checked out. The rankings were sensible. The mean was correct. So what happened?

It's a phenomenon called **variance shrinkage**, and it affects virtually every regression model. Here's the intuition:

The model trains on ~2,700 player-seasons. Most players are .240–.260 hitters. When the algorithm builds its decision trees, it optimizes for accuracy across *all* those players — and the safest prediction is always one closer to the middle. Predicting .310 for a career .318 hitter is "safer" (lower average error) than predicting .325, because even elite hitters sometimes hit .280. The model hedges.

The result: the predicted mean was right (.246, matching the real league average), but the **spread was compressed by half**. Real batting averages have a standard deviation of .029; the model's predictions had a std of .015. It was like watching baseball through a lens that made everyone look average.

The fix is a standard technique in statistical forecasting called **variance calibration**. For each stat, we measured how much the model compressed the spread, then rescaled the predictions to match the historical variance:

```
calibrated = league_mean + (raw_prediction - league_mean) × scale_factor
```

For batting average, the scale factor was **2.05×**. For ERA, it was **2.18×**. Stolen bases barely needed any correction (1.08×) — because steals are the most predictable stat, the model already captured most of the real variance.

This calibration preserves the ranking (Judge is still predicted above Ohtani), preserves the mean (the league average is unchanged), and preserves relative distances between players. It just restores the realistic scale that the algorithm compressed.

**Why this matters beyond baseball:** If you're using ML to predict anything — stock returns, sales forecasts, test scores — your model is almost certainly doing this same compression. The point predictions may rank correctly but understate how different the best and worst outcomes really are. Check your predicted variance against reality. You may need to calibrate.

## How the Models Work

Both models use the same input features — this is key for a fair comparison:

1. **Previous season stats** — last year's numbers
2. **Weighted 3-year rolling average** — recent seasons weighted higher (3x weight for most recent, 2x for year before, 1x for two years ago)
3. **Career averages** — lifetime performance baseline
4. **Year-over-year trend** — is the player improving or declining?
5. **Age and age²** — captures the well-known aging curve (players peak around 27-29, then gradually decline)
6. **Experience** — number of qualified MLB seasons

Each player is compared to **his own history**, not to other players. The model asks: "Given what this specific player has done over his career, what should we expect next year?" The league-wide data comes in only through training — the algorithm learns general patterns (like aging curves) from thousands of player-seasons, then applies those patterns to each individual.

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

How did the ~$40 black-box compare to the free, transparent model? For batting average and home runs:

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

All code and data are freely available at **[github.com/alexanderpanetta/baseball-ml-predictions](https://github.com/alexanderpanetta/baseball-ml-predictions)**. You need Python 3.10+ and a few packages:

```bash
pip install -r requirements.txt
python3 02_sklearn_models.py       # Train models, generate predictions
python3 verify_reproducibility.py  # Confirm your results match ours exactly
```

The data comes from the SABR Lahman Baseball Database (Creative Commons licensed), and every model uses `random_state=42` for exact reproducibility. A SHA-256 verification script confirms your output files are byte-identical to ours.

## The Spreadsheets

- **[Batting Predictions](link)** — All 348 qualified batters with predicted AVG, HR, RBI, R, H, 2B, SB, BB, OBP, SLG
- **[Pitching Predictions](link)** — All 339 qualified pitchers with predicted ERA, W, L, SO, WHIP, IP, SV
- **[Methodology](link)** — Full documentation of features, model parameters, evaluation metrics, and variance calibration

---

*Alex Panetta covers AI governance and data science. This analysis was built with assistance from Claude (Anthropic) for code development and scikit-learn/Google Vertex AI for machine learning. Full methodology and reproducibility documentation: [github.com/alexanderpanetta/baseball-ml-predictions](https://github.com/alexanderpanetta/baseball-ml-predictions)*
