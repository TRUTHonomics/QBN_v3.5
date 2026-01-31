# Position Delta Threshold Analysis Report

**Asset ID:** 1
**Generated:** 2026-01-25 07:32:53
**Lookback:** all data

---

## Optimal Thresholds

| Delta Type | Score Type | Threshold | MI Score | Distribution |
|------------|------------|-----------|----------|--------------|
| cumulative | coincident | 0.030 | 0.0000 | d:10% s:59% i:31% |
| cumulative | confirming | 0.030 | 0.0000 | d:13% s:58% i:29% |

---

## Methodology

1. **Data Source:** Event-labeled barrier outcomes with delta scores
2. **Delta Calculation:** Cumulative change since entry (direction-aware)
3. **Weighting:** Uniqueness weighting (1/N per event) to eliminate serial correlation
4. **Optimization:** MI Grid Search over threshold values
5. **Diversity Constraints:** Min 2 active states, max 80% stable

## Delta States

- **deteriorating:** delta < -threshold (situatie verslechtert)
- **stable:** -threshold <= delta <= +threshold
- **improving:** delta > +threshold (situatie verbetert)
