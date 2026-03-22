/// Compute the entropy-weighted KL divergence between a predicted distribution
/// and the ground-truth distribution.
///
/// For each cell:
///   1. Compute the entropy of the ground-truth distribution:
///      `H = -∑ p * ln(p)` for `p > 0`.
///   2. Skip cells whose entropy is below `1e-6` (static / deterministic terrain).
///   3. Compute the KL divergence `KL(ground_truth ‖ prediction)`:
///      `KL = ∑ p * ln(p / q)` where `q` is clamped to `1e-10` to avoid log(0).
///   4. Accumulate: `total_weighted_kl += H * KL`, `total_weight += H`.
///
/// Returns `total_weighted_kl / total_weight` (lower is better).
/// Returns `0.0` if no cell has positive entropy (all terrain is static).
pub fn score_prediction(
    prediction: &[Vec<[f64; 6]>],
    ground_truth: &[Vec<[f64; 6]>],
) -> f64 {
    let mut total_weighted_kl = 0.0_f64;
    let mut total_weight = 0.0_f64;

    for (gt_row, pred_row) in ground_truth.iter().zip(prediction.iter()) {
        for (gt_cell, pred_cell) in gt_row.iter().zip(pred_row.iter()) {
            // 1. Entropy of ground truth.
            let entropy: f64 = gt_cell
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -p * p.ln())
                .sum();

            // 2. Skip static cells.
            if entropy < 1e-6 {
                continue;
            }

            // 3. KL divergence KL(gt ‖ pred).
            let kl: f64 = gt_cell
                .iter()
                .zip(pred_cell.iter())
                .filter(|&(&p, _)| p > 0.0)
                .map(|(&p, &q)| {
                    let q_clamped = q.max(1e-10);
                    p * (p / q_clamped).ln()
                })
                .sum();

            // 4. Accumulate.
            total_weighted_kl += entropy * kl;
            total_weight += entropy;
        }
    }

    if total_weight == 0.0 {
        return 0.0;
    }

    total_weighted_kl / total_weight
}

/// Convert weighted KL divergence to the competition's 0–100 score.
///
/// Formula from the official docs:
///   `score = max(0, min(100, 100 × exp(-3 × weighted_kl)))`
///
/// - 100 = perfect prediction
/// - 0   = terrible prediction
pub fn competition_score(weighted_kl: f64) -> f64 {
    (100.0 * (-3.0 * weighted_kl).exp()).clamp(0.0, 100.0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a uniform 6-class distribution.
    fn uniform() -> Vec<Vec<[f64; 6]>> {
        vec![vec![[1.0 / 6.0; 6]; 4]; 4]
    }

    /// Helper: build a concentrated distribution (class 0 = 1.0).
    fn concentrated(class: usize) -> Vec<Vec<[f64; 6]>> {
        let mut cell = [0.0f64; 6];
        cell[class] = 1.0;
        vec![vec![cell; 4]; 4]
    }

    /// Helper: build a distribution with some spread across 2 classes.
    fn spread(p0: f64) -> Vec<Vec<[f64; 6]>> {
        let mut cell = [0.0f64; 6];
        cell[0] = p0;
        cell[1] = 1.0 - p0;
        vec![vec![cell; 4]; 4]
    }

    #[test]
    fn test_perfect_prediction_zero_kl() {
        // When prediction == ground_truth, KL should be 0.
        let gt = spread(0.6);
        let pred = gt.clone();
        let score = score_prediction(&pred, &gt);
        assert!(
            score.abs() < 1e-10,
            "expected KL ≈ 0 for perfect prediction, got {}",
            score
        );
    }

    #[test]
    fn test_uniform_prediction_worse_than_informed() {
        // ground truth is spread 0.8 / 0.2 over classes 0 and 1.
        let gt = spread(0.8);

        // "Informed" prediction: same as ground truth.
        let informed = gt.clone();

        // Uniform prediction: 1/6 for all 6 classes.
        let uninformed = uniform();

        let kl_informed = score_prediction(&informed, &gt);
        let kl_uniform = score_prediction(&uninformed, &gt);

        assert!(
            kl_uniform > kl_informed,
            "uniform KL ({}) should be > informed KL ({})",
            kl_uniform,
            kl_informed
        );
    }

    #[test]
    fn test_static_cells_skipped() {
        // Ground truth with all probability on one class → entropy ≈ 0.
        let gt = concentrated(2);
        let pred = uniform();
        // All cells have entropy 0, so score should be 0 (no dynamic cells).
        let score = score_prediction(&pred, &gt);
        assert_eq!(score, 0.0, "static cells should be skipped, score should be 0");
    }

    #[test]
    fn test_score_is_finite_and_non_negative() {
        let gt = spread(0.7);
        let pred = uniform();
        let score = score_prediction(&pred, &gt);
        assert!(score.is_finite(), "score must be finite");
        assert!(score >= 0.0, "score must be non-negative");
    }

    #[test]
    fn test_competition_score_perfect() {
        assert!((competition_score(0.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_competition_score_range() {
        // KL of 0.1 → 100*exp(-0.3) ≈ 74.1
        let s = competition_score(0.1);
        assert!(s > 73.0 && s < 75.0, "score for KL=0.1 should be ~74, got {s}");
        // KL of 1.0 → 100*exp(-3) ≈ 4.98
        let s = competition_score(1.0);
        assert!(s > 4.0 && s < 6.0, "score for KL=1.0 should be ~5, got {s}");
    }

    #[test]
    fn test_competition_score_clamped() {
        assert!(competition_score(100.0) < 1e-10);
    }
}
