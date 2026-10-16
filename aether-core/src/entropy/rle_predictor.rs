//! Context-class predictor with hierarchical decomposition for BWT+MTF+RLE streams.
//!
//! After BWT+MTF+RLE encoding, the byte stream contains:
//! - RUNA (0) / RUNB (1): bijective base-2 zero-run digits
//! - Values 2-255: shifted MTF values (original 1-254)
//!
//! The prediction decomposes each byte hierarchically:
//! 1. Binary: is it a run symbol (0-1) or literal (≥2)?
//! 2. If run: RUNA(0) vs RUNB(1)?
//! 3. If literal: which value (2-255)?
//!
//! Each sub-decision uses a separate model per context class.
//! This allows the "run vs literal" decision to be modeled very accurately
//! (it's highly predictable from context), while literal values get their
//! own dedicated model without interference from run symbols.

use super::traits::ProbabilityPredictor;
use crate::format::PredictorId;

/// Context classes.
const CTX_START: usize = 0; // First byte, no previous context
const CTX_IN_RUN: usize = 1; // Previous byte was RUNA(0) or RUNB(1)
const CTX_AFTER_LIT: usize = 2; // Previous byte was a literal (≥2)
const NUM_CONTEXTS: usize = 3;

/// Virtual pseudocount for binary models.
const ALPHA_BIN: f32 = 0.5;
/// Virtual pseudocount per symbol for literal model (254 possible values).
pub(crate) const ALPHA_LIT: f32 = 0.1;
/// Total virtual mass for literal model.
const ALPHA_LIT_TOTAL: f32 = 254.0 * ALPHA_LIT;
const NUM_LITS: usize = 254;

#[inline]
pub(crate) fn fenwick254_add(tree: &mut [u32; NUM_LITS], i: usize, delta: u32) {
    let mut idx = i + 1;
    while idx <= NUM_LITS {
        tree[idx - 1] += delta;
        idx += idx & idx.wrapping_neg();
    }
}

#[inline]
pub(crate) fn fenwick254_prefix(tree: &[u32; NUM_LITS], len: usize) -> u32 {
    let mut sum = 0u32;
    let mut idx = len;
    while idx > 0 {
        sum += tree[idx - 1];
        idx &= idx - 1;
    }
    sum
}

#[inline]
pub(crate) fn fenwick254_rebuild(tree: &mut [u32; NUM_LITS], counts: &[u32; NUM_LITS]) {
    *tree = *counts;
    for idx in 1..=NUM_LITS {
        let parent = idx + (idx & idx.wrapping_neg());
        if parent <= NUM_LITS {
            tree[parent - 1] += tree[idx - 1];
        }
    }
}

/// Simple binary counter with virtual pseudocounts.
#[derive(Clone)]
struct BinaryModel {
    count_yes: u32,
    count_total: u32,
}

impl BinaryModel {
    fn new() -> Self {
        Self {
            count_yes: 0,
            count_total: 0,
        }
    }

    /// Probability of the "yes" outcome.
    #[inline]
    fn p_yes(&self) -> f32 {
        (self.count_yes as f32 + ALPHA_BIN) / (self.count_total as f32 + 2.0 * ALPHA_BIN)
    }

    #[inline]
    fn update(&mut self, is_yes: bool) {
        if is_yes {
            self.count_yes += 1;
        }
        self.count_total += 1;
        // Rescale — shift total first, then clamp yes to maintain yes <= total.
        if self.count_total > 500_000 {
            self.count_total >>= 1;
            self.count_yes >>= 1;
            // Guarantee invariant: count_yes <= count_total
            if self.count_yes > self.count_total {
                self.count_yes = self.count_total;
            }
        }
    }
}

/// Predictor for BWT+MTF+RLE encoded streams.
///
/// Uses 3 context classes with hierarchical decomposition:
/// - Binary model: p(run_symbol | context)
/// - Binary model: p(RUNA | run, context)
/// - Multi-way model: p(value | literal, context) for values 2-255
///
/// Memory: ~6 KiB.
#[derive(Clone)]
pub struct RlePredictor {
    /// p(is_run_symbol | context) — is the next byte 0 or 1?
    run_vs_lit: [BinaryModel; NUM_CONTEXTS],
    /// p(RUNA | is_run, context) — given it's a run symbol, is it RUNA(0)?
    runa_vs_runb: [BinaryModel; NUM_CONTEXTS],
    /// p(value | is_literal, context) — literal value counts (indexed 0-253 for values 2-255).
    lit_counts: [[u32; NUM_LITS]; NUM_CONTEXTS],
    /// 1-indexed Fenwick tree over `lit_counts` for O(log 254) prefix queries.
    lit_fenwick: [[u32; NUM_LITS]; NUM_CONTEXTS],
    lit_totals: [u32; NUM_CONTEXTS],
    /// Current context class.
    ctx: usize,
}

impl RlePredictor {
    pub fn new() -> Self {
        Self {
            run_vs_lit: [BinaryModel::new(), BinaryModel::new(), BinaryModel::new()],
            runa_vs_runb: [BinaryModel::new(), BinaryModel::new(), BinaryModel::new()],
            lit_counts: [[0u32; NUM_LITS]; NUM_CONTEXTS],
            lit_fenwick: [[0u32; NUM_LITS]; NUM_CONTEXTS],
            lit_totals: [0u32; NUM_CONTEXTS],
            ctx: CTX_START,
        }
    }

    #[inline]
    fn context_of(byte: u8) -> usize {
        if byte <= 1 {
            CTX_IN_RUN
        } else {
            CTX_AFTER_LIT
        }
    }

    /// The next prediction in decomposed form, for callers that immediately
    /// fold this baseline into a larger model.
    ///
    /// Returns `(p_run, p_runa_given_run, literal_scale, literal_counts)`.
    /// These reconstruct [`predict`](ProbabilityPredictor::predict) exactly:
    ///
    /// ```text
    /// probs[0]     = p_run * p_runa_given_run
    /// probs[1]     = p_run * (1.0 - p_runa_given_run)
    /// probs[i + 2] = (literal_counts[i] as f32 + ALPHA_LIT) * literal_scale
    /// ```
    ///
    /// [`NeuralSsmPredictor`](crate::entropy::NeuralSsmPredictor) mixes this
    /// baseline symbol-by-symbol, so materialising the 256-entry array and
    /// copying it out on every predicted byte was pure memory traffic — over
    /// a kilobyte per byte of output on the decode hot path.
    #[inline]
    pub fn model_parts(&self) -> (f32, f32, f32, &[u32; NUM_LITS]) {
        let c = self.ctx;
        let p_run = self.run_vs_lit[c].p_yes();
        let p_runa_given_run = self.runa_vs_runb[c].p_yes();
        let lit_denom = self.lit_totals[c] as f32 + ALPHA_LIT_TOTAL;
        let literal_scale = (1.0 - p_run) / lit_denom;
        (p_run, p_runa_given_run, literal_scale, &self.lit_counts[c])
    }

    /// Decomposed prediction including the current context's total literal
    /// count and Fenwick tree for O(1) normalizer + O(log 254) CDF queries.
    #[inline]
    pub(crate) fn model_parts_fenwick(
        &self,
    ) -> (f32, f32, u32, &[u32; NUM_LITS], &[u32; NUM_LITS]) {
        let c = self.ctx;
        let p_run = self.run_vs_lit[c].p_yes();
        let p_runa_given_run = self.runa_vs_runb[c].p_yes();
        (
            p_run,
            p_runa_given_run,
            self.lit_totals[c],
            &self.lit_counts[c],
            &self.lit_fenwick[c],
        )
    }
}

impl Default for RlePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbabilityPredictor for RlePredictor {
    #[inline]
    fn predict(&mut self) -> [f32; 256] {
        // Expressed via `model_parts` so the array form and the decomposed
        // form cannot drift apart — `NeuralSsmPredictor` relies on them
        // producing bit-identical values.
        let (p_run, p_runa_given_run, scale, lit_entry) = self.model_parts();
        let mut probs = [0.0f32; 256];

        // Step 1+2: p(run_symbol) split into RUNA / RUNB.
        probs[0] = p_run * p_runa_given_run; // RUNA
        probs[1] = p_run * (1.0 - p_runa_given_run); // RUNB

        // Step 3: distribute the literal mass among values 2-255. `scale`
        // folds `p_lit / denom` so this is 254 multiplies, not 254 divides.
        for i in 0..NUM_LITS {
            probs[i + 2] = (lit_entry[i] as f32 + ALPHA_LIT) * scale;
        }

        probs
    }

    #[inline]
    fn update(&mut self, byte: u8) {
        let c = self.ctx;
        let b = byte as usize;

        if byte <= 1 {
            // Run symbol
            self.run_vs_lit[c].update(true);
            self.runa_vs_runb[c].update(byte == 0); // RUNA = yes
        } else {
            // Literal
            self.run_vs_lit[c].update(false);
            let idx = b - 2; // Map 2-255 → 0-253
            self.lit_counts[c][idx] += 1;
            self.lit_totals[c] += 1;
            if self.lit_totals[c] > 500_000 {
                self.lit_totals[c] = 0;
                for v in self.lit_counts[c].iter_mut() {
                    *v >>= 1;
                    self.lit_totals[c] += *v;
                }
                fenwick254_rebuild(&mut self.lit_fenwick[c], &self.lit_counts[c]);
            } else {
                fenwick254_add(&mut self.lit_fenwick[c], idx, 1);
            }
        }

        // Advance context
        self.ctx = Self::context_of(byte);
    }

    #[inline]
    fn predict_cdf(&mut self) -> [u16; 257] {
        use crate::coding::rans::PROB_TOTAL;
        const RESERVED: u32 = 256;
        let scalable = (PROB_TOTAL - RESERVED) as f64;

        let (p_run, p_runa, lit_total, lit_counts, _) = self.model_parts_fenwick();
        let w0 = (p_run * p_runa).max(1e-6) as f64;
        let w1 = (p_run * (1.0 - p_runa)).max(1e-6) as f64;
        let w01 = w0 + w1;
        let p_lit = (1.0 - p_run).max(1e-6) as f64;
        let lit_coef = p_lit / (lit_total as f64 + ALPHA_LIT_TOTAL as f64);
        let base_lit = lit_coef * (ALPHA_LIT as f64);
        let total = w01 + lit_coef * (lit_total as f64) + (NUM_LITS as f64) * base_lit;

        let mut cdf = [0u16; 257];
        cdf[0] = 0;
        cdf[1] = (1 + (w0 * scalable / total).floor() as u32).min(PROB_TOTAL) as u16;
        let mut rle_cum = 0u32;
        for (m, &cnt) in lit_counts.iter().enumerate().take(NUM_LITS) {
            let cum = w01 + lit_coef * (rle_cum as f64) + (m as f64) * base_lit;
            let sym = 2 + m;
            cdf[sym] =
                (sym as u32 + (cum * scalable / total).floor() as u32).min(PROB_TOTAL) as u16;
            rle_cum += cnt;
        }
        cdf[256] = PROB_TOTAL as u16;
        cdf
    }

    #[inline]
    fn query_cdf(&mut self, byte: u8) -> (u16, u16) {
        use crate::coding::rans::PROB_TOTAL;
        const RESERVED: u32 = 256;
        let scalable = (PROB_TOTAL - RESERVED) as f64;

        let (p_run, p_runa, lit_total, lit_counts, lit_fenwick) = self.model_parts_fenwick();
        let w0 = (p_run * p_runa).max(1e-6) as f64;
        let w1 = (p_run * (1.0 - p_runa)).max(1e-6) as f64;
        let w01 = w0 + w1;
        let p_lit = (1.0 - p_run).max(1e-6) as f64;
        let lit_coef = p_lit / (lit_total as f64 + ALPHA_LIT_TOTAL as f64);
        let base_lit = lit_coef * (ALPHA_LIT as f64);
        let total = w01 + lit_coef * (lit_total as f64) + (NUM_LITS as f64) * base_lit;

        let q = |cum: f64, sym: usize| -> u16 {
            (sym as u32 + (cum * scalable / total).floor() as u32).min(PROB_TOTAL) as u16
        };

        match byte {
            0 => (0, q(w0, 1)),
            1 => (q(w0, 1), q(w01, 2)),
            _ => {
                let m = (byte - 2) as usize;
                let rle_cum = fenwick254_prefix(lit_fenwick, m);
                let lo = q(
                    w01 + lit_coef * (rle_cum as f64) + (m as f64) * base_lit,
                    2 + m,
                );
                let hi = if byte == 255 {
                    PROB_TOTAL as u16
                } else {
                    let next_cum = rle_cum + lit_counts[m];
                    let m1 = m + 1;
                    q(
                        w01 + lit_coef * (next_cum as f64) + (m1 as f64) * base_lit,
                        2 + m1,
                    )
                };
                (lo, hi)
            }
        }
    }

    #[inline]
    fn decode_symbol(&mut self, freq: u32) -> (u8, u16, u16) {
        use crate::coding::rans::PROB_TOTAL;
        const RESERVED: u32 = 256;
        let scalable = (PROB_TOTAL - RESERVED) as f64;

        let (p_run, p_runa, lit_total, _, lit_fenwick) = self.model_parts_fenwick();
        let w0 = (p_run * p_runa).max(1e-6) as f64;
        let w1 = (p_run * (1.0 - p_runa)).max(1e-6) as f64;
        let w01 = w0 + w1;
        let p_lit = (1.0 - p_run).max(1e-6) as f64;
        let lit_coef = p_lit / (lit_total as f64 + ALPHA_LIT_TOTAL as f64);
        let base_lit = lit_coef * (ALPHA_LIT as f64);
        let total = w01 + lit_coef * (lit_total as f64) + (NUM_LITS as f64) * base_lit;

        let q = |cum: f64, sym: usize| -> u32 {
            (sym as u32 + (cum * scalable / total).floor() as u32).min(PROB_TOTAL)
        };

        let b1 = q(w0, 1);
        if freq < b1 {
            return (0, 0, b1 as u16);
        }
        let b2 = q(w01, 2);
        if freq < b2 {
            return (1, b1 as u16, b2 as u16);
        }

        let mut lo_m = 0usize;
        let mut rle_cum = 0u32;
        let mut lo_value = b2;
        let mut hi_value = PROB_TOTAL;

        for step in [128usize, 64, 32, 16, 8, 4, 2, 1] {
            let cand_m = lo_m + step;
            if cand_m < NUM_LITS {
                let cand_rle = rle_cum + lit_fenwick[cand_m - 1];
                let cum = w01 + lit_coef * (cand_rle as f64) + (cand_m as f64) * base_lit;
                let mid_value = q(cum, 2 + cand_m);
                if mid_value <= freq {
                    lo_m = cand_m;
                    rle_cum = cand_rle;
                    lo_value = mid_value;
                } else {
                    hi_value = mid_value;
                }
            }
        }

        ((2 + lo_m) as u8, lo_value as u16, hi_value as u16)
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    fn name(&self) -> &str {
        "rle-ctx"
    }

    fn predictor_id(&self) -> PredictorId {
        PredictorId::Rle
    }

    fn save_state(&self) -> Option<Vec<u8>> {
        // Format: [version: u8] per context [run_vs_lit(yes,total), runa_vs_runb(yes,total),
        //         lit_counts[254], lit_total], then ctx byte
        let mut buf = Vec::with_capacity(1 + NUM_CONTEXTS * (4 + 4 + 4 + 4 + 254 * 4 + 4) + 1);
        buf.push(1); // version 1
        for i in 0..NUM_CONTEXTS {
            buf.extend_from_slice(&self.run_vs_lit[i].count_yes.to_le_bytes());
            buf.extend_from_slice(&self.run_vs_lit[i].count_total.to_le_bytes());
            buf.extend_from_slice(&self.runa_vs_runb[i].count_yes.to_le_bytes());
            buf.extend_from_slice(&self.runa_vs_runb[i].count_total.to_le_bytes());
            for &c in &self.lit_counts[i] {
                buf.extend_from_slice(&c.to_le_bytes());
            }
            buf.extend_from_slice(&self.lit_totals[i].to_le_bytes());
        }
        buf.push(self.ctx as u8);
        Some(buf)
    }

    fn load_state(&mut self, data: &[u8]) -> bool {
        if data.is_empty() || data[0] != 1 {
            return false; // Missing or unknown version
        }
        let data = &data[1..];
        let per_ctx = 4 + 4 + 4 + 4 + 254 * 4 + 4;
        let expected = NUM_CONTEXTS * per_ctx + 1;
        if data.len() != expected {
            return false;
        }
        // Maximum allowed count per field. The BinaryModel rescales at 500K and
        // lit_counts rescale at 500K total, so 1M is a generous upper bound that
        // rejects adversarially extreme values.
        const MAX_BINARY_COUNT: u32 = 1_000_000;
        const MAX_LIT_COUNT: u32 = 1_000_000;

        // Read u32 LE from a 4-byte window. All reads are guaranteed in-bounds
        // by the exact length check above.
        fn read_u32(data: &[u8], off: &mut usize) -> u32 {
            let bytes = [data[*off], data[*off + 1], data[*off + 2], data[*off + 3]];
            *off += 4;
            u32::from_le_bytes(bytes)
        }

        let mut off = 0;
        // Parse into temporaries so we don't leave self in a half-loaded state on failure.
        let mut run_vs_lit = [BinaryModel::new(), BinaryModel::new(), BinaryModel::new()];
        let mut runa_vs_runb = [BinaryModel::new(), BinaryModel::new(), BinaryModel::new()];
        let mut lit_counts = [[0u32; 254]; NUM_CONTEXTS];
        let mut lit_totals = [0u32; NUM_CONTEXTS];

        for i in 0..NUM_CONTEXTS {
            let yes = read_u32(data, &mut off);
            let total = read_u32(data, &mut off);
            if yes > total || total > MAX_BINARY_COUNT {
                return false;
            }
            run_vs_lit[i].count_yes = yes;
            run_vs_lit[i].count_total = total;

            let yes = read_u32(data, &mut off);
            let total = read_u32(data, &mut off);
            if yes > total || total > MAX_BINARY_COUNT {
                return false;
            }
            runa_vs_runb[i].count_yes = yes;
            runa_vs_runb[i].count_total = total;

            let mut computed_total: u64 = 0;
            for count in lit_counts[i].iter_mut() {
                let c = read_u32(data, &mut off);
                if c > MAX_LIT_COUNT {
                    return false;
                }
                *count = c;
                computed_total += c as u64;
            }
            lit_totals[i] = read_u32(data, &mut off);
            // Validate that lit_totals is consistent with the sum of lit_counts
            if computed_total > u32::MAX as u64 || lit_totals[i] != computed_total as u32 {
                return false;
            }
        }

        let ctx = data[off] as usize;

        self.run_vs_lit = run_vs_lit;
        self.runa_vs_runb = runa_vs_runb;
        self.lit_counts = lit_counts;
        for i in 0..NUM_CONTEXTS {
            fenwick254_rebuild(&mut self.lit_fenwick[i], &self.lit_counts[i]);
        }
        self.lit_totals = lit_totals;
        self.ctx = if ctx >= NUM_CONTEXTS { 0 } else { ctx };
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_uniform() {
        let mut pred = RlePredictor::new();
        let probs = pred.predict();
        // Should sum to ~1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Sum should be ~1.0, got {sum}");
    }

    #[test]
    fn adapts_to_run_heavy_stream() {
        let mut pred = RlePredictor::new();
        // Feed lots of RUNA
        for _ in 0..100 {
            pred.predict();
            pred.update(0); // RUNA
        }
        let probs = pred.predict();
        // p(RUNA) should dominate
        assert!(
            probs[0] > 0.3,
            "After many RUNAs, p(RUNA) should be > 0.3, got {}",
            probs[0]
        );
        // p(RUNA) + p(RUNB) should be high
        let p_run = probs[0] + probs[1];
        assert!(p_run > 0.5, "p(run) should be high, got {p_run}");
    }

    #[test]
    fn hierarchical_probabilities_sum_to_one() {
        let mut pred = RlePredictor::new();
        let stream: Vec<u8> = vec![0, 1, 0, 3, 0, 0, 5, 0, 0, 0, 3, 0, 1, 4, 2, 0];
        for &b in &stream {
            let probs = pred.predict();
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Probabilities must sum to ~1.0, got {sum}"
            );
            pred.update(b);
        }
    }

    #[test]
    fn roundtrip_with_range_coder() {
        use crate::coding::rans;

        let rle_data: Vec<u8> = (0..500)
            .map(|i| match i % 7 {
                0..=2 => 0,
                3 => 1,
                4 => 3,
                5 => 0,
                _ => 5,
            })
            .collect();

        let mut enc = RlePredictor::new();
        let compressed = rans::encode_block(&rle_data, &mut enc).unwrap();

        let mut dec = RlePredictor::new();
        let decoded = rans::decode_block(&compressed, rle_data.len(), &mut dec).unwrap();

        assert_eq!(rle_data, decoded);
    }
}
