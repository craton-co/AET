//! Order-0 adaptive frequency model.
//!
//! The simplest possible predictor: it tracks how often each byte value has
//! appeared so far and uses the observed frequencies as the probability
//! distribution. No context is considered — hence "order-0".
//!
//! This serves as the baseline predictor and a building block for testing
//! the range coding pipeline.

use super::traits::ProbabilityPredictor;
use crate::format::PredictorId;

/// Largest `total` for which [`ExactDivisor`] is provably exact.
///
/// The magic-number bound below needs `total^2 * (PROB_TOTAL + 1) <= 2^SHIFT`.
/// The model rescales at 1_000_000, so this ceiling is never reached in
/// practice; [`ExactDivisor::div`] falls back to a hardware divide if it is.
const MAX_MAGIC_TOTAL: u32 = 1 << 20;

/// Exact `n / total` by multiply-and-shift, for a `total` that stays fixed
/// across a whole CDF sweep.
///
/// Both `predict_cdf` and `decode_symbol` divide by `self.total` once per
/// symbol — up to 256 times to resolve a single byte. A 64-bit hardware
/// divide is ~20-40 cycles; a widening multiply plus a shift is ~4, and this
/// is the dominant cost of decoding a byte-plane block.
///
/// With `M = floor(2^K / d) + 1` and `e = M*d - 2^K` (so `1 <= e <= d`),
/// `n*M / 2^K = n/d + n*e/(d * 2^K)`, which floors to `floor(n/d)` exactly
/// when `n * e < 2^K`. Here `n = cum * PROB_TOTAL + total/2 < total *
/// (PROB_TOTAL + 1)` and `e <= total`, so `total^2 * (PROB_TOTAL + 1) <= 2^K`
/// is sufficient — satisfied by `K = 56` for every `total <= 2^20`.
#[derive(Clone, Copy)]
struct ExactDivisor {
    total: u32,
    magic: u128,
    exact: bool,
}

impl ExactDivisor {
    const SHIFT: u32 = 56;

    #[inline]
    fn new(total: u32) -> Self {
        let exact = total > 0 && total <= MAX_MAGIC_TOTAL;
        let magic = if exact {
            ((1u128 << Self::SHIFT) / total as u128) + 1
        } else {
            0
        };
        Self {
            total,
            magic,
            exact,
        }
    }

    #[inline(always)]
    fn div(&self, n: u64) -> u64 {
        if self.exact {
            ((n as u128 * self.magic) >> Self::SHIFT) as u64
        } else {
            n / self.total as u64
        }
    }
}

/// Adaptive order-0 (unigram) frequency model with Laplace smoothing.
///
/// Maintains a 1-indexed 256-entry Fenwick tree alongside `counts` and uses
/// the overshoot-free `RESERVED = 256` monotone quantizer (`symbol + (cum * (PROB_TOTAL - 256)) / total`),
/// enabling O(log 256) `query_cdf` and O(log 256) `decode_symbol` without
/// chained monotonicity fixups or float fallbacks.
pub struct Order0Model {
    /// Frequency count for each byte value. Starts at 1 (Laplace prior).
    counts: [u32; 256],
    /// 1-indexed Fenwick tree over `counts` for O(log 256) prefix queries and binary lifting.
    fenwick: [u32; 256],
    /// Sum of all counts. Maintained incrementally for speed.
    total: u32,
}

impl Order0Model {
    pub fn new() -> Self {
        let counts = [1u32; 256];
        let mut fenwick = [0u32; 256];
        Self::rebuild_fenwick(&mut fenwick, &counts);
        Self {
            counts,
            fenwick,
            total: 256, // 256 * 1
        }
    }

    #[inline]
    fn rebuild_fenwick(fenwick: &mut [u32; 256], counts: &[u32; 256]) {
        *fenwick = *counts;
        for idx in 1..=256usize {
            let parent = idx + (idx & idx.wrapping_neg());
            if parent <= 256 {
                fenwick[parent - 1] += fenwick[idx - 1];
            }
        }
    }

    #[inline]
    fn fenwick_add(&mut self, i: usize, delta: u32) {
        let mut idx = i + 1;
        while idx <= 256 {
            self.fenwick[idx - 1] += delta;
            idx += idx & idx.wrapping_neg();
        }
    }

    #[inline]
    fn fenwick_prefix(&self, len: usize) -> u32 {
        let mut sum = 0u32;
        let mut idx = len;
        while idx > 0 {
            sum += self.fenwick[idx - 1];
            idx &= idx - 1;
        }
        sum
    }

    #[inline(always)]
    fn quantize_boundary(divisor: &ExactDivisor, cum: u32, symbol_index: usize) -> u32 {
        use crate::coding::rans::PROB_TOTAL;
        const RESERVED: u64 = 256;
        const SCALABLE: u64 = PROB_TOTAL as u64 - RESERVED;
        symbol_index as u32 + divisor.div((cum as u64) * SCALABLE) as u32
    }
}

impl Default for Order0Model {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbabilityPredictor for Order0Model {
    fn predict(&mut self) -> [f32; 256] {
        let mut probs = [0.0f32; 256];
        let total = self.total as f32;
        for (i, prob) in probs.iter_mut().enumerate() {
            *prob = self.counts[i] as f32 / total;
        }
        probs
    }

    /// Build 15-bit monotone CDF directly from integer counts using the
    /// overshoot-free `RESERVED = 256` quantizer.
    fn predict_cdf(&mut self) -> [u16; 257] {
        use crate::coding::rans::PROB_TOTAL;

        let mut cdf = [0u16; 257];
        let divisor = ExactDivisor::new(self.total);
        let mut cum = 0u32;
        for (i, cdf_val) in cdf.iter_mut().enumerate().take(256) {
            *cdf_val = Self::quantize_boundary(&divisor, cum, i) as u16;
            cum += self.counts[i];
        }
        cdf[256] = PROB_TOTAL as u16;
        cdf
    }

    fn update(&mut self, byte: u8) {
        let b = byte as usize;
        self.counts[b] += 1;
        self.total += 1;

        // Periodic rescaling to adapt to local statistics and prevent overflow.
        // Halve all counts (minimum 1) when total exceeds threshold.
        if self.total > 1_000_000 {
            self.total = 0;
            for c in self.counts.iter_mut() {
                *c = (*c >> 1).max(1);
                self.total += *c;
            }
            Self::rebuild_fenwick(&mut self.fenwick, &self.counts);
        } else {
            self.fenwick_add(b, 1);
        }
    }

    fn reset(&mut self) {
        self.counts = [1; 256];
        Self::rebuild_fenwick(&mut self.fenwick, &self.counts);
        self.total = 256;
    }

    /// Encode-side fast path: compute `(cdf[byte], cdf[byte + 1])` in O(log 256)
    /// using the Fenwick tree and `ExactDivisor`, bit-identical to `predict_cdf`.
    #[inline]
    fn query_cdf(&mut self, byte: u8) -> (u16, u16) {
        use crate::coding::rans::PROB_TOTAL;
        let s = byte as usize;
        let divisor = ExactDivisor::new(self.total);
        let cum = self.fenwick_prefix(s);
        let lo = Self::quantize_boundary(&divisor, cum, s) as u16;
        let hi = if s == 255 {
            PROB_TOTAL as u16
        } else {
            Self::quantize_boundary(&divisor, cum + self.counts[s], s + 1) as u16
        };
        (lo, hi)
    }

    /// Decode-side fast path: resolve `freq` in 8 Fenwick binary-lifting steps
    /// (O(log 256) multiply-shifts) without building the 257-entry CDF table.
    #[inline]
    fn decode_symbol(&mut self, freq: u32) -> (u8, u16, u16) {
        use crate::coding::rans::PROB_TOTAL;
        debug_assert!(freq < PROB_TOTAL);

        let divisor = ExactDivisor::new(self.total);
        let mut lo = 0usize;
        let mut cum = 0u32;
        let mut lo_value = 0u32;
        let mut hi_value = PROB_TOTAL;

        for step in [128usize, 64, 32, 16, 8, 4, 2, 1] {
            let cand = lo + step;
            let cand_cum = cum + self.fenwick[cand - 1];
            let mid_value = Self::quantize_boundary(&divisor, cand_cum, cand);
            if mid_value <= freq {
                lo = cand;
                cum = cand_cum;
                lo_value = mid_value;
            } else {
                hi_value = mid_value;
            }
        }

        (lo as u8, lo_value as u16, hi_value as u16)
    }

    fn name(&self) -> &str {
        "order-0"
    }

    fn predictor_id(&self) -> PredictorId {
        PredictorId::Order0
    }

    fn save_state(&self) -> Option<Vec<u8>> {
        // Format: [version: u8] [u32; 256] counts in little-endian
        let mut buf = Vec::with_capacity(1 + 256 * 4);
        buf.push(1); // version 1
        for &c in &self.counts {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        Some(buf)
    }

    fn load_state(&mut self, data: &[u8]) -> bool {
        if data.len() != 1 + 256 * 4 {
            return false;
        }
        if data[0] != 1 {
            return false; // Unknown version
        }
        let data = &data[1..];
        // Maximum allowed count per symbol. This bounds the skew an adversarial
        // payload can introduce. 2M per symbol × 256 symbols fits in u32 and
        // is well above any count the model would reach organically (rescale
        // fires at 1M total).
        const MAX_COUNT: u32 = 2_000_000;

        let mut counts = [0u32; 256];
        let mut total: u64 = 0;
        for (i, count) in counts.iter_mut().enumerate() {
            let offset = i * 4;
            let bytes = [
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ];
            let c = u32::from_le_bytes(bytes);
            if c == 0 {
                return false; // Laplace prior requires all counts >= 1
            }
            if c > MAX_COUNT {
                return false; // Reject adversarially skewed distributions
            }
            *count = c;
            total += c as u64;
        }
        if total > u32::MAX as u64 {
            return false; // Would overflow u32 total
        }
        self.counts = counts;
        Self::rebuild_fenwick(&mut self.fenwick, &self.counts);
        self.total = total as u32;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_is_uniform() {
        let mut model = Order0Model::new();
        let probs = model.predict();

        // All probabilities should be equal (1/256)
        let expected = 1.0 / 256.0;
        for &p in &probs {
            assert!((p - expected).abs() < 1e-6);
        }

        // Sum should be ~1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn prediction_adapts_after_updates() {
        let mut model = Order0Model::new();

        // Feed a bunch of 'A's
        for _ in 0..1000 {
            model.update(b'A');
        }

        let probs = model.predict();
        // 'A' should have much higher probability than other bytes
        let p_a = probs[b'A' as usize];
        let p_other = probs[0]; // 0x00, which was never fed
        assert!(p_a > p_other * 10.0, "P('A') = {p_a}, P(0x00) = {p_other}");
    }

    #[test]
    fn prediction_always_valid_distribution() {
        let mut model = Order0Model::new();

        for byte in 0..=255u8 {
            let probs = model.predict();

            // All positive
            for &p in &probs {
                assert!(p > 0.0, "All probabilities must be > 0");
            }

            // Sum to ~1.0
            let sum: f32 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "Sum = {sum}, expected ~1.0");

            model.update(byte);
        }
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut model = Order0Model::new();

        for _ in 0..500 {
            model.update(42);
        }

        model.reset();
        let probs = model.predict();
        let expected = 1.0 / 256.0;
        for &p in &probs {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    /// `ExactDivisor` must agree with hardware division across the whole
    /// operand range the CDF sweep can produce, including the boundaries
    /// where the magic-number bound is tightest.
    #[test]
    fn exact_divisor_matches_hardware_division() {
        use crate::coding::rans::PROB_TOTAL;

        let totals = [
            256u32,
            257,
            1_000,
            32_767,
            32_768,
            32_769,
            65_536,
            999_999,
            1_000_001,
            MAX_MAGIC_TOTAL,
            MAX_MAGIC_TOTAL + 1, // falls back to hardware divide
        ];

        for total in totals {
            let divisor = ExactDivisor::new(total);
            let half = (total / 2) as u64;
            // `cum` ranges over [0, total]; sample the ends densely and the
            // middle on a stride, since the operand is monotone in `cum`.
            let mut cums: Vec<u64> = (0..=64u64).collect();
            cums.extend((0..=64u64).map(|k| total as u64 - k.min(total as u64)));
            cums.extend((0..512u64).map(|k| k * (total as u64) / 512));
            for cum in cums {
                let cum = cum.min(total as u64);
                let n = cum * PROB_TOTAL as u64 + half;
                assert_eq!(
                    divisor.div(n),
                    n / total as u64,
                    "ExactDivisor mismatch: n={n}, total={total}",
                );
            }
        }
    }

    /// Differential guard for the encode-side `query_cdf` fast path.
    ///
    /// The decoder always calls `predict_cdf`, so any disagreement between
    /// the two silently desynchronises the range coder. This walks a set of
    /// count distributions chosen to exercise both `query_cdf` regimes —
    /// including heavily skewed ones where `predict_cdf`'s monotonicity
    /// fix-up overshoots and it rebuilds the table via `probs_to_cdf`.
    #[test]
    fn query_cdf_matches_predict_cdf_on_skewed_counts() {
        // (dominant symbol, repetitions) — large repetition counts push
        // `total` far past PROB_TOTAL, which is what collapses the rounded
        // gaps of the rare symbols to zero.
        let shapes: [(u8, usize); 6] = [
            (0, 100),
            (0, 32_000),
            (7, 200_000),
            (255, 500_000),
            (128, 900_000),
            (3, 1_200_000), // crosses the rescale threshold
        ];

        for (dominant, reps) in shapes {
            let mut model = Order0Model::new();
            for i in 0..reps {
                // Mostly the dominant symbol, with a thin tail of others so
                // the distribution is skewed rather than degenerate.
                if i % 4096 == 0 {
                    model.update(((i / 4096) % 256) as u8);
                } else {
                    model.update(dominant);
                }
            }

            let reference = model.predict_cdf();
            for symbol in 0..=255u8 {
                let (lo, hi) = model.query_cdf(symbol);
                let s = symbol as usize;
                assert_eq!(
                    (lo, hi),
                    (reference[s], reference[s + 1]),
                    "query_cdf disagrees with predict_cdf for symbol {symbol}                      (dominant={dominant}, reps={reps}, total={})",
                    model.total,
                );
            }
        }
    }

    /// Round-trip guard at the range-coder level: encode with `query_cdf`
    /// (what `rans::encode_block` uses) and decode with `predict_cdf`.
    #[test]
    fn skewed_stream_round_trips_through_range_coder() {
        // A float-exponent-like plane: one dominant value, a handful of
        // neighbours, and a long thin tail. This is the shape that broke
        // byte-plane decoding before the `query_cdf` overshoot fix.
        let mut data = Vec::with_capacity(200_000);
        let mut x: u32 = 0x1234_5678;
        for _ in 0..200_000 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let r = x >> 20; // 12 bits
            data.push(match r {
                0..=3800 => 130,
                3801..=4000 => 129,
                4001..=4050 => 131,
                _ => (r % 256) as u8,
            });
        }

        let mut encoder = Order0Model::new();
        let encoded = crate::coding::rans::encode_block(&data, &mut encoder).unwrap();

        let mut decoder = Order0Model::new();
        let decoded =
            crate::coding::rans::decode_block(&encoded, data.len(), &mut decoder).unwrap();

        assert_eq!(decoded, data, "Order0 range-coder round-trip mismatch");
    }

    #[test]
    fn rescaling_prevents_overflow() {
        let mut model = Order0Model::new();

        // Push past the rescaling threshold
        for _ in 0..2_000_000 {
            model.update(0);
        }

        // Model should still produce valid probabilities
        let probs = model.predict();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-2, "Sum after rescaling = {sum}");

        // Byte 0 should dominate
        assert!(probs[0] > 0.9);
    }
}
