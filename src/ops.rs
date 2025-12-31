//! Core operations for Llama inference.

/// RMSNorm epsilon, aligned with rms_norm_eps in Transformers.
pub const RMS_EPS: f32 = 1e-5;

/// RMS normalization, aligned with LlamaRMSNorm.forward.
#[inline]
pub fn rms_norm(dest: &mut [f32], src: &[f32], weight: &[f32]) {
    let n = src.len();
    let ss: f32 = src.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss / n as f32 + RMS_EPS).sqrt();
    for i in 0..dest.len() {
        dest[i] = weight[i] * (inv * src[i]);
    }
}

/// Matrix-vector multiplication: xout = x @ w.T (w is row-major flattened).
#[inline]
pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32]) {
    let in_dim = x.len();
    let out_dim = xout.len();
    for i in 0..out_dim {
        let off = i * in_dim;
        let mut val = 0.0f32;
        for j in 0..in_dim {
            val += w[off + j] * x[j];
        }
        xout[i] = val;
    }
}

/// Element-wise accumulation: a += b.
#[inline]
pub fn accum(a: &mut [f32], b: &[f32]) {
    for (ai, bi) in a.iter_mut().zip(b.iter()) {
        *ai += *bi;
    }
}

/// Softmax in-place.
#[inline]
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for xi in x.iter_mut() {
        *xi = (*xi - max_val).exp();
        sum += *xi;
    }
    for xi in x.iter_mut() {
        *xi /= sum;
    }
}

/// Apply rotary positional embeddings, aligned with apply_rotary_pos_emb.
#[inline]
pub fn apply_rotary_emb(x: &mut [f32], pos: i32, head_size: usize) {
    let head_size_f = head_size as f32;
    let mut i = 0;
    while i < x.len() {
        let head_dim = (i % head_size) as f32;
        let freq = 1.0 / (10000.0f32.powf(head_dim / head_size_f));
        let val = pos as f32 * freq;
        let (fci, fcr) = val.sin_cos();

        let x0 = x[i];
        let x1 = x[i + 1];
        x[i] = x0 * fcr - x1 * fci;
        x[i + 1] = x0 * fci + x1 * fcr;

        i += 2;
    }
}

/// SwiGLU activation: gate * sigmoid(gate) * up
#[inline]
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    for (g, u) in gate.iter_mut().zip(up.iter()) {
        let sigmoid = 1.0 / (1.0 + (-*g).exp());
        *g = *g * sigmoid * u;
    }
}
