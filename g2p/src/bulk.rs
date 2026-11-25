// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Bulk operations on buffers of a [`GaloisField8`](crate::GaloisField8)'s elements, as bytes
//!
//! Buffers are multiplied by constants with the best SIMD the CPU has: SSSE3, AVX2, AVX-512 and
//! GFNI on x86-64, NEON on aarch64. It's found at runtime with the `std` feature, or else is what
//! the build targets - unless [`set_isa()`] says.
//!
//! A coefficient is made into a [`MulTable`] once, then used for any number of buffers. A matrix
//! of them times a vector of buffers loads each input once for all the outputs: an erasure code's
//! encoding and decoding. It's row-major - `m[o * ins.len() + i]` is the coefficient of input `i`
//! in output `o` - and every buffer is one length.
//!
//! # Example
//! RAID6's parity of three data blocks: P is their sum, and Q the sum of `2^i` times block `i`.
//! ```rust
//! use g2p::{bulk, g2p, Matrix};
//!
//! g2p!(GF256, 8, modulus: 0b1_0001_1101);
//!
//! # fn main() {
//! let d = [[1, 2], [3, 4], [5, 6]];
//! let (mut p, mut q) = ([0; 2], [0; 2]);
//! let m: Matrix<GF256, 2, 3> = Matrix::from_fn(2, 3, |r, c| GF256::from(1 << (r * c)));
//! let m: Vec<_> = m.mul_tables().collect();
//!
//! bulk::mul_matrix(&mut [&mut p[..], &mut q[..]], &[&d[0][..], &d[1][..], &d[2][..]], &m);
//! assert_eq!(p, [1 ^ 3 ^ 5, 2 ^ 4 ^ 6]);
//! assert_eq!(q, [1 ^ 6 ^ 20, 2 ^ 8 ^ 24]);
//! # }
//! ```

use core::slice;
pub use g2p_kernels::{isa, set_isa, Isa, MulTable};

fn check(outs: &[&mut [u8]], ins: &[&[u8]], m: &[MulTable]) {
    let (k, n) = (outs.len(), ins.len());
    assert_eq!(
        m.len(),
        k * n,
        "g2p: {} coefficients for a {k}x{n} matrix",
        m.len()
    );
    let mut lens = outs
        .iter()
        .map(|o| o.len())
        .chain(ins.iter().map(|i| i.len()));
    let len = lens.next();
    assert!(
        lens.all(|l| Some(l) == len),
        "g2p: buffers of different lengths"
    );
}

/// Multiply `src` by `c`, and add it to `dst`
pub fn mul_add(dst: &mut [u8], src: &[u8], c: &MulTable) {
    mul_add_matrix(&mut [dst], &[src], slice::from_ref(c))
}

/// Multiply `src` by `c`, into `dst`
pub fn mul(dst: &mut [u8], src: &[u8], c: &MulTable) {
    mul_matrix(&mut [dst], &[src], slice::from_ref(c))
}

/// Multiply `ins` by the matrix `m`, and add the products to `outs`
///
/// # Panics
/// If the buffers' lengths differ, or `m` isn't `outs.len()` by `ins.len()`.
pub fn mul_add_matrix(outs: &mut [&mut [u8]], ins: &[&[u8]], m: &[MulTable]) {
    check(outs, ins, m);
    g2p_kernels::mul_add_matrix(outs, ins, m)
}

/// Multiply `ins` by the matrix `m`, into `outs`
///
/// # Panics
/// If the buffers' lengths differ, or `m` isn't `outs.len()` by `ins.len()`.
pub fn mul_matrix(outs: &mut [&mut [u8]], ins: &[&[u8]], m: &[MulTable]) {
    check(outs, ins, m);
    g2p_kernels::mul_matrix(outs, ins, m)
}
