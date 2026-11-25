// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The kernels of [g2p](https://docs.rs/g2p)'s bulk operations, `outs (+)= M * ins` over GF(2^8):
//! see `g2p::bulk`.
//!
//! A crate of its own so it can be built on its own - for a target whose code otherwise may not
//! use vector registers, a kernel's - and so nothing in it panics: a kernel does as many bytes as
//! its shortest buffer has, and a coefficient past the end of `m` is zero.
//!
//! Built on its own, it's built twice: with cfg `g2p_export` for a target with vector registers,
//! the kernels behind one unmangled symbol, `g2p_kernels_matrix`; and with cfg `g2p_extern` for
//! the target that hasn't, what g2p uses - the kernels' types, and `g2p_kernels_matrix`, called
//! through an `extern` block. Both are one rustc's, and what crosses between them is slices of
//! bytes, `MulTable`s and an `Isa`: their layout is pinned down.

#![no_std]
// What the kernels are made of isn't used where they're the other build's
#![cfg_attr(g2p_extern, allow(dead_code, unused_imports))]

#[cfg(feature = "std")]
extern crate std;

#[macro_use]
mod simd;
#[cfg(target_arch = "x86_64")]
mod x86;
#[cfg(target_arch = "x86_64")]
use x86 as arch;
#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
use aarch64 as arch;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod arch {
    isas! {}
}

pub use arch::Isa;
use core::sync::atomic::{AtomicU8, Ordering::Relaxed};

/// Multiplying bytes by a coefficient `c` of some GF(2^8): its products with each nibble, which
/// `pshufb` and `tbl` look bytes up in, and its 8x8 bit matrix, which GFNI multiplies by
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct MulTable {
    nibbles: [[u8; 16]; 2],
    /// Byte `7 - i` is the row of a product's bit `i`
    affine: u64,
}

impl MulTable {
    const ZERO: MulTable = MulTable::from_powers([0; 8]);
    /// In any field: `1 * 2^j` is `2^j`
    const ONE: MulTable = MulTable::from_powers([1, 2, 4, 8, 16, 32, 64, 128]);

    /// From `c`'s products with the powers of 2, `c * 2^j` for `j` in `0..8`: multiplying by `c`
    /// is linear, so they're all of it
    pub const fn from_powers(p: [u8; 8]) -> Self {
        // a nibble's product: its lowest bit's, and the rest's
        let mut nibbles = [[0; 16]; 2];
        let mut n: usize = 1;
        while n < 16 {
            let low = n.trailing_zeros() as usize;
            nibbles[0][n] = nibbles[0][n & (n - 1)] ^ p[low];
            nibbles[1][n] = nibbles[1][n & (n - 1)] ^ p[4 + low];
            n += 1;
        }

        // row i is bit i of each product: the products as rows of bits, transposed (Hacker's
        // Delight's transpose8), the last row first
        let mut x = u64::from_le_bytes(p);
        let t = (x ^ (x >> 7)) & 0x00aa_00aa_00aa_00aa;
        x ^= t ^ (t << 7);
        let t = (x ^ (x >> 14)) & 0x0000_cccc_0000_cccc;
        x ^= t ^ (t << 14);
        let t = (x ^ (x >> 28)) & 0x0000_0000_f0f0_f0f0;
        x ^= t ^ (t << 28);

        MulTable { nibbles, affine: x.swap_bytes() }
    }

    /// `x`'s product
    fn mul(&self, x: u8) -> u8 {
        self.nibbles[0][usize::from(x & 15)] ^ self.nibbles[1][usize::from(x >> 4)]
    }
}

/// The bytes a matrix kernel does: the shortest buffer's
fn matrix_len(outs: &[&mut [u8]], ins: &[&[u8]]) -> usize {
    let lens = outs
        .iter()
        .map(|o| o.len())
        .chain(ins.iter().map(|i| i.len()));
    lens.min().unwrap_or(0)
}

impl Isa {
    /// Every ISA this CPU has, the best first: found at runtime with the `std` feature, else what
    /// the build targets
    pub fn all() -> impl Iterator<Item = Isa> {
        Isa::ALL.iter().copied().filter(|&i| arch::has(i))
    }
}

/// The ISA [`set_isa()`] said, as its place in [`Isa::ALL`] - its discriminant - plus one
static ISA: AtomicU8 = AtomicU8::new(0);

/// The ISA the kernels run: the best this CPU has, unless [`set_isa()`] said
pub fn isa() -> Isa {
    let set = usize::from(ISA.load(Relaxed)).checked_sub(1);
    match set.and_then(|i| Isa::ALL.get(i)) {
        Some(&isa) => isa,
        None => Isa::all().next().unwrap_or(Isa::Scalar),
    }
}

/// Run `isa`'s kernels from now on
///
/// # Safety
/// The CPU must have it - [`Isa::features()`]: where there's no `std` to find out, it's the caller
/// who knows.
pub unsafe fn set_isa(isa: Isa) {
    ISA.store(isa as u8 + 1, Relaxed);
}

/// Multiply `ins` by the matrix `m`, and add the products to `outs`
pub fn mul_add_matrix(outs: &mut [&mut [u8]], ins: &[&[u8]], m: &[MulTable]) {
    // SAFETY: isa() is one the CPU has
    unsafe { arch::matrix(isa(), outs, ins, m, true) }
}

/// Multiply `ins` by the matrix `m`, into `outs`
pub fn mul_matrix(outs: &mut [&mut [u8]], ins: &[&[u8]], m: &[MulTable]) {
    // SAFETY: isa() is one the CPU has
    unsafe { arch::matrix(isa(), outs, ins, m, false) }
}
