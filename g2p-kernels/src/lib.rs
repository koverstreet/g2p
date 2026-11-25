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

#![no_std]

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
/// `pshufb` and `tbl` look bytes up in
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MulTable {
    nibbles: [[u8; 16]; 2],
}

impl MulTable {
    const ZERO: MulTable = MulTable::from_powers([0; 8]);

    /// From `c`'s products with the powers of 2, `c * 2^j` for `j` in `0..8`: multiplying by `c`
    /// is linear, so they're all of it
    pub const fn from_powers(p: [u8; 8]) -> Self {
        let mut nibbles = [[0; 16]; 2];
        let mut j = 0;
        while j < 8 {
            // in every nibble with bit j % 4
            let mut n = 0;
            while n < 16 {
                if n >> (j % 4) & 1 != 0 {
                    nibbles[j / 4][n] ^= p[j];
                }
                n += 1;
            }
            j += 1;
        }
        MulTable { nibbles }
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
        arch::ALL.iter().copied().filter(|&i| arch::has(i))
    }
}

/// The ISA [`set_isa()`] said, as its place in `arch::ALL` - its discriminant - plus one
static ISA: AtomicU8 = AtomicU8::new(0);

/// The ISA the kernels run: the best this CPU has, unless [`set_isa()`] said
pub fn isa() -> Isa {
    let set = usize::from(ISA.load(Relaxed)).checked_sub(1);
    match set.and_then(|i| arch::ALL.get(i)) {
        Some(&isa) => isa,
        None => Isa::all().next().unwrap_or(Isa::Scalar),
    }
}

/// Run `isa`'s kernels from now on
///
/// # Safety
/// The CPU must have it: where there's no `std` to find out, it's the caller who knows.
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
