// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The kernel, once: over a register of bytes ([`Reg`]), multiplied by a coefficient
//! ([`Multiply`]) by lookups in its nibble tables ([`Nibbles`] over a [`Shuffle`]) or by its 8x8
//! bit matrix ([`Gfni`] over an [`Affine`]). An arch says how its registers do those, a line an
//! operation, and which ISAs it has; the scalar kernel is the same one, a byte at a time.
//!
//! Everything here is `#[inline(always)]` into an ISA's `#[target_feature]` kernel - so it's
//! compiled, intrinsics and all, with that ISA.

use crate::{matrix_len, MulTable};
use core::marker::PhantomData;

/// Whether the CPU has a target feature: found at runtime with `std`, else what the build targets
#[cfg(all(feature = "std", target_arch = "x86_64"))]
macro_rules! has {
    ($f:tt) => {
        std::is_x86_feature_detected!($f)
    };
}
#[cfg(all(feature = "std", target_arch = "aarch64"))]
macro_rules! has {
    ($f:tt) => {
        std::arch::is_aarch64_feature_detected!($f)
    };
}
#[cfg(not(feature = "std"))]
macro_rules! has {
    ($f:tt) => {
        cfg!(target_feature = $f)
    };
}

/// An arch's ISAs, the best first: each one's variant of `Isa`, its kernel, how it multiplies,
/// and the target features it needs - as tokens, which `has!` matches on, where a `literal`
/// fragment would be opaque.
macro_rules! isas {
    ($($(#[$cfg:meta])* $isa:ident: $kernel:ident, $m:ty, $($feature:tt),+;)*) => {
        /// The kernels to run: an ISA's
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u8)]
        pub enum Isa {
            $($(#[$cfg])* $isa,)*
            Scalar,
        }

        impl Isa {
            /// Every ISA there are kernels for, the best first
            pub const ALL: &'static [Isa] = &[$($(#[$cfg])* Isa::$isa,)* Isa::Scalar];

            /// The target features its kernels need
            pub fn features(self) -> &'static [&'static str] {
                match self {
                    $($(#[$cfg])* Isa::$isa => &[$($feature),+],)*
                    Isa::Scalar => &[],
                }
            }
        }

        pub(crate) fn has(isa: Isa) -> bool {
            match isa {
                $($(#[$cfg])* Isa::$isa => true $(&& has!($feature))+,)*
                Isa::Scalar => true,
            }
        }

        $(
            $(#[$cfg])*
            #[cfg(not(g2p_extern))]
            $(#[target_feature(enable = $feature)])+
            unsafe fn $kernel(outs: &mut [&mut [u8]], ins: &[&[u8]], m: &[$crate::MulTable], add: bool) {
                $crate::simd::matrix::<$m>(outs, ins, m, add)
            }
        )*

        /// # Safety
        /// The CPU must have `isa`.
        #[cfg(not(g2p_extern))]
        #[cfg_attr(g2p_export, export_name = "g2p_kernels_matrix")]
        pub(crate) unsafe fn matrix(isa: Isa, outs: &mut [&mut [u8]], ins: &[&[u8]], m: &[$crate::MulTable], add: bool) {
            match isa {
                $($(#[$cfg])* Isa::$isa => $kernel(outs, ins, m, add),)*
                Isa::Scalar => $crate::simd::matrix::<$crate::simd::Scalar>(outs, ins, m, add),
            }
        }

        #[cfg(g2p_extern)]
        extern "Rust" {
            /// The g2p_export build's: the CPU must have `isa`
            #[link_name = "g2p_kernels_matrix"]
            pub(crate) fn matrix(isa: Isa, outs: &mut [&mut [u8]], ins: &[&[u8]], m: &[$crate::MulTable], add: bool);
        }
    };
}

/// A register of bytes
pub(crate) trait Reg: Copy {
    const BYTES: usize;
    /// `b`'s first `BYTES`: it has them
    unsafe fn load(b: &[u8]) -> Self;
    /// Into `b`'s first `BYTES`: it has them
    unsafe fn store(self, b: &mut [u8]);
    unsafe fn xor(self, r: Self) -> Self;
}

/// Looking bytes up in a table of 16, in each 16 bytes of a register: `pshufb`, `tbl`
pub(crate) trait Shuffle: Reg {
    /// The table in each 16 bytes
    unsafe fn table(t: &[u8; 16]) -> Self;
    unsafe fn lo_nibbles(self) -> Self;
    unsafe fn hi_nibbles(self) -> Self;
    /// Each byte of `i` looked up in `t`
    unsafe fn shuffle(t: Self, i: Self) -> Self;
}

/// Multiplying bytes by an 8x8 bit matrix: GFNI's `gf2p8affineqb`
#[cfg(target_arch = "x86_64")]
pub(crate) trait Affine: Reg {
    /// The matrix in each 8 bytes
    unsafe fn matrix(m: u64) -> Self;
    unsafe fn affine(self, m: Self) -> Self;
}

/// Multiplying a register of bytes by a coefficient
pub(crate) trait Multiply {
    type R: Reg;
    /// A coefficient, ready to multiply by
    type Coeff: Copy;
    /// A register, ready to be multiplied
    type Operand: Copy;
    unsafe fn coeff(t: &MulTable) -> Self::Coeff;
    unsafe fn operand(r: Self::R) -> Self::Operand;
    unsafe fn mul(c: Self::Coeff, x: Self::Operand) -> Self::R;
}

/// By lookups in the coefficient's nibble tables: a byte's product is its two nibbles'
pub(crate) struct Nibbles<R>(PhantomData<R>);

#[rustfmt::skip]
impl<R: Shuffle> Multiply for Nibbles<R> {
    type R = R;
    type Coeff = (R, R);
    type Operand = (R, R);
    #[inline(always)] unsafe fn coeff(t: &MulTable) -> (R, R) { (R::table(&t.nibbles[0]), R::table(&t.nibbles[1])) }
    #[inline(always)] unsafe fn operand(r: R) -> (R, R) { (r.lo_nibbles(), r.hi_nibbles()) }
    #[inline(always)] unsafe fn mul((lo, hi): (R, R), (x_lo, x_hi): (R, R)) -> R { R::shuffle(lo, x_lo).xor(R::shuffle(hi, x_hi)) }
}

/// By the coefficient's 8x8 bit matrix: one instruction
#[cfg(target_arch = "x86_64")]
pub(crate) struct Gfni<R>(PhantomData<R>);

#[cfg(target_arch = "x86_64")]
#[rustfmt::skip]
impl<R: Affine> Multiply for Gfni<R> {
    type R = R;
    type Coeff = R;
    type Operand = R;
    #[inline(always)] unsafe fn coeff(t: &MulTable) -> R { R::matrix(t.affine) }
    #[inline(always)] unsafe fn operand(r: R) -> R { r }
    #[inline(always)] unsafe fn mul(c: R, x: R) -> R { x.affine(c) }
}

/// A byte at a time, by its nibbles: the reference, and every ISA's tails
pub(crate) struct Scalar;

#[rustfmt::skip]
impl Reg for u8 {
    const BYTES: usize = 1;
    #[inline(always)] unsafe fn load(b: &[u8]) -> u8 { b.first().copied().unwrap_or(0) }
    #[inline(always)] unsafe fn store(self, b: &mut [u8]) { b.iter_mut().take(1).for_each(|d| *d = self) }
    #[inline(always)] unsafe fn xor(self, r: u8) -> u8 { self ^ r }
}

#[rustfmt::skip]
impl Multiply for Scalar {
    type R = u8;
    type Coeff = MulTable;
    type Operand = u8;
    #[inline(always)] unsafe fn coeff(t: &MulTable) -> MulTable { *t }
    #[inline(always)] unsafe fn operand(x: u8) -> u8 { x }
    #[inline(always)] unsafe fn mul(c: MulTable, x: u8) -> u8 { c.nibbles[0][usize::from(x & 15)] ^ c.nibbles[1][usize::from(x >> 4)] }
}

/// The most outputs a pass takes: their coefficients are held in registers
const OUTS: usize = 6;

/// The bytes of each buffer a pass does: an L1's worth, so the outputs stay there while each input
/// is added in
const TILE: usize = 2048;

/// `outs[o] (+)= c(o) * src`, over bytes `at..end`, a register at a time: each coefficient held in
/// registers, each of `src`'s registers made an operand once for all the outputs
#[inline(always)]
unsafe fn one_to_many<M: Multiply, const K: usize>(
    outs: &mut [&mut [u8]],
    src: &[u8],
    c: impl Fn(usize) -> MulTable,
    add: bool,
    (at, end): (usize, usize),
) {
    let w = M::R::BYTES;
    let c: [M::Coeff; K] = core::array::from_fn(|o| M::coeff(&c(o)));

    let mut x = at;
    for s in src.get(at..end).unwrap_or(&[]).chunks_exact(w) {
        let s = M::operand(M::R::load(s));
        for (out, &c) in outs.iter_mut().zip(&c) {
            if let Some(d) = out.get_mut(x..x + w) {
                let p = M::mul(c, s);
                (if add { M::R::load(d).xor(p) } else { p }).store(d);
            }
        }
        x += w;
    }
}

/// `outs (+)= M * ins` from byte `from`, as far as whole registers go: `OUTS` outputs a pass, a
/// `TILE` of them at a time, each input added in. The bytes it got to.
#[inline(always)]
unsafe fn passes<M: Multiply>(
    outs: &mut [&mut [u8]],
    ins: &[&[u8]],
    m: &[MulTable],
    add: bool,
    from: usize,
) -> usize {
    let n = ins.len();
    let to = from + (matrix_len(outs, ins).saturating_sub(from)) / M::R::BYTES * M::R::BYTES;

    for (g, outs) in outs.chunks_mut(OUTS).enumerate() {
        let m = m.get(g * OUTS * n..).unwrap_or(&[]);
        for at in (from..to).step_by(TILE) {
            let tile = (at, to.min(at + TILE));
            for (i, src) in ins.iter().enumerate() {
                let c = |o: usize| m.get(o * n + i).copied().unwrap_or(MulTable::ZERO);
                let add = add || i > 0;
                match outs.len() {
                    1 => one_to_many::<M, 1>(outs, src, c, add, tile),
                    2 => one_to_many::<M, 2>(outs, src, c, add, tile),
                    3 => one_to_many::<M, 3>(outs, src, c, add, tile),
                    4 => one_to_many::<M, 4>(outs, src, c, add, tile),
                    5 => one_to_many::<M, 5>(outs, src, c, add, tile),
                    _ => one_to_many::<M, 6>(outs, src, c, add, tile),
                }
            }
        }
    }
    to
}

/// `outs (+)= M * ins`: whole registers by `M`, the tails a byte at a time
#[inline(always)]
pub(crate) unsafe fn matrix<M: Multiply>(
    outs: &mut [&mut [u8]],
    ins: &[&[u8]],
    m: &[MulTable],
    add: bool,
) {
    if ins.is_empty() && !add {
        let len = matrix_len(outs, ins);
        outs.iter_mut()
            .for_each(|o| o.get_mut(..len).unwrap_or(&mut []).fill(0));
    }
    let done = passes::<M>(outs, ins, m, add, 0);
    passes::<Scalar>(outs, ins, m, add, done);
}
