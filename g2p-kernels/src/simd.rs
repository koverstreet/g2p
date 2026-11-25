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
//! compiled, intrinsics and all, with that ISA - and no intrinsic is called in a closure, which
//! isn't the kernel, and hasn't its target features.

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
    unsafe fn zero() -> Self;
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
    #[inline(always)] unsafe fn zero() -> u8 { 0 }
}

#[rustfmt::skip]
impl Multiply for Scalar {
    type R = u8;
    type Coeff = MulTable;
    type Operand = u8;
    #[inline(always)] unsafe fn coeff(t: &MulTable) -> MulTable { *t }
    #[inline(always)] unsafe fn operand(x: u8) -> u8 { x }
    #[inline(always)] unsafe fn mul(c: MulTable, x: u8) -> u8 { c.mul(x) }
}

/// The most outputs a pass holds in registers
const OUTS: usize = 6;

/// The most inputs a pass multiplies in: their coefficients are sorted out first
const INS: usize = 8;

/// A coefficient, sorted out: by most of an erasure code's - all of P's - nothing is multiplied
#[derive(Clone, Copy)]
enum Coeff<'a> {
    Zero,
    One,
    Mul(&'a MulTable),
}

/// `outs (+)= m * ins` over bytes `from..to`, `INS` inputs at a time - with their coefficients sorted
/// out first - and `U` registers at a time as far as that goes
#[inline(always)]
unsafe fn many_to_many<M: Multiply, const K: usize, const U: usize>(
    outs: &mut [&mut [u8]],
    ins: &[&[u8]],
    m: &[MulTable],
    add: bool,
    (from, to): (usize, usize),
) {
    let n = ins.len();

    // a pass with no inputs too, if there are none: it zeroes the outputs
    for pass in 0..((n + INS - 1) / INS).max(1) {
        let ins = ins.get(pass * INS..).unwrap_or(&[]);
        let ins = ins.get(..INS).unwrap_or(ins);
        let mut c = [[Coeff::Zero; K]; INS];
        for (i, c) in c.iter_mut().enumerate() {
            for (o, c) in c.iter_mut().enumerate() {
                *c = match m.get(o * n + pass * INS + i) {
                    Some(t) if t.affine == MulTable::ONE.affine => Coeff::One,
                    Some(t) if t.affine != MulTable::ZERO.affine => Coeff::Mul(t),
                    _ => Coeff::Zero,
                };
            }
        }
        let add = add || pass > 0;

        let at = held::<M, K, U>(outs, ins, &c, add, (from, to));
        held::<M, K, 1>(outs, ins, &c, add, (at, to));
    }
}

/// `outs (+)= c * ins` over bytes `from..to`, as far as `U` registers at a time go: the outputs'
/// held in registers while every input's is multiplied in, so each input is read once and each
/// output written once - and an input with nothing to multiply by, all of P's, is only added. Every
/// buffer is sliced by offset: a slice of `U` registers is known to have them, where an iterator a
/// buffer would be kept on the stack. The bytes it got to.
#[inline(always)]
unsafe fn held<M: Multiply, const K: usize, const U: usize>(
    outs: &mut [&mut [u8]],
    ins: &[&[u8]],
    c: &[[Coeff<'_>; K]; INS],
    add: bool,
    (from, to): (usize, usize),
) -> usize {
    let step = U * M::R::BYTES;
    let to = from + (to - from) / step * step;

    for x in (from..to).step_by(step) {
        let mut acc = [[M::R::zero(); U]; K];
        if add {
            for (acc, out) in acc.iter_mut().zip(outs.iter()) {
                if let Some(d) = out.get(x..x + step) {
                    for (acc, d) in acc.iter_mut().zip(d.chunks_exact(M::R::BYTES)) {
                        *acc = M::R::load(d);
                    }
                }
            }
        }
        for (src, c) in ins.iter().zip(c) {
            if let Some(s) = src.get(x..x + step) {
                let mut r = [M::R::zero(); U];
                for (r, s) in r.iter_mut().zip(s.chunks_exact(M::R::BYTES)) {
                    *r = M::R::load(s);
                }
                let mut op = [M::operand(M::R::zero()); U];
                if c.iter().any(|c| matches!(c, Coeff::Mul(_))) {
                    for (op, r) in op.iter_mut().zip(&r) {
                        *op = M::operand(*r);
                    }
                }
                for (acc, c) in acc.iter_mut().zip(c) {
                    match *c {
                        Coeff::Zero => {}
                        Coeff::One => {
                            for (acc, r) in acc.iter_mut().zip(&r) {
                                *acc = acc.xor(*r);
                            }
                        }
                        Coeff::Mul(t) => {
                            let t = M::coeff(t);
                            for (acc, op) in acc.iter_mut().zip(&op) {
                                *acc = acc.xor(M::mul(t, *op));
                            }
                        }
                    }
                }
            }
        }
        for (acc, out) in acc.iter().zip(outs.iter_mut()) {
            if let Some(d) = out.get_mut(x..x + step) {
                for (acc, d) in acc.iter().zip(d.chunks_exact_mut(M::R::BYTES)) {
                    acc.store(d);
                }
            }
        }
    }
    to
}

/// `outs (+)= m * ins` from byte `from`, as far as whole registers go, `OUTS` outputs a pass. The
/// bytes it got to.
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
        // as many registers at a time as there's room to hold every output's
        let range = (from, to);
        match outs.len() {
            1 => many_to_many::<M, 1, 4>(outs, ins, m, add, range),
            2 => many_to_many::<M, 2, 4>(outs, ins, m, add, range),
            3 => many_to_many::<M, 3, 2>(outs, ins, m, add, range),
            4 => many_to_many::<M, 4, 2>(outs, ins, m, add, range),
            5 => many_to_many::<M, 5, 1>(outs, ins, m, add, range),
            _ => many_to_many::<M, 6, 1>(outs, ins, m, add, range),
        }
    }
    to
}

/// `outs (+)= m * ins`: whole registers by `M`, the tails a byte at a time
#[inline(always)]
pub(crate) unsafe fn matrix<M: Multiply>(
    outs: &mut [&mut [u8]],
    ins: &[&[u8]],
    m: &[MulTable],
    add: bool,
) {
    let done = passes::<M>(outs, ins, m, add, 0);
    passes::<Scalar>(outs, ins, m, add, done);
}
