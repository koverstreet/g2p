// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! aarch64: NEON's `tbl`.

use crate::simd::{Nibbles, Reg, Shuffle};
use core::arch::aarch64::*;

#[rustfmt::skip]
impl Reg for uint8x16_t {
    const BYTES: usize = 16;
    #[inline(always)] unsafe fn load(b: &[u8]) -> Self { vld1q_u8(b.as_ptr()) }
    #[inline(always)] unsafe fn store(self, b: &mut [u8]) { vst1q_u8(b.as_mut_ptr(), self) }
    #[inline(always)] unsafe fn xor(self, r: Self) -> Self { veorq_u8(self, r) }
    #[inline(always)] unsafe fn zero() -> Self { vdupq_n_u8(0) }
}

#[rustfmt::skip]
impl Shuffle for uint8x16_t {
    #[inline(always)] unsafe fn table(t: &[u8; 16]) -> Self { Self::load(t) }
    #[inline(always)] unsafe fn lo_nibbles(self) -> Self { vandq_u8(self, vdupq_n_u8(0x0f)) }
    #[inline(always)] unsafe fn hi_nibbles(self) -> Self { vshrq_n_u8::<4>(self) }
    #[inline(always)] unsafe fn shuffle(t: Self, i: Self) -> Self { vqtbl1q_u8(t, i) }
}

isas! {
    Neon: neon, Nibbles<uint8x16_t>, "neon";
}
