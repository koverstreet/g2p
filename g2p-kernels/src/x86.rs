// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! x86-64: `pshufb` in 16 and 32 byte registers.

use crate::simd::{Nibbles, Reg, Shuffle};
use core::arch::x86_64::*;

#[rustfmt::skip]
impl Reg for __m128i {
    const BYTES: usize = 16;
    #[inline(always)] unsafe fn load(b: &[u8]) -> Self { _mm_loadu_si128(b.as_ptr().cast()) }
    #[inline(always)] unsafe fn store(self, b: &mut [u8]) { _mm_storeu_si128(b.as_mut_ptr().cast(), self) }
    #[inline(always)] unsafe fn xor(self, r: Self) -> Self { _mm_xor_si128(self, r) }
}

#[rustfmt::skip]
impl Shuffle for __m128i {
    #[inline(always)] unsafe fn table(t: &[u8; 16]) -> Self { Self::load(t) }
    #[inline(always)] unsafe fn lo_nibbles(self) -> Self { _mm_and_si128(self, _mm_set1_epi8(0x0f)) }
    #[inline(always)] unsafe fn hi_nibbles(self) -> Self { _mm_srli_epi64(self, 4).lo_nibbles() }
    #[inline(always)] unsafe fn shuffle(t: Self, i: Self) -> Self { _mm_shuffle_epi8(t, i) }
}

#[rustfmt::skip]
impl Reg for __m256i {
    const BYTES: usize = 32;
    #[inline(always)] unsafe fn load(b: &[u8]) -> Self { _mm256_loadu_si256(b.as_ptr().cast()) }
    #[inline(always)] unsafe fn store(self, b: &mut [u8]) { _mm256_storeu_si256(b.as_mut_ptr().cast(), self) }
    #[inline(always)] unsafe fn xor(self, r: Self) -> Self { _mm256_xor_si256(self, r) }
}

#[rustfmt::skip]
impl Shuffle for __m256i {
    #[inline(always)] unsafe fn table(t: &[u8; 16]) -> Self { _mm256_broadcastsi128_si256(__m128i::load(t)) }
    #[inline(always)] unsafe fn lo_nibbles(self) -> Self { _mm256_and_si256(self, _mm256_set1_epi8(0x0f)) }
    #[inline(always)] unsafe fn hi_nibbles(self) -> Self { _mm256_srli_epi64(self, 4).lo_nibbles() }
    #[inline(always)] unsafe fn shuffle(t: Self, i: Self) -> Self { _mm256_shuffle_epi8(t, i) }
}

isas! {
    Avx2:  avx2,  Nibbles<__m256i>, "avx2";
    Ssse3: ssse3, Nibbles<__m128i>, "ssse3";
}
