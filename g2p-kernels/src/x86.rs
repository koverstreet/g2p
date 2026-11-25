// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! x86-64: `pshufb` in 16, 32 and 64 byte registers, and GFNI - a multiply an instruction - in
//! 32 and 64.

use crate::simd::{Affine, Gfni, Nibbles, Reg, Shuffle};
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

#[cfg(g2p_avx512)]
#[rustfmt::skip]
impl Affine for __m256i {
    #[inline(always)] unsafe fn matrix(m: u64) -> Self { _mm256_set1_epi64x(m as i64) }
    #[inline(always)] unsafe fn affine(self, m: Self) -> Self { _mm256_gf2p8affine_epi64_epi8::<0>(self, m) }
}

#[cfg(g2p_avx512)]
#[rustfmt::skip]
impl Reg for __m512i {
    const BYTES: usize = 64;
    #[inline(always)] unsafe fn load(b: &[u8]) -> Self { _mm512_loadu_si512(b.as_ptr().cast()) }
    #[inline(always)] unsafe fn store(self, b: &mut [u8]) { _mm512_storeu_si512(b.as_mut_ptr().cast(), self) }
    #[inline(always)] unsafe fn xor(self, r: Self) -> Self { _mm512_xor_si512(self, r) }
}

#[cfg(g2p_avx512)]
#[rustfmt::skip]
impl Shuffle for __m512i {
    #[inline(always)] unsafe fn table(t: &[u8; 16]) -> Self { _mm512_broadcast_i32x4(__m128i::load(t)) }
    #[inline(always)] unsafe fn lo_nibbles(self) -> Self { _mm512_and_si512(self, _mm512_set1_epi8(0x0f)) }
    #[inline(always)] unsafe fn hi_nibbles(self) -> Self { _mm512_srli_epi64(self, 4).lo_nibbles() }
    #[inline(always)] unsafe fn shuffle(t: Self, i: Self) -> Self { _mm512_shuffle_epi8(t, i) }
}

#[cfg(g2p_avx512)]
#[rustfmt::skip]
impl Affine for __m512i {
    #[inline(always)] unsafe fn matrix(m: u64) -> Self { _mm512_set1_epi64(m as i64) }
    #[inline(always)] unsafe fn affine(self, m: Self) -> Self { _mm512_gf2p8affine_epi64_epi8::<0>(self, m) }
}

isas! {
    #[cfg(g2p_avx512)] GfniAvx512: gfni_avx512, Gfni<__m512i>, "gfni", "avx512f", "avx512bw";
    #[cfg(g2p_avx512)] Avx512:     avx512,      Nibbles<__m512i>, "avx512f", "avx512bw";
    #[cfg(g2p_avx512)] GfniAvx2:   gfni_avx2,   Gfni<__m256i>, "gfni", "avx2";
                       Avx2:       avx2,        Nibbles<__m256i>, "avx2";
                       Ssse3:      ssse3,       Nibbles<__m128i>, "ssse3";
}
