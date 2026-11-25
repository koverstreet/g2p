// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use g2p::bulk::{mul, mul_add, mul_add_matrix, mul_matrix, set_isa, Isa, MulTable};
use g2p::{g2p, GaloisField, GaloisField8, Matrix};

g2p!(GF256, 8, modulus: 0b1_0001_1101);
g2p!(Rijndael, 8, modulus: 0b1_0001_1011);

fn noise(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

/// Every coefficient times every byte, against the field's own product - long enough for the
/// vector kernels, with a tail.
fn mul_is_field_mul<F: GaloisField8>(isa: Isa) {
    let src: Vec<u8> = (0..=255).chain(0..=36).collect();
    for c in 0..=255_u8 {
        let t = F::from(c).mul_table();
        let want: Vec<u8> = src
            .iter()
            .map(|&b| (F::from(c) * F::from(b)).into())
            .collect();

        let mut d = vec![0x5a; src.len()];
        mul(&mut d, &src, &t);
        assert_eq!(d, want, "{:?}: {} times every byte", isa, c);

        mul_add(&mut d, &src, &t);
        assert!(
            d.iter().all(|&b| b == 0),
            "{:?}: {} times every byte, added",
            isa,
            c
        );
    }
}

/// A matrix's tables times buffers, against the matrix times each byte: more outputs than a pass
/// takes and no inputs, more bytes than a tile, tails, and offsets that aren't aligned - of random
/// coefficients, and an erasure code's: a row of 1s, P's, of powers of 2, Q's, then random rows,
/// whose sums, single inputs and Horner's rows the kernels have their own ways of doing.
fn matrix_is_matrix_mul(isa: Isa) {
    for (nout, nin, len, off, code) in [
        (1, 1, 1, 0, false),
        (2, 10, 4096 + 5, 3, false),
        (13, 5, 100, 7, false),
        (3, 0, 33, 1, false),
        (1, 6, 4096 + 5, 3, true),
        (2, 1, 1000, 0, true),
        (2, 6, 4096 + 37, 1, true),
        (3, 10, 600, 5, true),
    ] {
        let a: Matrix<GF256, 13, 10> = {
            let e = noise(nout * nin, len as u64);
            Matrix::from_fn(nout, nin, |r, c| match r {
                0 if code => GF256::from(1),
                1 if code => GF256::from(2).pow(c),
                _ => GF256::from(e[r * nin + c]),
            })
        };
        let ins: Vec<Vec<u8>> = (0..nin).map(|i| noise(len + off, i as u64)).collect();
        let before: Vec<Vec<u8>> = (0..nout)
            .map(|o| noise(len + off, 100 + o as u64))
            .collect();
        let m: Vec<MulTable> = a.mul_tables().collect();
        let i: Vec<&[u8]> = ins.iter().map(|i| &i[off..]).collect();

        let (mut got, mut added) = (before.clone(), before.clone());
        mul_matrix(
            &mut got.iter_mut().map(|o| &mut o[off..]).collect::<Vec<_>>(),
            &i,
            &m,
        );
        mul_add_matrix(
            &mut added.iter_mut().map(|o| &mut o[off..]).collect::<Vec<_>>(),
            &i,
            &m,
        );

        for x in 0..len {
            let p = &a * &Matrix::<GF256, 10, 1>::from_fn(nin, 1, |r, _| GF256::from(i[r][x]));
            for o in 0..nout {
                let at = format!(
                    "{:?}: {}x{} of {} bytes at {}, output {} byte {}",
                    isa, nout, nin, len, off, o, x
                );
                assert_eq!(GF256::from(got[o][off + x]), p[(o, 0)], "{}", at);
                assert_eq!(
                    GF256::from(added[o][off + x] ^ before[o][off + x]),
                    p[(o, 0)],
                    "{}, added",
                    at
                );
            }
        }
    }
}

/// Every ISA the CPU has does the field's arithmetic
#[test]
fn test_bulk() {
    for isa in Isa::all() {
        // SAFETY: Isa::all() found the CPU has it
        unsafe { set_isa(isa) };
        mul_is_field_mul::<GF256>(isa);
        mul_is_field_mul::<Rijndael>(isa);
        matrix_is_matrix_mul(isa);
    }
}
