// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use g2p::{g2p, Matrix};

g2p!(GF256, 8, modulus: 0b1_0001_1101);

type M = Matrix<GF256, 6, 6>;

fn random(rows: usize, cols: usize, mut x: u64) -> M {
    Matrix::from_fn(rows, cols, |_, _| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        GF256::from(x as u8)
    })
}

#[test]
fn test_mul() {
    let (a, b) = (random(3, 4, 1), random(4, 2, 2));
    let p = &a * &b;
    assert_eq!((p.rows(), p.cols()), (3, 2));
    for r in 0..3 {
        for c in 0..2 {
            let want: GF256 = (0..4).map(|k| a[(r, k)] * b[(k, c)]).sum();
            assert_eq!(p[(r, c)], want, "({}, {})", r, c);
        }
    }
    assert_eq!(&M::identity(3) * &a, a);
}

/// Random matrices' inverses, both ways round; and a singular one, its last row the sum of two
/// others
#[test]
fn test_inverse() {
    let mut inverted = 0;
    for seed in 1..=100 {
        for n in 1..=6 {
            let a = random(n, n, seed * 7 + n as u64);
            if let Some(inv) = a.inverse() {
                assert_eq!(
                    (&inv * &a, &a * &inv),
                    (M::identity(n), M::identity(n)),
                    "{:?}",
                    a
                );
                inverted += 1;
            }
        }
    }
    assert!(inverted > 550, "{} of 600 inverted", inverted);

    let r = random(3, 4, 3);
    let a = M::from_fn(4, 4, |i, c| {
        if i < 3 {
            r[(i, c)]
        } else {
            r[(1, c)] + r[(2, c)]
        }
    });
    assert_eq!(a.inverse(), None);
}
