// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Matrices of a field's elements

use crate::bulk::MulTable;
use crate::{GaloisField, GaloisField8};
use core::ops::{Index, Mul};

/// A matrix of a field's elements, of up to `R` rows and `C` columns
///
/// Its size is chosen when it's made - an erasure code's matrices are as big as its stripe - and
/// it needs no allocation.
///
/// # Example
/// Two data bytes, found again from their P and Q - their sum, and the sum of `2^i` times byte `i`
/// - by inverting those coefficients.
/// ```rust
/// use g2p::{g2p, Matrix};
///
/// g2p!(GF256, 8, modulus: 0b1_0001_1101);
///
/// # fn main() {
/// let (d0, d1) = (GF256::from(7), GF256::from(9));
/// let (p, q) = (d0 + d1, d0 + GF256::from(2) * d1);
///
/// let m: Matrix<GF256, 2, 2> = Matrix::from_fn(2, 2, |r, c| GF256::from(1 << (r * c)));
/// let pq: Matrix<GF256, 2, 1> = Matrix::from_fn(2, 1, |r, _| [p, q][r]);
/// let d = &m.inverse().expect("invertible") * &pq;
/// assert_eq!((d[(0, 0)], d[(1, 0)]), (d0, d1));
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Matrix<F, const R: usize, const C: usize> {
    rows: usize,
    cols: usize,
    /// Zero outside `rows` and `cols`, so the derived comparisons are the matrix's
    e: [[F; C]; R],
}

impl<F: GaloisField, const R: usize, const C: usize> Matrix<F, R, C> {
    /// Make a `rows` by `cols` matrix, of `f(r, c)` in row `r`, column `c`
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> F) -> Self {
        assert!(
            rows <= R && cols <= C,
            "g2p: a {rows}x{cols} matrix in one of {R}x{C}"
        );
        let mut e = [[F::ZERO; C]; R];
        for (r, row) in e.iter_mut().enumerate().take(rows) {
            for (c, x) in row.iter_mut().enumerate().take(cols) {
                *x = f(r, c);
            }
        }
        Matrix { rows, cols, e }
    }

    pub fn identity(n: usize) -> Self {
        Self::from_fn(n, n, |r, c| if r == c { F::ONE } else { F::ZERO })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The inverse, if it has one: by Gauss-Jordan elimination
    pub fn inverse(&self) -> Option<Self> {
        assert_eq!(self.rows, self.cols, "g2p: inverting a non-square matrix");
        let (mut a, mut inv) = (*self, Self::identity(self.rows));

        for c in 0..self.rows {
            let pivot = (c..self.rows).find(|&r| a.e[r][c] != F::ZERO)?;
            a.e.swap(c, pivot);
            inv.e.swap(c, pivot);

            let s = F::ONE / a.e[c][c];
            let (ac, ic) = (a.e[c].map(|x| x * s), inv.e[c].map(|x| x * s));
            a.e[c] = ac;
            inv.e[c] = ic;
            for r in (0..self.rows).filter(|&r| r != c) {
                let f = a.e[r][c];
                a.e[r].iter_mut().zip(&ac).for_each(|(x, &y)| *x -= f * y);
                inv.e[r].iter_mut().zip(&ic).for_each(|(x, &y)| *x -= f * y);
            }
        }
        Some(inv)
    }
}

impl<F: GaloisField8, const R: usize, const C: usize> Matrix<F, R, C> {
    /// The tables [`bulk`](crate::bulk)'s matrix operations multiply by this with, row by row
    pub fn mul_tables(&self) -> impl Iterator<Item = MulTable> + '_ {
        let rows = self.e[..self.rows].iter();
        rows.flat_map(move |row| row[..self.cols].iter().map(|&x| x.mul_table()))
    }
}

impl<F, const R: usize, const C: usize> Index<(usize, usize)> for Matrix<F, R, C> {
    type Output = F;

    fn index(&self, (r, c): (usize, usize)) -> &F {
        let (rows, cols) = (self.rows, self.cols);
        assert!(
            r < rows && c < cols,
            "g2p: element ({r}, {c}) of a {rows}x{cols} matrix"
        );
        &self.e[r][c]
    }
}

impl<F: GaloisField, const R: usize, const K: usize, const K2: usize, const C: usize>
    Mul<&Matrix<F, K2, C>> for &Matrix<F, R, K>
{
    type Output = Matrix<F, R, C>;

    fn mul(self, rhs: &Matrix<F, K2, C>) -> Matrix<F, R, C> {
        assert_eq!(
            self.cols, rhs.rows,
            "g2p: multiplying matrices that don't fit"
        );
        Matrix::from_fn(self.rows, rhs.cols, |r, c| {
            (0..self.cols).map(|k| self.e[r][k] * rhs.e[k][c]).sum()
        })
    }
}
