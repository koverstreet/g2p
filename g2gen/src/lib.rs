// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Procedural macro to generate finite field types
//!
//! This is just the procedural macro, for more information look at [g2p](https://docs.rs/g2p).

#![recursion_limit = "128"]
extern crate proc_macro;

use proc_macro::TokenStream as P1TokenStream;
use proc_macro2::{Ident, Span, TokenStream as P2TokenStream};

use g2poly::G2Poly;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Token,
};

/// Generate a newtype of the given name and implement finite field arithmetic on it.
///
/// The generated type have implementations for [`Add`](::core::ops::Add),
/// [`Sub`](::core::ops::Sub), [`Mul`](::core::ops::Mul) and [`Div`](::core::ops::Div).
///
/// There are also implementations for equality, copy and debug. Conversion from and to the base
/// type are implemented via the From trait.
/// Depending on the size of `p` the underlying type is u8, u16 or u32.
///
/// # Example
/// ```ignore
/// g2gen::g2p!(
///     GF256,                  // Name of the newtype
///     8,                      // The power of 2 specifying the field size 2^8 = 256 in this
///                             // case.
///     modulus: 0b1_0001_1101, // The reduction polynomial to use, each bit is a coefficient.
///                             // Can be left out in case it is not needed.
/// );
///
/// # fn main() {
/// let a: GF256 = 255.into();  // Conversion from the base type
/// assert_eq!(a - a, a + a);   // Finite field arithmetic.
/// assert_eq!(format!("{}", a), "255_GF256");
/// # }
/// ```
#[proc_macro]
pub fn g2p(input: P1TokenStream) -> P1TokenStream {
    let args = parse_macro_input!(input as ParsedInput);
    let settings = Settings::from_input(args).unwrap();
    let ident = settings.ident;
    let ident_name = settings.ident_name;
    let modulus = settings.modulus;
    let generator = settings.generator;
    let p = settings.p_val;
    let field_size = 1_usize << p;
    let mask = (1_u64 << p).wrapping_sub(1);

    let ty = match p {
        0 => panic!("p must be > 0"),
        1..=8 => quote!(u8),
        9..=16 => quote!(u16),
        17..=32 => quote!(u32),
        _ => unimplemented!("p > 32 is not implemented right now"),
    };

    let mod_name = Ident::new(&format!("{}_mod", ident_name), Span::call_site());

    let struct_def = quote![
        #[derive(Clone, Copy, Eq, PartialEq, Hash)]
        pub struct #ident(pub #ty);
    ];

    let struct_impl = quote![
        impl #ident {
            pub const MASK: #ty = #mask as #ty;
        }
    ];

    let from = quote![
        impl ::core::convert::From<#ident> for #ty {
            fn from(v: #ident) -> #ty {
                v.0
            }
        }
    ];

    let into = quote![
        impl ::core::convert::From<#ty> for #ident {
            fn from(v: #ty) -> #ident {
                #ident(v & #ident::MASK)
            }
        }
    ];

    let tmpl = format!("{{}}_{}", ident_name);
    let debug = quote![
        impl ::core::fmt::Debug for #ident {
            fn fmt<'a>(&self, f: &mut ::core::fmt::Formatter<'a>) -> ::core::fmt::Result {
                write!(f, #tmpl, self.0)
            }
        }
    ];
    let display = quote![
        impl ::core::fmt::Display for #ident {
            fn fmt<'a>(&self, f: &mut ::core::fmt::Formatter<'a>) -> ::core::fmt::Result {
                write!(f, #tmpl, self.0)
            }
        }
    ];
    let add = quote![
        impl ::core::ops::Add for #ident {
            type Output = Self;

            #[allow(clippy::suspicious_arithmetic_impl)]
            fn add(self, rhs: Self) -> Self {
                Self(self.0 ^ rhs.0)
            }
        }
        impl ::core::ops::AddAssign for #ident {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
    ];
    let sum = quote![
        impl ::core::iter::Sum for #ident {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(<Self as ::g2p::GaloisField>::ZERO, ::core::ops::Add::add)
            }
        }
    ];
    let sub = quote![
        impl ::core::ops::Sub for #ident {
            type Output = Self;


            #[allow(clippy::suspicious_arithmetic_impl)]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 ^ rhs.0)
            }
        }
        impl ::core::ops::SubAssign for #ident {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }
        impl ::core::ops::Neg for #ident {
            type Output = Self;

            fn neg(self) -> Self::Output {
                self
            }
        }
    ];
    let gen = generator.0;
    let modulus_val = modulus.0;
    let galois_trait_impl = quote![
        impl ::g2p::GaloisField for #ident {
            const SIZE: usize = #field_size;
            const MODULUS: ::g2p::G2Poly = ::g2p::G2Poly(#modulus_val);
            const ZERO: Self = Self(0);
            const ONE: Self = Self(1);
            const GENERATOR: Self = Self(#gen as #ty);
        }
    ];

    let field8 = if p == 8 {
        quote![impl ::g2p::GaloisField8 for #ident {}]
    } else {
        quote![]
    };

    let (tables, mul, div) = generate_mul_impl(
        ident.clone(),
        &ident_name,
        modulus,
        generator,
        ty,
        field_size,
    );
    let product = quote![
        impl ::core::iter::Product for #ident {
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(<Self as ::g2p::GaloisField>::ONE, ::core::ops::Mul::mul)
            }
        }
    ];

    P1TokenStream::from(quote![
        #struct_def

        mod #mod_name {
            use super::#ident;
            #struct_impl
            #tables
            #from
            #into
            #debug
            #display
            #add
            #sum
            #sub
            #mul
            #product
            #div
            #galois_trait_impl
            #field8
        }
    ])
}

struct ParsedInput {
    ident: syn::Ident,
    p: syn::LitInt,
    modulus: Option<syn::LitInt>,
}

impl Parse for ParsedInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ident = input.parse()?;
        let _sep: Token![,] = input.parse()?;
        let p = input.parse()?;

        let mut modulus = None;

        loop {
            let sep: Option<Token![,]> = input.parse()?;
            if sep.is_none() || input.is_empty() {
                break;
            }
            let ident: syn::Ident = input.parse()?;
            let ident_name = ident.to_string();
            let _sep: Token![:] = input.parse()?;
            match ident_name.as_str() {
                "modulus" => {
                    if modulus.is_some() {
                        Err(syn::parse::Error::new(
                            ident.span(),
                            "Double declaration of 'modulus'",
                        ))?
                    }
                    modulus = Some(input.parse()?);
                }
                _ => Err(syn::parse::Error::new(ident.span(), "Expected 'modulus'"))?,
            }
        }

        Ok(ParsedInput { ident, p, modulus })
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct Settings {
    ident: syn::Ident,
    ident_name: String,
    p_val: u64,
    modulus: G2Poly,
    generator: G2Poly,
}

fn find_modulus_poly(p: u64) -> G2Poly {
    assert!(p < 64);

    let start = (1 << p) + 1;
    let end = (1_u64 << (p + 1)).wrapping_sub(1);

    for m in start..=end {
        let p = G2Poly(m);
        if p.is_irreducible() {
            return p;
        }
    }

    unreachable!("There are irreducible polynomial for any degree!")
}

fn find_generator(m: G2Poly) -> G2Poly {
    let max = m.degree().expect("Modulus must have positive degree");

    for g in 1..(2 << max) {
        let g = G2Poly(g);
        if g.is_generator(m) {
            return g;
        }
    }

    unreachable!("There must be a generator element")
}

/// Generate the log and exp tables
///
/// Every element but zero is a power of the generator: `exp[i]` is `generator^i`, and `log[x]` the
/// `i` with `generator^i == x`. A product's log is the sum of its factors' logs, so exp has the
/// powers twice over and `exp[log[a] + log[b]]` needs no reduction - three tables' worth of
/// entries for a field, where a full multiplication table would have the field size squared.
fn generate_log_exp_tables(modulus: G2Poly, generator: G2Poly) -> (Vec<u64>, Vec<u64>) {
    let field_size = 1_usize
        << modulus
            .degree()
            .expect("Irreducible polynomial has positive degree");
    let order = field_size - 1;

    let mut log = vec![0; field_size];
    let mut exp = Vec::with_capacity(2 * order);
    let mut x = G2Poly(1);
    for i in 0..order {
        exp.push(x.0);
        log[x.0 as usize] = i as u64;
        x = x * generator % modulus;
    }
    exp.extend_from_within(..order);
    (log, exp)
}

/// A table, as a token stream: faster to generate than with quote
fn table_tokens(table: &[u64]) -> P2TokenStream {
    use std::fmt::Write;
    let mut res = String::with_capacity(8 * table.len());
    res.push('[');
    for v in table {
        write!(&mut res, "{},", v).unwrap();
    }
    res.push(']');
    res.parse().unwrap()
}

fn generate_mul_impl(
    ident: syn::Ident,
    ident_name: &str,
    modulus: G2Poly,
    generator: G2Poly,
    ty: P2TokenStream,
    field_size: usize,
) -> (P2TokenStream, P2TokenStream, P2TokenStream) {
    let (log, exp) = generate_log_exp_tables(modulus, generator);
    let order = field_size - 1;
    let exp_len = exp.len();

    // A field of up to 2^8 elements gets a table of every product too, at most 64KB, so a product
    // is one lookup - unless the small-tables feature says log and exp will do.
    let mul = if field_size <= 256 && !cfg!(feature = "small-tables") {
        let product = |a: usize, b: usize| match (a, b) {
            (0, _) | (_, 0) => 0,
            _ => exp[(log[a] + log[b]) as usize],
        };
        let rows: Vec<String> = (0..field_size)
            .map(|a| {
                let row: Vec<String> = (0..field_size).map(|b| product(a, b).to_string()).collect();
                format!("[{}]", row.join(","))
            })
            .collect();
        let products: P2TokenStream = format!("[{}]", rows.join(",")).parse().unwrap();
        quote![
            pub static MUL: [[#ty; #field_size]; #field_size] = #products;

            impl ::core::ops::Mul for #ident {
                type Output = Self;
                fn mul(self, rhs: Self) -> Self {
                    Self(MUL[(self.0 & Self::MASK) as usize][(rhs.0 & Self::MASK) as usize])
                }
            }
        ]
    } else {
        quote![
            impl ::core::ops::Mul for #ident {
                type Output = Self;
                fn mul(self, rhs: Self) -> Self {
                    let (a, b) = ((self.0 & Self::MASK) as usize, (rhs.0 & Self::MASK) as usize);
                    if a == 0 || b == 0 {
                        return Self(0);
                    }
                    Self(EXP[LOG[a] as usize + LOG[b] as usize])
                }
            }
        ]
    };

    let (log, exp) = (table_tokens(&log), table_tokens(&exp));

    // NB: We generate static arrays, as they are guaranteed to have a fixed location in memory.
    //     Using const would mean the compiler is free to create copies on the stack etc. Since
    //     The arrays are quite large, this could lead to stack overflows.
    let tables = quote! {
        pub static LOG: [#ty; #field_size] = #log;
        pub static EXP: [#ty; #exp_len] = #exp;
    };

    let mul = quote![
        #mul
        impl ::core::ops::MulAssign for #ident {
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }
    ];

    let err_msg = format!("Division by 0 in {}", ident_name);

    let div = quote![
        impl ::core::ops::Div for #ident {
            type Output = Self;

            fn div(self, rhs: Self) -> Self {
                let (a, b) = ((self.0 & Self::MASK) as usize, (rhs.0 & Self::MASK) as usize);
                if b == 0 {
                    panic!(#err_msg);
                }
                if a == 0 {
                    return Self(0);
                }
                Self(EXP[LOG[a] as usize + #order - LOG[b] as usize])
            }
        }
        impl ::core::ops::DivAssign for #ident {
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }
    ];

    (tables, mul, div)
}

impl Settings {
    pub fn from_input(input: ParsedInput) -> syn::Result<Self> {
        let ident = input.ident;
        let ident_name = ident.to_string();
        let p_val = input.p.base10_parse()?;
        let modulus = match input.modulus {
            Some(lit) => G2Poly(lit.base10_parse()?),
            None => find_modulus_poly(p_val),
        };

        if !modulus.is_irreducible() {
            Err(syn::Error::new(
                Span::call_site(),
                format!("Modulus {} is not irreducible", modulus),
            ))?;
        }

        let generator = find_generator(modulus);

        if !generator.is_generator(modulus) {
            Err(syn::Error::new(
                Span::call_site(),
                format!("{} is not a generator", generator),
            ))?;
        }

        Ok(Settings {
            ident,
            ident_name,
            p_val,
            modulus,
            generator,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_parser() {
        let span = Span::call_site();

        let input = ParsedInput {
            ident: Ident::new("foo", span),
            p: syn::LitInt::new("3", span),
            modulus: None,
        };

        let r = Settings::from_input(input);
        assert!(r.is_ok());
        assert_eq!(
            r.unwrap(),
            Settings {
                ident: syn::Ident::new("foo", span),
                ident_name: "foo".to_string(),
                p_val: 3,
                modulus: G2Poly(0b1011),
                generator: G2Poly(0b10),
            }
        );
    }

    #[test]
    fn test_generate_log_exp_tables() {
        // Rijndael's field, whose generator isn't 2
        let m = G2Poly(0b1_0001_1011);
        let g = find_generator(m);
        assert_eq!(g, G2Poly(3));
        let (log, exp) = generate_log_exp_tables(m, g);

        assert_eq!((log.len(), exp.len()), (256, 510));
        assert_eq!(exp[..255], exp[255..]);
        for a in 1..256_u64 {
            assert_eq!(exp[log[a as usize] as usize], a);
            for b in 1..256_u64 {
                let p = exp[(log[a as usize] + log[b as usize]) as usize];
                assert_eq!(G2Poly(p), G2Poly(a) * G2Poly(b) % m, "{} * {}", a, b);
            }
        }
    }
}
