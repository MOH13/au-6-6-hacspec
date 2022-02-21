use hacspec_lib::*;
#[allow(unused_imports)]
use hacspec_secp256k1::*;
use hacspec_dev::prelude::*;
use hacspec_lib::U8;

extern crate quickcheck;
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

/// Helper type for generating quickcheck Arbitrary trait
#[derive(Clone, Debug)]
struct AffineGenerator(Affine);

impl From<AffineGenerator> for Affine {
    fn from(e: AffineGenerator) -> Affine {
        e.0
    }
}

impl From<&Affine> for AffineGenerator {
    fn from(e: &Affine) -> AffineGenerator {
        AffineGenerator(*e)
    }
}

#[derive(Clone, Debug)]
struct Secp256k1ScalarGenerator(Secp256k1Scalar);

impl From<Secp256k1ScalarGenerator> for Secp256k1Scalar {
    fn from(e: Secp256k1ScalarGenerator) -> Secp256k1Scalar {
        e.0
    }
}

impl From<Secp256k1Scalar> for Secp256k1ScalarGenerator {
    fn from(e: Secp256k1Scalar) -> Secp256k1ScalarGenerator {
        Secp256k1ScalarGenerator(e)
    }
}

use quickcheck::*;

impl Arbitrary for AffineGenerator {
    fn arbitrary(g: &mut Gen) -> AffineGenerator {
        let p = BASE_POINT();
        let res = scalar_multiplication(Secp256k1ScalarGenerator::arbitrary(g).into(), p).unwrap();
        match g.choose(&[res, INFINITY()]) {
            Some(v) => v.into(),
            None => panic!("Whoops"),
        }
    }
}

impl Arbitrary for Secp256k1ScalarGenerator {
    fn arbitrary(g: &mut Gen) -> Secp256k1ScalarGenerator {
        let mut a: [u64; 4] = [0; 4];
        for i in 0..4 {
            a[i] = u64::arbitrary(g);
        }
        let mut b: [u8; 32] = [0; 32];
        for i in 0..4 {
            let val: u64 = a[i];
            b[(i * 8)..((i + 1) * 8)].copy_from_slice(&(val.to_le_bytes()));
        }
        Secp256k1Scalar::from_byte_seq_le(Seq::<U8>::from_public_slice(&b)).into()
    }
}

#[test]
fn test_infty_add_1() {
    fn helper(p: AffineGenerator) -> bool {
        let p = p.into();
        let res = add_points(p, INFINITY()).unwrap();
        res == p
    }
    QuickCheck::new()
        .tests(5)
        .quickcheck(helper as fn(AffineGenerator) -> bool);
}


#[test]
fn test_infty_add_2() {
    fn helper(p: AffineGenerator) -> bool {
        let p = p.into();
        let res = add_points(INFINITY(), p).unwrap();
        res == p
    }
    QuickCheck::new()
        .tests(5)
        .quickcheck(helper as fn(AffineGenerator) -> bool);
}

#[test]
fn test_infty_add_3() {
    let res = add_points(INFINITY(), INFINITY()).unwrap();
    assert!(res == INFINITY())
}

#[test]
fn test_add_negatives_gives_infty() {
    fn helper(p: AffineGenerator) -> bool {
        let p = p.into();
        let minus_p = neg_point(p).unwrap();
        let res = add_points(p, minus_p).unwrap();
        res == INFINITY()
    }
    QuickCheck::new()
        .tests(5)
        .quickcheck(helper as fn(AffineGenerator) -> bool);
}