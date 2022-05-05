use hacspec_lib::*;
#[allow(unused_imports)]
use hacspec_secp256k1::*;
use hacspec_schnorr_sig_secp256k1_sha256::*;

extern crate quickcheck;
//#[macro_use(quickcheck)]
extern crate quickcheck_macros;

include!("../../secp256k1/src/secp256k1_generators.txt");

#[test]
#[allow(non_snake_case)]
fn test_lots_of_tests() {
  fn helper(a: Secp256k1ScalarGenerator, v: Secp256k1ScalarGenerator, m: Vec<u8>) -> TestResult {
    let a = a.into();
    let v = v.into(); 
    if a == Secp256k1Scalar::ZERO() {
      return TestResult::discard()
    }
    if v == Secp256k1Scalar::ZERO() {
      return TestResult::discard()
    }
    let m = ByteSeq::from_vec(m.iter().map(|i| (*i).into()).collect());
    let A = scalar_multiplication(a, GENERATOR());
    let sig = sign(a, A, v, &m);
    TestResult::from_bool(verify(A, &m, sig))
  }
  QuickCheck::new()
      .tests(5)
      .quickcheck(helper as fn(Secp256k1ScalarGenerator, Secp256k1ScalarGenerator, Vec<u8>) -> TestResult);
}

#[test]
#[allow(non_snake_case)]
// Not technically always true, but incredibly likely
fn test_wrong_r() {
  fn helper(a: Secp256k1ScalarGenerator, v: Secp256k1ScalarGenerator, m: Vec<u8>) -> TestResult {
    let a = a.into();
    let v = v.into();
    if a == Secp256k1Scalar::ZERO() {
      return TestResult::discard()
    }
    if v == Secp256k1Scalar::ZERO() {
      return TestResult::discard()
    }
    let m = ByteSeq::from_vec(m.iter().map(|i| (*i).into()).collect());
    let A = scalar_multiplication(a, GENERATOR());
    let (V,r) = sign(a, A, v, &m);
    TestResult::from_bool(!verify(A, &m, (add_points(V, GENERATOR()), r)))
  }
  QuickCheck::new()
      .tests(5)
      .quickcheck(helper as fn(Secp256k1ScalarGenerator, Secp256k1ScalarGenerator, Vec<u8>) -> TestResult);
}

#[test]
#[allow(non_snake_case)]
// Not technically always true, but incredibly likely
fn test_wrong_s() {
  fn helper(a: Secp256k1ScalarGenerator, v: Secp256k1ScalarGenerator, m: Vec<u8>) -> TestResult {
    let a = a.into();
    let v = v.into();
    if a == Secp256k1Scalar::ZERO() {
      return TestResult::discard()
    }
    if v == Secp256k1Scalar::ZERO() {
      return TestResult::discard()
    }
    let m = ByteSeq::from_vec(m.iter().map(|i| (*i).into()).collect());
    let A = scalar_multiplication(a, GENERATOR());
    let (V,r) = sign(a, A, v, &m);
    TestResult::from_bool(!verify(A, &m, (V, r - Secp256k1Scalar::ONE())))
  }
  QuickCheck::new()
      .tests(5)
      .quickcheck(helper as fn(Secp256k1ScalarGenerator, Secp256k1ScalarGenerator, Vec<u8>) -> TestResult);
}

#[test]
#[allow(non_snake_case)]
fn test_multi_sig() {
  fn helper(av: Vec<(Secp256k1ScalarGenerator, Secp256k1ScalarGenerator)>, m: Vec<u8>) -> TestResult {
    let av: Vec<(Secp256k1Scalar, Secp256k1Scalar)> = av.into();
    
    let m = ByteSeq::from_vec(m.iter().map(|i| (*i).into()).collect());

    TestResult::from_bool(false)
  }
  QuickCheck::new()
      .tests(5)
      .quickcheck(helper as fn(Vec<(Secp256k1ScalarGenerator, Secp256k1ScalarGenerator)>, Vec<u8>) -> TestResult)

}