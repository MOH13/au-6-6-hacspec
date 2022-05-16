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
      .tests(50)
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
      .tests(50)
      .quickcheck(helper as fn(Secp256k1ScalarGenerator, Secp256k1ScalarGenerator, Vec<u8>) -> TestResult)
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
      .tests(50)
      .quickcheck(helper as fn(Secp256k1ScalarGenerator, Secp256k1ScalarGenerator, Vec<u8>) -> TestResult);
}

#[test]
#[allow(non_snake_case)]
fn test_multi_sig() {
  fn helper(avs: Vec<(Secp256k1ScalarGenerator, Secp256k1ScalarGenerator, Secp256k1ScalarGenerator)>, m: Vec<u8>) -> TestResult {
    if avs.len() == 0 {
      return TestResult::discard()
    }
    let mut secret_keys = Seq::<Secp256k1Scalar>::new(avs.len());
    let mut public_keys = Seq::<Affine>::new(avs.len());
    let mut rands = Seq::<Secp256k1Scalar>::new(avs.len());
    let mut random_points = Seq::<Affine>::new(rands.len());
    for i in 0..avs.len() {
      let a: Secp256k1Scalar = avs[i].0.into();
      let v: Secp256k1Scalar = avs[i].1.into();
      if a == Secp256k1Scalar::ZERO() {
        return TestResult::discard()
      }
      if v == Secp256k1Scalar::ZERO() {
        return TestResult::discard()
      }
      let A = scalar_multiplication(a, GENERATOR());
      secret_keys[i] = a;
      public_keys[i] = A;
      rands[i] = avs[i].2.into();
      random_points[i] = scalar_multiplication(rands[i], GENERATOR());
    }
    let m = ByteSeq::from_vec(m.iter().map(|i| (*i).into()).collect());

    // begin actual signing process
    let a_values = compute_a_values(&public_keys);
    let agg_pk = compute_agg_pk(&public_keys, &a_values);
    let agg_R = compute_agg_R(&random_points);

    let mut s_values = Seq::<Secp256k1Scalar>::new(avs.len());
    for i in 0..avs.len(){
      s_values[i] = compute_own_s(secret_keys[i], agg_pk, agg_R, m.clone(), &public_keys, rands[i]);
    }
    let agg_s = compute_agg_s(s_values);
    let signature = (agg_R, agg_s);

    TestResult::from_bool(multi_sig_verify(public_keys, &m, signature))
  }
  QuickCheck::new()
      .gen(Gen::new(5))
      .tests(5)
      .quickcheck(helper as fn(Vec<(Secp256k1ScalarGenerator, Secp256k1ScalarGenerator, Secp256k1ScalarGenerator)>, Vec<u8>) -> TestResult)
}

#[test]
#[allow(non_snake_case)]
fn test_lots_of_batch_verification() {
  fn helper(avms: Vec<(Secp256k1ScalarGenerator, Secp256k1ScalarGenerator, Vec<u8>, Secp256k1ScalarGenerator)>) -> TestResult {
    let mut messages = Seq::<ByteSeq>::new(avms.len());
    let mut public_keys = Seq::<Affine>::new(avms.len());
    let mut signatures = Seq::<(Affine, Secp256k1Scalar)>::new(avms.len());
    let mut rands = Seq::<Secp256k1Scalar>::new(avms.len());
    for i in 0..avms.len() {
      let a = avms[i].0.into();
      let v = avms[i].1.into();
      let m = &avms[i].2;
      if a == Secp256k1Scalar::ZERO() {
        return TestResult::discard()
      }
      if v == Secp256k1Scalar::ZERO() {
        return TestResult::discard()
      }
      let m = ByteSeq::from_vec(m.iter().map(|i| (*i).into()).collect());
      let A = scalar_multiplication(a, GENERATOR());
      messages[i] = m.clone();
      public_keys[i] = A;
      signatures[i] = sign(a, A, v, &m);
      rands[i] = avms[i].3.into()
    }
    TestResult::from_bool(batch_verification(messages, public_keys, signatures, rands))
  }
  QuickCheck::new()
      .gen(Gen::new(20))
      .tests(50)
      .quickcheck(helper as fn(Vec<(Secp256k1ScalarGenerator, Secp256k1ScalarGenerator, Vec<u8>, Secp256k1ScalarGenerator)>) -> TestResult);
}