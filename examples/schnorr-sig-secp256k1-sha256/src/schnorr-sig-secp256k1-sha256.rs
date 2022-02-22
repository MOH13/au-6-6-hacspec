use hacspec_lib::*;
use hacspec_secp256k1::*;
use hacspec_sha256::*;

pub fn sign(x: Secp256k1Scalar, m: &ByteSeq) {
    let g = BASE_POINT();
    let k = Secp256k1Scalar::ZERO() //Replace this with fresh scalar!
    let r = scalar_multiplication()
}