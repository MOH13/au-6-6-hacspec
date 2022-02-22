use hacspec_lib::*;
use hacspec_secp256k1::*;
use hacspec_sha256::*;

pub fn sign(x: Secp256k1Scalar, k: Secp256k1Scalar, m: &ByteSeq) -> (Secp256k1Scalar, Secp256k1Scalar) {
    let g = BASE_POINT();
    let (rx,ry) = scalar_multiplication(k, g).unwrap();
    let e = hash(&rx.to_byte_seq_le().concat(&ry.to_byte_seq_le().concat(m)));
    let e_as_scalar = Secp256k1Scalar::from_byte_seq_le(e);
    let s = k - x * e_as_scalar;
    (s,e_as_scalar)
}

pub fn verify(y: Affine, m: &ByteSeq, s: Secp256k1Scalar, e: Secp256k1Scalar) -> bool {
    let g = BASE_POINT();
    let gs = scalar_multiplication(s, g).unwrap();
    let ye = scalar_multiplication(e, y).unwrap();
    let (rvx, rvy) = add_points(gs, ye).unwrap();
    let ev = hash(&rvx.to_byte_seq_le().concat(&rvy.to_byte_seq_le().concat(m)));
    let ev_as_scalar = Secp256k1Scalar::from_byte_seq_le(ev);
    e == ev_as_scalar
} 