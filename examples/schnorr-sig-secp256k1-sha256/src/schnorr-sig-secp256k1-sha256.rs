use hacspec_lib::*;
use hacspec_secp256k1::*;
use hacspec_sha256::*;

// todo: rename variables to be consistent with IETF standard
#[allow(non_snake_case)]
pub fn sign(a: Secp256k1Scalar, A: Affine, v: Secp256k1Scalar, m: &ByteSeq) -> (Affine, Secp256k1Scalar) {
    let g = GENERATOR();
    let (Vx,Vy) = scalar_multiplication(v, g);
    let (Ax, Ay) = A;
    let c = hash(&Vx.to_byte_seq_le().concat(&Vy.to_byte_seq_le().concat(&Ax.to_byte_seq_le().concat(&Ay.to_byte_seq_le().concat(m)))));
    let c_as_scalar = Secp256k1Scalar::from_byte_seq_le(c);
    let r = v - a * c_as_scalar;
    ((Vx, Vy),r)
}

#[allow(non_snake_case)]
pub fn verify(A: Affine, m: &ByteSeq, V: Affine, r: Secp256k1Scalar) -> bool {
    let g = GENERATOR();
    let (Vx, Vy) = V;
    let (Ax, Ay) = A;
    let c = hash(&Vx.to_byte_seq_le().concat(&Vy.to_byte_seq_le().concat(&Ax.to_byte_seq_le().concat(&Ay.to_byte_seq_le().concat(m)))));
    let c_as_scalar = Secp256k1Scalar::from_byte_seq_le(c);
    let gr = scalar_multiplication(r, g);
    let cA = scalar_multiplication(c_as_scalar, A);
    V == add_points(gr, cA) && is_point_on_curve(A) && !is_infinity(A)
} 