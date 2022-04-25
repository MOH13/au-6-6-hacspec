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
pub fn multi_sig(v: Secp256k1Scalar, m: &ByteSeq, L: Seq<Affine>) {
    let L_as_bytes: Seq<ByteSeq> = Seq::<ByteSeq>::new(L.len());
    let a: Seq<Sha256Digest> = Seq::<Sha256Digest>::new(L.len());
    let g = GENERATOR();
    let (Vx, Vy) = scalar_multiplication(v, g);
    let signer_pk_as_bytes = Vx.to_byte_seq_le().concat(&Vy.to_byte_seq_le());
    

    // Transform all public keys into byte sequences
    for i in 0..L.len() {
        let (x, y) = L[i];
        L_as_bytes.push(&x.to_byte_seq_le().concat(&y.to_byte_seq_le()));
    }

    // Concatenate all byte sequences into one sequence
    let mut L_byte_concat: Seq<U8> = Seq::<U8>::new(L_as_bytes.len());
    for i in 0..L_as_bytes.len(){
        L_byte_concat = L_byte_concat.concat(&L_as_bytes[i]);
    }

    // compute a_i's by hashing L with each X_i
    for i in 0..L.len() {
        a.push(&hash(&L_byte_concat.concat(&L_as_bytes[i])));
    }
    // this is also stored in a above, but how to know which of the entries belong to our sk?
    let signers_a = hash(&L_byte_concat.concat(&signer_pk_as_bytes));
}

#[allow(non_snake_case)]
pub fn verify(A: Affine, m: &ByteSeq, signature : (Affine, Secp256k1Scalar)) -> bool {
    let (V, r) = signature;
    let g = GENERATOR();
    let (Vx, Vy) = V;
    let (Ax, Ay) = A;
    let c = hash(&Vx.to_byte_seq_le().concat(&Vy.to_byte_seq_le().concat(&Ax.to_byte_seq_le().concat(&Ay.to_byte_seq_le().concat(m)))));
    let c_as_scalar = Secp256k1Scalar::from_byte_seq_le(c);
    let gr = scalar_multiplication(r, g);
    let cA = scalar_multiplication(c_as_scalar, A);
    V == add_points(gr, cA) && is_point_on_curve(A) && !is_infinity(A)
} 