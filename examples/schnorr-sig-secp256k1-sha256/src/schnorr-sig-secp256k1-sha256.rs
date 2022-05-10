use hacspec_lib::*;
use hacspec_secp256k1::*;
use hacspec_sha256::*;

#[allow(non_snake_case)]
pub fn generate_c(V: Affine, A: Affine, m: &ByteSeq) -> Secp256k1Scalar {
    let (Vx, Vy) = V;
    let (Ax, Ay) = A;
    let c = hash(&Vx.to_byte_seq_le().concat(&Vy.to_byte_seq_le().concat(&Ax.to_byte_seq_le().concat(&Ay.to_byte_seq_le().concat(m)))));
    Secp256k1Scalar::from_byte_seq_le(c)
}

/// Creates a Schnorr signature for a single signer
#[allow(non_snake_case)]
pub fn sign(a: Secp256k1Scalar, A: Affine, v: Secp256k1Scalar, m: &ByteSeq) -> (Affine, Secp256k1Scalar) {
    let g = GENERATOR();
    let V = scalar_multiplication(v, g);
    let c = generate_c(V, A, m);
    let r = v - a * c;
    (V,r)
}

/// Given a public key and a message, this method verifies the signature of the message
#[allow(non_snake_case)]
pub fn verify(A: Affine, m: &ByteSeq, signature : (Affine, Secp256k1Scalar)) -> bool {
    let (V, r) = signature;
    let g = GENERATOR();
    let c = generate_c(V, A, m);
    let gr = scalar_multiplication(r, g);
    let cA = scalar_multiplication(c, A);
    V == add_points(gr, cA) && is_point_on_curve(A) && !is_infinity(A)
}

///Helper method that computes the 'a' values in the MuSig scheme
#[allow(non_snake_case)]
pub fn compute_a_values(L: &Seq<Affine>) -> Seq<Sha256Digest>{
    let mut a: Seq<Sha256Digest> = Seq::<Sha256Digest>::new(L.len());
    let L_as_bytes = public_keys_to_byte_seqs(L);
    let L_byte_concat = concat_byte_seqs_to_single_byte_seq(&L_as_bytes);
    for i in 0..L.len() {
        a[i] = hash(&L_byte_concat.concat(&L_as_bytes[i]));
    }
    a
}

#[allow(non_snake_case)]
/// Computes the "aggregate" public key from the signers public keys and their respective a values. Assumes L and a are sorted similarly
pub fn compute_agg_pk (L: Seq<Affine>, a: Seq<Sha256Digest>) -> Affine {
    let mut agg_pk = INFINITY();
    for i in 0..L.len() {
        let pk_i_a_i = scalar_multiplication(Secp256k1Scalar::from_byte_seq_le(a[i]), L[i]); //a_i * pk_i
        agg_pk = add_points(agg_pk, pk_i_a_i);
    }
    agg_pk
}


#[allow(non_snake_case)]
/// Hashes the points in R_seq and checks them against t. Assumes R_seq and t are sorted similarly i.e. if point x is on index y in R_seq, the hash of x must be on index y in t
pub fn check_ti_match_Ri (t: Seq<Sha256Digest>, R_seq: Seq<Affine>) -> bool {
    let mut check = true;
    for i in 0..R_seq.len() {
        let (Rx, Ry) = R_seq[i];
        if !(t[i].equal(hash(&Rx.to_byte_seq_le().concat(&Ry.to_byte_seq_le())))) {
            check = false;
        }
    }
    check
}

#[allow(non_snake_case)]
/// Computes the aggregate point of all random points used in the MuSig scheme and returns this point
pub fn compute_agg_R (R_seq: &Seq<Affine>) -> Affine {
    let mut R = INFINITY();
    for i in 0..R_seq.len() { // compute the random point to be used in signature, as an "aggreate" of all random points
        R = add_points(R, R_seq[i]);
    }
    R
}

#[allow(non_snake_case)]
///Computes the specific signer's s value
pub fn compute_own_s (v: Secp256k1Scalar, agg_pk: Affine, R: Affine, m: ByteSeq, L: Seq<Affine>, r: Secp256k1Scalar) -> Secp256k1Scalar {
    let g = GENERATOR();
    let (Vx, Vy) = scalar_multiplication(v, g); // Signer's public key
    let V_as_bytes = Vx.to_byte_seq_le().concat(&Vy.to_byte_seq_le());

    let (Rx, Ry) = R; // The aggregate random point
    let R_as_bytes = Rx.to_byte_seq_le().concat(&Ry.to_byte_seq_le());

    let (Px, Py) = agg_pk; // Aggregate public key of all signers
    let agg_pk_as_bytes = Px.to_byte_seq_le().concat(&Py.to_byte_seq_le());

    let pk_R_m_as_bytes = agg_pk_as_bytes.concat(&R_as_bytes.concat(&m));

    let c = hash(&pk_R_m_as_bytes);
    let c_as_scalar = Secp256k1Scalar::from_byte_seq_le(c);

    let L_byte_concat = concat_byte_seqs_to_single_byte_seq(&public_keys_to_byte_seqs(&L));
    let a_1 = hash(&L_byte_concat.concat(&V_as_bytes));
    let a_1_as_scalar = Secp256k1Scalar::from_byte_seq_le(a_1);

    let s_1 = r + c_as_scalar * a_1_as_scalar * v;
    s_1
}

pub fn compute_agg_s (s_seq: Seq<Secp256k1Scalar>) -> Secp256k1Scalar {
    let mut s = Secp256k1Scalar::ZERO();
    for i in 0..s_seq.len() {
        s = s + s_seq[i];
    }
    s
}

#[allow(non_snake_case)]
pub fn multi_sig_verify(L: Seq<Affine>, m: &ByteSeq, signature: (Affine, Secp256k1Scalar)) -> bool {
    let a: Seq<Sha256Digest> = Seq::<Sha256Digest>::new(0);
    let L_as_bytes: Seq<ByteSeq> = Seq::<ByteSeq>::new(0);
    let (R, s) = signature;
    let g = GENERATOR();

    // Transform all public keys into byte sequences
    for i in 0..L.len() {
        let (x, y) = L[i];
        L_as_bytes.push(&x.to_byte_seq_le().concat(&y.to_byte_seq_le()));
    }

    // Concatenate all byte sequences of public keys into one sequence
    let mut L_byte_concat: Seq<U8> = Seq::<U8>::new(0);
    for i in 0..L_as_bytes.len(){
        L_byte_concat = L_byte_concat.concat(&L_as_bytes[i]);
    }

    for i in 0..L.len() {
        a.push(&hash(&L_byte_concat.concat(&L_as_bytes[i])));
    }

    let mut agg_pk = INFINITY();
    for i in 0..L.len() {
        let pk_i_a_i = scalar_multiplication(Secp256k1Scalar::from_byte_seq_le(a[i]), L[i]); //pk_i * a_i
        agg_pk = add_points(agg_pk, pk_i_a_i); // aggregated key is addition of all (pk_i * a_i)
    }

    let (Rx, Ry) = R;
    let R_as_bytes = Rx.to_byte_seq_le().concat(&Ry.to_byte_seq_le());

    let (Px, Py) = agg_pk;
    let agg_pk_as_bytes = Px.to_byte_seq_le().concat(&Py.to_byte_seq_le());

    let agg_pk_R_m_as_bytes = agg_pk_as_bytes.concat(&R_as_bytes.concat(m));

    let c = hash(&agg_pk_R_m_as_bytes);

    let mut agg_pk_a_c = INFINITY();
    for i in 0..L.len() {
        let pk_i_a_i_c = scalar_multiplication(Secp256k1Scalar::from_byte_seq_le(c), scalar_multiplication(Secp256k1Scalar::from_byte_seq_le(a[i]), L[i])); //pk_i * a_i * c
        agg_pk_a_c = add_points(agg_pk_a_c, pk_i_a_i_c);
    }

    let agg_pk_c = scalar_multiplication(Secp256k1Scalar::from_byte_seq_le(c), agg_pk);


    // signature is correct if g x s = R + sum(pk_i x a_i x c) = R + agg_pk x c
    scalar_multiplication(s, g).eq(&add_points(R, agg_pk_a_c)) && add_points(R, agg_pk_a_c).eq(&add_points(R, agg_pk_c))
}

///Helper method that transforms a sequence of Affine points into byte sequences
#[allow(non_snake_case)]
pub fn public_keys_to_byte_seqs (L: &Seq<Affine>) -> Seq<ByteSeq> {
    let mut L_as_bytes = Seq::<ByteSeq>::new(L.len());
    for i in 0..L.len() {
        let (x, y) = L[i];
        L_as_bytes[i] = x.to_byte_seq_le().concat(&y.to_byte_seq_le());
    }
    L_as_bytes
}

///Helper method that transforms a sequence of byte sequences into one byte sequence
#[allow(non_snake_case)]
pub fn concat_byte_seqs_to_single_byte_seq (L_as_bytes: &Seq<ByteSeq>) -> ByteSeq {
    let mut L_byte_concat: ByteSeq = ByteSeq::new(0);
    for i in 0..L_as_bytes.len(){
        L_byte_concat = L_byte_concat.concat(&L_as_bytes[i]);
    }
    L_byte_concat
}

#[allow(non_snake_case)]
pub fn valid_As(As: &Seq<Affine>) -> bool {
    let mut res = true;
    for i in 0..As.len(){
        let A = As[i];
        res = res && is_point_on_curve (A) && !is_infinity(A)
    }
    res
}

/// Verifies a batch of signatures and corresponding messages based on https://en.bitcoin.it/wiki/BIP_0340
#[allow(non_snake_case)]
pub fn batch_verification (m: Seq<ByteSeq>, As: Seq<Affine>, sigs: Seq<(Affine, Secp256k1Scalar)>, rand: Seq<Secp256k1Scalar>) -> bool {
    #[allow(unused_assignments)]
    let mut res = false;
    if m.len() != As.len() || As.len() != sigs.len() || sigs.len() != rand.len() || !valid_As(&As) {
        res = false;
    } else {
        let mut Vs = Seq::<Affine>::new(sigs.len());
        let mut rs = Seq::<Secp256k1Scalar>::new(sigs.len());
        for i in 0..sigs.len() {
            let (V, r) = sigs[i];
            Vs[i] = V;
            rs[i] = r;
        }
        let g = GENERATOR();
        let mut cs = Seq::<Secp256k1Scalar>::new(sigs.len());
        let mut grs = Seq::<(Secp256k1Scalar, Affine)>::new(sigs.len());
        let mut cAs = Seq::<(Secp256k1Scalar, Affine)>::new(sigs.len());
        let mut Vmults = Seq::<(Secp256k1Scalar, Affine)>::new(sigs.len());
        for i in 0..sigs.len() {
            cs[i] = generate_c(Vs[i], As[i], &m[i]);
            grs[i] = (rs[i] * rand[i], g);
            cAs[i] = (cs[i] * rand[i], As[i]);
            Vmults[i] = (rand[i], Vs[i])
        }
        res = batch_scalar_multiplication (&Vmults) == add_points(batch_scalar_multiplication (&grs), batch_scalar_multiplication(&cAs))
    }
    res
}