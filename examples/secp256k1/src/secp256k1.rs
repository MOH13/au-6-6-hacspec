use hacspec_lib::*;

#[derive(Debug)]
pub enum Error {
    InvalidAddition,
    NoValue,
}

public_nat_mod!(
    type_name: Secp256k1FieldElement,
    type_of_canvas: FieldCanvas,
    bit_size_of_field: 256,
    modulo_value: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"
);

const SCALAR_BITS: usize = 256;

public_nat_mod!(
    type_name: Secp256k1Scalar,
    type_of_canvas: ScalarCanvas,
    bit_size_of_field: 256,
    modulo_value: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"
);

pub type AffineResult = Result<Affine, Error>;
pub type Affine = (Secp256k1FieldElement, Secp256k1FieldElement);

/// Checks whether the given point is the point at infinity
pub fn is_infinity(p: &Affine) -> bool {
    p == &INFINITY()
}

/// Generates an affine representation of point at infinity (uses a placeholder off the curve)
pub fn INFINITY() -> Affine {
    (Secp256k1FieldElement::ONE(), Secp256k1FieldElement::ZERO())
}

/// Returns the base point, G, for the Secp256k1 curve in affine coordinates
pub fn BASE_POINT() -> Affine {
    (Secp256k1FieldElement::from_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"),
    Secp256k1FieldElement::from_hex("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"))
}

pub fn neg_point(p: Affine) -> AffineResult {
    let (x,y) = p;
    AffineResult::Ok((x, Secp256k1FieldElement::ZERO() - y))
}

pub fn add_points(p: Affine, q: Affine) -> AffineResult {
    let mut result = AffineResult::Err(Error::NoValue);
    if is_infinity(&p) {
        result = AffineResult::Ok(q);
    } else {
        if is_infinity(&q) {
            result = AffineResult::Ok(p);
        } else {
            if p == q {
                result = double_point(p);
            } else {
                let (px,py) = p;
                let (qx,qy) = q;
                if px == qx && py + qy == Secp256k1FieldElement::ZERO() {
                    result = AffineResult::Ok(INFINITY());
                } else {
                    result = add_different_points(p,q);
                }
            }
        }
    };
    result
}

/// Helper function for add_points
fn add_different_points(p: Affine, q: Affine) -> AffineResult {
    let (px,py) = p;
    let (qx,qy) = q;
    let s = (qy - py) / (qx - px);
    let s2 = s * s;
    let x3 = s2 - px - qx;
    let y3 = s * (px - x3) - py;
    AffineResult::Ok((x3,y3))
}

/// Doubles the given point in affine coordinates
pub fn double_point(p: Affine) -> AffineResult {
    let (x,y) = p.into();
    let mut result = AffineResult::Err(Error::NoValue);
    if y.equal(Secp256k1FieldElement::ZERO()) {
        result = AffineResult::Ok(INFINITY())
    } else {
        let t = (Secp256k1FieldElement::from_literal(3u128) * x * x) / (Secp256k1FieldElement::from_literal(2u128) * y); //Equal to (3 * x * x + CURVE_A) / (2 * y), since a = 0
        let t2 = t * t;
        let x3 = t2 - Secp256k1FieldElement::TWO() * x;
        let y3 = t2 * (x - x3) - y;
        result = AffineResult::Ok((x3, y3))
    };
    result
}

/// Performs scalar multiplication on the given point in affine coordinates
pub fn scalar_multiplication(k: Secp256k1Scalar, p: Affine) -> AffineResult {
    let mut q = INFINITY();
    for i in 0..SCALAR_BITS {
        q = double_point(q)?;
        if k.get_bit(SCALAR_BITS - 1 - i).equal(Secp256k1Scalar::ONE()) {
            q = add_points(p, q)?;
        }
    }
    AffineResult::Ok(q)
}