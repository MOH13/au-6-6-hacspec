use hacspec_lib::*;

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

public_nat_mod!(
    type_name: Secp256k1Scalar,
    type_of_canvas: ScalarCanvas,
    bit_size_of_field: 256,
    modulo_value: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"
);

pub type Affine = (Secp256k1FieldElement, Secp256k1FieldElement);
pub type AffineResult = Result<Affine, Error>;

bytes!(Secp256k1SerializedPoint, 32);
bytes!(Secp256k1SerializedScalar, 32);

fn is_infinity(p: Affine) -> bool {
    p == infinity()
}

fn infinity() -> Affine {
    (Secp256k1FieldElement::ONE(), Secp256k1FieldElement::ZERO())
}

fn add_points(p: Affine, q: Affine) -> AffineResult {
    let mut result = AffineResult::Err(Error::NoValue);
    if is_infinity(p) {
        result = AffineResult::Ok(infinity());
    } else {
        if is_infinity(q) {
            result = AffineResult::Ok(infinity());
        } else {
            if p == q {
                result = double_point(p);
            } else {
                let (px,py) = p;
                let (qx,qy) = q;
                if px == qx && py + qy == Secp256k1FieldElement::ZERO() {
                    result = AffineResult::Ok(infinity());
                } else {
                    result = add_different_points(p,q);
                }
            }
        }
    };
    result
}

fn add_different_points(p: Affine, q: Affine) -> AffineResult {
    let (px,py) = p;
    let (qx,qy) = q;
    let s = (qy - py) / (qx - px);
    let s2 = s * s;
    let x3 = s2 - px - qx;
    let y3 = s * (px - x3) - py;
    AffineResult::Ok((x3,y3))
}

fn double_point(p: Affine) -> AffineResult {
    let (x,y) = p;
    let mut result = AffineResult::Err(Error::NoValue);
    if y == Secp256k1FieldElement::ZERO() {
        result = AffineResult::Ok(infinity())
    } else {
        let t = (Secp256k1FieldElement::from_literal(3u128) * x * x) / (Secp256k1FieldElement::from_literal(2u128) * y); //Equal to (3 * x * x + CURVE_A) / (2 * y), since a = 0
        let t2 = t * t;
        let x3 = t2 - Secp256k1FieldElement::TWO() * x;
        let y3 = t2 * (x - x3) - y;
        result = AffineResult::Ok((x3, y3))
    };
    result
}