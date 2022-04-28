use hacspec_lib::*;

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

pub type Affine = (Secp256k1FieldElement, Secp256k1FieldElement);

/// Generates an affine representation of point at infinity (uses a placeholder off the curve)
#[allow(non_snake_case)]
pub fn INFINITY() -> Affine {
    (Secp256k1FieldElement::ONE(), Secp256k1FieldElement::ZERO())
}

/// Checks whether the given point is the point at infinity
pub fn is_infinity(p: Affine) -> bool {
    p == INFINITY()
}


/// Checks if the given point is a valid point on the curve
pub fn is_point_on_curve(p: Affine) -> bool {
    let (x,y) = p;
    is_infinity(p) || y.exp(2u32) == x.exp(3u32) + Secp256k1FieldElement::from_literal(7u128)
}
/// Returns the base point, G, for the Secp256k1 curve in affine coordinates
#[allow(non_snake_case)]
pub fn GENERATOR() -> Affine {
    (Secp256k1FieldElement::from_byte_seq_be(&byte_seq!(
        0x79u8, 0xBEu8, 0x66u8, 0x7Eu8, 0xF9u8, 0xDCu8, 0xBBu8, 0xACu8,
        0x55u8, 0xA0u8, 0x62u8, 0x95u8, 0xCEu8, 0x87u8, 0x0Bu8, 0x07u8,
        0x02u8, 0x9Bu8, 0xFCu8, 0xDBu8, 0x2Du8, 0xCEu8, 0x28u8, 0xD9u8,
        0x59u8, 0xF2u8, 0x81u8, 0x5Bu8, 0x16u8, 0xF8u8, 0x17u8, 0x98u8
    )),
    Secp256k1FieldElement::from_byte_seq_be(&byte_seq!(
        0x48u8, 0x3Au8, 0xDAu8, 0x77u8, 0x26u8, 0xA3u8, 0xC4u8, 0x65u8,
        0x5Du8, 0xA4u8, 0xFBu8, 0xFCu8, 0x0Eu8, 0x11u8, 0x08u8, 0xA8u8,
        0xFDu8, 0x17u8, 0xB4u8, 0x48u8, 0xA6u8, 0x85u8, 0x54u8, 0x19u8,
        0x9Cu8, 0x47u8, 0xD0u8, 0x8Fu8, 0xFBu8, 0x10u8, 0xD4u8, 0xB8u8
    )))
}

pub fn neg_point(p: Affine) -> Affine {
    let (x,y) = p;
    (x, y.neg())
}

/// Helper function for add_points
fn add_different_points(p: Affine, q: Affine) -> Affine {
    let (px,py) = p;
    let (qx,qy) = q;
    let s = (qy - py) * (qx - px).inv();
    let s2 = s * s;
    let x3 = s2 - px - qx;
    let y3 = s * (px - x3) - py;
    (x3,y3)
}

/// Doubles the given point in affine coordinates
pub fn double_point(p: Affine) -> Affine {
    #[allow(unused_assignments)]
    let mut result = INFINITY();
    let neg_p = neg_point(p);
    if p == neg_p {
        result = INFINITY()
    } else {
        let (x,y) = p;
        let t = (Secp256k1FieldElement::from_literal(3u128) * x * x) * (Secp256k1FieldElement::from_literal(2u128) * y).inv(); //Equal to (3 * x * x + CURVE_A) / (2 * y), since a = 0
        let t2 = t * t;
        let x3 = t2 - Secp256k1FieldElement::TWO() * x;
        let y3 = t * (x - x3) - y;
        result = (x3, y3)
    };
    result
}

pub fn add_points(p: Affine, q: Affine) -> Affine {
    #[allow(unused_assignments)]
    let mut result = INFINITY();
    if is_infinity(p) {
        result = q;
    } else {
        if is_infinity(q) {
            result = p;
        } else {
            if p == q {
                result = double_point(p);
            } else {
                let neg_q = neg_point(q);
                if p == neg_q {
                    result = INFINITY();
                } else {
                    result = add_different_points(p,q);
                }
            }
        }
    }
    result
}

/// Performs scalar multiplication on the given point in affine coordinates
pub fn scalar_multiplication(k: Secp256k1Scalar, p: Affine) -> Affine {
    let mut q = INFINITY();
    for i in 0..SCALAR_BITS {
        q = double_point(q);
        if k.bit(SCALAR_BITS - 1 - i) {
            q = add_points(p, q);
        }
    }
    q
}

/// Calculates the sum a_1 * P_1 + ... + a_i * P_i + ... + A_m * P_m
pub fn batch_scalar_multiplication(elems: Seq<(Secp256k1Scalar, Affine)>) -> Affine {
    #[allow(unused_assignments)]
    let mut res = INFINITY();
    if elems.len() == 0 {
        res = INFINITY()
    }
    else {
        let mut new_elems = elems;
        for i in 0..new_elems.len()-2 {
            let (ai, pi) = new_elems[i];
            let (aiplus1, piplus1) = new_elems[i+1];
            new_elems[i] = (ai - aiplus1, pi);
            new_elems[i+1] = (aiplus1, add_points(pi, piplus1));
        }
        for i in 0..new_elems.len()-1 {
            let (ai, pi) = new_elems[i];
            res = add_points(res, scalar_multiplication(ai, pi))
        }
    }
    res
}
