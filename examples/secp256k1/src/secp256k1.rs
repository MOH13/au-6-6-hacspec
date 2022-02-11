use hacspec_lib::*;

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
    modulo_value: "?"
);

type Point = (Secp256k1FieldElement, Secp256k1FieldElement);

bytes!(Secp256k1SerializedPoint, 32);
bytes!(Secp256k1SerializedScalar, 32);

