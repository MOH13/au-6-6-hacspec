(** This file was automatically generated using Hacspec **)
Require Import Lib MachineIntegers.
From Coq Require Import ZArith.
Import List.ListNotations.
Open Scope Z_scope.
Open Scope bool_scope.
Open Scope hacspec_scope.
Require Import Hacspec.Lib.

Require Import Hacspec.Secp256k1.

Require Import Hacspec.Sha256.

Definition sign
  (a_0 : secp256k1_scalar_t)
  (a_1 : affine_t)
  (v_2 : secp256k1_scalar_t)
  (m_3 : byte_seq)
  : (affine_t × secp256k1_scalar_t) :=
  let g_4 : (secp256k1_field_element_t × secp256k1_field_element_t) :=
    generator  in 
  let '(vx_5, vy_6) :=
    scalar_multiplication (v_2) (g_4) in 
  let '(ax_7, ay_8) :=
    a_1 in 
  let c_9 : sha256_digest_t :=
    hash (seq_concat (nat_mod_to_byte_seq_le (vx_5)) (seq_concat (
          nat_mod_to_byte_seq_le (vy_6)) (seq_concat (nat_mod_to_byte_seq_le (
              ax_7)) (seq_concat (nat_mod_to_byte_seq_le (ay_8)) (m_3))))) in 
  let c_as_scalar_10 : secp256k1_scalar_t :=
    nat_mod_from_byte_seq_le (c_9) : secp256k1_scalar_t in 
  let r_11 : secp256k1_scalar_t :=
    (v_2) -% ((a_0) *% (c_as_scalar_10)) in 
  ((vx_5, vy_6), r_11).

Definition verify
  (a_12 : affine_t)
  (m_13 : byte_seq)
  (signature_14 : (affine_t × secp256k1_scalar_t))
  : bool :=
  let '(v_15, r_16) :=
    signature_14 in 
  let g_17 : (secp256k1_field_element_t × secp256k1_field_element_t) :=
    generator  in 
  let '(vx_18, vy_19) :=
    v_15 in 
  let '(ax_20, ay_21) :=
    a_12 in 
  let c_22 : sha256_digest_t :=
    hash (seq_concat (nat_mod_to_byte_seq_le (vx_18)) (seq_concat (
          nat_mod_to_byte_seq_le (vy_19)) (seq_concat (nat_mod_to_byte_seq_le (
              ax_20)) (seq_concat (nat_mod_to_byte_seq_le (ay_21)) (
              m_13))))) in 
  let c_as_scalar_23 : secp256k1_scalar_t :=
    nat_mod_from_byte_seq_le (c_22) : secp256k1_scalar_t in 
  let gr_24 : (secp256k1_field_element_t × secp256k1_field_element_t) :=
    scalar_multiplication (r_16) (g_17) in 
  let c_a_25 : (secp256k1_field_element_t × secp256k1_field_element_t) :=
    scalar_multiplication (c_as_scalar_23) (a_12) in 
  (((v_15) =.? (add_points (gr_24) (c_a_25))) && (is_point_on_curve (
        a_12))) && (negb (is_infinity (a_12))).

