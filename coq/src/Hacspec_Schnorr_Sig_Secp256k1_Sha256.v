(** This file was automatically generated using Hacspec **)
Require Import Hacspec_Lib MachineIntegers.
From Coq Require Import ZArith.
Import List.ListNotations.
Open Scope Z_scope.
Open Scope bool_scope.
Open Scope hacspec_scope.
Require Import Hacspec_Lib Field.

Require Import Hacspec_Secp256k1.

Require Import Hacspec_Sha256.

Add Field field_elem_FZpZ : field_elem_FZpZ.
Add Field scalar_FZpZ : scalar_FZpZ.

(** 
This file contains the coq export of the hacspec-schnorr-sig-secp256k1-sha256 implementation and its corresponding proofs.

Proven properties of the curve implementation include:
- [schnorr_correctness]: Proof of correctness for single signatures.

There are currently no proofs of correctness for batch verification or multi-signatures.

*)

(** * hacspec-to-coq definitions *)

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

(** * Schnorr properties *)

Lemma schnorr_correctness: forall (a v : secp256k1_scalar_t) (m : byte_seq), a <> nat_mod_zero /\ v <> nat_mod_zero -> verify (a *' generator) m (sign a (a *' generator) v m) =  true.
Proof.
  intros a v m H.
  destruct H as [H0 H1].
  unfold verify.
  unfold sign.
  remember (v *' generator) as V.
  remember (a *' generator) as A.
  destruct V as (Vx, Vy).
  destruct A as (Ax, Ay). 
  remember (nat_mod_from_byte_seq_le
    (hash
      (seq_concat (nat_mod_to_byte_seq_le Vx)
          (seq_concat (nat_mod_to_byte_seq_le Vy)
            (seq_concat (nat_mod_to_byte_seq_le Ax)
                (seq_concat (nat_mod_to_byte_seq_le Ay) m)))))) as H.
  remember (Vx, Vy) as V.
  remember (Ax, Ay) as A.
  rewrite Bool.andb_true_iff.
  rewrite Bool.andb_true_iff.
  split.
  - split.
    + rewrite HeqA.
      remember (mkoncurve generator generator_on_curve) as g.
      assert (generator = point g) as ->. { rewrite Heqg. reflexivity. }
      rewrite (scalar_mult_assoc2 H a g).
      rewrite scalar_mult_distributivity.
      assert (v = ((v -% (a *% H)) +% (H *% a))). {
        unfold secp256k1_scalar_t in v, a, H.
        field_simplify.
        reflexivity.
      }
      rewrite <- H2.
      rewrite HeqV, eqb_leibniz.
      rewrite Heqg.
      reflexivity.
    + remember (mkoncurve generator generator_on_curve) as g.
      pose proof scalar_mult_closed g a.
      destruct H2.
      rewrite Heqg in H2.
      simpl in H2.
      rewrite <- HeqA in H2.
      pose proof on_curve x.
      rewrite H2 in H3.
      exact H3.
  - pose proof scalar_mult_generator_not_zero a H0.
    rewrite <- HeqA in H2.
    rewrite H2.
    reflexivity.
Qed.