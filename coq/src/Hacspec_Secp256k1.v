(** This file was automatically generated using Hacspec **)
Require Import Hacspec_Lib MachineIntegers.
From Coq Require Import ZArith.
Import List.ListNotations.
Open Scope Z_scope.
Open Scope bool_scope.
Open Scope hacspec_scope.
Require Import Hacspec_Lib.
Import Bool.
Import GZnZ.

Definition field_canvas_t := nseq (int8) (32).
Definition secp256k1_field_element_t :=
  nat_mod 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F.

Definition scalar_bits_v : uint_size :=
  usize 256.

Definition scalar_canvas_t := nseq (int8) (32).
Definition secp256k1_scalar_t :=
  nat_mod 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141.

Notation "'affine_t'" := ((
  secp256k1_field_element_t ×
  secp256k1_field_element_t
)) : hacspec_scope.

Definition infinity  : affine_t :=
  (nat_mod_one , nat_mod_zero ).

Definition is_infinity (p_3 : affine_t) : bool :=
  (p_3) =.? (infinity ).

Definition is_point_on_curve (p_0 : affine_t) : bool :=
  let '(x_1, y_2) :=
    p_0 in 
  (is_infinity (p_0)) || (((y_2) *% (y_2)) =.? ((((x_1) *% (x_1)) *% (x_1)) +% (
        nat_mod_from_literal (
          0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F) (
          @repr WORDSIZE128 7) : secp256k1_field_element_t))).

Definition generator  : affine_t :=
  (
    nat_mod_from_byte_seq_be ([
        secret (@repr WORDSIZE8 121) : int8;
        secret (@repr WORDSIZE8 190) : int8;
        secret (@repr WORDSIZE8 102) : int8;
        secret (@repr WORDSIZE8 126) : int8;
        secret (@repr WORDSIZE8 249) : int8;
        secret (@repr WORDSIZE8 220) : int8;
        secret (@repr WORDSIZE8 187) : int8;
        secret (@repr WORDSIZE8 172) : int8;
        secret (@repr WORDSIZE8 85) : int8;
        secret (@repr WORDSIZE8 160) : int8;
        secret (@repr WORDSIZE8 98) : int8;
        secret (@repr WORDSIZE8 149) : int8;
        secret (@repr WORDSIZE8 206) : int8;
        secret (@repr WORDSIZE8 135) : int8;
        secret (@repr WORDSIZE8 11) : int8;
        secret (@repr WORDSIZE8 7) : int8;
        secret (@repr WORDSIZE8 2) : int8;
        secret (@repr WORDSIZE8 155) : int8;
        secret (@repr WORDSIZE8 252) : int8;
        secret (@repr WORDSIZE8 219) : int8;
        secret (@repr WORDSIZE8 45) : int8;
        secret (@repr WORDSIZE8 206) : int8;
        secret (@repr WORDSIZE8 40) : int8;
        secret (@repr WORDSIZE8 217) : int8;
        secret (@repr WORDSIZE8 89) : int8;
        secret (@repr WORDSIZE8 242) : int8;
        secret (@repr WORDSIZE8 129) : int8;
        secret (@repr WORDSIZE8 91) : int8;
        secret (@repr WORDSIZE8 22) : int8;
        secret (@repr WORDSIZE8 248) : int8;
        secret (@repr WORDSIZE8 23) : int8;
        secret (@repr WORDSIZE8 152) : int8
      ]) : secp256k1_field_element_t,
    nat_mod_from_byte_seq_be ([
        secret (@repr WORDSIZE8 72) : int8;
        secret (@repr WORDSIZE8 58) : int8;
        secret (@repr WORDSIZE8 218) : int8;
        secret (@repr WORDSIZE8 119) : int8;
        secret (@repr WORDSIZE8 38) : int8;
        secret (@repr WORDSIZE8 163) : int8;
        secret (@repr WORDSIZE8 196) : int8;
        secret (@repr WORDSIZE8 101) : int8;
        secret (@repr WORDSIZE8 93) : int8;
        secret (@repr WORDSIZE8 164) : int8;
        secret (@repr WORDSIZE8 251) : int8;
        secret (@repr WORDSIZE8 252) : int8;
        secret (@repr WORDSIZE8 14) : int8;
        secret (@repr WORDSIZE8 17) : int8;
        secret (@repr WORDSIZE8 8) : int8;
        secret (@repr WORDSIZE8 168) : int8;
        secret (@repr WORDSIZE8 253) : int8;
        secret (@repr WORDSIZE8 23) : int8;
        secret (@repr WORDSIZE8 180) : int8;
        secret (@repr WORDSIZE8 72) : int8;
        secret (@repr WORDSIZE8 166) : int8;
        secret (@repr WORDSIZE8 133) : int8;
        secret (@repr WORDSIZE8 84) : int8;
        secret (@repr WORDSIZE8 25) : int8;
        secret (@repr WORDSIZE8 156) : int8;
        secret (@repr WORDSIZE8 71) : int8;
        secret (@repr WORDSIZE8 208) : int8;
        secret (@repr WORDSIZE8 143) : int8;
        secret (@repr WORDSIZE8 251) : int8;
        secret (@repr WORDSIZE8 16) : int8;
        secret (@repr WORDSIZE8 212) : int8;
        secret (@repr WORDSIZE8 184) : int8
      ]) : secp256k1_field_element_t
  ).

Definition neg_point (p_4 : affine_t) : affine_t :=
  let '(x_5, y_6) :=
    p_4 in 
  (x_5, (nat_mod_zero ) -% (y_6)).

Definition add_different_points
  (p_11 : affine_t)
  (q_12 : affine_t)
  : affine_t :=
  let '(px_13, py_14) :=
    p_11 in 
  let '(qx_15, qy_16) :=
    q_12 in 
  let s_17 : secp256k1_field_element_t :=
    ((qy_16) -% (py_14)) *% (nat_mod_inv ((qx_15) -% (px_13))) in 
  let s2_18 : secp256k1_field_element_t :=
    (s_17) *% (s_17) in 
  let x3_19 : secp256k1_field_element_t :=
    ((s2_18) -% (px_13)) -% (qx_15) in 
  let y3_20 : secp256k1_field_element_t :=
    ((s_17) *% ((px_13) -% (x3_19))) -% (py_14) in 
  (x3_19, y3_20).

Definition double_point (p_21 : affine_t) : affine_t :=
  let result_22 : (secp256k1_field_element_t × secp256k1_field_element_t) :=
    infinity  in 
  let neg_p_23 : (secp256k1_field_element_t × secp256k1_field_element_t) :=
    neg_point (p_21) in 
  let '(result_22) :=
    if (p_21) =.? (neg_p_23):bool then (let result_22 :=
        infinity  in 
      (result_22)) else (let '(x_24, y_25) :=
        p_21 in 
      let t_26 : secp256k1_field_element_t :=
        (((nat_mod_from_literal (
                0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F) (
                @repr WORDSIZE128 3) : secp256k1_field_element_t) *% (
              x_24)) *% (x_24)) *% (nat_mod_inv ((nat_mod_from_literal (
                0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F) (
                @repr WORDSIZE128 2) : secp256k1_field_element_t) *% (
              y_25))) in 
      let t2_27 : secp256k1_field_element_t :=
        (t_26) *% (t_26) in 
      let x3_28 : secp256k1_field_element_t :=
        (t2_27) -% ((nat_mod_two ) *% (x_24)) in 
      let y3_29 : secp256k1_field_element_t :=
        ((t_26) *% ((x_24) -% (x3_28))) -% (y_25) in 
      let result_22 :=
        (x3_28, y3_29) in 
      (result_22)) in 
  result_22.


  Definition add_points (p_7 : affine_t) (q_8 : affine_t) : affine_t :=
    let result_9 : (secp256k1_field_element_t × secp256k1_field_element_t) :=
      infinity  in 
    let '(result_9) :=
      if is_infinity (p_7):bool then (let result_9 :=
          q_8 in 
        (result_9)) else (let '(result_9) :=
          if is_infinity (q_8):bool then (let result_9 :=
              p_7 in 
            (result_9)) else (let '(result_9) :=
              if (p_7) =.? (q_8):bool then (let result_9 :=
                  double_point (p_7) in 
                (result_9)) else (let neg_q_10 : (
                    secp256k1_field_element_t ×
                    secp256k1_field_element_t
                  ) :=
                  neg_point (q_8) in 
                let '(result_9) :=
                  if (p_7) =.? (neg_q_10):bool then (let result_9 :=
                      infinity  in 
                    (result_9)) else (let result_9 :=
                      add_different_points (p_7) (q_8) in 
                    (result_9)) in 
                (result_9)) in 
            (result_9)) in 
        (result_9)) in 
    result_9.

Definition scalar_multiplication
  (k_30 : secp256k1_scalar_t)
  (p_31 : affine_t)
  : affine_t :=
  let q_32 : (secp256k1_field_element_t × secp256k1_field_element_t) :=
    infinity  in 
  let q_32 :=
    foldi (usize 0) (scalar_bits_v) (fun i_33 q_32 =>
      let q_32 :=
        double_point (q_32) in 
      let '(q_32) :=
        if nat_mod_equal (nat_mod_get_bit (k_30) (((scalar_bits_v) - (
                usize 1)) - (i_33))) (nat_mod_one ):bool then (let q_32 :=
            add_points (p_31) (q_32) in 
          (q_32)) else ((q_32)) in 
      (q_32))
    q_32 in 
  q_32.

Notation "p '+'' q" := (add_points p q) (at level 5, left associativity).

Lemma curve_eq_symm: forall (px py qx qy : secp256k1_field_element_t),
  (px, py) =.? (qx, qy) = (qx, qy) =.? (px, py).
Proof.
  intros.
  simpl.
  unfold nat_mod_val.
  destruct (px =? qx) eqn:eq1. {
    destruct (py =? qy) eqn:eq2. {
      rewrite -> Z.eqb_sym in eq1.
      rewrite -> Z.eqb_sym in eq2.
      rewrite -> eq1.
      rewrite -> eq2.
      reflexivity.
    } {
      rewrite -> Z.eqb_sym in eq2.
      rewrite -> eq2.
      rewrite -> andb_false_r.
      rewrite -> andb_false_r.
      reflexivity.
    }
  } {
    rewrite -> Z.eqb_sym in eq1.
    rewrite -> eq1.
    simpl.
    reflexivity.
  }
Qed.

Lemma infty_is_infty: is_infinity infinity = true.
Proof.
  unfold is_infinity.
  simpl.
  reflexivity.
Qed.

Lemma is_infty_means_infty: forall (p: affine_t), is_infinity p = true -> p =.? infinity = true.
Proof.
  intros p H.
  unfold is_infinity in H.
  apply H.
Qed.

Lemma double_neg: forall (p : affine_t), (p = neg_point (neg_point p)).
Proof.
  intros p.
  unfold neg_point.
  destruct p as (px, py).
  assert (nat_mod_zero -% (nat_mod_zero -% py) = py). {
    unfold "-%".
    unfold sub.
    unfold "nat_mod_zero - py".
    unfold nat_mod_zero.
    unfold "- p".
    simpl.
    destruct (val 115792089237316195423570985008687907853269984665640564039457584007908834671663 py) eqn:eq1. 
    - simpl.
      subst py.
    Search(GZnZ.znz).
  } {

  }
  simpl.

Lemma neg_symm: forall (p q : affine_t), (neg_point p =.? q) = (p =.? neg_point q).
Proof.
  intros p q.
  destruct (neg_point p =.? q) eqn:eq1. {
    rewrite -> eqb_leibniz in eq1.
    rewrite <- eq1.
    Search (?a =.? ?b = true).
  }

Lemma add_infty_1: forall (p: affine_t), infinity +' p =.? p = true.
Proof.
  intros p.
  unfold add_points.
  destruct (is_infinity infinity) eqn:eq.
  - rewrite -> Hacspec_Lib.eqb_refl.
    + reflexivity.
  - discriminate eq.
Qed.

Lemma add_comm: forall (p q : affine_t), add_points p q = add_points q p.
Proof.
  intros p q.
  unfold add_points.
  unfold nat_mod_val.
  destruct (is_infinity p) eqn:eq1. {
    destruct (is_infinity q) eqn:eq2. {
      apply is_infty_means_infty in eq1. {
        apply is_infty_means_infty in eq2. {
          rewrite -> eqb_leibniz in eq1. {
            rewrite -> eqb_leibniz in eq2. {
              rewrite -> eq1.
              rewrite -> eq2.
              reflexivity.
            }}}}} {
              reflexivity.
            }
  } {
    destruct (is_infinity q) eqn:eq2. {
      reflexivity.
    } {
      destruct p as (px, py) eqn:P1.
      destruct q as (qx, qy) eqn:Q1.
      rewrite -> curve_eq_symm.
      destruct ((qx, qy) =.? (px, py)) eqn:H1. {
        assert (H2: (px, py) = (qx, qy)). {
          Search (?a =.? ?b = true).
          rewrite -> eqb_leibniz in H1.
          rewrite -> H1.
          reflexivity.
        }
        rewrite H2.
        reflexivity.
      } {
        destruct ((px, py) =.? neg_point (qx, qy)) eqn:H2. {

        }
      }
      (*Missing case: P = -Q*)
      (*Missing case: P != Q /\ P != -Q /\ P != infty /\ Q != infty*)
    }
  }
Qed.

Lemma add_infty_2: forall (p: affine_t), p +' infinity =.? p = true.
Proof.
  intros p.
  unfold add_points.
  destruct (is_infinity p) eqn:eq1.
  - destruct (infinity =.? p) eqn:eq2.
    + reflexivity.
    + apply is_infty_means_infty in eq1. 
