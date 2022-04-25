(** This file was automatically generated using Hacspec **)
Require Import Hacspec_Lib MachineIntegers.
From Coq Require Import ZArith Nat.
Import List.ListNotations.
Open Scope Z_scope.
Open Scope bool_scope.
Open Scope hacspec_scope.
Require Import Hacspec_Lib.
Import Bool.
Import GZnZ.
Import Coq.ZArith.Zdiv.
Require Import ZDivEucl BinIntDef.
Require Import Lia.
Require Import Znumtheory Field_theory Field.
From Coqprime Require GZnZ.

Definition elem_max := 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F.
Definition scalar_max := 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141.

Definition field_canvas_t := nseq (int8) (32).
Definition secp256k1_field_element_t :=
  nat_mod elem_max.

Definition scalar_bits_v : uint_size :=
  usize 256.

Definition scalar_canvas_t := nseq (int8) (32).
Definition secp256k1_scalar_t :=
  nat_mod scalar_max.

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
  (is_infinity (p_0)) || (nat_mod_exp (y_2) (@repr WORDSIZE32 2) =.? ((nat_mod_exp (x_1) (@repr WORDSIZE32 3)) +% (
        nat_mod_from_literal (
          elem_max) (
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
  (x_5, nat_mod_neg (y_6)).

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
                elem_max) (
                @repr WORDSIZE128 3) : secp256k1_field_element_t) *% (
              x_24)) *% (x_24)) *% (nat_mod_inv ((nat_mod_from_literal (
                elem_max) (
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

Lemma zero_less_than_elem_max: 0 < elem_max.
Proof.
  unfold elem_max.
  intuition.
Qed.

Lemma zero_less_than_scalar_max: 0 < scalar_max.
Proof.
  unfold scalar_max.
  intuition.
Qed.

Notation "p '+'' q" := (add_points p q) (at level 5, left associativity).
Notation "k '*'' p" := (scalar_multiplication k p) (at level 4, right associativity).

Section nat_mod.

Variable max: Z.
Variable max_prime: prime max.

Definition nat_mod_eq (a b : nat_mod max) := a = b.

Add Field FZpZ : (GZnZ.FZpZ max max_prime).

Theorem max_pos: 0 < max.
  generalize (prime_ge_2 _ max_prime); auto with zarith.
Qed.

Definition nat_mod_FZpZ: field_theory nat_mod_zero nat_mod_one nat_mod_add nat_mod_mul nat_mod_sub nat_mod_neg nat_mod_div nat_mod_inv (@Logic.eq (nat_mod max)).
Proof.
  split.
  - split.
    + unfold nat_mod.
      unfold nat_mod_eq.
      unfold nat_mod_zero.
      unfold "+%".
      intros x.
      ring.
    + unfold nat_mod.
      intros x y.
      unfold "+%".
      ring.
    + unfold nat_mod.
      intros x y z.
      unfold "+%".
      ring.
    + unfold nat_mod.
      intros x.
      unfold "*%".
      unfold nat_mod_one.
      ring.
    + unfold nat_mod.
      intros x y.
      unfold "*%".
      ring.
    + unfold nat_mod.
      intros x y z.
      unfold "*%".
      ring.
    + unfold nat_mod.
      intros x y z.
      unfold "+%".
      unfold "*%".
      ring.
    + unfold nat_mod.
      intros x y.
      unfold "-%".
      unfold nat_mod_neg.
      unfold "+%".
      ring.
    + unfold nat_mod.
      intros x.
      unfold "+%".
      unfold nat_mod_neg.
      unfold nat_mod_zero.
      ring.
  - pose proof prime_ge_2 _ max_prime.
    unfold nat_mod_one.
    unfold nat_mod_zero.
    unfold one.
    unfold zero.  
    assert (H1: 1 < max). { lia. }
    Check (Zmod_1_l max H1).
    pose proof (Zmod_1_l max H1).
    assert (1 mod max <> 0 mod max). {
      rewrite H0.
      unfold "mod".
      simpl.
      intuition.
    }
    intuition.
    inversion H3.
    apply H2 in H5.
    contradiction.
  - unfold nat_mod.
    intros p q.
    unfold "/%", "*%", nat_mod_div, nat_mod_inv.
    unfold div.
    reflexivity.
  - unfold nat_mod.
    intros p H.
    unfold nat_mod_zero in H.
    unfold nat_mod_inv, nat_mod_one, "*%".
    field.
    exact H.
Qed.

Add Field nat_mod_FZpZ : nat_mod_FZpZ.

Lemma nat_mod_small: forall (n : nat_mod max), 0 <= n < max.
Proof.
  intros n.
  pose proof inZnZ max n.
  pose proof Z_mod_lt n max (Z.lt_gt _ _ max_pos).
  rewrite <- H in H0.
  exact H0.
Qed.

Lemma small_to_nat_mod: forall (a : Z), 0 <= a < max -> exists (b : nat_mod max), a =? b = true.
Proof.
  intros a H.
  pose proof (Zmod_small a max H).
  symmetry in H0.
  remember (mkznz max a H0) as b.
  exists b.
  rewrite Heqb.
  simpl.
  rewrite Z.eqb_eq.
  reflexivity.
Qed.

Lemma nat_mod_double_neg: forall (x : nat_mod max), nat_mod_neg (nat_mod_neg x) = x.
Proof.
  intros x.
  ring.
Qed.

Lemma nat_mod_neg_inj: forall (n m : nat_mod max), nat_mod_neg n = nat_mod_neg m -> n = m.
Proof.
  intros n m H.
  rewrite <- nat_mod_double_neg.
  rewrite <- H.
  rewrite nat_mod_double_neg.
  reflexivity.
Qed.

Lemma nat_mod_neg_both: forall (n m : nat_mod max), n = m <-> nat_mod_neg n = nat_mod_neg m.
Proof.
  intros n m.
  split.
  - intros H.
    rewrite H.
    reflexivity.
  - intros H.
    apply nat_mod_neg_inj.
    exact H.
Qed.

Lemma nat_mod_neg_symm: forall (x y : nat_mod max), nat_mod_neg x = y <-> x = nat_mod_neg y.
Proof.
  intros x y.
  split.
  - intros H.
    rewrite <- H.
    rewrite nat_mod_double_neg.
    reflexivity.
  - intros H.
    rewrite H.
    rewrite nat_mod_double_neg.
    reflexivity.
Qed.

Lemma one_minus_mod2: forall (x : Z), x mod 2 <> (1 - x mod 2) mod 2.
Proof.
  intros x.
  destruct (x mod 2 =? 0) eqn:eq1.
  - rewrite Z.eqb_eq in eq1.
    rewrite eq1.
    simpl.
    unfold "mod".
    simpl.
    intuition.
  - rewrite Z.eqb_neq in eq1.
    assert (x mod 2 = 1). {
      assert (0 < 2). { lia. }
      pose proof Zmod_pos_bound x 2 H.
      lia.
    }
    rewrite H.
    unfold "mod".
    simpl.
    intuition. 
Qed.

Lemma nat_mod_neg_not_zero: forall (x : nat_mod max), (max mod 2 = 1) -> (x =? nat_mod_neg x) = (x =? 0).
Proof.
  intros x H.
  destruct (x =? nat_mod_neg x) eqn:eq1. {
    rewrite Z.eqb_eq in eq1.
    unfold nat_mod_neg in eq1.
    unfold opp in eq1.
    simpl in eq1.
    destruct (x =? 0) eqn:eq2.
    - reflexivity.
    - rewrite Z.eqb_neq in eq2.
      rewrite inZnZ in eq2.
      pose proof Z_mod_nz_opp_full x max eq2.
      rewrite H0 in eq1.
      rewrite <- inZnZ in eq1.
      assert (x mod 2 = (max - x) mod 2). {
        rewrite <- eq1. reflexivity.
      }
      rewrite Zminus_mod in H1.
      rewrite H in H1.
      pose proof one_minus_mod2 x.
      intuition.
  } {
    rewrite Z.eqb_neq in eq1.
    simpl in eq1.
    destruct (x =? 0) eqn:eq2.
      - rewrite Z.eqb_eq in eq2.
        rewrite eq2 in eq1.
        unfold "mod" in eq1.
        simpl in eq1.
        intuition.
      - reflexivity.
  }
Qed.

Lemma mod_diff_helper: forall (a b : nat_mod max), (a - b <> 0) -> (a - b) mod max <> 0.
Proof.
  intros a b H1.
  pose proof nat_mod_small a as H2.
  pose proof nat_mod_small b as H3.
  assert (H4: Z.opp max < a - b < max). { lia. }
  destruct (a - b >? 0) eqn:eq1. {
    assert (H5: 0 <= a - b < max). { lia. }
    pose proof Zmod_small _ _ H5.
    intuition.
  } {
    assert (H5: 0 <= b - a < max). { lia. }
    assert (H6: a - b = Z.opp (b - a)). { lia. }
    pose proof Zmod_small _ _ H5 as H7.
    rewrite H6.
    assert (H8: (b - a) mod max <> 0). { lia. }
    pose proof Z_mod_nz_opp_full _ _ H8 as H9.
    rewrite H9.
    rewrite H7.
    lia.
  }
Qed.

Lemma nat_mod_neq_diff: forall (a b : nat_mod max), (a <> b) -> (a -% b <> nat_mod_zero).
Proof.
  intros a b H.
  unfold "-%".
  unfold sub.
  unfold nat_mod_zero.
  assert (a - b <> 0). {
    simpl.
    intuition.
    destruct a as (a', inZnZa).
    destruct b as (b', inZnZb).
    simpl in H0.
    assert (a' = b'). { lia. }
    pose proof (zirr _ _ _ inZnZa inZnZb H1).
    intuition.
  }
  pose proof mod_diff_helper a b H0.
  intuition.
  inversion H2.
  rewrite Zmod_0_l in H4.
  intuition.
Qed.

Lemma flip_division: forall (a b c d : nat_mod max), (c <> d) -> (a -% b) *% nat_mod_inv (c -% d) = (b -% a) *% nat_mod_inv (d -% c).
Proof.
  intros a b c d H.
  field.
  pose proof nat_mod_neq_diff c d H.
  apply not_eq_sym in H.
  pose proof nat_mod_neq_diff d c H.
  apply (conj H1) in H0.
  exact H0.
Qed.

Definition nat_mod_root (x : nat_mod max) (a : int128) := nat_mod_exp_def x (val max (nat_mod_inv (nat_mod_from_literal max a))).

End nat_mod.

Axiom elem_max_prime: prime elem_max.
Axiom scalar_max_prime: prime scalar_max.

Definition field_elem_FZpZ := (nat_mod_FZpZ elem_max elem_max_prime).
Definition scalar_FZpZ := (nat_mod_FZpZ scalar_max scalar_max_prime).

Add Field field_elem_FZpZ : field_elem_FZpZ.
Add Field scalar_FZpZ : scalar_FZpZ.

Lemma field_elem_small: forall (n : secp256k1_field_element_t), 0 <= n < elem_max.
Proof.
  intros n.
  apply (nat_mod_small elem_max elem_max_prime n).
Qed.

Lemma scalar_small: forall (n : secp256k1_scalar_t), 0 <= n < scalar_max.
Proof.
  intros n. 
  apply (nat_mod_small scalar_max scalar_max_prime n).
Qed.

Lemma curve_eq_symm: forall (p q : affine_t),
  p =.? q = q =.? p.
Proof.
  intros.
  simpl.
  unfold nat_mod_val.
  destruct p as (px, py).
  destruct q as (qx, qy).
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

Lemma curve_eq_reflect: forall (p : affine_t), p =.? p = true.
Proof.
  intros.
  simpl.
  destruct p as (px, py).
  rewrite -> Z.eqb_refl.
  rewrite -> Z.eqb_refl.
  reflexivity.
Qed.

Lemma curve_eq_red: forall (p q : affine_t), p =.? q = (fst p =? fst q) && (snd p =? snd q).
Proof.
  intros p q.
  simpl.
  unfold nat_mod_val.
  destruct p as (px, py).
  destruct q as (qx, qy).
  simpl.
  reflexivity.
Qed.

Lemma infty_is_infty: is_infinity infinity = true.
Proof.
  unfold is_infinity.
  simpl.
  reflexivity.
Qed.

Lemma is_infty_means_infty: forall (p: affine_t), is_infinity p = true -> p = infinity.
Proof.
  intros p H.
  unfold is_infinity in H.
  rewrite eqb_leibniz in H.
  exact H.
Qed.

Lemma pos_diff: forall a b, a < b -> b - a > 0.
Proof.
  intros a b H.
  auto with zarith.
Qed.

Lemma double_neg: forall (p : affine_t), p = neg_point (neg_point p).
Proof.
  intros p.
  unfold neg_point.
  destruct p as (px, py).
  pose proof nat_mod_double_neg _ elem_max_prime py.
  rewrite pair_equal_spec.
  split.
  - reflexivity.
  - intuition.
Qed.

Lemma neg_both: forall (p q : affine_t), p = q <-> neg_point p = neg_point q.
Proof.
  intros p q.
  split.
  - intros H.
    rewrite H.
    reflexivity.
  - intros H.
    destruct p as (px, py).
    destruct q as (qx, qy).
    unfold neg_point in H.
    rewrite pair_equal_spec.
    rewrite pair_equal_spec in H.
    split.
    + apply H.
    + destruct H as [_ H1].
      pose proof nat_mod_neg_inj _ elem_max_prime py qy.
      rewrite H1 in H.
      intuition.
Qed.

Lemma neg_symm: forall (p q : affine_t), neg_point p = q <-> p = neg_point q.
Proof.
  intros p q.
  destruct p as (px, py).
  destruct q as (qx, qy).
  unfold neg_point.
  rewrite pair_equal_spec.
  split.
  - intros H.
    rewrite pair_equal_spec.
    rewrite <- (nat_mod_double_neg _ elem_max_prime qy) in H.
    destruct H as [H1 H2].
    pose proof nat_mod_neg_inj _ elem_max_prime _ _ H2.
    Search (?a -> ?b -> ?a /\ ?b).
    apply (conj H1) in H.
    exact H.
  - intros H.
    rewrite pair_equal_spec in H.
    rewrite <- (nat_mod_double_neg _ elem_max_prime py) in H.
    destruct H as [H1 H2].
    pose proof nat_mod_neg_inj _ elem_max_prime _ _ H2.
    apply (conj H1) in H.
    exact H.
Qed.

Lemma neg_symm_bool: forall (p q : affine_t), neg_point p =.? q = p =.? neg_point(q).
Proof.
  intros p q.
  destruct (neg_point p =.? q) eqn:eq1.
  - rewrite eqb_leibniz in eq1.
    rewrite neg_symm in eq1.
    rewrite <- eqb_leibniz in eq1.
    rewrite eq1.
    reflexivity.
  - destruct p as (px, py).
    destruct q as (qx, qy).
    unfold neg_point.
    unfold neg_point in eq1.
    unfold "=.?", Dec_eq_prod.
    unfold "=.?", Dec_eq_prod in eq1.
    destruct (px =.? qx) eqn:eq2.
    + destruct py as (py', inZnZpy) eqn:Heqpy.
      destruct qy as (qy', inZnZqy) eqn:Heqqy.
      simpl.
      simpl in eq1.
      destruct (py' =? 0) eqn:eq3. {
        rewrite Z.eqb_eq in eq3.
        destruct (qy' =? 0) eqn:eq4. {
          rewrite Z.eqb_eq in eq4.
          rewrite eq4 in eq1.
          rewrite eq3 in eq1.
          simpl in eq1.
          discriminate eq1.
        } {
          rewrite Z.eqb_neq in eq4.
          rewrite inZnZqy in eq4.
          rewrite (Z_mod_nz_opp_full _ _ eq4).
          rewrite <- inZnZqy.
          pose proof field_elem_small qy.
          rewrite Heqqy in H.
          simpl in H.
          lia.
        }
      } {
        destruct (qy' =? 0) eqn:eq4. {
          rewrite Z.eqb_eq in eq4.
          rewrite eq4.
          unfold "mod".
          simpl.
          lia.
        } {
          rewrite Z.eqb_neq in eq3.
          rewrite inZnZpy in eq3.
          rewrite (Z_mod_nz_opp_full _ _ eq3) in eq1.
          rewrite <- inZnZpy in eq1.
          rewrite Z.eqb_neq in eq4.
          rewrite inZnZqy in eq4.
          rewrite (Z_mod_nz_opp_full _ _ eq4).
          rewrite <- inZnZqy.
          lia.
        }
      }
    + rewrite andb_false_l.
      reflexivity.
Qed.

Lemma same_x_cases: forall (p q : affine_t), (fst p = fst q) -> p = q \/ p = neg_point q.
Proof.
Admitted.

Lemma add_infty_1: forall (p: affine_t), infinity +' p = p.
Proof.
  intros p.
  unfold add_points.
  destruct (is_infinity infinity) eqn:eq.
  - reflexivity.
  - discriminate eq.
Qed.

Lemma double_point_not_infty: forall (p : affine_t), snd p =? 0 = false -> is_infinity (double_point p) = false.
Proof.
  destruct p as (px, py).
  simpl.
  intros H.
  unfold double_point.
Admitted.

Lemma add_different_comm: forall (p q : affine_t), (fst p <> fst q) -> add_different_points p q = add_different_points q p.
Proof.
  intros p q H.
  unfold add_different_points.
  destruct p as (px, py).
  destruct q as (qx, qy).
  unfold fst in H.
  unfold secp256k1_field_element_t in px, py, qx, qy.
  apply pair_equal_spec.
  assert (H1: forall (a b c : nat_mod elem_max), a -% b -% c = a -% c -% b). { intros a b c. ring. }
  split.
  - apply not_eq_sym in H.
    rewrite (flip_division _ elem_max_prime _ _ _ _ H).
    apply H1.
  - apply not_eq_sym in H.
    rewrite (flip_division _ elem_max_prime _ _ _ _ H).
    rewrite H1.
    remember (((((py -% qy) *% nat_mod_inv (px -% qx)) *% ((py -% qy) *% nat_mod_inv (px -% qx))) -% qx) -% px) as x3.
    field_simplify.
    + reflexivity.
    + apply not_eq_sym in H.
      apply (nat_mod_neq_diff _ elem_max_prime _ _ H).
    + apply not_eq_sym in H.
      apply (nat_mod_neq_diff _ elem_max_prime _ _ H).
Qed.

Lemma add_comm: forall (p q : affine_t), add_points p q = add_points q p.
Proof.
  intros p q.
  unfold add_points.
  unfold nat_mod_val.
  destruct (is_infinity p) eqn:eq1. {
    destruct (is_infinity q) eqn:eq2.
    - apply is_infty_means_infty in eq1.
      apply is_infty_means_infty in eq2.
      rewrite -> eq1.
      rewrite -> eq2.
      reflexivity.
    - reflexivity.
  } {
    destruct (is_infinity q) eqn:eq2. {
      reflexivity.
    } {
      destruct p as (px, py) eqn:P1.
      destruct q as (qx, qy) eqn:Q1.
      rewrite -> curve_eq_symm.
      destruct ((qx, qy) =.? (px, py)) eqn:H1. {
        assert (H2: (px, py) = (qx, qy)). {
          rewrite -> eqb_leibniz in H1.
          rewrite -> H1.
          reflexivity.
        }
        rewrite H2.
        reflexivity.
      } {
        destruct ((px, py) =.? neg_point (qx, qy)) eqn:H2. {
          rewrite eqb_leibniz in H2.
          rewrite <- neg_symm in H2.
          rewrite H2.
          simpl.
          rewrite Z.eqb_refl.
          rewrite Z.eqb_refl.
          simpl.
          reflexivity.
        } {
          destruct ((qx, qy) =.? neg_point (px, py)) eqn:eq3. {
            rewrite eqb_leibniz in eq3.
            symmetry in eq3.
            rewrite neg_symm in eq3.
            rewrite eq3 in H2.
            pose proof (curve_eq_reflect (neg_point (qx, qy))) as H.
            rewrite H in H2.
            discriminate H2.
          } {
            apply add_different_comm.
            simpl.
            assert (px <> qx). {
              intuition.
              pose proof same_x_cases p q.
              rewrite P1, Q1 in H0.
              unfold fst in H0.
              apply H0 in H.
              destruct H.
              - symmetry in H.
                rewrite <- eqb_leibniz in H.
                rewrite H in H1.
                discriminate H1.
              - rewrite <- eqb_leibniz in H.
                rewrite H in H2.
                discriminate H2.
            }
            exact H.
          }
        }
      }
    }
  }
Qed.

Lemma add_infty_2: forall (p: affine_t), p +' infinity = p.
Proof.
  intros p.
  apply add_comm.
Qed.

Lemma add_assoc: forall (p q r : affine_t), (p +' q) +' r = p +' (q +' r).
Proof.
  intros p q r.
  destruct (is_infinity p) eqn:eq1. {
    apply (is_infty_means_infty p) in eq1.
    rewrite eq1.
    apply add_infty_1.
  } {
    destruct (is_infinity q) eqn:eq2. {
      apply (is_infty_means_infty q) in eq2.
      rewrite eq2.
      rewrite (add_infty_2 p).
      rewrite (add_infty_1 r).
      reflexivity.
    } {
      destruct (is_infinity r) eqn:eq3. {
        apply (is_infty_means_infty r) in eq3.
        rewrite eq3.
        rewrite add_infty_2, add_infty_2.
        reflexivity.
      } {
        destruct (p =.? q) eqn:eq4. {
          unfold "+'".
          rewrite eq1, eq2, eq3, eq4.
          destruct (q =.? r) eqn:eq5. {
            rewrite eqb_leibniz in eq4.
            rewrite eqb_leibniz in eq5.
            rewrite eq5 in eq4.
            rewrite eq4, eq5.
            rewrite curve_eq_symm.
            destruct (r =.? double_point r) eqn:eq6. {
              rewrite eqb_leibniz in eq6.
              rewrite <- eq6.
              rewrite <- eq6.
              reflexivity.
            } {
              destruct (double_point r =.? neg_point r) eqn:eq7.  {
                rewrite eqb_leibniz in eq7.
                rewrite <- neg_symm in eq7.
                symmetry in eq7.
                rewrite <- eqb_leibniz in eq7.
                rewrite eq7.
                reflexivity.
              } {
                rewrite <- neg_symm_bool.
                rewrite curve_eq_symm, eq7.
                rewrite add_different_comm. (*
                reflexivity.
              }
            }
          } {
            rewrite eqb_leibniz in eq4.
            rewrite eq4.
            destruct (q =.? neg_point r) eqn:eq6. {
              rewrite infty_is_infty.
              assert (H1: is_infinity (double_point q) = false). {
                destruct q as (qx, qy).
                assert (H2: qy =? 0 = false). {
                  rewrite eqb_leibniz in eq6.
                  rewrite eq6 in eq5.
                  simpl in eq5.
                  destruct (neg_point r) as (nrx, nry) eqn:eq7.
                  destruct r as (rx, ry).
                  unfold neg_point in eq7.
                  inversion eq7.
                  unfold nat_mod_val in eq5.
                  symmetry in H0.
                  destruct nrx as (nrxval, inZnZrnx) eqn:Heqnrx.
                  destruct rx as (rxval, inZnZrx) eqn:Heqrx.
                  inversion H0.
                  simpl in eq5.
                  rewrite <- Z.eqb_eq in H2.
                  rewrite H2 in eq5.
                  rewrite andb_true_l in eq5.
                  rewrite <- H1 in eq5.
                  rewrite Z.eqb_sym in eq5.
                  rewrite nat_mod_neg_not_zero in eq5.
                  inversion eq7.
                  rewrite <- H4 in eq6.
                  inversion eq6.
                  rewrite Z.eqb_neq in eq5.
                  rewrite inZnZ in eq5.
                  unfold nat_mod_neg.
                  unfold opp.
                  simpl.
                  rewrite (Z_mod_nz_opp_full _ _ eq5).
                  rewrite <- inZnZ.
                  pose proof field_elem_small ry.
                  lia.
                }
                rewrite double_point_not_infty.
                - reflexivity.
                - unfold snd.
                  exact H2.
              }
              rewrite H1. 
              
            }
          }
          apply is_infty_means_infty in eq4.
          rewrite eq3, add_infty_1.
        }
      }
      unfold "+'".
      rewrite -> eq1.
      rewrite -> eq2.
      simpl.
    }
  }*)
Admitted.

Fixpoint simple_scalar_mult (k : nat) (p : affine_t) : affine_t :=
  match k with
  | 0%nat => infinity
  | S k1  => (simple_scalar_mult (k1) p) +' p
  end.

Definition nat_to_scalar (n : nat) : secp256k1_scalar_t.
Proof.
  unfold secp256k1_scalar_t.
  unfold nat_mod.
  remember (Z.of_nat n mod scalar_max) as x.
  apply (mkznz scalar_max x).
  rewrite Heqx.
  rewrite Zmod_mod.
  reflexivity.
Defined.

Lemma nat_to_scalar_id: forall (n : secp256k1_scalar_t), nat_to_scalar (Z.to_nat n) = n.
Proof.
  intros n.
  destruct n as (n', inZnZ') eqn:eq1.
  assert (H0: 0 <= n'). {
    pose proof scalar_small n.
    intuition.
    unfold "<=".
    unfold "<=" in H0.
    unfold "?=".
    unfold "?=" in H0.
    unfold val in H0.
    rewrite eq1 in H0.
    exact H0.
  }
  unfold val.
  unfold nat_to_scalar.
  rewrite (Z2Nat.id _ H0).
  apply GZnZ.zirr.
  intuition.
Qed.

Lemma nat_to_scalar_add: forall (a b : nat), nat_to_scalar a +% nat_to_scalar b = nat_to_scalar (a + b).
Proof.
  intros a b.
  unfold "+%".
  unfold add.
  unfold nat_to_scalar.
  simpl.
  assert ((Z.of_nat a mod scalar_max + Z.of_nat b mod scalar_max) mod scalar_max = Z.of_nat (a + b) mod scalar_max). {
    rewrite <- Zplus_mod.
    intuition.
  }
  apply zirr.
  exact H. 
Qed.

Lemma nat_to_scalar_mult: forall (a b : nat), nat_to_scalar a *% nat_to_scalar b = nat_to_scalar (a * b).
Proof.
  intros a b.
  unfold "*%".
  unfold mul.
  unfold nat_to_scalar.
  simpl.
  assert (((Z.of_nat a mod scalar_max) * (Z.of_nat b mod scalar_max)) mod scalar_max = Z.of_nat (a * b) mod scalar_max). {
    rewrite <- Zmult_mod.
    intuition.
  }
  apply zirr.
  exact H. 
Qed.

Lemma nat_to_scalar_lem: forall (n : nat), exists (n' : secp256k1_scalar_t), n' =? (Z.of_nat n) mod scalar_max = true.
Proof.
  intros n.
  remember (Z.of_nat n mod scalar_max) as x.
  pose proof (Z.mod_pos_bound (Z.of_nat n) scalar_max zero_less_than_scalar_max) as H2.
  rewrite <- Heqx in H2.
  pose proof (small_to_nat_mod scalar_max x H2) as H3.
  destruct H3 as [b G].
  rewrite -> Z.eqb_sym in G.
  exists b.
  exact G.
Qed.

Lemma scalar_mult_def: forall (k : nat) (p : affine_t), (nat_to_scalar k) *' p = simple_scalar_mult (k) p.
Proof.
  intros k p.
  induction (k). {
    unfold simple_scalar_mult.
    unfold "*'".
    unfold nat_mod_get_bit.
    unfold nat_mod_bit.
    assert (H: forall n, BinInt.Z.testbit (nat_to_scalar 0) n = false). {
      unfold nat_to_scalar.
      simpl.
      rewrite -> Zmod_0_l.
      exact Z.testbit_0_l.
    }
    auto. (*
  } {
    simpl.
  }*)
Admitted.

Lemma simple_scalar_mult_distributivity: forall (k1 k2 : nat) (p: affine_t), (simple_scalar_mult k1 p) +' (simple_scalar_mult k2 p) = (simple_scalar_mult (k1 + k2) p).
Proof.
  intros k1 k2 p.
  induction k2 as [|k' IHk']. {
    simpl.
    pose proof add_infty_2 (simple_scalar_mult k1 p) as H1.
    rewrite H1, Nat.add_0_r.
    reflexivity.
  } {
    rewrite <- plus_Snm_nSm.
    simpl.
    rewrite <- add_assoc.
    rewrite IHk'.
    reflexivity.
  }
Qed.

(* Follows from Langrange's Theorem  since the order of the group is prime *)
Lemma simple_scalar_mult_mod: forall (k : nat) (p: affine_t), simple_scalar_mult k p = simple_scalar_mult (k mod (Z.to_nat scalar_max)) p.
Proof.
Admitted.

Lemma scalar_mult_distributivity: forall (k1 k2 : secp256k1_scalar_t) (p: affine_t), k1 *' p +' k2 *' p = (k1 +% k2) *' p.
Proof.
  intros k1 k2 p.
  rewrite <- (nat_to_scalar_id k1).
  rewrite <- (nat_to_scalar_id k2).
  rewrite scalar_mult_def.
  rewrite scalar_mult_def.
  rewrite nat_to_scalar_add.
  rewrite scalar_mult_def.
  rewrite simple_scalar_mult_distributivity.
  reflexivity.
Qed.

Lemma scalar_mult_fold_once: forall (a b : nat) (p : affine_t), (simple_scalar_mult (a * b) p) +' (simple_scalar_mult b p) = simple_scalar_mult (S(a) * b) p.
Proof.
  intros a b p.
  simpl.
  rewrite <- simple_scalar_mult_distributivity.
  rewrite add_comm.
  reflexivity.
Qed.

Lemma scalar_mult_assoc2: forall (a b : secp256k1_scalar_t) (p : affine_t), a *' b *' p = (a *% b) *' p.
Proof.
  intros a b p.
  rewrite <- (nat_to_scalar_id a).
  rewrite <- (nat_to_scalar_id b).
  rewrite scalar_mult_def.
  rewrite scalar_mult_def.
  rewrite nat_to_scalar_mult.
  rewrite scalar_mult_def.
  induction (Z.to_nat a).
  - unfold simple_scalar_mult.
    simpl.
    reflexivity.
  - unfold simple_scalar_mult.
    fold simple_scalar_mult.
    rewrite IHn.
    fold (simple_scalar_mult (S n * Z.to_nat b) p).
    rewrite scalar_mult_fold_once.
    reflexivity.
Qed.

Structure on_curve_t: Set:=
 mkoncurve {point: affine_t;
        on_curve: is_point_on_curve point = true }.

Lemma infty_on_curve: is_point_on_curve infinity = true.
Proof.
  intuition.
Qed.

Lemma generator_on_curve: is_point_on_curve generator = true.
Proof.
  unfold is_point_on_curve.
  assert (is_infinity generator = false). {
    assert (fst generator <> fst infinity). {
      unfold fst, infinity, generator.
      auto. (*
    }
    unfold generator, is_infinity, infinity.
  }
  destruct generator as (x, y) eqn:eq1.*)
Admitted.

Lemma on_curve_x_from_y: forall (p : on_curve_t),
  is_infinity (point p) = true \/ (fst (point p)) =  nat_mod_root elem_max ((nat_mod_exp (snd (point p)) (@repr WORDSIZE128 2)) -%  nat_mod_from_literal elem_max (@repr WORDSIZE128 7)) (@repr WORDSIZE128 3).
Proof.
  intros p.
  destruct (is_infinity (point p)) eqn:eq1.
  - intuition.
  - pose proof on_curve p.
    unfold is_point_on_curve in H.
    rewrite eq1 in H.
    simpl in H.
    destruct (point p) as (px, py).
    simpl.
    assert (px = nat_mod_root elem_max (nat_mod_exp_def py 2 -% nat_mod_from_literal elem_max (@repr WORDSIZE128 7)) (@repr WORDSIZE128 3)).
    + unfold secp256k1_field_element_t in px, py.
      unfold nat_mod_root.
      field_simplify.
Admitted.

Lemma add_different_closure: forall (p q : on_curve_t), is_infinity (point p) = false -> is_infinity (point q) = false -> (fst (point p) <> fst (point q)) -> exists r, point r = add_different_points (point p) (point q).
Proof.
  intros p q H1 H2 H3.
  remember (add_different_points (point p) (point q)) as r'.
  destruct (is_point_on_curve r') eqn:eq1.
  - remember (mkoncurve r' eq1) as r.
    exists r.
    rewrite Heqr.
    simpl.
    reflexivity.
  - remember (is_point_on_curve r') as res.
    unfold is_point_on_curve in Heqres.
    destruct r' as (rx, ry) eqn:eqr.
    destruct (is_infinity (rx, ry)) eqn:eq2.
    + simpl in Heqres.
      rewrite Heqres in eq1.
      discriminate.
    + simpl in Heqres.
Admitted.

Lemma double_point_closure: forall (p : on_curve_t), exists (r : on_curve_t), point r = double_point (point p).
Proof.
Admitted.

Lemma add_points_closure: forall (p q : on_curve_t), exists (r : on_curve_t), point r = (point p) +' (point q).
Proof.
  intros p q.
  unfold "+'".
  destruct (is_infinity (point p)) eqn:eq1. {
    exists q.
    intuition.
  } {
    destruct (is_infinity (point q)) eqn:eq2. {
      exists p.
      intuition.
    } {
      destruct (point p =.? point q) eqn:eq3. {
        exact (double_point_closure p).
      } {
        destruct (point p =.? neg_point (point q)) eqn:eq4. {
          remember (mkoncurve infinity (infty_on_curve)) as i.
          exists i.
          rewrite Heqi.
          intuition.
        } {
          assert (eq5: fst (point p) <> fst (point q)). {
            pose proof same_x_cases (point p) (point q).
            intuition.
            - rewrite <- eqb_leibniz in H.
              rewrite H in eq3.
              discriminate.
            - rewrite <- eqb_leibniz in H.
              rewrite H in eq4.
              discriminate.
          }
          exact (add_different_closure p q eq1 eq2 eq5).
        }
      }
    }
  }
Qed.

Lemma scalar_mult_closure: forall (p : on_curve_t) (k : secp256k1_scalar_t), exists (q : on_curve_t), is_point_on_curve point q = k *' (point p).
Proof.
  intros p k.
  rewrite <- (nat_to_scalar_id k).
  rewrite scalar_mult_def.
  induction (Z.to_nat k).
  - unfold simple_scalar_mult.
    remember (mkoncurve infinity infty_on_curve) as i.
    exists i.
    rewrite Heqi.
    intuition.
  - unfold simple_scalar_mult.
    fold simple_scalar_mult.
    destruct IHn.
    rewrite <- H.
    exact (add_points_closure x p).
Qed.

Lemma scalar_mult_generator_not_zero: forall (a : secp256k1_scalar_t), a <> nat_mod_zero -> is_infinity (a *' generator) = false.
Proof.
Admitted.