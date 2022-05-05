(** This file was automatically generated using Hacspec **)
Require Import Hacspec_Lib MachineIntegers.
From Coq Require Import ZArith Nat List.
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

(* Set Printing Coercions. *)

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
        if nat_mod_bit (k_30) (((scalar_bits_v) - (usize 1)) - (
            i_33)):bool then (let q_32 :=
            add_points (p_31) (q_32) in 
          (q_32)) else ((q_32)) in 
      (q_32))
    q_32 in 
  q_32.
  
Definition batch_scalar_optimization
  (elems_34 : seq (secp256k1_scalar_t × affine_t))
  : seq (secp256k1_scalar_t × affine_t) :=
  let new_elems_35 : seq (secp256k1_scalar_t × affine_t) :=
    elems_34 in 
  let '(new_elems_35) :=
    if (seq_len (new_elems_35)) =.? (usize 0):bool then (let new_elems_35 :=
        new_elems_35 in 
      (new_elems_35)) else (let new_elems_35 :=
        foldi (usize 0) ((seq_len (new_elems_35)) - (
              usize 1)) (fun i_36 new_elems_35 =>
          let '(ai_37, pi_38) :=
            seq_index (new_elems_35) (i_36) in 
          let '(aiplus1_39, piplus1_40) :=
            seq_index (new_elems_35) ((i_36) + (usize 1)) in 
          let new_elems_35 :=
            seq_upd new_elems_35 (i_36) (((ai_37) -% (aiplus1_39), pi_38)) in 
          let new_elems_35 :=
            seq_upd new_elems_35 ((i_36) + (usize 1)) ((
                aiplus1_39,
                add_points (pi_38) (piplus1_40)
              )) in 
          (new_elems_35))
        new_elems_35 in 
      (new_elems_35)) in 
  new_elems_35.

Definition product_sum
  (elems_41 : seq (secp256k1_scalar_t × affine_t))
  : affine_t :=
  let res_42 : (secp256k1_field_element_t × secp256k1_field_element_t) :=
    infinity  in 
  let res_42 :=
    foldi (usize 0) (seq_len (elems_41)) (fun i_43 res_42 =>
      let '(ai_44, pi_45) :=
        seq_index (elems_41) (i_43) in 
      let res_42 :=
        add_points (res_42) (scalar_multiplication (ai_44) (pi_45)) in 
      (res_42))
    res_42 in 
  res_42.

Definition batch_scalar_multiplication
  (elems_46 : seq (secp256k1_scalar_t × affine_t))
  : affine_t :=
  let optimized_47 : seq (secp256k1_scalar_t × affine_t) :=
    batch_scalar_optimization (elems_46) in 
  product_sum (optimized_47).

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

Lemma Zpos_helper0: forall (n : nat_mod max) (p : positive), val _ n = Z.pos p~0 -> Z.pos p = Z.pos p mod max.
Proof.
  intros n p H.
  pose proof nat_mod_small n.
  rewrite H in H0.
  assert (0 <= Z.pos p < max). { lia. }
  pose proof small_to_nat_mod (Z.pos p) H1.
  destruct H2 as (x, same).
  destruct x as (xval, inZnZ).
  unfold val in same.
  rewrite Z.eqb_eq in same.
  rewrite <- same in inZnZ.
  exact inZnZ.
Qed.

Lemma Zpos_helper1: forall (n : nat_mod max) (p : positive), val _ n = Z.pos p~1 -> Z.pos p = Z.pos p mod max.
Proof.
  intros n p H.
  pose proof nat_mod_small n.
  rewrite H in H0.
  assert (0 <= Z.pos p < max). { lia. }
  pose proof small_to_nat_mod (Z.pos p) H1.
  destruct H2 as (x, same).
  destruct x as (xval, inZnZ).
  unfold val in same.
  rewrite Z.eqb_eq in same.
  rewrite <- same in inZnZ.
  exact inZnZ.
Qed.

Lemma znz_duplicate: forall (a : nat_mod max) (b : Z) (H : val _ a = b), exists H2, a = mkznz _ b H2.
Proof.
  intros a b H.
  destruct a as (a', inZnZa).
  simpl in H.
  assert (inZnZb := inZnZa).
  rewrite H in inZnZb.
  exists inZnZb.
  apply (zirr _ _ _ inZnZa inZnZb H).
Qed.

End nat_mod.

Section foldi.

Lemma foldi_empty: forall (A : Type) (a b : uint_size) (f : uint_size -> A -> A) (acc : A), (unsigned a > unsigned b) -> foldi a b f acc = acc.
Proof.
  intros A a b f acc H.
  unfold foldi.
  destruct (unsigned b - unsigned a) eqn:eq1.
  - reflexivity.
  - remember (unsigned b - unsigned a) as res.
    assert (exists v, unsigned b - unsigned a = Z.neg v). { lia. }
    destruct H0.
    rewrite H0 in Heqres.
    rewrite eq1 in Heqres.
    discriminate Heqres.
  - reflexivity.
Qed.

Lemma foldi_equiv: forall (A : Type) (a b : uint_size) (f g : uint_size -> A -> A) (acc : A),
  (forall (i0 : uint_size) acc0, (unsigned a <= unsigned i0 < unsigned b) -> f i0 acc0 = g i0 acc0) ->
  foldi a b f acc = foldi a b g acc.
Proof.
  intros.
  revert H.
  unfold foldi.
  destruct a as (a', a_small).
  destruct b as (b', b_small).
  simpl.
  destruct (b' - a') eqn:eq1.
  - reflexivity.
  - unfold foldi_.
    remember (mkint a' a_small) as a.
    assert (Z.of_nat (Pos.to_nat p) + unsigned a = b'). {
      rewrite Heqa.
      simpl.
      lia.
    }
    assert (a' <= unsigned a). {
      rewrite Heqa.
      simpl.
      lia.
    }
    fold (foldi_ (Pos.to_nat p) a f acc).
    fold (foldi_ (Pos.to_nat p) a g acc).
    clear Heqa.
    revert H. revert H0. revert g. revert f. revert acc. revert a.
    induction (Pos.to_nat p).
    + reflexivity.
    + intros.
      assert (a' <= unsigned a < b'). {
        destruct a.
        simpl.
        unfold unsigned in H.
        unfold MachineIntegers.intval in H.
        unfold unsigned in H0.
        unfold MachineIntegers.intval in H0.
        lia.
      }
      unfold foldi_.
      fold (foldi_ n (a .+ MachineIntegers.one) f (f a acc)).
      fold (foldi_ n (a .+ MachineIntegers.one) g (g a acc)).
      rewrite (H1 _ acc H2).
      remember (fun i => f (i .+ MachineIntegers.one)) as f'.
      remember (fun i => g (i .+ MachineIntegers.one)) as g'.
      assert (forall i0 acc0, a' <= unsigned i0 < b' -1 -> f' i0 acc0 = g' i0 acc0). {
        intros.
        assert (a' <= unsigned (i0 .+ MachineIntegers.one) < b'). {
          unfold ".+".
          assert (@unsigned WORDSIZE32 MachineIntegers.one = 1). { easy. }
          rewrite H4.
          simpl.
          assert (0 <= unsigned i0 + 1 < @modulus WORDSIZE32). { lia. }
          rewrite Z_mod_modulus_eq.
          rewrite (Zmod_small _ _ H5).
          intuition.
        }
        pose proof (H1 (i0 .+ MachineIntegers.one) acc0 H4).
        rewrite Heqf'.
        rewrite Heqg'.
        exact H5.
      }
      assert (unsigned (a .+ MachineIntegers.one) = unsigned a + 1). {
        unfold ".+".
        assert (@unsigned WORDSIZE32 MachineIntegers.one = 1). { easy. }
        rewrite H4.
        assert (@max_unsigned WORDSIZE32 = @modulus WORDSIZE32 - 1). { easy. }
        assert (0 <= unsigned a + 1 <= @max_unsigned WORDSIZE32). { lia. }
        rewrite (unsigned_repr _ H6).
        reflexivity.
      }
      assert (a' <= unsigned (a .+ MachineIntegers.one)). {
        lia.
      }
      assert (Z.of_nat n + unsigned (a .+ MachineIntegers.one) = b'). { lia. }
      exact (IHn (a .+ MachineIntegers.one) (g a acc) f g H5 H6 H1).
  - reflexivity.
Qed.

Lemma foldi_offset: forall (A : Type) (n : nat) i j (f g : uint_size -> A -> A) acc,
  0 <= Z.of_nat n + unsigned i <= @max_unsigned WORDSIZE32 ->
  0 <= Z.of_nat n + unsigned j <= @max_unsigned WORDSIZE32 ->
  (forall i0 acc0, 0 <= i0 <= n -> f ((repr i0) .+ i) acc0 = g ((repr i0) .+ j) acc0) ->
  foldi_ n i f acc = foldi_ n j g acc.
Proof.
  intros.
  (*
  remember (repr 0) as a.
  assert (n + uint_size_to_nat a = n)%nat. {
    rewrite Heqa.
    simpl.
    Set Printing All.
    unfold uint_size_to_nat.
    unfold from_uint_size.
    unfold nat_uint_sizable.
    assert (0 <= 0 <= @max_unsigned WORDSIZE32). { easy. }
    rewrite (unsigned_repr _ H0).
    intuition.
  }
  revert H0.*) revert H1. revert H0. revert H.
  (*clear Heqa.*)
  revert acc. revert g. revert f. (*revert a.*) revert j. revert i.
  induction n.
  - reflexivity.
  - intros.
    unfold foldi_.
    fold (foldi_ n (i .+ MachineIntegers.one) f (f i acc)).
    fold (foldi_ n (j .+ MachineIntegers.one) g (g j acc)).
    pose proof unsigned_range_2 i.
    pose proof unsigned_range_2 j.
    assert (@unsigned WORDSIZE32 MachineIntegers.one = 1). { easy. }
    assert (0 <= Z.of_nat n + unsigned (i .+ MachineIntegers.one) <= @max_unsigned WORDSIZE32). {
      unfold ".+".
      rewrite H4.
      assert (unsigned i < @max_unsigned WORDSIZE32). { lia. }
      assert (0 <= (unsigned i) + 1 <= @max_unsigned WORDSIZE32). { lia. }
      rewrite (unsigned_repr _ H6).
      lia.
    }
    assert (0 <= Z.of_nat n + unsigned (j .+ MachineIntegers.one) <= @max_unsigned WORDSIZE32). {
      unfold ".+".
      rewrite H4.
      assert (unsigned j < @max_unsigned WORDSIZE32). { lia. }
      assert (0 <= (unsigned j) + 1 <= @max_unsigned WORDSIZE32). { lia. }
      rewrite (unsigned_repr _ H7).
      lia.
    }
    assert (forall i0 acc0, 0 <= i0 <= n ->
        f (repr i0 .+ (i .+ MachineIntegers.one)) acc0 = g (repr i0 .+ (j .+ MachineIntegers.one)) acc0). {
      intros.
      remember (i0 + 1) as i0'.
      assert (0 <= i0' <= S n). { lia. }
      pose proof (H1 i0' acc0 H8).
      rewrite Heqi0' in H9.
      unfold ".+" in H9.
      assert (0 <= i0 + 1 <= @max_unsigned WORDSIZE32). { lia. }
      rewrite (unsigned_repr _ H10) in H9.
      unfold ".+".
      assert (0 <= i0 <= @max_unsigned WORDSIZE32). { lia. }
      rewrite (unsigned_repr _ H11).
      rewrite H4.
      pose proof @repr_unsigned WORDSIZE32.
      assert (0 <= unsigned i + 1 <= @max_unsigned WORDSIZE32). { lia. }
      rewrite (unsigned_repr _ H13).
      assert (0 <= unsigned j + 1 <= @max_unsigned WORDSIZE32). { lia. }
      rewrite (unsigned_repr _ H14).
      assert (i0 + 1 + unsigned i = i0 + (unsigned i + 1)). { lia. }
      assert (i0 + 1 + unsigned j = i0 + (unsigned j + 1)). { lia. }
      rewrite H15, H16 in H9.
      exact H9.
    }
    assert (0 <= 0 <= S n). { lia. }
    pose proof (H1 0 acc H8).
    unfold ".+" in H9.
    assert (0 <= 0 <= @max_unsigned WORDSIZE32). { easy. }
    rewrite (unsigned_repr _ H10) in H9.
    simpl in H9.
    rewrite repr_unsigned, repr_unsigned in H9.
    rewrite H9.
    exact (IHn (i .+ MachineIntegers.one) (j .+ MachineIntegers.one) f g (g j acc) H5 H6 H7).
Qed.

End foldi.

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

Structure on_curve_t: Set:=
 mkoncurve {point: affine_t;
        on_curve: is_point_on_curve point = true }.

Coercion point : on_curve_t >-> prod.

Lemma infty_on_curve: is_point_on_curve infinity = true.
Proof.
  intuition.
Qed.

(* How do we get the actual values? *)
Lemma generator_on_curve: is_point_on_curve generator = true.
Proof.
  unfold is_point_on_curve.
  destruct generator as (x, y).
  destruct (is_infinity (x, y)).
  - intuition.
  - simpl.
Admitted.

Lemma same_x_cases: forall (p q : on_curve_t), (fst p = fst q) -> point p = point q \/ point p = neg_point (q).
Proof.
  intros p q H.

Admitted.

Lemma add_infty_1: forall (p: affine_t), infinity +' p = p.
Proof.
  intros p.
  unfold add_points.
  destruct (is_infinity infinity) eqn:eq.
  - reflexivity.
  - discriminate eq.
Qed.

Lemma double_point_preserves_infinity: forall (p : affine_t), is_infinity p = is_infinity (double_point p).
Proof.
  intros p.
  destruct (is_infinity p) eqn:eq1.
  - unfold double_point.
    apply is_infty_means_infty in eq1.
    rewrite eq1.
    simpl.
    rewrite infty_is_infty.
    reflexivity.
  - unfold double_point.
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

Lemma add_comm: forall (p q : on_curve_t), add_points p q = add_points q p.
Proof.
  intros p q.
  Set Printing All.
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
      destruct (point p) as (px, py) eqn:P1.
      destruct (point q) as (qx, qy) eqn:Q1.
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
  unfold "+'".
  simpl.
  destruct (is_infinity p) eqn:eq1.
  - rewrite (is_infty_means_infty _ eq1).
    reflexivity.
  - reflexivity.
Qed.

Lemma double_infty: double_point infinity = infinity.
Proof.
  unfold double_point.
  simpl.
  reflexivity.
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

Lemma add_to_double: forall (p : affine_t), p +' p = double_point p.
Proof.
  intros p.
  unfold add_points.
  destruct (is_infinity p) eqn:eq1.
  - unfold double_point.
    apply is_infty_means_infty in eq1.
    rewrite eq1.
    simpl.
    reflexivity.
  - rewrite curve_eq_reflect.
    reflexivity.
Qed.

Fixpoint simple_scalar_mult (k : nat) (p : affine_t) : affine_t :=
  match k with
  | 0%nat => infinity
  | S k1  => (simple_scalar_mult (k1) p) +' p
  end.

Lemma double_point_closed: forall (p : on_curve_t), exists (r : on_curve_t), point r = double_point (point p).
Proof.
Admitted.

Lemma add_different_closed: forall (p q : on_curve_t), is_infinity (point p) = false -> is_infinity (point q) = false -> (fst (point p) <> fst (point q)) -> exists r, point r = add_different_points (point p) (point q).
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

Lemma add_points_closed: forall (p q : on_curve_t), exists (r : on_curve_t), point r = (point p) +' (point q).
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
        exact (double_point_closed p).
      } {
        destruct (point p =.? neg_point (point q)) eqn:eq4. {
          remember (mkoncurve infinity (infty_on_curve)) as i.
          exists i.
          rewrite Heqi.
          intuition.
        } {
          assert (eq5: fst (point p) <> fst (point q)). {
            pose proof same_x_cases p q.
            intuition.
            - rewrite <- eqb_leibniz in H.
              rewrite H in eq3.
              discriminate.
            - rewrite <- eqb_leibniz in H.
              rewrite H in eq4.
              discriminate.
          }
          exact (add_different_closed p q eq1 eq2 eq5).
        }
      }
    }
  }
Defined.

Lemma simple_scalar_mult_closed: forall (p : on_curve_t) (k : nat), exists (q : on_curve_t), point q = simple_scalar_mult k p.
Proof.
  intros p k.
  induction (k).
  - unfold simple_scalar_mult.
    remember (mkoncurve infinity infty_on_curve) as i.
    exists i.
    rewrite Heqi.
    intuition.
  - unfold simple_scalar_mult.
    fold simple_scalar_mult.
    destruct IHn.
    rewrite <- H.
    exact (add_points_closed x p).
Qed.

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

Lemma simple_scalar_mult_distributivity2: forall (k : nat) (p1 p2 : on_curve_t), simple_scalar_mult k (p1 +' p2) = (simple_scalar_mult k p1) +' (simple_scalar_mult k p2).
Proof.
  intros.
  induction k.
  - simpl. apply add_infty_1.
  - unfold simple_scalar_mult.
    fold simple_scalar_mult.
    rewrite IHk.
    rewrite (add_assoc (simple_scalar_mult k p1) p1).
    destruct (simple_scalar_mult_closed p2 k).
    destruct (add_points_closed x p2).
    rewrite <- H.
    rewrite <- H0.
    rewrite (add_comm p1 x0).
    rewrite H0.
    rewrite (add_assoc x p2 p1).
    rewrite (add_comm p2 p1).
    rewrite add_assoc.
    reflexivity.
Qed.

Lemma simple_scalar_mult_fold: forall (k : nat) (p : affine_t), (simple_scalar_mult k p) +' p = simple_scalar_mult (S k) p.
Proof.
  intros k p.
  unfold simple_scalar_mult.
  fold simple_scalar_mult.
  reflexivity.
Qed.

Fixpoint simple_scalar_mult2 (k : positive) (p : affine_t) : affine_t :=
  match k with
  | xH => p
  | xO r => double_point (simple_scalar_mult2 r p)
  | xI r => (double_point (simple_scalar_mult2 r p)) +' p
  end.

Definition bitlist := list bool.

Fixpoint simple_scalar_mult3 (l : bitlist) (p acc : affine_t) : affine_t :=
  match l with
  | nil => acc
  | false :: l' => simple_scalar_mult3 l' p (double_point acc)
  | true :: l' => simple_scalar_mult3 l' p ((double_point acc) +' p)
  end.

Fixpoint pos_to_bitlist (v : positive) : bitlist :=
  match v with
    | xH => [true]
    | xO r => pos_to_bitlist r ++ [false]
    | xI r => pos_to_bitlist r ++ [true]
  end.

Definition z_to_bitlist (v : Z) : bitlist :=
  match v with
  | Zpos v' => pos_to_bitlist v'
  | _ => nil
  end.

Fixpoint norm_bitlist (l : bitlist) : bitlist :=
  match l with
  | false :: l' => norm_bitlist l'
  | _ => l
  end.

Fixpoint bitlist_to_pos_helper (l : bitlist) (acc : positive) : positive :=
  match l with
  | nil => acc
  | false :: l' => bitlist_to_pos_helper l' (xO acc)
  | true :: l' => bitlist_to_pos_helper l' (xI acc)
  end.

Fixpoint bitlist_to_pos (l : bitlist) : positive :=
  match l with
  | false :: l' => bitlist_to_pos l'
  | true :: l' => bitlist_to_pos_helper l' xH
  | _ => xH
  end.

Fixpoint bitlist_to_z (l : bitlist) : Z :=
  match l with
  | false :: l' => bitlist_to_z l'
  | true :: l' => Zpos (bitlist_to_pos l)
  | _ => 0
  end.

Lemma pos_bitfold: forall (l : bitlist) a, pos_to_bitlist( bitlist_to_pos_helper l a) = (pos_to_bitlist a) ++ l.
Proof.
  intros l. induction l.
  - simpl. intros a. rewrite List.app_nil_r. reflexivity.
  - intros b. destruct a; simpl; rewrite IHl; simpl; rewrite <- List.app_assoc; auto.
Qed.

Lemma bitlist_to_z_id: forall (l : bitlist), z_to_bitlist (bitlist_to_z l) = norm_bitlist l.
Proof.
  intros l. induction l.
  - simpl. reflexivity.
  - simpl.
    destruct a.
    + simpl. apply pos_bitfold.
    + simpl. apply IHl.
Qed.

Lemma bitlist_to_pos_append_lsb1: forall (l : bitlist) a, bitlist_to_pos_helper (l ++ [true]) a = (bitlist_to_pos_helper l a)~1%positive.
Proof.
  intros l. induction l.
  - simpl. reflexivity.
  - intros b; destruct a; simpl; rewrite IHl; reflexivity.
Qed.

Lemma bitlist_to_z_append_lsb1: forall (l : bitlist), bitlist_to_z (l ++ [true]) = (2 * (bitlist_to_z l) + 1).
Proof.
  intros l. induction l.
  - simpl. reflexivity.
  - destruct a.
    + simpl. rewrite bitlist_to_pos_append_lsb1. reflexivity.
    + simpl. rewrite IHl. reflexivity.
Qed.

Lemma bitlist_to_pos_append_lsb0: forall (l : bitlist) a, bitlist_to_pos_helper (l ++ [false]) a = (bitlist_to_pos_helper l a)~0%positive.
Proof.
  intros l. induction l.
  - simpl. reflexivity.
  - intros b; destruct a; simpl; rewrite IHl; reflexivity.
Qed.

Lemma bitlist_to_z_append_lsb0: forall (l : bitlist), bitlist_to_z (l ++ [false]) = 2 * (bitlist_to_z l).
Proof.
  intros l. induction l.
  - simpl. reflexivity.
  - destruct a.
    + simpl. rewrite bitlist_to_pos_append_lsb0. reflexivity.
    + simpl. rewrite IHl. reflexivity.
Qed.

Lemma z_to_bitlist_id: forall (v : Z), (0 <= v) -> bitlist_to_z (z_to_bitlist v) = v.
Proof.
  intros v H. destruct v.
  - intuition.
  - clear H. induction p.
    + simpl. rewrite bitlist_to_z_append_lsb1. simpl in IHp. rewrite IHp. lia.
    + simpl. rewrite bitlist_to_z_append_lsb0. simpl in IHp. rewrite IHp. lia.
    + reflexivity.
  - lia.
Qed.

Definition scalar_mult_foldi_helper (k : secp256k1_scalar_t) p :=
  (fun (i_33 : uint_size)
           (q_32 : prod secp256k1_field_element_t secp256k1_field_element_t)
         =>
         match
           BinInt.Z.testbit k
             (@from_uint_size Z Z_uint_sizable
                (Z_to_uint_size
                   (BinInt.Z.sub
                      (BinInt.Z.sub
                         (BinInt.Z.of_N
                            (N.of_nat (uint_size_to_nat scalar_bits_v)))
                         (BinInt.Z.of_N
                            (N.of_nat
                               (uint_size_to_nat
                                  (@usize Z Z_uint_sizable (Zpos xH))))))
                      (BinInt.Z.of_N (N.of_nat (uint_size_to_nat i_33))))))
           return (prod secp256k1_field_element_t secp256k1_field_element_t)
         with
         | true => add_points p (double_point q_32)
         | false => double_point q_32
         end).

Lemma scalar_mult_body_eq_foldi_helper: forall k p, scalar_mult_foldi_helper k p =
  (fun (i_33 : uint_size) (q_32 : affine_t) =>
    if nat_mod_bit k (scalar_bits_v - usize 1 - i_33)
    then p +' (double_point q_32)
    else double_point q_32).
Proof.
  intros k p.
  reflexivity.
Qed.

Lemma foldi_helper_zero: forall (k : secp256k1_scalar_t) p i acc, val _ k = 0 -> scalar_mult_foldi_helper k p i acc = double_point acc.
Proof.
  intros.
  unfold scalar_mult_foldi_helper.
  rewrite H.
  rewrite Z.bits_0.
  reflexivity.
Qed.

(* From BLSProof.v *)
Lemma max_unsigned32 : @max_unsigned WORDSIZE32 = 4294967295.
Proof. reflexivity. Qed.

Lemma modulus32 : @modulus WORDSIZE32 = 4294967296.
Proof. reflexivity. Qed.

Lemma foldi_step: forall (accT : Type) (n : nat) (i : Z) (f : uint_size -> accT -> accT) (acc : accT), 0 <= i <= @max_unsigned WORDSIZE32 -> foldi_ (S n) (repr i) f acc = foldi_ n (repr (i +1)) f (f (repr i) acc).
Proof.
  intros.
  unfold foldi_.
  fold (foldi_ n (repr i .+ MachineIntegers.one) f (f (repr i) acc)).
  fold (foldi_ n (repr (i + 1)) f (f (repr i) acc)).
  assert (@MachineIntegers.add WORDSIZE32 (repr i) (@MachineIntegers.one WORDSIZE32) = repr (i + 1)). {
    unfold ".+".
    rewrite unsigned_one.
    rewrite (unsigned_repr _ H).
    reflexivity.
  }
  rewrite H0. reflexivity.
Qed.

Lemma scalar_mult_zero: forall (k : secp256k1_scalar_t) p, val _ k = 0 -> k *' p = infinity.
Proof.
  intros.
  unfold scalar_multiplication.
  rewrite <- scalar_mult_body_eq_foldi_helper.
  unfold foldi.
  destruct (unsigned scalar_bits_v - unsigned (usize 0)).
  - reflexivity.
  - assert (forall c, foldi_ (Pos.to_nat p0) c (scalar_mult_foldi_helper k p) infinity = infinity). {
      induction (Pos.to_nat p0).
      - unfold foldi_.
        reflexivity.
      - intros c.
        unfold foldi_.
        rewrite (foldi_helper_zero _ _ _ _ H).
        fold (foldi_ n (c .+ MachineIntegers.one) (scalar_mult_foldi_helper k p) (double_point infinity)).
        rewrite double_infty.
        apply IHn.
    }
    apply H0.
  - reflexivity.
Qed.

Lemma list_length_nonneg: forall (l : bitlist), 0 <= length l.
Proof.
  induction l.
  - simpl. reflexivity.
  - simpl. lia.
Qed.

Lemma usize_sub_helper: forall n, usize (usize 256 - n) = usize (256 - n).
Proof.
  intros n.
  reflexivity.
Qed.

Lemma usize_sub_helper2: forall n m, 0 <= n <= 256 -> 0 <= m <= 256 -> usize(usize n - usize m) = usize (n - m).
Proof.
  intros n m H1 H2. (*
  rewrite unsigned_repr.
  assert (uint_size_to_Z (usize n) = n). { simpl.  reflexivity. }
  reflexivity.*)
Admitted.

Lemma scalar_mult_foldi_skip_leading_zeros:
  forall (n m : nat) (l : bitlist) p, (0 <= bitlist_to_z l < scalar_max) -> (length l <= n <= 256) -> (m = length l) ->
  (foldi_ n (repr (scalar_bits_v - n)) (scalar_mult_foldi_helper (mkznz _ _ (modz scalar_max (bitlist_to_z l))) p) infinity) = (foldi_ m (repr (scalar_bits_v - m)) (scalar_mult_foldi_helper (mkznz _ _ (modz scalar_max (bitlist_to_z l))) p) infinity).
Proof.
  intros.
  unfold scalar_mult_foldi_helper.
  simpl.
  rewrite (Zmod_small _ _ H).
  induction n.
  - destruct m.
    + reflexivity.
    + lia.
  - destruct (S n =? m)%nat eqn:eq1.
    + rewrite Nat.eqb_eq in eq1.
      rewrite <- eq1, eq1.
      reflexivity.
    + rewrite Nat.eqb_neq in eq1. 
      rewrite foldi_step.
      assert (@repr WORDSIZE32 1 = usize 1). { reflexivity. }
      rewrite H2.
      rewrite H2 in IHn.
      assert ((scalar_bits_v - usize 1) = usize 255). { reflexivity. }
      rewrite H3.
      rewrite H3 in IHn.
      assert (@repr WORDSIZE32 (scalar_bits_v - S n) = usize (scalar_bits_v - S n)). { reflexivity. }
      rewrite H4.
      assert (usize 255 - usize (scalar_bits_v - S n) = n). { unfold scalar_bits_v. rewrite usize_sub_helper. }(*
      assert (Z.of_N (N.of_nat (uint_size_to_nat (@repr WORDSIZE32 (scalar_bits_v - S n)))) = usize (255) - usize(n)). {

      }
      assert (BinInt.Z.testbit (bitlist_to_z l)
      (@Z_mod_modulus WORDSIZE32
         (scalar_bits_v - uint_size_to_Z (repr 1) - uint_size_to_Z(repr (scalar_bits_v - S n)))) = false). {
           assert ((scalar_bits_v - usize 1) = usize 255). { reflexivity. }
           assert (@repr WORDSIZE32 1 = usize 1). { reflexivity. }
           rewrite H3.
           Set Printing All.
           rewrite H2. 
           unfold sca
           rewrite H2.
           assert (usize 256 - uint_size_to_Z (repr 1) = 255)
         }
  remember (k *' p) as sdfs.
  unfold scalar_multiplication in Heqsdfs.
  unfold foldi in Heqsdfs.
  rewrite <- scalar_mult_body_eq_foldi_helper in Heqsdfs.*)
Admitted.


Definition simple_scalar_mult2_def: forall (k : positive) (p : affine_t), simple_scalar_mult2 k p = simple_scalar_mult (Pos.to_nat k) p.
Proof.
  intros k p.
  induction k.
  - unfold simple_scalar_mult2.
    fold simple_scalar_mult2.
    rewrite <- add_to_double.
    rewrite IHk.
    rewrite simple_scalar_mult_distributivity.
    rewrite simple_scalar_mult_fold.
    assert (S (Pos.to_nat k + Pos.to_nat k) = Pos.to_nat k~1). { lia. }
    rewrite H.
    reflexivity.
  - unfold simple_scalar_mult2.
    fold simple_scalar_mult2.
    rewrite <- add_to_double.
    rewrite IHk.
    rewrite simple_scalar_mult_distributivity.
    assert (Nat.add (Pos.to_nat k) (Pos.to_nat k) = Pos.to_nat k~0). { lia. }
    rewrite H.
    reflexivity.
  - simpl.
    rewrite add_infty_1.
    reflexivity.
Qed.

Lemma simple_scalar_mult3_lsb1: forall (l : bitlist) (p acc : affine_t), simple_scalar_mult3 (l ++ [true]) p acc = (double_point (simple_scalar_mult3 l p acc)) +' p.
Proof.
  intros l.
  induction l.
  - simpl. reflexivity.
  - intros. destruct a; apply IHl.
Qed.

Lemma simple_scalar_mult3_lsb0: forall (l : bitlist) (p acc : affine_t), simple_scalar_mult3 (l ++ [false]) p acc = (double_point (simple_scalar_mult3 l p acc)).
Proof.
  intros l.
  induction l.
  - simpl. reflexivity.
  - intros. destruct a; apply IHl.
Qed.

Lemma simple_scalar_mult3_def: forall (k : nat) (p : affine_t), simple_scalar_mult3 (z_to_bitlist (Z.of_nat k)) p infinity = simple_scalar_mult k p.
Proof.
  intros k.
  destruct k.
  - intros p. simpl. reflexivity.
  - unfold z_to_bitlist.
    fold z_to_bitlist.
    unfold Z.of_nat.
    assert (S k = Pos.to_nat (Pos.of_succ_nat k)). { intuition. }
    rewrite H.
    intros p.
    rewrite <- (simple_scalar_mult2_def (Pos.of_succ_nat k) p).
    clear H.
    revert p.
    remember (Pos.of_succ_nat k) as k'.
    clear Heqk'.
    induction k'; intros p;
    unfold pos_to_bitlist, simple_scalar_mult3, simple_scalar_mult2;
    fold pos_to_bitlist; fold simple_scalar_mult3; fold simple_scalar_mult2.
    + rewrite <- IHk'. apply simple_scalar_mult3_lsb1.
    + rewrite <- IHk'. apply simple_scalar_mult3_lsb0.
    + rewrite double_infty, add_infty_1. reflexivity.
Qed.

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

Lemma scalar_mult_def: forall (k : secp256k1_scalar_t) (p : affine_t), k *' p = simple_scalar_mult3 (z_to_bitlist k) p infinity.
Proof.
  intros k.
  remember (bitlist_to_z (z_to_bitlist k)) as g'.
  pose proof scalar_small k as [H0 _].
  pose proof z_to_bitlist_id k H0.
  rewrite <- Heqg' in H.
  symmetry in H.
  pose proof znz_duplicate _ k g' H.
  destruct H1.
  remember (mkznz _ g' x) as g.
  intros p.
  assert (k *' p = g *' p). { rewrite H1. reflexivity. }
  rewrite H2.
  clear H2.
  revert p.
  subst.
  clear H. clear H0. clear H1.
  induction (z_to_bitlist k).
  - remember (mkznz _ (bitlist_to_z []) x) as g.
    assert (val _ g = 0). { rewrite Heqg. simpl. reflexivity. }
    intros p.
    apply (scalar_mult_zero _ _ H).
  - destruct a;
    unfold simple_scalar_mult3;
    fold simple_scalar_mult3;
    rewrite double_infty.
    unfold scalar_multiplication.
    intros p. (*
    rewrite <- (scalar_mult_body_eq_foldi_helper g p).
    pose proof foldi_helper_zero g p.
    .*)
Admitted.

(* Follows from Langrange's Theorem  since the order of the group is prime *)
Lemma simple_scalar_mult_mod: forall (k : nat) (p: affine_t), simple_scalar_mult k p = simple_scalar_mult (k mod (Z.to_nat scalar_max)) p.
Proof.
Admitted.

Lemma scalar_mult_distributivity: forall (k1 k2 : secp256k1_scalar_t) (p: affine_t), k1 *' p +' k2 *' p = (k1 +% k2) *' p.
Proof.
  intros k1 k2 p.
  rewrite scalar_mult_def.
  rewrite scalar_mult_def.
  rewrite scalar_mult_def.
  pose proof scalar_small k1 as [H1 _].
  pose proof scalar_small k2 as [H2 _].
  pose proof scalar_small (k1 +% k2) as [H3 _].
  rewrite <- (Z2Nat.id k1 H1).
  rewrite <- (Z2Nat.id k2 H2).
  rewrite <- (Z2Nat.id (k1 +% k2) H3).
  rewrite simple_scalar_mult3_def.
  rewrite simple_scalar_mult3_def.
  rewrite simple_scalar_mult3_def.
  rewrite simple_scalar_mult_distributivity.
  unfold "+%".
  unfold add.
  simpl.
  assert (0 <= (k1 + k2)). {
    pose proof scalar_small k1.
    pose proof scalar_small k2.
    lia.
  }
  assert (0 <= scalar_max). { unfold scalar_max. intuition. }
  rewrite (Z2Nat.inj_mod _ _ H H0).
  rewrite simple_scalar_mult_mod.
  rewrite (Z2Nat.inj_add _ _ H1 H2).
  reflexivity.
Qed.

Lemma scalar_mult_distributivity2: forall (k : secp256k1_scalar_t) (p1 p2 : on_curve_t), k *' (p1 +' p2) = k *' p1 +' k *' p2.
Proof.
  intros.
  rewrite scalar_mult_def.
  rewrite scalar_mult_def.
  rewrite scalar_mult_def.
  pose proof scalar_small k as [H1 _].
  rewrite <- (Z2Nat.id k H1).
  rewrite simple_scalar_mult3_def.
  rewrite simple_scalar_mult3_def.
  rewrite simple_scalar_mult3_def.
  apply simple_scalar_mult_distributivity2.
Qed.

Lemma scalar_mult_closed: forall (p : on_curve_t) (k : secp256k1_scalar_t), exists (q : on_curve_t), point q = k *' (point p).
Proof.
  intros p k.
  rewrite scalar_mult_def.
  pose proof scalar_small k as [H1 _].
  rewrite <- (Z2Nat.id k H1).
  rewrite simple_scalar_mult3_def.
  apply simple_scalar_mult_closed.
Qed.

Lemma scalar_mult_fold_once: forall (a b : nat) (p : on_curve_t), (simple_scalar_mult (a * b) p) +' (simple_scalar_mult b p) = simple_scalar_mult (S(a) * b) p.
Proof.
  intros a b p.
  simpl.
  rewrite <- simple_scalar_mult_distributivity.
  pose proof simple_scalar_mult_closed p (b).
  pose proof simple_scalar_mult_closed p (a * b).
  destruct H.
  destruct H0.
  rewrite <- H.
  rewrite <- H0.
  rewrite add_comm.
  reflexivity.
Qed.

Lemma scalar_mult_assoc2: forall (a b : secp256k1_scalar_t) (p : on_curve_t), a *' b *' p = (a *% b) *' p.
Proof.
  intros a b p.
  rewrite scalar_mult_def, scalar_mult_def, scalar_mult_def.
  unfold "*%", mul.
  simpl.
  pose proof scalar_small a as [H _].
  pose proof scalar_small b as [H1 _].
  rewrite <- (Z2Nat.id a H).
  rewrite <- (Z2Nat.id b H1).
  rewrite simple_scalar_mult3_def.
  rewrite simple_scalar_mult3_def.
  rewrite (Z2Nat.id a H).
  rewrite (Z2Nat.id b H1).
  assert (scalar_max > 0). { unfold scalar_max. intuition. }
  pose proof Z_mod_lt (a * b) _ H0 as [H2 _].
  rewrite <- (Z2Nat.id _ H2).
  rewrite simple_scalar_mult3_def.
  assert (0 <= a * b). { intuition. }
  assert (0 <= scalar_max). { intuition. }
  rewrite (Z2Nat.inj_mod _ _ H3 H4).
  rewrite <- simple_scalar_mult_mod.
  assert (Z.to_nat (Z.mul a b) = Nat.mul (Z.to_nat a) (Z.to_nat b)). { intuition. }
  rewrite H5.
  clear H5.
  induction (Z.to_nat a).
  - unfold simple_scalar_mult.
    simpl.
    reflexivity.
  - unfold simple_scalar_mult, simple_scalar_mult.
    fold simple_scalar_mult.
    rewrite IHn.
    rewrite scalar_mult_fold_once.
    reflexivity.
Qed.

Lemma scalar_mult_generator_zero: forall (a : secp256k1_scalar_t), a = nat_mod_zero <-> is_infinity (a *' generator) = true.
Proof.
  intros a.
  split.
  - intros H.
    rewrite H.
    rewrite scalar_mult_def.
    simpl.
    exact infty_is_infty.
  - intros H.
    apply is_infty_means_infty in H.
Admitted.


Lemma scalar_mult_generator_not_zero: forall (a : secp256k1_scalar_t), a <> nat_mod_zero -> is_infinity (a *' generator) = false.
Proof.
Admitted.

Fixpoint simple_batch_scalar_multiplication
(elems : seq (secp256k1_scalar_t × affine_t))
: affine_t :=
  match elems with
  | [] => infinity
  | (a, p) :: r => a *' p +' (simple_batch_scalar_multiplication r)
  end.

Fixpoint batch_scalar_opt_helper
(head : (secp256k1_scalar_t × affine_t)) (elems : seq (secp256k1_scalar_t × affine_t))
: seq (secp256k1_scalar_t × affine_t) :=
  match (head, elems) with
  | ((a1, p1), (a2, p2) :: r) => (a1 -% a2,p1) :: batch_scalar_opt_helper (a2, p1 +' p2) r
  | _ => head :: elems
  end.

Definition batch_scalar_opt (elems : seq (secp256k1_scalar_t × affine_t)) :=
  match elems with
  | head :: r => batch_scalar_opt_helper head r
  | _ => elems
  end.

Fixpoint on_curve_list_to_affines (elems : seq (secp256k1_scalar_t × on_curve_t)) : seq (secp256k1_scalar_t × affine_t) :=
  match elems with
  | (a, P) :: r => (a, point P) :: on_curve_list_to_affines r
  | nil => nil
  end.

Lemma seq_len_eq: forall A B (l1 : seq A) (l2 : seq B), seq_len l1 = seq_len l2 <-> length l1 = length l2.
Proof.
  intros.
  split.
  - intros H. unfold seq_len in H. apply Nnat.Nat2N.inj. exact H.
  - intros H. unfold seq_len. rewrite H. reflexivity.
Qed.

Lemma on_curve_list_to_affines_length: forall (elems : seq (secp256k1_scalar_t × on_curve_t)),
  seq_len (on_curve_list_to_affines elems) = seq_len elems.
Proof.
  intros elems.
  apply seq_len_eq.
  induction elems.
    - simpl. reflexivity.
    - destruct a as (a,P). simpl. rewrite IHelems. reflexivity.
Qed.

Lemma on_curve_list_to_affines_length2: forall a b (elems : seq (secp256k1_scalar_t × on_curve_t)),
  seq_len (a :: on_curve_list_to_affines elems) = seq_len (b :: elems).
Proof.
  intros.
  rewrite seq_len_eq.
  simpl.
  Search (S ?a = S ?b <-> ?a = ?b).
  rewrite Nat.succ_inj_wd.
  rewrite <- seq_len_eq.
  apply on_curve_list_to_affines_length.
Qed.

(*Lemma on_curve_list_to_affines_concat1: forall a P b c,
  on_curve_list_to_affines (((a, P) :: b) ++ c) = (a, point P) :: on_curve_list_to_affines (b ++ c).
Proof.
  intros.
  revert c. revert P. revert a.
  induction b.
  - intros. simpl. reflexivity.
  - intros. *)

Lemma on_curve_list_to_affines_concat2: forall a b,
  on_curve_list_to_affines (a ++ b) = on_curve_list_to_affines a ++ on_curve_list_to_affines b.
Proof.
  intros a.
  induction a.
  - simpl. reflexivity.
  - intros b.
    destruct a as (a, P).
    unfold on_curve_list_to_affines.
    fold on_curve_list_to_affines.
    rewrite <- app_comm_cons.
    simpl.
    rewrite IHa.
    reflexivity.
Qed.

Lemma usize_eq: forall (x : Z), 0 <= x <= @max_unsigned WORDSIZE32 -> (x = usize x).
Proof.
  intros.
  simpl.
  unfold Z.of_N, N.of_nat, uint_size_to_nat, from_uint_size, nat_uint_sizable.
  pose proof @unsigned_repr WORDSIZE32 _ H.
  rewrite H0.
  fold (N.of_nat (Z.to_nat x)).
  fold (Z.of_N (N.of_nat (Z.to_nat x))).
  lia.
Qed.

Definition product_sum_foldi_helper (elems : seq (secp256k1_scalar_t × affine_t)) : uint_size -> affine_t -> affine_t :=
  fun i_43 res_42 =>
    let '(ai_44, pi_45) :=
      seq_index elems (i_43) in 
    let res_42 :=
      add_points res_42 (scalar_multiplication ai_44 pi_45) in 
    res_42.

Lemma product_sum_eq_foldi_helper: forall (elems : seq (secp256k1_scalar_t × affine_t)),
  product_sum_foldi_helper elems = fun i_43 res_42 =>
  let '(ai_44, pi_45) :=
    seq_index elems (i_43) in 
  let res_42 :=
    add_points res_42 (scalar_multiplication ai_44 pi_45) in 
  res_42.
Proof.
  intros elems. reflexivity.
Qed.

Lemma product_sum_foldi_helper_zero: forall a P elems,
  product_sum_foldi_helper ((a, P) :: elems) (repr 0) infinity = a *' P.
Proof.
  intros.
  unfold product_sum_foldi_helper.
  unfold seq_index.
  simpl.
  rewrite add_infty_1.
  reflexivity.
Qed.

Lemma product_sum_foldi_ignore_last: forall elems1 elems2 acc,
  length elems1 <= @max_unsigned WORDSIZE32 ->
  foldi (repr 0) (length elems1) (product_sum_foldi_helper (on_curve_list_to_affines elems1 ++ on_curve_list_to_affines elems2)) acc =
  foldi (repr 0) (length elems1) (product_sum_foldi_helper (on_curve_list_to_affines elems1)) acc.
Proof.
  intros.
  remember (product_sum_foldi_helper (on_curve_list_to_affines elems1 ++ on_curve_list_to_affines elems2)) as f.
  remember (product_sum_foldi_helper (on_curve_list_to_affines elems1)) as g.
  pose proof (foldi_equiv _ (repr 0) (length elems1) f g acc).
  Set Printing All.
  assert (forall i acc, @unsigned WORDSIZE32 (repr 0) <= unsigned i < unsigned (Z_to_uint_size (Z.of_N (N.of_nat (length elems1)))) -> f i acc = g i acc). {
    intros.
    rewrite Heqf, Heqg.
    unfold product_sum_foldi_helper.
    unfold seq_index.
    assert (uint_size_to_nat i < length (on_curve_list_to_affines elems1))%nat. {
      destruct i as (i', i_small).
      unfold Z_to_uint_size in H1.
      assert (0 <= Z.of_N (length elems1) <= @max_unsigned WORDSIZE32). { lia. }
      rewrite (@unsigned_repr WORDSIZE32 (Z.of_N (length elems1)) H2) in H1.
      simpl in H1.
      unfold uint_size_to_nat.
      simpl.
      pose proof on_curve_list_to_affines_length elems1.
      unfold seq_len in H3.
      lia.
    }
    rewrite (app_nth1 _ _ _  H2).
    reflexivity.
  }
  rewrite (H0 H1). reflexivity.
Qed.

Lemma product_sum_foldi_skip_first: forall a elems acc,
  foldi_ (length elems) (repr (1)) (product_sum_foldi_helper (a :: on_curve_list_to_affines elems)) acc =
  foldi_ (length elems) (repr 0) (product_sum_foldi_helper (on_curve_list_to_affines elems)) acc.
Proof.
  intros.
  revert a. revert acc.
  rewrite <- (rev_involutive elems).
  induction (rev elems).
  - simpl. reflexivity.
  - intros acc a0.
    rewrite rev_length.
    unfold rev.
    fold (rev l).
    rewrite on_curve_list_to_affines_concat2.
    assert (0 <= 0 <= @max_unsigned WORDSIZE32). { easy. }
    rewrite (foldi_step _ _ _ _ _ H).
    assert (0 <= 1 <= @max_unsigned WORDSIZE32). { easy. }
    rewrite (foldi_step _ _ _ _ _ H0).
    rewrite IHn.
    (*rewrite <- IHn0. *)
Admitted.

Lemma product_sum_foldi_add_acc: forall elems n acc,
  0 <= n < @max_unsigned WORDSIZE32 ->
  foldi_ (length elems) (repr n) (product_sum_foldi_helper (on_curve_list_to_affines elems)) acc =
  acc +' (foldi_ (length elems) (repr n) (product_sum_foldi_helper (on_curve_list_to_affines elems)) infinity).
Proof.
  intros elems.
  induction elems.
  - intros. simpl. rewrite add_infty_2. reflexivity.
  - intros.
    unfold length.
    fold (length elems).
    assert (0 <= n <= @max_unsigned WORDSIZE32). { intuition. }
    rewrite (foldi_step _ _ _ _ _ H0).
    rewrite (foldi_step _ _ _ _ _ H0). (*
    assert (0 <= n0 + 1)
    rewrite IHn.*)
Admitted.

Lemma product_sum_def: forall (elems : seq (secp256k1_scalar_t × on_curve_t)),
  0 <= seq_len elems <= @max_unsigned WORDSIZE32 -> product_sum (on_curve_list_to_affines elems) = simple_batch_scalar_multiplication (on_curve_list_to_affines elems).
Proof.
  intros elems.
  induction elems.
  - reflexivity.
  - intros H.
    destruct a as (a, P). 
    unfold on_curve_list_to_affines.
    fold on_curve_list_to_affines.
    unfold simple_batch_scalar_multiplication.
    fold simple_batch_scalar_multiplication.
    assert (0 <= seq_len elems <= seq_len ((a, P) :: elems)). { unfold seq_len. simpl. intuition. }
    assert (0 <= seq_len elems <= @max_unsigned WORDSIZE32). { lia. }
    rewrite <- (IHelems H1).
    unfold product_sum.
    unfold foldi.
    rewrite <- product_sum_eq_foldi_helper.
    rewrite <- product_sum_eq_foldi_helper.
    unfold foldi.
    pose proof @unsigned_repr WORDSIZE32.
    unfold Z_to_uint_size.
    rewrite (on_curve_list_to_affines_length2 _ (a,P) _).
    rewrite (@unsigned_repr WORDSIZE32 _ H).
    rewrite on_curve_list_to_affines_length.
    rewrite (@unsigned_repr WORDSIZE32 _ H1).
    assert (0 <= 0 <= @max_unsigned WORDSIZE32). { easy. }
    unfold usize.
    unfold Z_uint_sizable.
    rewrite (@unsigned_repr WORDSIZE32 _ H3).
    simpl.
    assert (Pos.to_nat (Pos.of_succ_nat (length elems)) = S (length elems)). { lia. }
    rewrite H4.
    rewrite (foldi_step _ (length elems)).
    rewrite product_sum_foldi_helper_zero.
    + destruct (seq_len elems) eqn:eq1.
     * simpl.
        unfold seq_len in eq1.
        assert (0%N = N.of_nat 0). { lia. }
        rewrite H5 in eq1.
        pose proof (Nnat.Nat2N.inj (length elems) (0%nat) eq1).
        rewrite H6.
        simpl.
        rewrite add_infty_2. reflexivity.
      * simpl.
        assert (Pos.to_nat p = length elems). {
          unfold seq_len in eq1.
          intuition.
        }
        rewrite H5.
        rewrite product_sum_foldi_skip_first.
        assert (0 <= 0 < @max_unsigned WORDSIZE32). { easy. }
        apply (product_sum_foldi_add_acc elems 0 (a *' P) H6).
    + exact H3.
Qed.
    

Lemma batch_scalar_opt_eq: forall (elems : seq (secp256k1_scalar_t × on_curve_t)),
  simple_batch_scalar_multiplication (batch_scalar_opt (on_curve_list_to_affines elems)) = simple_batch_scalar_multiplication (on_curve_list_to_affines elems).
Proof.
  intros elems.
  destruct elems.
  - auto.
  - revert p.
    induction elems.
    + intros p. unfold batch_scalar_opt, batch_scalar_opt_helper. destruct p as (a, P). reflexivity.
    + intros p.
      destruct a as (a1, p1).
      destruct p as (a2, p2).
      unfold batch_scalar_opt, batch_scalar_opt_helper.
      unfold on_curve_list_to_affines.
      fold on_curve_list_to_affines.
      fold batch_scalar_opt_helper.
      unfold simple_batch_scalar_multiplication.
      fold simple_batch_scalar_multiplication.
      destruct (add_points_closed p2 p1).
      rewrite <- H.
      assert (batch_scalar_opt_helper (a1, x) (on_curve_list_to_affines elems) = batch_scalar_opt (on_curve_list_to_affines ((a1, x) :: elems))). {
        unfold batch_scalar_opt, on_curve_list_to_affines.
        fold on_curve_list_to_affines. reflexivity.
      }
      rewrite H0.
      rewrite IHelems.
      unfold on_curve_list_to_affines.
      fold on_curve_list_to_affines.
      unfold simple_batch_scalar_multiplication.
      fold simple_batch_scalar_multiplication.
      rewrite <- add_assoc.
      assert (a2 -% a1 = a2 +% nat_mod_neg a1). { field. }
      rewrite H1.
      rewrite <- scalar_mult_distributivity.
      rewrite H.
      rewrite scalar_mult_distributivity2.
      remember (simple_batch_scalar_multiplication (on_curve_list_to_affines elems)) as r.
      rewrite <- (add_assoc).
      rewrite <- (add_assoc).
      rewrite (add_assoc (a2 *' p2) _ _).
      rewrite scalar_mult_distributivity.
      assert (nat_mod_neg a1 +% a1 = nat_mod_zero). { field. }
      rewrite H2.
      rewrite (scalar_mult_zero nat_mod_zero p2).
      * rewrite add_infty_2.
        reflexivity.
      * simpl. apply Zmod_0_l.
Qed.

Lemma wordsize32_eq: @wordsize WORDSIZE32 = 32.
Proof.
  reflexivity.
Qed.

Lemma batch_scalar_mult_reduc: forall (elems : seq (secp256k1_scalar_t × on_curve_t)),
  batch_scalar_multiplication (on_curve_list_to_affines elems) = simple_batch_scalar_multiplication (on_curve_list_to_affines elems).
Proof.
  intros elems.
  unfold batch_scalar_multiplication.

Lemma batch_scalar_mult_def: forall (elems :  seq (secp256k1_scalar_t × affine_t)),
  batch_scalar_multiplication elems = simple_batch_scalar_multiplication elems.
Proof.
  intros elems.
  induction elems as [ | h t IHn].
  - reflexivity.
  - destruct t.
    + unfold batch_scalar_multiplication.
      assert (0 <= 1 <= @max_unsigned WORDSIZE32). {  easy. }
      rewrite <- (usize_eq 1 H).
      simpl.
      unfold foldi.
      simpl.
      assert (Zbits.P_mod_two_p 1 (@wordsize WORDSIZE32) = 1). { easy. }
      rewrite H0.
      simpl.
      assert (0%nat = uint_size_to_nat(@repr WORDSIZE32 0)). { easy. }
      rewrite <- H1.
      destruct h as (a, P).
      simpl.
      rewrite add_infty_1, add_infty_2.
      reflexivity.
    + unfold batch_scalar_multiplication.
      assert (0 <= 1 <= @max_unsigned WORDSIZE32). {  easy. }
      rewrite <- (usize_eq 1 H).
      unfold foldi.
      assert (seq_len (h :: p :: t) - 1 > 0). { unfold seq_len, length. lia. }
      simpl.
      simpl.
    
  - destruct h as (a, p).
    unfold simple_batch_scalar_multiplication.
    fold simple_batch_scalar_multiplication.
    unfold batch_scalar_multiplication.
    remember (fun (i_37 : uint_size)
    (new_elems_36 : seq (nat_mod scalar_max × affine_t)) =>
  let
  '(ai_38, pi_39) := seq_index new_elems_36 i_37 in
   let
   '(aiplus1_40, piplus1_41) :=
    seq_index new_elems_36 (i_37 + usize 1) in
    seq_upd
      (seq_upd new_elems_36 i_37 (ai_38 -% aiplus1_40, pi_39))
      (i_37 + usize 1) (aiplus1_40, pi_39 +' piplus1_41)) as fhelper.
    remember (foldi (usize 0) (seq_len ((a, p) :: t) - usize 2) fhelper ((a, p) :: t)) as updated.
    assert (foldi (usize 0) (seq_len updated - usize 1) (fun i_42 res_35 =>
      let '(ai_43, pi_44) := seq_index updated i_42 in res_35 +' ai_43 *' pi_44) infinity = simple_batch_scalar_multiplication updated).
      {
      induction updated.
      - simpl.
        Search (foldi ?a ?b ?c).
        unfold foldi.
        simpl.
    }
    simpl.