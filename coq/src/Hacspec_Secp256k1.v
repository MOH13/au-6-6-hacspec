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

(** 
This file contains the coq export of the hacspec-secp256k1 implementation and its corresponding proofs.
*)

(** * hacspec-to-coq definitions *)

(** ** Main curve implementation *)

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

 (**  ** Batch scalar multiplication *) 

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

(** * Notation *)

Notation "p '+'' q" := (add_points p q) (at level 69, left associativity).
Notation "k '*'' p" := (scalar_multiplication k p) (at level 68, right associativity).

(** * Nat_mod properties *)

Section nat_mod.

Variable max: Z.
Variable max_prime: prime max.

Definition nat_mod_eq (a b : nat_mod max) := a = b.

Add Field FZpZ : (GZnZ.FZpZ max max_prime).

Lemma max_pos: 0 < max.
  generalize (prime_ge_2 _ max_prime); auto with zarith.
Qed.

(** ** Field properties *)
(** Proof of field properties of nat_mod (simple wrapper of GZnZ.znz) *)
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

(** ** Nat_mod helper lemmas *)

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

(** * foldi (for-loop) helper lemmas *)

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
          rewrite unsigned_one.
          simpl.
          assert (0 <= unsigned i0 + 1 < @modulus WORDSIZE32). { lia. }
          rewrite Z_mod_modulus_eq.
          rewrite (Zmod_small _ _ H4).
          intuition.
        }
        pose proof (H1 (i0 .+ MachineIntegers.one) acc0 H4).
        rewrite Heqf'.
        rewrite Heqg'.
        exact H5.
      }
      assert (unsigned (a .+ MachineIntegers.one) = unsigned a + 1). {
        unfold ".+".
        rewrite unsigned_one.
        assert (@max_unsigned WORDSIZE32 = @modulus WORDSIZE32 - 1). { easy. }
        assert (0 <= unsigned a + 1 <= @max_unsigned WORDSIZE32). { lia. }
        rewrite (unsigned_repr _ H5).
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
  revert H1. revert H0. revert H.
  revert acc. revert g. revert f. revert j. revert i.
  induction n.
  - reflexivity.
  - intros.
    unfold foldi_.
    fold (foldi_ n (i .+ MachineIntegers.one) f (f i acc)).
    fold (foldi_ n (j .+ MachineIntegers.one) g (g j acc)).
    pose proof unsigned_range_2 i.
    pose proof unsigned_range_2 j.
    assert (0 <= Z.of_nat n + unsigned (i .+ MachineIntegers.one) <= @max_unsigned WORDSIZE32). {
      unfold ".+".
      rewrite unsigned_one.
      assert (unsigned i < @max_unsigned WORDSIZE32). { lia. }
      assert (0 <= (unsigned i) + 1 <= @max_unsigned WORDSIZE32). { lia. }
      rewrite (unsigned_repr _ H5).
      lia.
    }
    assert (0 <= Z.of_nat n + unsigned (j .+ MachineIntegers.one) <= @max_unsigned WORDSIZE32). {
      unfold ".+".
      rewrite unsigned_one.
      assert (unsigned j < @max_unsigned WORDSIZE32). { lia. }
      assert (0 <= (unsigned j) + 1 <= @max_unsigned WORDSIZE32). { lia. }
      rewrite (unsigned_repr _ H6).
      lia.
    }
    assert (forall i0 acc0, 0 <= i0 <= n ->
        f (repr i0 .+ (i .+ MachineIntegers.one)) acc0 = g (repr i0 .+ (j .+ MachineIntegers.one)) acc0). {
      intros.
      remember (i0 + 1) as i0'.
      assert (0 <= i0' <= S n). { lia. }
      pose proof (H1 i0' acc0 H7).
      rewrite Heqi0' in H8.
      unfold ".+" in H8.
      assert (0 <= i0 + 1 <= @max_unsigned WORDSIZE32). { lia. }
      rewrite (unsigned_repr _ H9) in H8.
      unfold ".+".
      assert (0 <= i0 <= @max_unsigned WORDSIZE32). { lia. }
      rewrite (unsigned_repr _ H10).
      rewrite unsigned_one.
      pose proof @repr_unsigned WORDSIZE32.
      assert (0 <= unsigned i + 1 <= @max_unsigned WORDSIZE32). { lia. }
      rewrite (unsigned_repr _ H12).
      assert (0 <= unsigned j + 1 <= @max_unsigned WORDSIZE32). { lia. }
      rewrite (unsigned_repr _ H13).
      assert (i0 + 1 + unsigned i = i0 + (unsigned i + 1)). { lia. }
      assert (i0 + 1 + unsigned j = i0 + (unsigned j + 1)). { lia. }
      rewrite H14, H15 in H8.
      exact H8.
    }
    assert (0 <= 0 <= S n). { lia. }
    pose proof (H1 0 acc H7).
    unfold ".+" in H8.
    assert (0 <= 0 <= @max_unsigned WORDSIZE32). { easy. }
    rewrite (unsigned_repr _ H9) in H8.
    simpl in H8.
    rewrite repr_unsigned, repr_unsigned in H8.
    rewrite H8.
    exact (IHn (i .+ MachineIntegers.one) (j .+ MachineIntegers.one) f g (g j acc) H4 H5 H6).
Qed.

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

Lemma foldi_step2: forall (A : Type) (a b : uint_size) (f : uint_size -> A -> A) (acc : A),
  (unsigned a < unsigned b) ->
  foldi a b f acc = foldi (a .+ MachineIntegers.one) b f (f a acc).
Proof.
  intros.
  unfold foldi.
  assert (unsigned (@MachineIntegers.one WORDSIZE32) = 1). { easy. }
  pose proof unsigned_range_2 a.
  pose proof unsigned_range_2 b.
  assert (0 <= unsigned a + 1 <= @max_unsigned WORDSIZE32). { lia. }
  destruct (unsigned b - unsigned a) eqn:eq1. lia.
  destruct (unsigned b - (unsigned (a .+ MachineIntegers.one))) eqn:eq2.
  - unfold ".+" in eq2.
    rewrite H0 in eq2.
    rewrite (unsigned_repr _ H3) in eq2.
    assert (Pos.to_nat p = 1)%nat. { lia. }
    rewrite H4.
    simpl. reflexivity.
  - destruct (Pos.to_nat p) eqn:eq3.
    + lia.
    + rewrite <- (repr_unsigned a).
      rewrite (foldi_step _ _ _ _ _ H1).
      assert (Pos.to_nat p0 = n). {
        unfold ".+" in eq2.
        rewrite H0 in eq2.
        rewrite (unsigned_repr _ H3) in eq2.
        lia.
      }
      rewrite H4. 
      unfold ".+".
      rewrite H0.
      rewrite (repr_unsigned a). reflexivity.
  - unfold ".+" in eq2.
    rewrite H0 in eq2.
    rewrite (unsigned_repr _ H3) in eq2.
    lia.
  - lia.
Qed.

Lemma foldi_split: forall (A : Type) (a b c : uint_size) (f : uint_size -> A -> A) (acc : A),
  (unsigned a <= unsigned b < unsigned c) ->
  foldi a c f acc = foldi b c f (foldi a b f acc).
Proof.
  intros.
  destruct (unsigned c - unsigned a) eqn:eq1.
  - intros.
    assert (unsigned c - unsigned b = 0). { intuition. }
    assert (unsigned b - unsigned a = 0). { intuition. }
    unfold foldi.
    rewrite eq1, H0, H1. reflexivity.
  - assert (Z.pos p = Z.of_nat (Pos.to_nat p)). { intuition. }
    rewrite H0 in eq1.
    revert eq1. revert H. revert acc. revert c. revert b. revert a.
    clear H0.
    induction (Pos.to_nat p).
    + intros. 
      assert (unsigned c - unsigned b = 0). { intuition. }
      assert (unsigned b - unsigned a = 0). { intuition. }
      unfold foldi.
      rewrite eq1, H0, H1.
      reflexivity.
    + intros.
      assert (unsigned a < unsigned c). { lia. }
      destruct (unsigned a <? unsigned b) eqn:eq2.
      * assert (unsigned a < unsigned b). { lia. }
        rewrite (foldi_step2 _ a c _ _ H0).
        rewrite (foldi_step2 _ a b _ _ H1).
        assert (0 <= unsigned a + 1 <= @max_unsigned WORDSIZE32). {
          pose proof unsigned_range_2 a.
          pose proof unsigned_range_2 c.
          lia.
        }
        destruct (unsigned (a .+ MachineIntegers.one) =? unsigned c) eqn:eq3.
        ++rewrite Z.eqb_eq in eq3.
          unfold foldi.
          rewrite eq3.
          unfold ".+" in eq3.
          rewrite unsigned_one in eq3.
          rewrite (unsigned_repr _ H2) in eq3.
          lia.
        ++
          assert (unsigned (a .+ MachineIntegers.one) <= unsigned b < unsigned c). {
            unfold ".+".
            rewrite unsigned_one.
            rewrite (unsigned_repr _ H2).
            lia.
          }
          assert (unsigned c - unsigned (a .+ MachineIntegers.one) = Z.of_nat n). {
            unfold ".+".
            rewrite unsigned_one.
            rewrite (unsigned_repr _ H2).
            lia.
          }
          exact (IHn (a .+ MachineIntegers.one) b c (f a acc) H3 H4).
      * assert (a = b). {
          assert (unsigned a = unsigned b). { lia. }
          rewrite <- (repr_unsigned a).
          rewrite <- (repr_unsigned b).
          rewrite H1. reflexivity.
        }
        rewrite H1.
        unfold foldi.
        assert (unsigned b - unsigned b = 0). { lia. }
        rewrite H2. reflexivity.
    - lia.
Qed.

Lemma foldi_simple_invariant: forall (A : Type) (a b : uint_size) (f : uint_size -> A -> A) (acc : A) (P : A -> Prop),
  P acc ->
  (forall i0 acc0, unsigned a <= unsigned i0 < unsigned b -> P acc0 -> P (f i0 acc0)) ->
  P (foldi a b f acc).
Proof.
  intros.
  revert H.
  revert acc.
  unfold foldi.
  destruct (unsigned b - unsigned a) eqn:eq1.
  - intros. exact H.
  - assert (Z.of_nat (Pos.to_nat p) + unsigned a = unsigned b). { lia. }
    clear eq1.
    revert H. revert H0. revert a.
    induction (Pos.to_nat p).
    + intros. simpl. exact H1.
    + intros. simpl.
      pose proof unsigned_range_2 b.
      assert (unsigned (a .+ MachineIntegers.one) = unsigned (a) + 1). {
        unfold ".+".
        pose proof unsigned_range_2 a.
        rewrite unsigned_one.
        assert (0 <= unsigned a + 1 <= @max_unsigned WORDSIZE32). { lia. }
        rewrite (unsigned_repr _ H4). reflexivity.
      }
      assert (P (f a acc)). {
        assert (unsigned a <= unsigned a < unsigned b). { lia. }
        exact (H0 a acc H4 H1).
      }
      destruct (unsigned (a .+ MachineIntegers.one) <? unsigned b) eqn:eq2.
      * 
        assert (forall i0 acc0, unsigned (a .+ MachineIntegers.one) <= unsigned i0 < unsigned b -> P acc0 -> P (f i0 acc0)). {
          intros.
          assert (unsigned a <= unsigned i0 < unsigned b). { intuition. }
          exact (H0 i0 acc0 H7 H6).
        }
        assert (Z.of_nat n + unsigned (a .+ MachineIntegers.one) = unsigned b). { lia. }
        exact (IHn (a .+ MachineIntegers.one) H5 H6 (f a acc) H4).
      * unfold foldi_.
        destruct n eqn:eq3.
        ++exact H4.
        ++lia.
  - intros. exact H.
Qed.

End foldi.

(** * Field definitions *)

(** p and n are prime (actual proofs needed) *)

Axiom elem_max_prime: prime elem_max.
Axiom scalar_max_prime: prime scalar_max.

(** Specialize field proof to the two concrete fields *)

Definition field_elem_FZpZ := (nat_mod_FZpZ elem_max elem_max_prime).
Definition scalar_FZpZ := (nat_mod_FZpZ scalar_max scalar_max_prime).

Add Field field_elem_FZpZ : field_elem_FZpZ.
Add Field scalar_FZpZ : scalar_FZpZ.

(** * Curve properties *)

(** ** General properties and helper lemmas *)

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

(** ** Proofs of closed operations *)

Structure on_curve_t: Set:=
 mkoncurve {point: affine_t;
        on_curve: is_point_on_curve point = true }.

Coercion point : on_curve_t >-> prod.

Lemma infty_on_curve: is_point_on_curve infinity = true.
Proof.
  intuition.
Qed.

(** Could not be completed since [nat_mod_from_byte_seq_be] is not defined *)
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

(** Proofs of remaining cases missing *)
Lemma add_assoc: forall (p q r : on_curve_t), (p +' q) +' r = p +' (q +' r).
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
      } (*{
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
                rewrite add_different_comm. 
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
    + simpl in Heqres. (* Show that this still obeys curve equation *)
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

(** * Scalar multiplication proofs

General strategy to show equivalences:
[simple_scalar_mult] <-> [simple_scalar_mult2] <-> [simple_scalar_mult3] <-> [scalar_multiplication].

Missing equivalence proof between [simple_scalar_mult3] and [scalar_multiplication].

*)

(** ** Simple scalar multiplication function 1 *)
Fixpoint simple_scalar_mult (k : nat) (p : affine_t) : affine_t :=
  match k with
  | 0%nat => infinity
  | S k1  => (simple_scalar_mult k1 p) +' p
  end.

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

Lemma simple_scalar_mult_distributivity: forall (k1 k2 : nat) (p: on_curve_t), (simple_scalar_mult k1 p) +' (simple_scalar_mult k2 p) = (simple_scalar_mult (k1 + k2) p).
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
    destruct (simple_scalar_mult_closed p k1).
    destruct (simple_scalar_mult_closed p k').
    destruct (simple_scalar_mult_closed p (k1 + k')).
    rewrite <- H, <- H0, <- H1 in *.
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
    destruct (simple_scalar_mult_closed p1 k).
    destruct (simple_scalar_mult_closed p2 k).
    rewrite <- H, <- H0 in *.
    destruct (add_points_closed x x0).
    rewrite <- H1.
    rewrite <- add_assoc.
    rewrite H1.
    rewrite add_assoc.
    rewrite (add_comm x0 p1).
    rewrite <- add_assoc.
    destruct (add_points_closed x p1).
    rewrite <- H2.
    rewrite add_assoc. reflexivity.
Qed.

Lemma simple_scalar_mult_fold: forall (k : nat) (p : affine_t), (simple_scalar_mult k p) +' p = simple_scalar_mult (S k) p.
Proof.
  intros k p.
  unfold simple_scalar_mult.
  fold simple_scalar_mult.
  reflexivity.
Qed.

(** ** Simple scalar multiplication function 2 *)
Fixpoint simple_scalar_mult2 (k : positive) (p : affine_t) : affine_t :=
  match k with
  | xH => p
  | xO r => double_point (simple_scalar_mult2 r p)
  | xI r => (double_point (simple_scalar_mult2 r p)) +' p
  end.

Definition bitlist := list bool.

(** ** Simple scalar multiplication function 3 *)
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
      assert (usize 255 - usize (scalar_bits_v - S n) = n). { unfold scalar_bits_v. rewrite usize_sub_helper. (*}
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


Definition simple_scalar_mult2_def: forall (k : positive) (p : on_curve_t), simple_scalar_mult2 k p = simple_scalar_mult (Pos.to_nat k) p.
Proof.
  intros k p.
  induction k.
  - unfold simple_scalar_mult2.
    fold simple_scalar_mult2.
    rewrite <- add_to_double.
    rewrite IHk.
    rewrite simple_scalar_mult_distributivity.
    rewrite simple_scalar_mult_fold.
    assert (S (Pos.to_nat k + Pos.to_nat k) = Pos.to_nat k~1) as -> by lia.
    reflexivity.
  - unfold simple_scalar_mult2.
    fold simple_scalar_mult2.
    rewrite <- add_to_double.
    rewrite IHk.
    rewrite simple_scalar_mult_distributivity.
    assert (Nat.add (Pos.to_nat k) (Pos.to_nat k) = Pos.to_nat k~0) as -> by lia.
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

Lemma simple_scalar_mult3_def: forall (k : nat) (p : on_curve_t), simple_scalar_mult3 (z_to_bitlist (Z.of_nat k)) p infinity = simple_scalar_mult k p.
Proof.
  intros k.
  destruct k.
  - intros p. simpl. reflexivity.
  - unfold z_to_bitlist.
    fold z_to_bitlist.
    unfold Z.of_nat.
    assert (S k = Pos.to_nat (Pos.of_succ_nat k)) as -> by intuition.
    intros p.
    rewrite <- (simple_scalar_mult2_def (Pos.of_succ_nat k) p).
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

(** ** [scalar_multiplication] equivalence with function 3 (incomplete) *)

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
  assert (k *' p = g *' p) as ->. { rewrite H1. reflexivity. }
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

(** Follows from Langrange's Theorem since the order of the group is prime *)
Lemma simple_scalar_mult_mod: forall (k : nat) (p: affine_t), simple_scalar_mult k p = simple_scalar_mult (k mod (Z.to_nat scalar_max)) p.
Proof.
Admitted.

Lemma scalar_mult_distributivity: forall (k1 k2 : secp256k1_scalar_t) (p: on_curve_t), k1 *' p +' k2 *' p = (k1 +% k2) *' p.
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
  destruct (add_points_closed p1 p2).
  rewrite <- H in *.
  rewrite simple_scalar_mult3_def.
  rewrite H.
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
  destruct (simple_scalar_mult_closed p (Z.to_nat b)).
  rewrite <- H0.
  rewrite simple_scalar_mult3_def.
  rewrite (Z2Nat.id a H).
  rewrite (Z2Nat.id b H1).
  assert (scalar_max > 0). { unfold scalar_max. intuition. }
  pose proof Z_mod_lt (a * b) _ H2 as [H3 _].
  rewrite <- (Z2Nat.id _ H3).
  rewrite simple_scalar_mult3_def.
  assert (0 <= a * b). { intuition. }
  assert (0 <= scalar_max). { intuition. }
  rewrite (Z2Nat.inj_mod _ _ H4 H5).
  rewrite <- simple_scalar_mult_mod.
  assert (Z.to_nat (Z.mul a b) = Nat.mul (Z.to_nat a) (Z.to_nat b)). { intuition. }
  rewrite H6.
  clear H6.
  induction (Z.to_nat a).
  - unfold simple_scalar_mult.
    simpl.
    reflexivity.
  - unfold simple_scalar_mult, simple_scalar_mult.
    fold simple_scalar_mult.
    rewrite IHn.
    rewrite H0.
    rewrite scalar_mult_fold_once.
    reflexivity.
Qed.

Lemma scalar_mult_generator_not_zero: forall (a : secp256k1_scalar_t), a <> nat_mod_zero -> is_infinity (a *' generator) = false.
Proof.
Admitted.

(** * Batch scalar multiplication proofs *)

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

(** Helper function converting a list of tuples of scalars and points on curve to list of tuples of scalars and corresponding points without the proof. *)
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

Lemma on_curve_list_to_affines_length0: forall (elems : seq (secp256k1_scalar_t × on_curve_t)),
  length (on_curve_list_to_affines elems) = length elems.
Proof.
  intros elems.
  induction elems.
    - simpl. reflexivity.
    - destruct a as (a,P). simpl. rewrite IHelems. reflexivity.
Qed.

Lemma on_curve_list_to_affines_length: forall (elems : seq (secp256k1_scalar_t × on_curve_t)),
  seq_len (on_curve_list_to_affines elems) = seq_len elems.
Proof.
  intros elems.
  apply seq_len_eq.
  apply on_curve_list_to_affines_length0.
Qed.

Lemma on_curve_list_to_affines_length2: forall a b (elems : seq (secp256k1_scalar_t × on_curve_t)),
  seq_len (a :: on_curve_list_to_affines elems) = seq_len (b :: elems).
Proof.
  intros.
  rewrite seq_len_eq.
  simpl.
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

Lemma on_curve_list_to_affines_rev: forall a,
  on_curve_list_to_affines (rev a) = rev (on_curve_list_to_affines a).
Proof.
  intros a.
  induction a.
  - simpl. reflexivity.
  - destruct a as (a, P).
    simpl.
    rewrite <- IHa.
    rewrite on_curve_list_to_affines_concat2.
    reflexivity.
Qed.

Lemma on_curve_list_to_affines_nth: forall i l default,
  (i < length l)%nat -> exists (a : secp256k1_scalar_t) (P : on_curve_t), nth i (on_curve_list_to_affines l) default = (a, P).
Proof.
  intros.
  revert H. revert i.
  induction l.
  - intros. simpl in H. lia.
  - intros.
    destruct a as (a, P).
    simpl.
    destruct i.
    + exists a.
      exists P.
      reflexivity.
    + assert (i < length l)%nat. { simpl in H. intuition. }
      exact (IHl i H0).
Qed.

Lemma on_curve_list_to_affines_contra: forall (l : seq (secp256k1_scalar_t × affine_t)) default,
  (forall (i : nat), (i < length l)%nat -> is_point_on_curve (snd (nth i l default)) = true)
  -> exists l2, l = on_curve_list_to_affines l2.
Proof.
  intros.
  induction l.
  - remember (@nil (secp256k1_scalar_t × on_curve_t)) as l2.
    exists l2.
    rewrite Heql2.
    reflexivity.
  - assert (forall i : nat, (i < length (l))%nat -> is_point_on_curve (snd (nth i (l) default)) = true). {
      intros.
      assert (nth (S i) (a :: l) default = nth i l default). { reflexivity. }
      rewrite <- H1.
      assert (S i < length (a :: l))%nat. {
        simpl. lia. 
      }
      exact (H (S i) H2).
    }
    destruct (IHl H0).
    assert (0 < length (a :: l))%nat. { simpl. intuition. }
    specialize (H 0%nat H2).
    destruct a as (a, p).
    simpl in H.
    remember (mkoncurve p H) as p'.
    remember ((a, p') :: x) as l2.
    exists l2.
    rewrite Heql2.
    simpl.
    rewrite <- H1.
    rewrite Heqp'.
    reflexivity.
Qed.

Lemma all_on_curve: forall l i default,
  (i < length l)%nat -> is_point_on_curve (snd (nth i (on_curve_list_to_affines l) default)) = true.
Proof.
  intros.
  pose proof on_curve_list_to_affines_nth i l default H.
  destruct H0.
  destruct H0.
  rewrite H0.
  simpl.
  exact (on_curve x0).
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

Lemma simple_batch_scalar_mult_closed: forall (elems1 : seq (secp256k1_scalar_t × on_curve_t)),
  exists (p : on_curve_t), point p = simple_batch_scalar_multiplication (on_curve_list_to_affines elems1).
Proof.
  intros elems1.
  induction elems1.
  - simpl.
    exists (mkoncurve infinity infty_on_curve).
    reflexivity.
  - destruct a as (a, P).
    simpl.
    destruct (scalar_mult_closed P a).
    destruct IHelems1.
    destruct (add_points_closed x x0).
    exists x1.
    rewrite H1.
    rewrite H.
    rewrite H0.
    reflexivity.
Qed.

Lemma simple_batch_scalar_mult_concat: forall (elems1 elems2 : seq (secp256k1_scalar_t × on_curve_t)),
simple_batch_scalar_multiplication (on_curve_list_to_affines elems1 ++ on_curve_list_to_affines elems2)
  = (simple_batch_scalar_multiplication (on_curve_list_to_affines elems1)) +' (simple_batch_scalar_multiplication (on_curve_list_to_affines elems2)).
Proof.
  intros elems1.
  induction elems1.
  - intros. simpl. rewrite add_infty_1. reflexivity.
  - intros. destruct a as (a, P).
    unfold on_curve_list_to_affines.
    fold on_curve_list_to_affines.
    assert (((a, point P) :: on_curve_list_to_affines elems1) ++ on_curve_list_to_affines elems2 = (a, point P) :: (on_curve_list_to_affines elems1 ++ on_curve_list_to_affines elems2)). { intuition. }
    rewrite H.
    unfold simple_batch_scalar_multiplication.
    fold simple_batch_scalar_multiplication.
    rewrite IHelems1.
    destruct (simple_batch_scalar_mult_closed elems1).
    rewrite <- H0.
    destruct (simple_batch_scalar_mult_closed elems2).
    rewrite <- H1.
    destruct (scalar_mult_closed P a).
    rewrite <- H2.
    rewrite add_assoc. reflexivity.
Qed.

Lemma simple_batch_scalar_prepend: forall elems1 elems2 pre,
  simple_batch_scalar_multiplication elems1 = simple_batch_scalar_multiplication elems2 -> simple_batch_scalar_multiplication (pre ++ elems1) = simple_batch_scalar_multiplication (pre ++ elems2).
Proof.
  intros.

  induction pre.
  - simpl. exact H.
  - simpl. rewrite IHpre. reflexivity.
Qed.

Definition product_sum_foldi_helper (elems : seq (secp256k1_scalar_t × affine_t)) : uint_size -> affine_t -> affine_t :=
  fun i_43 res_42 =>
    let '(ai_44, pi_45) :=
      seq_index elems (i_43) in 
    let res_42 :=
      add_points res_42 (scalar_multiplication ai_44 pi_45) in 
    res_42.

Lemma product_sum_eq_foldi_helper: forall (elems : seq (secp256k1_scalar_t × affine_t)),
  product_sum elems = foldi (usize 0) (seq_len elems) (product_sum_foldi_helper elems) infinity.
Proof.
  intros.
  unfold product_sum.
  assert ((fun (i_43 : uint_size) (res_42 : affine_t) =>
      let '(ai_44, pi_45) := seq_index elems i_43 in res_42 +' ai_44 *' pi_45) = product_sum_foldi_helper elems). {
    reflexivity.
  }
  rewrite H. reflexivity.
Qed.

Lemma product_sum_foldi_helper_ignore_last: forall elems1 elems2 acc,
  foldi (usize 0) (seq_len elems1) (product_sum_foldi_helper (elems1 ++ elems2)) acc = foldi (usize 0) (seq_len elems1) (product_sum_foldi_helper elems1) acc.
Proof.
  intros.
  remember (usize 0) as a.
  remember (Z_to_uint_size (seq_len elems1)) as b.
  assert (forall (i0 : uint_size) acc0, unsigned (a) <= unsigned (i0) < (unsigned (b)) ->
      (product_sum_foldi_helper (elems1 ++ elems2)) i0 acc0 = (product_sum_foldi_helper (elems1)) i0 acc0). {
    intros.
    unfold product_sum_foldi_helper, seq_index.
    rewrite Heqb in H.
    unfold Z_to_uint_size in H.
    destruct (seq_len elems1 <=? @max_unsigned WORDSIZE32) eqn:eq1.
    - assert (0 <= Z.of_N (seq_len elems1) <= @max_unsigned WORDSIZE32). { lia. }
      rewrite (unsigned_repr _ H0) in H.
      assert ((uint_size_to_nat i0) < length elems1)%nat. {
        unfold uint_size_to_nat. simpl.
        unfold seq_len in H.
        pose proof unsigned_range_2 i0.
        lia.
      }
      rewrite (app_nth1 _ _ _ H1).
      reflexivity.
    - assert (unsigned (@repr WORDSIZE32 (seq_len elems1)) < Z.of_N (seq_len elems1)). {
        unfold unsigned, repr, intval.
        rewrite Z_mod_modulus_eq.
        assert (0 < @modulus WORDSIZE32). { easy. }
        pose proof Zmod_pos_bound (Z.of_N (seq_len elems1)) _ H0.
        assert (@modulus WORDSIZE32 - 1 = @max_unsigned WORDSIZE32). { easy. }
        lia.
      }
      assert (uint_size_to_nat i0 < length elems1)%nat. {
        unfold seq_len in H0, H.
        unfold uint_size_to_nat.
        simpl.
        pose proof unsigned_range_2 i0.
        lia.
      }
      rewrite (app_nth1 _ _ _ H1). reflexivity.
  }
  rewrite (foldi_equiv _ a b (product_sum_foldi_helper (elems1 ++ elems2)) _ _ H). reflexivity.
Qed.

(** Equivalence proof between [product_sum] and [simple_batch_scalar_multiplication]. *)
Lemma product_sum_def: forall (elems : seq (secp256k1_scalar_t × on_curve_t)),
  0 <= seq_len elems <= @max_unsigned WORDSIZE32 -> product_sum (on_curve_list_to_affines elems) = simple_batch_scalar_multiplication (on_curve_list_to_affines elems).
Proof.
  intros.
  rewrite <- (rev_involutive (elems)).
  rewrite <- on_curve_list_to_affines_length in H.
  rewrite <- (rev_involutive elems) in H.
  revert H.
  induction (rev (elems)).
  - intros. easy.
  - intros.
    assert (rev (a :: l) = rev (l) ++ [a]). { intuition. }
    rewrite H0.
    assert (seq_len (on_curve_list_to_affines(rev l)) + 1 = seq_len (on_curve_list_to_affines(rev l ++ [a]))). {
      unfold seq_len.
      rewrite on_curve_list_to_affines_concat2.
      rewrite app_length.
      destruct a as (a, P).
      simpl.
      lia.
    }
    rewrite H0 in H.
    assert (unsigned (usize 0) <= unsigned (Z_to_uint_size(seq_len (on_curve_list_to_affines (rev l)))) < unsigned (Z_to_uint_size(seq_len (on_curve_list_to_affines (rev l ++ [a]))))). {
      assert (unsigned (usize 0) = 0). { easy. }
      rewrite H2.
      unfold seq_len, Z_to_uint_size.
      unfold seq_len in H.
      rewrite (unsigned_repr _ H).
      assert (0 <= length (on_curve_list_to_affines (rev l)) <= @max_unsigned WORDSIZE32). { unfold seq_len in H1. lia. }
      rewrite (unsigned_repr _ H3).
      rewrite on_curve_list_to_affines_concat2.
      rewrite app_length.
      destruct a as (a, P).
      simpl.
      lia.
    }
    rewrite product_sum_eq_foldi_helper.
    rewrite (foldi_split _ (usize 0) _ _ _ infinity H2).
    rewrite on_curve_list_to_affines_concat2.
    rewrite (product_sum_foldi_helper_ignore_last (on_curve_list_to_affines (rev l))).
    rewrite <- on_curve_list_to_affines_concat2.
    rewrite <- (product_sum_eq_foldi_helper (on_curve_list_to_affines (rev l))).
    rewrite IHl.
    + unfold foldi.
      unfold seq_len.
      assert (unsigned (Z_to_uint_size (length (on_curve_list_to_affines (rev l ++ [a])))) - unsigned (Z_to_uint_size(length (on_curve_list_to_affines(rev l)))) = 1). {
        unfold Z_to_uint_size.
        unfold seq_len in H.
        rewrite (unsigned_repr _ H).
        assert (0 <= length (on_curve_list_to_affines (rev l)) <= @max_unsigned WORDSIZE32). { unfold seq_len in H1. lia. }
        rewrite (unsigned_repr _ H3).
        unfold seq_len in H1.
        lia.
      }
      rewrite H3.
      simpl.
      unfold product_sum_foldi_helper.
      unfold seq_index.
      rewrite on_curve_list_to_affines_concat2.
      rewrite (app_nth2).
      * assert (uint_size_to_nat (Z_to_uint_size (length (on_curve_list_to_affines (rev l)))) - length (on_curve_list_to_affines (rev l)) = 0)%nat. {
          unfold Z_to_uint_size.
          unfold uint_size_to_nat.
          unfold from_uint_size.
          unfold nat_uint_sizable.
          assert (0 <= length (on_curve_list_to_affines (rev l)) <= @max_unsigned WORDSIZE32). { unfold seq_len in H, H1. lia. }
          rewrite (unsigned_repr _ H4).
          intuition.
        }
        rewrite H4.
        destruct a as (a, P).
        simpl.
        fold (on_curve_list_to_affines [(a, P)]).
        rewrite simple_batch_scalar_mult_concat.
        simpl.
        rewrite add_infty_2.
        reflexivity.
      * unfold Z_to_uint_size.
        unfold uint_size_to_nat.
        unfold from_uint_size.
        unfold nat_uint_sizable.
        assert (0 <= length (on_curve_list_to_affines (rev l)) <= @max_unsigned WORDSIZE32). {
          unfold seq_len in H, H1. lia.
        }
        rewrite (unsigned_repr _ H4).
        intuition.
    + unfold seq_len.
      unfold seq_len in H.
      rewrite on_curve_list_to_affines_concat2 in H.
      destruct a as (a, P).
      simpl in H.
      rewrite app_length in H.
      intuition.
Qed.

Definition opt_foldi_helper := (fun (i_36 : uint_size) (new_elems_35 : seq (secp256k1_scalar_t × affine_t)) =>
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
(new_elems_35)).

(** Admitted since this it must be a pretty trivial property and proving it is outside the scope of this project. *)
Lemma seq_upd_length: forall (A : Type) `{Default A} (l : seq A) i elem, length (seq_upd l i elem) = length l.
Proof.
Admitted.

(** Admitted since this it must be a pretty trivial property and proving it is outside the scope of this project. *)
Lemma seq_upd_split: forall (A : Type) `{Default A} (l1 l2 : seq A) elem,
  seq_upd (l1 ++ l2) (length l1) elem = l1 ++ (seq_upd l2 0 elem).
Proof.
Admitted.

(** Admitted since this it must be a pretty trivial property and proving it is outside the scope of this project. *)
Lemma seq_upd_zero: forall (A : Type) `{Default A} a l b,
  seq_upd (a :: l) 0 b = b :: l.
Proof.
Admitted.

(** Admitted since this it must be a pretty trivial property and proving it is outside the scope of this project. *)
Lemma seq_upd_on_curve: forall l i b,
  is_point_on_curve (snd b) = true -> exists l2, seq_upd (on_curve_list_to_affines l) i b = on_curve_list_to_affines l2.
Proof.
Admitted.

(*Lemma batch_scalar_opt_eq: forall (elems : seq (secp256k1_scalar_t × on_curve_t)),
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
Qed. *)

(** Large helper function that makes the next three properties trivial *)
Lemma batch_scalar_optimization_induction_helper: forall (elems : seq (secp256k1_scalar_t × on_curve_t)),
  simple_batch_scalar_multiplication (batch_scalar_optimization (on_curve_list_to_affines elems)) = simple_batch_scalar_multiplication (on_curve_list_to_affines elems)
  /\  length (batch_scalar_optimization (on_curve_list_to_affines elems)) = length (on_curve_list_to_affines elems)
  /\ exists elems2, batch_scalar_optimization (on_curve_list_to_affines elems) = on_curve_list_to_affines elems2.
Proof.
  intros elems.
  unfold batch_scalar_optimization.
  remember (fun (i_36 : uint_size) (new_elems_35 : seq (nat_mod scalar_max × affine_t)) =>
  let
  '(ai_37, pi_38) := seq_index new_elems_35 i_36 in
   let
   '(aiplus1_39, piplus1_40) := seq_index new_elems_35 (i_36 + usize 1) in
    seq_upd (seq_upd new_elems_35 i_36 (ai_37 -% aiplus1_39, pi_38))
      (i_36 + usize 1) (aiplus1_39, pi_38 +' piplus1_40)) as f.
  destruct (seq_len (on_curve_list_to_affines elems) =.? usize 0) eqn:eq1.
  - split. reflexivity. split. reflexivity. exists elems. reflexivity.
  - remember (fun seq =>
    forall (default_a : secp256k1_scalar_t × affine_t) (default_c : secp256k1_scalar_t × on_curve_t), simple_batch_scalar_multiplication seq = simple_batch_scalar_multiplication (on_curve_list_to_affines elems)
    /\ length seq = length (on_curve_list_to_affines elems) /\ exists seq2, seq = on_curve_list_to_affines seq2).
    assert (P (on_curve_list_to_affines elems)). {
      rewrite HeqP. split.
      - reflexivity.
      - split.
        + reflexivity.
        + exists elems. reflexivity.
    }
    unfold seq_len in *.
    destruct (1 <=? length (on_curve_list_to_affines elems))%nat eqn:eq2.
    + assert (forall i0 acc0, unsigned (usize 0) <= unsigned i0 < unsigned (Z_to_uint_size (length (on_curve_list_to_affines elems) - usize 1)) -> P acc0 -> P (f i0 acc0)). {
        intros.
        unfold seq_len in *.
        rewrite HeqP.
        intros.
        rewrite HeqP in H1.
        specialize (H1 default_a default_c).
        destruct H1.
        destruct H2.
        destruct H3.
        rewrite H3.
        rewrite Heqf.
        simpl.
        unfold uint_size_to_nat, from_uint_size, nat_uint_sizable.
        assert (0 <= 1 <= @max_unsigned WORDSIZE32). { easy. }
        rewrite (unsigned_repr _ H4).
        unfold usize in H0.
        simpl in H0.
        unfold uint_size_to_nat, from_uint_size, nat_uint_sizable in H0.
        rewrite (unsigned_repr _ H4) in H0.
        rewrite Z_mod_modulus_eq in H0.
        assert (0 < @modulus WORDSIZE32). { easy. }
        destruct H0.
        assert (Z.of_N (N.of_nat (Z.to_nat 1)) = 1). { lia. }
        rewrite H7 in *.
        apply (leb_complete 1 (length (on_curve_list_to_affines elems))) in eq2.
        assert (0 <= length (on_curve_list_to_affines elems) -  1). { lia. }
        pose proof Zmod_le (length (on_curve_list_to_affines elems) - 1) (@modulus WORDSIZE32) H5 H8.
        assert (Z.to_nat (unsigned i0) < length x)%nat. { rewrite <- on_curve_list_to_affines_length0. rewrite <- H3. lia. }
        rewrite <- H2 in H9.
        unfold seq_index.
        destruct (nth_split x default_c H10).
        destruct H11.
        destruct H11.
        rewrite H11.
        rewrite on_curve_list_to_affines_concat2.
        rewrite <- H12.
        rewrite <- on_curve_list_to_affines_length0.
        assert (length (on_curve_list_to_affines x0) = length (on_curve_list_to_affines x0) + 0)%nat. { intuition. }
        rewrite H13.
        rewrite (app_nth2_plus).
        rewrite <- H13.
        rewrite on_curve_list_to_affines_length0.
        rewrite H12.
        destruct (nth (Z.to_nat (unsigned i0)) x default_c) as (ai, Pi).
        simpl.
        assert (on_curve_list_to_affines x0 ++ (ai, point Pi) :: on_curve_list_to_affines x1 = (on_curve_list_to_affines x0 ++ [(ai, point Pi)]) ++ on_curve_list_to_affines x1). {
          rewrite <- app_assoc.
          intuition.
        }
        rewrite H14.
        assert (Z.to_nat (unsigned i0) + Pos.to_nat 1 = length (on_curve_list_to_affines x0 ++ [(ai, point Pi)]) + 0)%nat. {
          rewrite app_length.
          simpl.
          rewrite on_curve_list_to_affines_length0.
          lia.
        }
        rewrite H15.
        rewrite app_nth2_plus.
        destruct x1 eqn:eq3.
        - assert (Z.to_nat (unsigned i0) < length (on_curve_list_to_affines elems) - 1)%nat. { rewrite <- H2 in H6. lia. }
          rewrite <- H12 in H16.
          rewrite <- H2 in H16.
          rewrite H3, H11, on_curve_list_to_affines_concat2, app_length in H16.
          simpl in H16.
          rewrite on_curve_list_to_affines_length0 in H16.
          lia.
        - simpl.
          destruct p as (aiplus1, piplus1).
          simpl.
          split.
          + rewrite <- H12.
            rewrite <- on_curve_list_to_affines_length0.
            rewrite <- app_assoc.
            rewrite seq_upd_split.
            simpl.
            rewrite seq_upd_zero.
            assert (on_curve_list_to_affines x0 ++ (ai -% aiplus1, point Pi) :: (aiplus1, point piplus1) :: on_curve_list_to_affines l = (on_curve_list_to_affines x0 ++ [(ai -% aiplus1, point Pi)]) ++ ((aiplus1, point piplus1) :: on_curve_list_to_affines l)). {
              rewrite <- app_assoc.
              intuition.
            }
            fold secp256k1_scalar_t in *.
            rewrite H16.
            assert ((length (on_curve_list_to_affines x0 ++ [(ai, point Pi)]) + 0)%nat = length (on_curve_list_to_affines x0 ++ [(ai -% aiplus1, point Pi)])). {
              rewrite app_length, app_length. simpl. intuition.  
            }
            rewrite H17.
            rewrite seq_upd_split.
            rewrite seq_upd_zero.
            rewrite <- H1.
            rewrite H3, H11, on_curve_list_to_affines_concat2.
            rewrite <- app_assoc.
            apply simple_batch_scalar_prepend.
            simpl.
            destruct (simple_batch_scalar_mult_closed l).
            rewrite <- H18.
            rewrite scalar_mult_distributivity2.
            assert (ai -% aiplus1 = ai +% (nat_mod_neg aiplus1)). { ring. }
            rewrite H19.
            rewrite <- scalar_mult_distributivity.
            destruct (scalar_mult_closed Pi aiplus1).
            rewrite <- H20.
            destruct (scalar_mult_closed Pi (nat_mod_neg aiplus1)).
            rewrite <- H21.
            destruct (scalar_mult_closed piplus1 aiplus1).
            rewrite <- H22.
            rewrite (add_assoc x3 x5 x2).
            destruct (scalar_mult_closed Pi ai).
            destruct (add_points_closed x5 x2).
            rewrite <- H23, <- H24.
            destruct (add_points_closed x3 x7).
            rewrite <- H25.
            rewrite add_assoc.
            rewrite H25.
            rewrite <- add_assoc.
            rewrite H20, H21.
            rewrite scalar_mult_distributivity.
            assert (nat_mod_neg aiplus1 +% aiplus1 = nat_mod_zero). { ring. }
            rewrite H26.
            rewrite scalar_mult_zero.
            rewrite add_infty_1.
            reflexivity.
            reflexivity.
          + split.
            * rewrite seq_upd_length, seq_upd_length.
              rewrite <- H2.
              rewrite H3, H11, on_curve_list_to_affines_concat2. simpl. rewrite <- app_assoc. reflexivity.
            * assert ((on_curve_list_to_affines x0 ++ [(ai, point Pi)]) ++ (aiplus1, point piplus1) :: on_curve_list_to_affines l = on_curve_list_to_affines x). {
                rewrite H11, on_curve_list_to_affines_concat2. simpl. rewrite <- app_assoc. reflexivity.
              }
              rewrite H16.
              destruct (seq_upd_on_curve x (Z.to_nat (unsigned i0)) (ai -% aiplus1, point Pi)).
              --simpl. destruct Pi as (Pi, pioncurve). simpl. exact pioncurve.
              --fold secp256k1_scalar_t in *.
                rewrite H17.
                destruct (seq_upd_on_curve x2 (length (on_curve_list_to_affines x0 ++ [(ai, point Pi)]) + 0) (aiplus1, Pi +' piplus1)).
                ++destruct (add_points_closed Pi piplus1).
                  rewrite <- H18.
                  destruct x3.
                  simpl.
                  exact on_curve0.
                ++rewrite H18.
                  exists x3. reflexivity.
      }
      pose proof foldi_simple_invariant _ (usize 0) (seq_len (on_curve_list_to_affines elems) - usize 1) f (on_curve_list_to_affines elems) P H H0.
      unfold seq_len in *.
      rewrite HeqP in H1.
      specialize (H1 (nat_mod_zero, (nat_mod_zero, nat_mod_zero)) (nat_mod_zero, (mkoncurve infinity infty_on_curve))).
      exact H1.
    + unfold "=.?", N_eqdec, usize in eq1.
      simpl in eq1.
      apply (leb_complete_conv (length (on_curve_list_to_affines elems)) 1) in eq2.
      assert (length (on_curve_list_to_affines elems) = 0)%nat. { lia. }
      rewrite H0 in eq1.
      simpl in eq1.
      discriminate eq1.
Qed.

Lemma batch_scalar_optimization_eq: forall (elems : seq (secp256k1_scalar_t × on_curve_t)),
simple_batch_scalar_multiplication (batch_scalar_optimization (on_curve_list_to_affines elems)) = simple_batch_scalar_multiplication (on_curve_list_to_affines elems).
Proof.
  intros elems.
  pose proof (batch_scalar_optimization_induction_helper elems) as [H _].
  exact H.
Qed.

Lemma batch_scalar_opt_length: forall l,
  seq_len (batch_scalar_optimization (on_curve_list_to_affines l)) = seq_len l.
Proof.
  intros elems.
  pose proof (batch_scalar_optimization_induction_helper elems) as [_ [H _]].
  rewrite <- on_curve_list_to_affines_length.
  unfold seq_len.
  intuition.
Qed.

Lemma batch_scalar_opt_on_curve_list_to_affines: forall l,
  exists l2, batch_scalar_optimization (on_curve_list_to_affines l) = on_curve_list_to_affines l2.
Proof.
  intros elems.
  pose proof (batch_scalar_optimization_induction_helper elems) as [_ [_ H]].
  exact H.
Qed.

(** Equivalence proof between [batch_scalar_multiplication] and [simple_batch_scalar_multiplication]. *)
Lemma batch_scalar_mult_def: forall (elems :  seq (secp256k1_scalar_t × on_curve_t)),
  0 <= seq_len elems <= @max_unsigned WORDSIZE32 ->
  batch_scalar_multiplication (on_curve_list_to_affines elems) = simple_batch_scalar_multiplication (on_curve_list_to_affines elems).
Proof.
  intros.
  unfold batch_scalar_multiplication.
  destruct (batch_scalar_opt_on_curve_list_to_affines elems).
  rewrite H0.
  rewrite product_sum_def.
  - rewrite <- H0.
    exact (batch_scalar_optimization_eq elems).
  - rewrite <- on_curve_list_to_affines_length. rewrite <- H0. rewrite batch_scalar_opt_length. exact H.
Qed.