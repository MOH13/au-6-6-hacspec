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
Require ZArithRing.
From Coqprime Require GZnZ.

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
Notation "k '*'' p" := (scalar_multiplication k p) (at level 4, right associativity).

Notation elem_max := 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F.
Notation scalar_max := 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141.

Lemma nat_mod_small: forall (max : Z), forall (n : nat_mod max), max > 0 -> 0 <= n < max.
Proof.
  intros max n H.
  pose proof inZnZ max n.
  pose proof Z_mod_lt n max H.
  rewrite <- H0 in H1.
  exact H1.
Qed.

Lemma field_elem_small: forall (n : secp256k1_field_element_t), 0 <= n < elem_max.
Proof.
  intros n.
  assert (H: elem_max > 0). { lia. }
  apply (nat_mod_small elem_max n H).
Qed.

Lemma scalar_small: forall (n : secp256k1_scalar_t), 0 <= n < scalar_max.
Proof.
  intros n.
  assert (H: scalar_max > 0). { lia. }
  apply (nat_mod_small scalar_max n H).
Qed.

Lemma small_to_nat_mod: forall (n : Z), forall (a : Z), 0 <= a < n -> exists (b : nat_mod n), a =? b = true.
Proof.
  intros n a H.
  pose proof (Zmod_small a n H).
  symmetry in H0.
  remember (mkznz n a H0) as b.
  assert (H1: (a =? b) = true). {
    unfold "=?".
    unfold val.
    rewrite -> Heqb.
    destruct a eqn:eq1. {
      reflexivity.
    } {
      rewrite -> Pos.eqb_refl.
      reflexivity.
    } {
      rewrite -> Pos.eqb_refl.
      reflexivity.
    }
  }
  exists b.
  exact H1.
Qed.

Lemma nat_mod_val_in_range: forall (n : Z) (a : nat_mod n) (b : Z), 0 <= a < b -> 0 <= val n a < b.
Proof.
  intros n a b H.
  exact H.
Qed.

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

Lemma zero_less_than_elem_max: 0 < elem_max.
Proof.
  lia.
Qed.

Lemma nat_mod_double_neg: forall (n : secp256k1_field_element_t), nat_mod_neg (nat_mod_neg n) = n.
Proof.
  intros n.
  destruct (nat_mod_val elem_max n =? 0) eqn:eq2. {
    unfold nat_mod_val in eq2.
    rewrite -> Z.eqb_eq in eq2.
    unfold nat_mod_neg.
    unfold opp.
    rewrite -> eq2.
    simpl.
    symmetry in eq2.
    destruct n as [n' inZnZ'].
    pose proof (zirr elem_max (0 mod elem_max) n' (modz elem_max 0) inZnZ' eq2).
    exact H.
  } {
    unfold nat_mod_val in eq2.
    rewrite -> Z.eqb_neq in eq2.
    rewrite -> inZnZ in eq2.
    unfold nat_mod_neg.
    unfold opp.
    simpl.
    rewrite -> (Z_mod_nz_opp_full (val elem_max n) elem_max eq2).
    rewrite <- (inZnZ _ n).
    pose proof (field_elem_small n).
    destruct H as [_ H1].
    pose proof (pos_diff n elem_max H1).
    assert (H2: 0<= elem_max - (val elem_max n) < elem_max). {
      rewrite -> Z.gt_lt_iff in H.
      apply Z.lt_le_incl in H.
      rewrite <- inZnZ in eq2.
      pose proof field_elem_small n.
      lia.
    }
    apply Zmod_small in H2.
    rewrite <- H2 in H.
    assert (H3: (elem_max - (val elem_max n)) <> 0). { lia. }
    rewrite <- H2 in H3.
    assert (H4: Z.opp (elem_max - n) mod elem_max = n). {
      rewrite (Z_mod_nz_opp_full (elem_max - (val elem_max n)) elem_max H3).
      rewrite H2.
      lia.
    }
    destruct n as [n' inZnZ'].
    pose proof (zirr _ (Z.opp (elem_max - n') mod elem_max) n' (modz elem_max (Z.opp (elem_max - n'))) (inZnZ') H4) as H5.
    apply H5.
  }
Qed.

Lemma nat_mod_neg_inj: forall (n m : secp256k1_field_element_t), nat_mod_neg n = nat_mod_neg m -> n = m.
Proof.
  intros n m H.
  destruct n as [n' inZnZn'] eqn:eqN.
  destruct m as [m' inZnZm'] eqn:eqM.
  destruct (n' =? m') eqn:eq1. {
    rewrite Z.eqb_eq in eq1.
    apply zirr.
    apply eq1.
  } {
    destruct (n' =? 0) eqn:eq2. {
      rewrite Z.eqb_eq in eq2.
      unfold nat_mod_neg in H.
      unfold opp in H.
      simpl in H.
      rewrite -> eq2 in H.
      destruct (n' =? m') eqn:eq3. {
        discriminate eq1.
      } {
        simpl in H.
        assert (H1: m' <> 0). {
          rewrite -> Z.eqb_neq in eq3.
          pose proof Z.neq_sym n' m' eq3 as eq4.
          rewrite -> eq2 in eq4.
          exact eq4.
        }
        rewrite -> inZnZm' in H1.
        pose proof (Z_mod_nz_opp_full (m') elem_max H1) as H2.
        assert (H3: 0 mod elem_max <> -m' mod elem_max). {
          rewrite Zmod_0_l.
          rewrite H2.
          rewrite <- inZnZm'.
          assert (H4: m' < elem_max). {
            pose proof (field_elem_small m).
            rewrite eqM in H0.
            simpl in H0.
            lia.
          }
          lia.
        }
        rewrite Zmod_0_l in H3.
        inversion H.
        intuition.
      }
    } {
      unfold nat_mod_neg in H.
      unfold opp in H.
      simpl in H.
      rewrite -> Z.eqb_neq in eq2.
      rewrite inZnZn' in eq2.
      pose proof (Z_mod_nz_opp_full n' elem_max eq2) as H1.
      inversion H.
      rewrite H1 in H2.
      rewrite <- inZnZn' in H2.
      destruct (m' =? 0) eqn:eq3. {
        rewrite Z.eqb_eq in eq3.
        rewrite eq3 in H2.
        rewrite Zmod_0_l in H2.
        pose proof field_elem_small n.
        rewrite eqN in H0.
        simpl in H0.
        lia.
      } {
        rewrite Z.eqb_neq in eq3.
        rewrite inZnZm' in eq3.
        pose proof (Z_mod_nz_opp_full m' elem_max eq3) as H3.
        rewrite H3 in H2.
        rewrite <- inZnZm' in H2.
        lia.
      } 
    }
  }
Qed.

Lemma nat_mod_neg_both: forall (n m : secp256k1_field_element_t), n = m <-> nat_mod_neg n = nat_mod_neg m.
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

Lemma double_neg: forall (p : affine_t), p = neg_point (neg_point p).
Proof.
  intros p.
  unfold neg_point.
  destruct p as (px, py).
  pose proof nat_mod_double_neg py.
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
      pose proof nat_mod_neg_inj py qy.
      rewrite H1 in H.
      intuition.
Qed.

Lemma neg_symm_left: forall (p q : affine_t), neg_point p =.? q = true -> p =.? neg_point q = true.
Proof.
  intros p q H.
  rewrite -> eqb_leibniz in H.
  rewrite <- H.
  apply double_neg.
Qed.

Lemma neg_symm: forall (p q : affine_t), (neg_point p =.? q) = (p =.? neg_point q).
Proof.
  intros p q.
  destruct (neg_point p =.? q) eqn:eq1. {
    symmetry.
    apply neg_symm_left.
    apply eq1.
  } {
    pose proof double_neg p as eq2.
    rewrite -> eqb_leibniz in eq2.
    rewrite -> eq2.
    rewrite <- neg_both.
    rewrite -> eq1.
    reflexivity.
  }
Qed.

Lemma add_infty_1: forall (p: affine_t), infinity +' p = p.
Proof.
  intros p.
  unfold add_points.
  destruct (is_infinity infinity) eqn:eq.
  - reflexivity.
  - discriminate eq.
Qed.

Lemma add_different_comm: forall (p q : affine_t), add_different_points p q = add_different_points q p.
Proof.
  intros p q.
  unfold add_different_points.
Admitted.

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
          destruct (neg_point (px, py)) as (a0, b0) eqn:eq3.
          rewrite -> curve_eq_symm with (px:= qx) (py:= qy).
          rewrite <- eq3.
          rewrite -> neg_symm.
          rewrite -> H2.
          reflexivity.
        } {
          destruct (neg_point (px, py)) as (a0, b0) eqn:eq3.
          rewrite -> curve_eq_symm with (px:= qx) (py:= qy).
          rewrite <- eq3.
          rewrite -> neg_symm.
          rewrite -> H2.
          apply add_different_comm.
          (*Missing case: P != Q /\ P != -Q /\ P != infty /\ Q != infty*)
          (*I.e. add_different_points*)
        }
      }
    }
  }
Qed.

Lemma add_infty_2: forall (p: affine_t), p +' infinity =.? p = true.
Proof.
  intros p.
  Search (Z.to_nat ?a).
  apply add_comm.
Qed.

Lemma add_assoc: forall (p q r : affine_t), (p +' q) +' r = p +' (q +' r).
Proof.
  intros p q r.
  destruct (is_infinity p) eqn:eq1. {
    apply (is_infty_means_infty p) in eq1.
    rewrite eqb_leibniz in eq1.
    rewrite eq1.
    rewrite <- eqb_leibniz.
    apply add_infty_1.
  } {
    destruct (is_infinity q) eqn:eq2. {
      apply (is_infty_means_infty q) in eq2.
      rewrite eqb_leibniz in eq2.
      rewrite eq2.
      rewrite <- eqb_leibniz.
      apply add_infty_2.
    }
  }
Admitted.

Fixpoint simple_scalar_mult (k : nat) (p : affine_t) : affine_t :=
  match k with
  | 0%nat => infinity
  | S k1  => (simple_scalar_mult (k1) p) +' p
  end.

Lemma nat_mod_to_nat_id: forall (max : Z), forall (n : nat_mod max), max > 0 -> Z.of_nat (Z.to_nat n) = n.
Proof.
  intros max n H.
  pose proof inZnZ max n.
  apply (nat_mod_small max n) in H.
  destruct H as [H1 _].
  apply (Z2Nat.id n H1).
Qed.

Lemma scalar_to_nat_id: forall (n : secp256k1_scalar_t), Z.of_nat (Z.to_nat n) = n.
Proof.
  intros n.
  assert (H: scalar_max > 0). { lia. }
  apply (nat_mod_to_nat_id scalar_max n H).
Qed.

Search (?a mod ?b < ?b).

Lemma nat_to_scalar_lem: forall (n : nat), exists (n' : secp256k1_scalar_t), n' =? (Z.of_nat n) mod scalar_max = true.
Proof.
  intros n.
  remember (Z.of_nat n mod scalar_max) as x.
  assert (H1: 0 < scalar_max). { lia. }
  pose proof (Z.mod_pos_bound (Z.of_nat n) scalar_max H1) as H2.
  rewrite <- Heqx in H2.
  pose proof (small_to_nat_mod scalar_max x H2) as H3.
  destruct H3 as [b G].
  rewrite -> Z.eqb_sym in G.
  exists b.
  exact G.
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
    auto.
  } {
    simpl.
  }
Admitted.

Lemma simple_scalar_mult_distributivity: forall (k1 k2 : nat) (p: affine_t), (simple_scalar_mult k1 p) +' (simple_scalar_mult k2 p) = (simple_scalar_mult (k1 + k2) p).
Proof.
  intros k1 k2 p.
  induction k2 as [|k' IHk']. {
    simpl.
    pose proof add_infty_2 (simple_scalar_mult k1 p) as H1.
    rewrite eqb_leibniz in H1.
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

Lemma scalar_mult_helper: forall (k1 k2 : secp256k1_scalar_t) (p: affine_t), nat_mod_val scalar_max k1 = nat_mod_val scalar_max k2 -> k1 *' p = k2 *' p.
Proof.
  intros k1 k2 p H.
  unfold nat_mod_val in H.
  unfold "*'".
  unfold nat_mod_get_bit.
  unfold nat_mod_bit.
  rewrite -> H.
  reflexivity.
Qed.

Lemma scalar_mult_distributivity: forall (k1 k2 : secp256k1_scalar_t) (p: affine_t), k1 *' p +' k2 *' p = (k1 +% k2) *' p.
Proof.
  intros k1 k2 p.
  assert (H0: nat_mod_val scalar_max (nat_to_scalar (Z.to_nat k1)) = nat_mod_val scalar_max k1). {
    simpl.
    unfold nat_to_scalar.
    rewrite -> Z2Nat.id.
    - rewrite <- (inZnZ scalar_max k1).
      unfold nat_mod_val.
      reflexivity.
    - apply nat_mod_small.
      lia.
  }
  assert (H1: nat_mod_val scalar_max (nat_to_scalar (Z.to_nat k2)) = nat_mod_val scalar_max k2). {
    simpl.
    unfold nat_to_scalar.
    rewrite -> Z2Nat.id.
    - rewrite <- (inZnZ scalar_max k2).
      unfold nat_mod_val.
      reflexivity.
    - apply nat_mod_small.
      lia.
  }
  rewrite <- (scalar_mult_helper (nat_to_scalar (Z.to_nat k1)) k1 p H0).
  rewrite <- (scalar_mult_helper (nat_to_scalar (Z.to_nat k2)) k2 p H1).
  rewrite scalar_mult_def.
  rewrite scalar_mult_def.
  assert (H2: nat_mod_val scalar_max (nat_to_scalar (Z.to_nat (k1 +%k2))) = nat_mod_val scalar_max (k1 +% k2)). {
    simpl.
    rewrite -> Z2Nat.id.
    - rewrite -> Zmod_mod.
      reflexivity.
    - apply Z.mod_pos_bound.
      lia.
  }
  rewrite <- (scalar_mult_helper (nat_to_scalar (Z.to_nat (k1 +% k2))) (k1 +% k2) p H2).
  rewrite scalar_mult_def.
  rewrite simple_scalar_mult_distributivity.
  rewrite simple_scalar_mult_mod.
  unfold "+%".
  unfold add.
  unfold Z.to_nat.
  unfold val.
  assert (H3: scalar_max > 0). { lia. }
  destruct k1 as [k1' k1inz] eqn:eq1.
  destruct k2 as [k2' k2inz] eqn:eq2.
  assert (H4: 0 <= k1'). {
    pose proof nat_mod_small scalar_max k1 H3 as [H _].
    unfold "<=".
    unfold "?=".
    unfold "<=" in H.
    unfold "?=" in H.
    unfold val in H.
    rewrite eq1 in H.
    exact H.
  }
  assert (H5: 0 <= k2'). {
    pose proof nat_mod_small scalar_max k2 H3 as [H _].
    unfold "<=".
    unfold "?=".
    unfold "<=" in H.
    unfold "?=" in H.
    unfold val in H.
    rewrite eq2 in H.
    exact H.
  }
  assert (H6: 0 <= k1' + k2'). { lia. }
  assert (H7: 0 <= scalar_max). { lia. }
  fold (Z.to_nat k1').
  fold (Z.to_nat k2').
  fold (Z.to_nat ((k1'+k2') mod scalar_max)).
  rewrite -> (Z2Nat.inj_mod (k1' + k2') scalar_max H6 H7).
  rewrite -> (Z2Nat.inj_add k1' k2' H4 H5).
  unfold BinInt.Z.to_nat.
  reflexivity.
Qed.