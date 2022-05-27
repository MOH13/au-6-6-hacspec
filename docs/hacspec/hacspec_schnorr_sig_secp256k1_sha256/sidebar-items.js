initSidebarItems({"fn":[["batch_verification","Verifies a batch of signatures and corresponding messages based on https://en.bitcoin.it/wiki/BIP_0340."],["check_ti_match_Ri","Hashes the points in R_seq and checks them against t. Assumes R_seq and t are sorted similarly i.e. if point x is on index y in R_seq, the hash of x must be on index y in t."],["compute_a_values","Helper method that computes the ‘a’ values in the MuSig scheme"],["compute_agg_R","Computes the aggregate point of all random points used in the MuSig scheme and returns this point"],["compute_agg_pk","Computes the “aggregate” public key from the signers public keys and their respective a values. Assumes L and a are sorted similarly."],["compute_agg_s",""],["compute_own_s","Computes the specific signer’s s value"],["concat_byte_seqs_to_single_byte_seq","Helper method that transforms a sequence of byte sequences into one byte sequence"],["multi_sig_verify",""],["public_keys_to_byte_seqs","Helper method that transforms a sequence of Affine points into byte sequences"],["sign","Creates a Schnorr signature for a single signer."],["valid_As",""],["verify","Given a public key and a message, this method verifies the signature of the message"]]});