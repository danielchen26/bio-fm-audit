"""The public STFlow code (commit 880c2ee) does not start: TransformerBlock passes non_negative=... to GeneUpdate, whose
constructor has no such argument and no non-negativity code. Smallest repair: drop the unused keyword, keeping GeneUpdate
exactly as released."""
import shutil
P = "STFlow/stflow/model/transformer.py"
shutil.copy(P, P + ".orig")
s = open(P).read()
a = "GeneUpdate(d_model, n_genes, proj_drop=proj_drop, non_negative=gene_exp_non_negative)"
assert s.count(a) == 1
open(P, "w").write(s.replace(a, "GeneUpdate(d_model, n_genes, proj_drop=proj_drop)"))
print("patched")
