"""
RNA 3D Structure Prediction — Baseline (Template-Based Modeling)
Adapted from: antonoof/rna-3d-baseline-cpu (Kaggle)

Strategy:
  1. For each test sequence, find the most similar training sequences (templates)
  2. Align query → template, copy + interpolate 3D coordinates
  3. Refine geometry (bond lengths, clash removal, base-pair distances)
  4. Generate 5 diverse predictions per target
  5. Fall back to de novo helix generation when no good template exists
"""

import warnings
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning if False else DeprecationWarning)

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist, squareform

# ── Pure-Python Needleman-Wunsch (replaces Bio.pairwise2) ───────────────────
# Gotoh algorithm: global alignment with affine gap penalties O(nm) time/space

def needleman_wunsch(seq_a, seq_b, match=2, mismatch=-1, gap_open=-10, gap_extend=-0.5):
    """
    Global sequence alignment with affine gap penalties (Gotoh 1982).
    Returns (aligned_a, aligned_b, score).

    Three DP matrices:
      M[i,j] = best score when seq_a[i-1] and seq_b[j-1] are aligned
      X[i,j] = best score with a gap in seq_b (deletion in b)
      Y[i,j] = best score with a gap in seq_a (insertion in b)
    """
    n, m = len(seq_a), len(seq_b)
    NEG_INF = float('-inf')

    M = np.full((n+1, m+1), NEG_INF)
    X = np.full((n+1, m+1), NEG_INF)
    Y = np.full((n+1, m+1), NEG_INF)

    M[0, 0] = 0.0
    for i in range(1, n+1):
        X[i, 0] = gap_open + (i-1) * gap_extend
    for j in range(1, m+1):
        Y[0, j] = gap_open + (j-1) * gap_extend

    for i in range(1, n+1):
        for j in range(1, m+1):
            s = match if seq_a[i-1] == seq_b[j-1] else mismatch
            M[i,j] = max(M[i-1,j-1], X[i-1,j-1], Y[i-1,j-1]) + s
            X[i,j] = max(M[i-1,j] + gap_open, X[i-1,j] + gap_extend)
            Y[i,j] = max(M[i,j-1] + gap_open, Y[i,j-1] + gap_extend)

    score = max(M[n,m], X[n,m], Y[n,m])

    # Traceback
    a_aln, b_aln = [], []
    i, j = n, m
    state = np.argmax([M[n,m], X[n,m], Y[n,m]])  # 0=M, 1=X, 2=Y

    while i > 0 or j > 0:
        if state == 0:                             # came from M (match/mismatch)
            a_aln.append(seq_a[i-1]); b_aln.append(seq_b[j-1])
            prev = np.argmax([M[i-1,j-1], X[i-1,j-1], Y[i-1,j-1]])
            i -= 1; j -= 1; state = prev
        elif state == 1:                           # came from X (gap in seq_b)
            a_aln.append(seq_a[i-1]); b_aln.append('-')
            state = 0 if M[i-1,j] + gap_open >= X[i-1,j] + gap_extend else 1
            i -= 1
        else:                                      # came from Y (gap in seq_a)
            a_aln.append('-'); b_aln.append(seq_b[j-1])
            state = 0 if M[i,j-1] + gap_open >= Y[i,j-1] + gap_extend else 2
            j -= 1

    return ''.join(reversed(a_aln)), ''.join(reversed(b_aln)), score

# ── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/Kaggle/stanford-rna-3d-folding-2/data"
KAGGLE_DATA_PATH = "/kaggle/input/competitions/stanford-rna-3d-folding-2"  # actual mount path on Kaggle
OUT_PATH  = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/Kaggle/stanford-rna-3d-folding-2"

class Config:
    # Template search
    MAX_RELATIVE_LENGTH_DIFF = 0.5   # skip templates >50% longer/shorter than query
    MAX_ALIGN_LENGTH         = 500   # skip pairwise alignment for sequences longer than this
    ALIGNMENT_MATCH          = 2
    ALIGNMENT_MISMATCH       = -1
    ALIGNMENT_GAP_OPEN       = -10   # heavy penalty → prefers matches over gaps
    ALIGNMENT_GAP_EXTEND     = -0.5

    # Geometry refinement
    BOND_DISTANCE_TARGET       = 6.0   # Å between consecutive C1' atoms
    BOND_DISTANCE_TOL          = 0.5
    MIN_NONBOND_DISTANCE       = 3.8   # Å steric clash threshold
    BASE_PAIRING_DISTANCE_IDEAL = 12.5 # Å ideal Watson-Crick C1'–C1' distance
    BASE_PAIRING_DISTANCE_RANGE = (8.0, 14.0)

    # De novo helix fallback
    HELIX_RADIUS       = 10.0
    HELIX_RISE_PER_BASE = 2.5
    HELIX_ANGLE_STEP   = 0.6
    PAIRING_PROB_THRESHOLD = 0.7
    STEP_LENGTH_RANGE  = (3.5, 4.5)

    # Prediction diversity
    NUM_PREDICTIONS    = 5
    RANDOM_SEED_OFFSET = 1000
    DEFAULT_RELIABILITY = 0.2
    NOISE_SCALE_MIN    = 0.01
    TEMPLATE_WEIGHT    = 0.55
    RANDOM_WEIGHT      = 0.45

# ── Data loading ─────────────────────────────────────────────────────────────

import time
t0 = time.time()

print("Loading data...")
train        = pd.read_csv(f"{DATA_PATH}/train_sequences.csv")
val          = pd.read_csv(f"{DATA_PATH}/validation_sequences.csv")
test         = pd.read_csv(f"{DATA_PATH}/test_sequences.csv")
train_labels = pd.read_csv(f"{DATA_PATH}/train_labels.csv", low_memory=False)
val_labels   = pd.read_csv(f"{DATA_PATH}/validation_labels.csv", low_memory=False)

print(f"  Train: {len(train)} sequences | Val: {len(val)} | Test: {len(test)}")

# ── Step 1: Build structure lookup dict ──────────────────────────────────────
# Groups label rows by PDB ID → numpy array of shape (n_residues, 3)

def extract_structures(labels_df):
    """
    Convert flat labels CSV into dict: {target_id -> np.array shape (L,3)}
    The ID column is like '4TNA_1', '4TNA_2', etc. — we strip the residue suffix.
    Uses vectorized string ops + groupby on column (not lambda) — fast on 7.8M rows.
    """
    df = labels_df.copy()
    df['target_id'] = df['ID'].str.rsplit('_', n=1).str[0]   # vectorised
    structs = {}
    for target_id, group in df.groupby('target_id', sort=False):
        coords = group.sort_values('resid')[['x_1', 'y_1', 'z_1']].values.astype(np.float32)
        structs[target_id] = coords
    return structs

print(f"  CSVs loaded in {time.time()-t0:.1f}s")

print("Building structure lookup tables...")
t1 = time.time()
train_structs = extract_structures(train_labels)
val_structs   = extract_structures(val_labels)
print(f"  {len(train_structs)} training structures loaded in {time.time()-t1:.1f}s")

# ── Step 2: Template search ───────────────────────────────────────────────────
# Filter by length, secondary structure compatibility, then score by alignment

def secondary_structure_filter(seq_a, seq_b, min_bp=2):
    """Quick check: do seq_a and seq_b have complementary bases at ≥min_bp positions?"""
    pairing = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    count = sum(
        1 for i in range(min(len(seq_a), len(seq_b)))
        if seq_a[i] in pairing and seq_b[i] == pairing[seq_a[i]]
    )
    return count >= min_bp

def compute_alignment_score(query_seq, template_seq):
    """
    Global Needleman-Wunsch alignment (pure Python, no biopython).
    Returns ((aligned_query, aligned_template), normalised_score).
    Score is normalised by 2×min_length so it's in roughly [0, 1].
    """
    a_aln, b_aln, raw_score = needleman_wunsch(
        query_seq, template_seq,
        match=Config.ALIGNMENT_MATCH,
        mismatch=Config.ALIGNMENT_MISMATCH,
        gap_open=Config.ALIGNMENT_GAP_OPEN,
        gap_extend=Config.ALIGNMENT_GAP_EXTEND,
    )
    norm_score = raw_score / (2 * min(len(query_seq), len(template_seq)))
    return (a_aln, b_aln), norm_score

def find_comparable_seqs(query_seq, template_df, struct_dict, date_cutoff=None, top_k=5):
    """
    Return top_k (target_id, seq, score, coords) tuples sorted by alignment score.
    date_cutoff prevents using structures published after the test sequence cutoff.

    For long sequences (> MAX_ALIGN_LENGTH) we skip pairwise alignment entirely
    and return an empty list — the caller falls back to de novo generation.
    """
    if len(query_seq) > Config.MAX_ALIGN_LENGTH:
        return []   # too long to align cheaply → de novo fallback

    if date_cutoff:
        candidates = template_df[template_df['temporal_cutoff'] < date_cutoff]
    else:
        candidates = template_df

    matches = []
    for _, row in candidates.iterrows():
        tid  = row['target_id']
        tseq = row['sequence']
        if tid not in struct_dict:
            continue
        # Length filter — very short/long templates are poor models
        rel_diff = abs(len(tseq) - len(query_seq)) / max(len(tseq), len(query_seq))
        if rel_diff > Config.MAX_RELATIVE_LENGTH_DIFF:
            continue
        # Quick secondary-structure compatibility check (cheap pre-filter)
        if not secondary_structure_filter(query_seq, tseq):
            continue
        _, score = compute_alignment_score(query_seq, tseq)
        matches.append((tid, tseq, score, struct_dict[tid]))

    matches.sort(key=lambda x: x[2], reverse=True)
    return matches[:top_k]

# ── Step 3: Template morphing ─────────────────────────────────────────────────
# Align query to template, copy coordinates for matched positions,
# interpolate gaps with numpy's interp

def morph_template(dest_seq, source_seq, source_coords):
    """
    Warp source_coords to match dest_seq via global sequence alignment.
    Gaps in dest are skipped; gaps in source are filled by linear interpolation.
    Returns (L, 3) float32 array.
    """
    alignment, _ = compute_alignment_score(dest_seq, source_seq)
    if alignment is None:
        return make_de_novo_structure(dest_seq)

    aligned_dest, aligned_source = alignment

    morphed = np.full((len(dest_seq), 3), np.nan, dtype=np.float32)
    d_i = s_i = 0
    for a, b in zip(aligned_dest, aligned_source):
        if a != '-' and b != '-':                   # match/mismatch → copy coord
            if s_i < len(source_coords):
                morphed[d_i] = source_coords[s_i]
            d_i += 1; s_i += 1
        elif a != '-':                               # gap in template → d_i advances
            d_i += 1
        else:                                        # gap in query → s_i advances
            s_i += 1

    valid = ~np.isnan(morphed[:, 0])
    if not valid.any():
        return make_de_novo_structure(dest_seq)

    # Fill NaN gaps by linear interpolation along each axis
    idx = np.arange(len(morphed))
    for dim in range(3):
        morphed[:, dim] = np.interp(
            idx, idx[valid], morphed[valid, dim],
            left=morphed[valid, dim][0], right=morphed[valid, dim][-1]
        )

    return morphed

# ── Step 4: Geometry refinement ───────────────────────────────────────────────
# Push consecutive atoms to ~6 Å, repel clashing atoms, nudge base pairs

def refine_geometry(positions, sequence, reliability=1.0):
    """
    Apply light physics-based constraints to clean up morphed coordinates:
      - Bond length correction (consecutive C1' atoms ~6 Å apart)
      - Angle smoothing (no hairpin bends > ~143°)
      - Steric clash removal (no two atoms < 3.8 Å unless bonded)
      - Watson-Crick base-pair distance nudging (~12.5 Å C1'–C1')

    reliability: 0–1, from alignment score. Low reliability → stronger corrections.
    """
    refined = positions.copy()
    n = len(sequence)
    strength = 0.8 * (1.0 - min(reliability, 0.8))   # low confidence → push harder

    # Bond length
    for i in range(n - 1):
        vec  = refined[i+1] - refined[i]
        dist = np.linalg.norm(vec)
        if abs(dist - Config.BOND_DISTANCE_TARGET) > Config.BOND_DISTANCE_TOL:
            unit  = vec / (dist + 1e-10)
            delta = (Config.BOND_DISTANCE_TARGET - dist) * strength
            refined[i+1] = refined[i] + unit * (dist + delta)

    # Angle smoothing
    if n > 3:
        for i in range(1, n - 1):
            prev  = refined[i] - refined[i-1]
            nxt   = refined[i+1] - refined[i]
            denom = np.linalg.norm(prev) * np.linalg.norm(nxt) + 1e-10
            angle = np.arccos(np.clip(np.dot(prev, nxt) / denom, -1, 1))
            if angle > 2.5:    # ~143°
                smoothed    = (refined[i-1] + refined[i+1]) / 2
                refined[i]  = refined[i] * 0.3 + smoothed * 0.7

    # Steric clash removal
    dmat   = squareform(pdist(refined))
    clashes = (dmat < Config.MIN_NONBOND_DISTANCE) & (dmat > 0)
    for i in range(n):
        for j in range(i + 2, n):
            if clashes[i, j]:
                vec  = refined[j] - refined[i]
                unit = vec / (np.linalg.norm(vec) + 1e-10)
                push = (Config.MIN_NONBOND_DISTANCE - dmat[i, j]) * strength
                refined[i] -= unit * (push / 2)
                refined[j] += unit * (push / 2)

    # Watson-Crick base-pair distance
    pairs = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    for i in range(n):
        partner = pairs.get(sequence[i])
        if not partner:
            continue
        for j in range(i + 3, min(i + 20, n)):
            if sequence[j] == partner:
                d = np.linalg.norm(refined[i] - refined[j])
                lo, hi = Config.BASE_PAIRING_DISTANCE_RANGE
                if lo < d < hi:
                    delta = (Config.BASE_PAIRING_DISTANCE_IDEAL - d) * strength * 0.3
                    unit  = (refined[j] - refined[i]) / (d + 1e-10)
                    refined[i] -= unit * (delta / 2)
                    refined[j] += unit * (delta / 2)
                    break

    return refined

# ── Step 5: De novo fallback structure ────────────────────────────────────────
# When no template exists: build an idealised helix with probabilistic base pairing

def make_de_novo_structure(seq, seed=None):
    """
    Build a rough 3D structure from scratch using:
      - Idealised A-form helix parameters (radius=10Å, rise=2.5Å/base)
      - Probabilistic Watson-Crick base-pairing placement
      - Random walk for unpaired nucleotides
    """
    if seed is not None:
        np.random.seed(seed)

    n      = len(seq)
    coords = np.zeros((n, 3), dtype=np.float32)
    pairs  = {'G': 'C', 'C': 'G', 'A': 'U', 'U': 'A'}

    if n <= 3:
        for i in range(n):
            coords[i] = [i * 6.0, 0.0, 0.0]
        return coords

    # Seed first few residues in a helix
    start = min(4, n // 2)
    for i in range(start):
        angle     = i * Config.HELIX_ANGLE_STEP * 0.8
        coords[i] = [
            Config.HELIX_RADIUS * 0.7 * np.cos(angle),
            Config.HELIX_RADIUS * 0.7 * np.sin(angle),
            i * Config.HELIX_RISE_PER_BASE * 1.2
        ]

    direction = np.array([0.0, 0.1, 0.995])
    direction /= np.linalg.norm(direction)

    for i in range(start, n):
        # Look back for a complementary base within stem-forming distance
        found_pair = False
        for j in range(max(0, i - 12), i):
            if seq[j] == pairs.get(seq[i]) and (i - j) <= 8:
                if np.random.rand() < Config.PAIRING_PROB_THRESHOLD:
                    vec = coords[i-1] - coords[max(0, i-2)]
                    if np.linalg.norm(vec) > 1e-6:
                        perp = np.cross(vec, np.array([0, 0, 1]))
                        perp /= np.linalg.norm(perp) + 1e-10
                        coords[i] = coords[i-1] + perp * (Config.BASE_PAIRING_DISTANCE_IDEAL * 0.6)
                    else:
                        r = np.random.normal(0, 1, 3)
                        coords[i] = coords[i-1] + r / np.linalg.norm(r) * Config.BASE_PAIRING_DISTANCE_IDEAL * 0.6
                    direction = np.random.normal(0, 0.2, 3)
                    direction /= np.linalg.norm(direction) + 1e-10
                    found_pair = True
                    break

        if not found_pair:
            if np.random.rand() < 0.25:
                axis = np.random.normal(0, 1, 3); axis /= np.linalg.norm(axis)
                rot  = Rotation.from_rotvec(np.random.uniform(-0.3, 0.3) * axis)
                direction = rot.apply(direction)
            else:
                direction += np.random.normal(0, 0.1, 3)
                direction /= np.linalg.norm(direction) + 1e-10
            step     = np.random.uniform(*Config.STEP_LENGTH_RANGE)
            coords[i] = coords[i-1] + direction * step

    return coords

# ── Step 6: Generate 5 diverse predictions ────────────────────────────────────

def predict_structure(seq, target_id, template_df, struct_dict,
                      n_preds=5, date_cutoff=None):
    """
    Returns list of n_preds arrays, each shape (L, 3).
    Prediction 0..k: morphed templates (best alignments)
    Remaining slots: de novo fallback
    All predictions are centred at origin (mean-subtracted).
    """
    preds = []

    templates = find_comparable_seqs(seq, template_df, struct_dict,
                                     date_cutoff=date_cutoff, top_k=n_preds)

    for tid, tseq, score, tcoords in templates:
        morphed  = morph_template(seq, tseq, tcoords)
        refined  = refine_geometry(morphed, seq, reliability=score)
        noise_s  = max(Config.NOISE_SCALE_MIN, 0.8 - score)
        if score > 0.5:
            noise_s *= 0.5
        preds.append(refined + np.random.normal(0, noise_s, refined.shape))
        if len(preds) >= n_preds:
            break

    while len(preds) < n_preds:
        seed   = (hash(target_id) % 10000) + len(preds) * Config.RANDOM_SEED_OFFSET
        denovo = make_de_novo_structure(seq, seed=seed)
        refined = refine_geometry(denovo, seq, reliability=Config.DEFAULT_RELIABILITY)
        if preds:
            c0 = np.mean(preds[0], axis=0)
            refined = c0 + (refined - np.mean(refined, axis=0)) * np.random.uniform(0.8, 1.2)
        preds.append(refined)

    # Centre every prediction at origin
    return [p - np.mean(p, axis=0) for p in preds[:n_preds]]

# ── Step 7: Run predictions on test set ──────────────────────────────────────

print("\nGenerating predictions for test sequences...")
records = []

for i, row in test.iterrows():
    target_id = row['target_id']
    sequence  = row['sequence']
    cutoff    = row.get('temporal_cutoff', None)

    print(f"  [{i+1:2d}/28] {target_id:8s}  len={len(sequence):5d}  cutoff={cutoff}")

    structs = predict_structure(
        sequence, target_id,
        train, train_structs,
        n_preds=Config.NUM_PREDICTIONS,
        date_cutoff=cutoff
    )

    for resid in range(len(sequence)):
        record = {'ID': f"{target_id}_{resid+1}", 'resname': sequence[resid], 'resid': resid+1}
        for m in range(Config.NUM_PREDICTIONS):
            record[f'x_{m+1}'] = structs[m][resid][0]
            record[f'y_{m+1}'] = structs[m][resid][1]
            record[f'z_{m+1}'] = structs[m][resid][2]
        records.append(record)

# ── Step 8: Build and save submission ─────────────────────────────────────────

cols = ['ID', 'resname', 'resid']
for m in range(1, Config.NUM_PREDICTIONS + 1):
    for ax in ['x', 'y', 'z']:
        cols.append(f'{ax}_{m}')

submission = pd.DataFrame(records)[cols]
out_file   = f"{OUT_PATH}/submission_baseline.csv"
submission.to_csv(out_file, index=False)

print(f"\nSaved {len(submission)} rows → {out_file}")
print(submission.head(3).to_string())
