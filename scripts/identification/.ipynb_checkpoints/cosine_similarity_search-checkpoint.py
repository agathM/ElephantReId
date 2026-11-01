"""
Recherche des plus proches voisins par similarité cosinus sur les features
précalculées, sans tête de classification.

Pour chaque crop de l'ensemble de VALIDATION, ce script retrouve les TOP_K
features les plus proches (similarité cosinus) dans la galerie retenue et
écrit un CSV de type `val_pred.csv` enrichi de :

    label_1 ... label_K   : individu du K-ième plus proche voisin
    sim_1  ... sim_K      : similarité cosinus correspondante
    pred_label           : label_1 (permet de réutiliser tel quel
                           scripts/classifier/results/analyze_splits.py,
                           qui lit la colonne `pred_label`)

Toutes les colonnes de métadonnées de `val_pred.csv` (run, indiv_name, camera,
mode, video, entry_day, split, crop_id) sont conservées, dans le même ordre
que dans train_runs.py.

Les features sont lues dans le cache déjà calculé par train_runs.py :
    <FEATURES_PATH>/features.pt    torch.Tensor (N, dim) aligné sur
    <FEATURES_PATH>/crop_ids.pt    liste des crop_ids, ligne i <-> feature i
Aucun modèle n'est chargé et aucune image n'est relue : le script ne fait que
lire ce cache, il doit donc avoir été produit au préalable par train_runs.py.

Choix de la galerie (paramètre GALLERY_SPLIT) :
    "train" : les crops TRAIN du même run (protocole classique de
              ré-identification : requête = validation, galerie = train).
    "all"   : tous les crops du run, validation comprise. Dans ce cas chaque
              requête figure elle-même dans la galerie ; ces appariements sont
              exclus (similarité mise à -inf) avant le tri.

Pour une autre expérience : modifier les paramètres ci-dessous, puis lancer
le script.
"""

import os

import numpy as np
import pandas as pd
import torch

from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

# ============================================================
# PARAMETERS
# ============================================================
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pca_reduc = "1800"
features_name = "brightness_normalize_pad_miewid"
head = "_entry_day_FC"
OUT_DIR_NAME = features_name + head
version = "v9"
crop_type = "Ear"
mode = "Day"
RESULTS_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/results/classifiers/") / version / crop_type / OUT_DIR_NAME
os.makedirs(RESULTS_PATH, exist_ok=True)
FEATURES_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/precomputed_features") / version / crop_type / features_name
os.makedirs(FEATURES_PATH, exist_ok=True)

# CSV des runs : colonnes run, entry, split (train/val)
SPLIT_FILE = Path(
    "/sps/mnhn/ameurisse/ElephantReId/"
    "data/processed/crops/classifiers/split/"
    "all_0.8_h_1_v_1.6_nc_2.7.csv"
)

RUNS = [97, 57, 98, 94, 6]

CROPS_FILE = Path(
    "/sps/mnhn/ameurisse/ElephantReId/"
    "data/processed/crops/classifiers/split/crops_0.8_h_1_v_1.6_nc_2.7.csv"
)
entry_col = "entry_day"
# Nombre de plus proches voisins à conserver.
TOP_K = 10

# Galerie dans laquelle les voisins sont cherchés : "train" ou "all".
GALLERY_SPLIT = "train"

# Nom du fichier écrit. Volontairement différent de "val_pred.csv" pour ne
# pas écraser les prédictions de la tête FC produites par train_runs.py.
OUT_NAME = f"val_pred_knn_{pca_reduc}.csv"

# Nombre de requêtes traitées en parallèle. La matrice de similarité d'un
# bloc vaut chunk_size x len(gallery) ; 2048 x 10 000 floats = ~80 Mo.
CHUNK_SIZE = 2048

# Décimales conservées pour les similarités (allège le CSV).
SIM_DECIMALS = 4

# Colonnes de métadonnées, dans l'ordre de train_runs.py.
METADATA_COLS = [
    "run",
    "indiv_name",
    "camera",
    "mode",
    "video",
    "entry_day",
    "split",
    "crop_id",
]


# ============================================================
# DATA
# ============================================================

def get_entry_col(df: pd.DataFrame, source: Path) -> str:
    """Nom de la colonne d'entry, acceptée sous `entry_day` ou `entry`.

    Les fichiers récents écrivent `entry_day`, les plus anciens `entry` : on
    accepte les deux pour pouvoir réutiliser des fichiers déjà produits.
    """
    for col in ("entry_day", "entry"):
        if col in df.columns:
            return col
    raise ValueError(
        f"Colonne d'entry absente de {source} — colonnes attendues : "
        f"'entry_day' ou 'entry'"
    )


def load_crops() -> tuple:
    """Crops (crop_id, individu, caméra, mode, vidéo, entry), nettoyés.

    Mêmes filtres que train_runs.py : individu non vide, exclusion de "X",
    filtre sur le mode, un seul crop_id par crop.
    """
    crops_df = pd.read_csv(CROPS_FILE).dropna(subset=["indiv_name", "crop_id"])
    crops_df = crops_df[crops_df.indiv_name != "X"]
    if mode and "mode" in crops_df.columns:
        crops_df = crops_df[crops_df["mode"] == mode]
    # Un crop n'appartient qu'à une seule entry : on retire les doublons
    crops_df = crops_df.drop_duplicates(subset=["crop_id"])

    entry_col = get_entry_col(crops_df, CROPS_FILE)
    print(f"Crops : {len(crops_df)} ({crops_df['indiv_name'].nunique()} individus, "
          f"{crops_df[entry_col].nunique()} entries), colonne d'entry '{entry_col}'")

    return crops_df, entry_col


def load_split_dfs(crops_df: pd.DataFrame, entry_col: str) -> tuple:
    """Un dataframe de crops par run, avec la colonne `split` (train/val).

    Le merge se fait sur la colonne d'entry, comme dans train_runs.py.
    """
    split_df = pd.read_csv(SPLIT_FILE).filter(['run',entry_col,'split']).drop_duplicates()
    for col in ("run", entry_col, "split"):
        if col not in split_df.columns:
            raise ValueError(f"Colonne '{col}' absente de {SPLIT_FILE}")
    split_df = split_df[split_df.run.isin(RUNS)]

    missing = sorted(set(RUNS) - set(split_df["run"].unique()))
    if missing:
        raise ValueError(
            f"Run(s) {missing} absents de {SPLIT_FILE} — runs disponibles : "
            f"{sorted(split_df['run'].unique())}"
        )

    runs = sorted(split_df["run"].unique())
    print(f"Nombre de runs trouvés dans {SPLIT_FILE.name} : {len(runs)}")

    split_dfs = {}
    for run in runs:
        split_dfs[run] = split_df[split_df["run"] == run][
            ["run", entry_col, "split"]
        ].merge(
            crops_df,
            on=entry_col,
            how="inner",
        )

    return runs, split_dfs


def load_features() -> tuple:
    """Features précalculées et table crop_id -> indice de ligne.

    Le cache peut contenir plus de crop_ids que ceux de cette expérience
    (train_runs.py l'alimente au fil des lancements) : seuls ceux présents
    dans le split courant seront sélectionnés ensuite.
    """
    features_path = FEATURES_PATH / "features.pt"
    crop_ids_path = FEATURES_PATH / "crop_ids.pt"

    for path in (features_path, crop_ids_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} introuvable. Lancer d'abord scripts/train_runs.py "
                f"pour calculer les features ({features_name})."
            )

    print(f"Chargement des features : {features_path}")
    features = torch.load(features_path, weights_only=False)
    cached_crop_ids = torch.load(crop_ids_path, weights_only=False)

    if not torch.is_tensor(features) or features.ndim != 2:
        raise RuntimeError(
            f"{features_path} ne contient pas un tenseur 2D : "
            f"{type(features)} de forme {getattr(features, 'shape', None)}"
        )
    if features.shape[0] != len(cached_crop_ids):
        raise RuntimeError(
            f"Cache désaligné : {features.shape[0]} features pour "
            f"{len(cached_crop_ids)} crop_ids. Relancer train_runs.py."
        )

    print(f"  {features.shape[0]} features de dimension {features.shape[1]}")

    return features, {cid: i for i, cid in enumerate(cached_crop_ids)}


# ============================================================
# RECHERCHE DES PLUS PROCHES VOISINS
# ============================================================

def topk_cosine(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_crop_ids: list,
    gallery_crop_ids: list,
    top_k: int,
) -> tuple:
    """K plus proches voisins de chaque query dans la galerie (cosinus).

    Les features sont normalisées (L2), la similarité cosinus est donc le
    simple produit scalaire. Les requêtes sont traitées par blocs pour ne pas
    materialiser une matrice query x galerie de taille prohibitive.

    Une requête présente elle-même dans la galerie (cas de GALLERY_SPLIT
    "all") est exclue de ses propres voisins.

    Returns
    -------
    (top_idx, top_sim) : deux np.ndarray de forme (n_queries, k)
        top_idx[:, j] : indice de colonne dans la galerie
        top_sim[:, j] : similarité correspondante, décroissante
    """
    n_gallery = gallery_features.shape[0]
    k = min(top_k, n_gallery)
    if k < top_k:
        print(f"  Galerie de {n_gallery} éléments : k réduit à {k}")

    device = DEVICE
    query_features = torch.as_tensor(query_features, dtype=torch.float32)
    queries = torch.nn.functional.normalize(
        query_features, dim=1 #.to(device=device, dtype=torch.float32)
    )
    gallery_features = torch.as_tensor(gallery_features, dtype=torch.float32)
    gallery = torch.nn.functional.normalize(
        gallery_features, dim=1 #.to(device=device, dtype=torch.float32)
    )

    # Position de chaque crop dans la galerie, pour pouvoir s'exclure soi-même.
    gallery_position = {cid: i for i, cid in enumerate(gallery_crop_ids)}

    top_idx = np.empty((queries.shape[0], k), dtype=np.int64)
    top_sim = np.empty((queries.shape[0], k), dtype=np.float32)

    for start in range(0, queries.shape[0], CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, queries.shape[0])
        sims = queries[start:stop] @ gallery.T

        # Un crop ne doit pas être son propre voisin : on pousse sa similarité
        # à -inf pour que le tri l'ignore.
        for local, cid in enumerate(query_crop_ids[start:stop]):
            pos = gallery_position.get(cid)
            if pos is not None:
                sims[local, pos] = -torch.inf

        values, indices = torch.topk(sims, k, dim=1)
        top_idx[start:stop] = indices.cpu().numpy()
        top_sim[start:stop] = values.cpu().numpy()

        del sims

    return top_idx, top_sim


def build_rows(
    val_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    entry_col: str,
) -> pd.DataFrame:
    """Un dataframe de voisins par crop de validation, pour un run.

    `val_df` doit être déjà filtré (features disponibles, un seul row par
    crop) et aligné avec `query_features` ; `gallery_df` aligné avec
    `gallery_features`.
    """
    query_crop_ids = val_df["crop_id"].tolist()
    gallery_crop_ids = gallery_df["crop_id"].tolist()
    gallery_labels = gallery_df["indiv_name"].to_numpy()

    top_idx, top_sim = topk_cosine(
        query_features,
        gallery_features,
        query_crop_ids,
        gallery_crop_ids,
        TOP_K,
    )

    metadata_cols = [
        entry_col if col == "entry_day" else col for col in METADATA_COLS
    ]
    missing = [col for col in metadata_cols if col not in val_df.columns]
    if missing:
        raise ValueError(
            f"Colonnes de métadonnées absentes des crops : {missing}"
        )

    rows = val_df[metadata_cols].reset_index(drop=True).copy()

    for rank in range(top_idx.shape[1]):
        pos = rank + 1
        rows[f"label_{pos}"] = gallery_labels[top_idx[:, rank]]
        rows[f"sim_{pos}"] = np.round(top_sim[:, rank], SIM_DECIMALS)

    # `pred_label` = label_1 : les scripts d'analyse existants (analyze_splits.py)
    # lisent cette colonne et peuvent donc être lancés sur ce fichier tel quel.
    rows["pred_label"] = rows["label_1"]

    return rows


def get_rank_k(rows: pd.DataFrame) -> int:
    """Nombre de voisins effectivement écrits (galerie < TOP_K)."""
    return len([c for c in rows.columns if c.startswith("label_")])


def print_run_accuracy(rows: pd.DataFrame) -> None:
    """Top-1 / top-K : individu attendu présent dans les K voisins."""
    k = get_rank_k(rows)
    truth = rows["indiv_name"].to_numpy()
    labels = rows[[f"label_{i + 1}" for i in range(k)]].to_numpy()

    top1 = float((labels[:, 0] == truth).mean() * 100)
    topk = float((labels == truth[:, None]).any(axis=1).mean() * 100)

    print(f"  Top-1 : {top1:.2f}%  |  Top-{k} : {topk:.2f}%")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(f"{crop_type} - {mode}")
    print(f"Device : {DEVICE}")
    print(f"Galerie : {GALLERY_SPLIT}  |  top_k = {TOP_K}")

    if GALLERY_SPLIT not in ("train", "all"):
        raise ValueError(
            f"GALLERY_SPLIT={GALLERY_SPLIT} inconnu : utiliser 'train' ou 'all'."
        )

    crops_df, entry_col = load_crops()
    runs, split_dfs = load_split_dfs(crops_df, entry_col)
    all_features, crop_to_global_idx = load_features()

    all_rows = []

    for run in runs:
        print(f"\n===== Run {run}/{len(runs)} =====")

        df = split_dfs[run]

        val_df = df[df["split"] == "val"]
        # "all" = train + val : les queries se trouvent alors elles-mêmes dans
        # la galerie, ce que topk_cosine exclut.
        gallery_df = df if GALLERY_SPLIT == "all" else df[df["split"] == GALLERY_SPLIT]

        # (run, entry_day) n'est pas unique dans le fichier de splits (une
        # ligne par individu) : le merge sur l'entry duplique donc certains
        # crops. Un même crop val apparaît plusieurs fois, ce qui n'a pas de
        # sens pour une recherche de voisins -> on n'en garde qu'une ligne.
        n_before = len(val_df)
        val_df = val_df.drop_duplicates(subset=["crop_id"])
        gallery_df = gallery_df.drop_duplicates(subset=["crop_id"])

        # Ne garder que les crop_ids dont on a les features
        val_df = val_df[val_df["crop_id"].isin(crop_to_global_idx)]
        gallery_df = gallery_df[gallery_df["crop_id"].isin(crop_to_global_idx)]

        if val_df.empty or gallery_df.empty:
            raise ValueError(
                f"Run {run} : split vide après filtrage des features "
                f"(val={len(val_df)}, galerie={len(gallery_df)}). "
                f"Vérifier le contenu de {CROPS_FILE.name} pour les "
                f"entries de ce run."
            )

        if n_before > len(val_df):
            print(f"  {n_before - len(val_df)} lignes de val dupliquees "
                  f"(entry en double dans {SPLIT_FILE.name}) : un seul row "
                  f"par crop conserve")

        val_gidx = torch.tensor(
            [crop_to_global_idx[cid] for cid in val_df["crop_id"]],
            dtype=torch.long,
        )
        gallery_gidx = torch.tensor(
            [crop_to_global_idx[cid] for cid in gallery_df["crop_id"]],
            dtype=torch.long,
        )
        pca = PCA(n_components=int(pca_reduc))
        X = pca.fit_transform(all_features[gallery_gidx])
        Y = pca.transform(all_features[val_gidx])
        print(
            f"  Requêtes (val) : {len(val_df)}  |  Galerie : "
            f"{len(gallery_df)}  |  Individus : "
            f"{df['indiv_name'].nunique()}"
        )

        rows = build_rows(
            val_df,
            gallery_df,
            Y,
            X,
            entry_col,
        )

        print_run_accuracy(rows)

        all_rows.append(rows)

    val_pred_df = pd.concat(all_rows, ignore_index=True)

    out_dir = RESULTS_PATH
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / OUT_NAME
    val_pred_df.to_csv(out_path, index=False)

    k = get_rank_k(val_pred_df)

    print(f"\nVoisins les plus proches écrits dans : {out_path}")
    print(f"  {len(val_pred_df)} lignes, {val_pred_df['run'].nunique()} runs")
    print(f"  Colonnes : {', '.join(val_pred_df.columns)}")
    print_run_accuracy(val_pred_df)


if __name__ == "__main__":
    main()