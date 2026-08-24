from pathlib import Path


OUTPUT_ROOT = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/processed/"
    "crops/classifiers/Head/v3"
)

SOURCE_ROOT = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/processed/"
    "crops/classifiers/Head/v2"
)


# ============================================================
# Fichiers
# ============================================================

source_files = [
    p for p in SOURCE_ROOT.rglob("*")
    if p.is_file()
]

output_files = [
    p for p in OUTPUT_ROOT.iterdir()
    if p.is_file()
]


# ============================================================
# Extraire uniquement :
#   video ID + frame ID
#
# Exemples :
#
# video_SEB_125081_frame_000015_Head_0.png
#       -> ("125081", "000015")
#
# SEB_125081_frame_000015_head_5345.png
#       -> ("125081", "000015")
# ============================================================

def source_key(name):
    parts = name.split("_")

    # SEB_125081_frame_000015_head_5345.png
    return parts[2], parts[4]


def output_key(name):
    parts = name.split("_")

    # video_SEB_125081_frame_000015_Head_0.png
    return parts[1], parts[3]


# ============================================================
# Construire le dictionnaire des fichiers v2
# ============================================================

source_by_key = {}

for p in source_files:
    key = source_key(p.name)

    source_by_key.setdefault(key, []).append(p)


# ============================================================
# Renommer les fichiers v3
# ============================================================

matched = 0
missing = 0
ambiguous = 0
already_correct = 0

for p in sorted(output_files):

    key = output_key(p.name)

    matches = source_by_key.get(key, [])

    # Aucun fichier v2 correspondant
    if len(matches) == 0:
        print(
            f"[MISSING] {p.name} "
            f"(clé = {key})"
        )
        missing += 1
        continue

    # Plusieurs fichiers v2 avec même vidéo + frame
    if len(matches) > 1:
        print(
            f"[AMBIGUOUS] {p.name} "
            f"(clé = {key})"
        )

        for match in matches:
            print(f"    -> {match.name}")

        ambiguous += 1
        continue

    # Correspondance unique
    wanted_name = matches[0].name
    new_path = p.with_name(wanted_name)

    if p.name == wanted_name:
        already_correct += 1
        continue

    print(
        f"[RENAME] {p.name} -> {wanted_name}"
    )

    p.rename(new_path)
    matched += 1


# ============================================================
# Résumé
# ============================================================

print()
print("=" * 60)
print(f"Renommés       : {matched}")
print(f"Déjà corrects  : {already_correct}")
print(f"Manquants      : {missing}")
print(f"Ambigus        : {ambiguous}")
print("=" * 60)