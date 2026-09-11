import pandas as pd
from rdkit import Chem


FILE_PATH = "Dataset/ci600312d_si_001.xls"


# ============================================================
# 1. LOAD THE TWO TABLES
# ============================================================

adenot = pd.read_excel(
    FILE_PATH,
    sheet_name="Table S1-Adenot",
    header=2
)

li = pd.read_excel(
    FILE_PATH,
    sheet_name="Table S2-Li",
    header=2
)


# ============================================================
# 2. SELECT IMPORTANT COLUMNS
# ============================================================

# -------------------------
# Adenot dataset
# -------------------------

adenot = adenot[
    ["SMILES", "SMILES-B", "Name", "BBB-crossing"]
].copy()

adenot.columns = [
    "SMILES",
    "Alternative_SMILES",
    "Name",
    "BBB"
]

adenot["Source"] = "Adenot"


# -------------------------
# Li dataset
# -------------------------

li = li[
    ["No", "No-Li", "Name", "SMILES-Li", "SMILES", "Class", "Ref."]
].copy()

li.columns = [
    "No",
    "No-Li",
    "Name",
    "Original_SMILES",
    "SMILES",
    "BBB",
    "Ref"
]

li["Source"] = "Li"


# ============================================================
# 3. MAKE COMMON STRUCTURE
# ============================================================

# Add missing columns to Adenot
adenot["No"] = None
adenot["No-Li"] = None
adenot["Original_SMILES"] = None
adenot["Ref"] = None


# Add missing column to Li
li["Alternative_SMILES"] = None


# Select same column order
columns = [
    "Source",
    "No",
    "No-Li",
    "Name",
    "SMILES",
    "Alternative_SMILES",
    "Original_SMILES",
    "BBB",
    "Ref"
]

adenot = adenot[columns]
li = li[columns]


# ============================================================
# 4. COMBINE BOTH DATASETS
# ============================================================

dataset = pd.concat(
    [adenot, li],
    ignore_index=True
)


# ============================================================
# 5. BASIC CLEANING
# ============================================================

# Remove rows without SMILES or BBB label
dataset = dataset.dropna(
    subset=["SMILES", "BBB"]
)


# Remove duplicate molecules based on common SMILES
dataset = dataset.drop_duplicates(
    subset=["SMILES"]
).reset_index(drop=True)


# ============================================================
# 6. CONVERT BBB LABELS TO INTEGER
# ============================================================

def convert_label(value):

    value = str(value).strip()

    if value == "0/1":
        return 0

    return int(float(value))


dataset["BBB"] = dataset["BBB"].apply(
    convert_label
)


# ============================================================
# 7. CONVERT SMILES → RDKit MOLECULE
# ============================================================

def smiles_to_molecule(smiles):

    return Chem.MolFromSmiles(smiles)


dataset["Molecule"] = dataset["SMILES"].apply(
    smiles_to_molecule
)


# ============================================================
# 8. REMOVE INVALID MOLECULES
# ============================================================

dataset = dataset.dropna(
    subset=["Molecule"]
).reset_index(drop=True)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("=" * 60)
print("BBB DATASET")
print("=" * 60)

print("Total valid compounds:", len(dataset))

print("\nSource distribution:")
print(dataset["Source"].value_counts())

print("\nBBB Class Distribution:")
print(dataset["BBB"].value_counts())

print("\nFirst 5 compounds:")
print(
    dataset[
        [
            "Source",
            "Name",
            "SMILES",
            "Alternative_SMILES",
            "Original_SMILES",
            "BBB"
        ]
    ].head()
)

print("\nRDKit Molecular Objects:")
print(dataset["Molecule"].head())
# ============================================================
# 10. GENERATE DETAILED IMAGE FOR EVERY COMPOUND
# ============================================================

import os
from PIL import Image, ImageDraw, ImageFont
from rdkit.Chem import Draw


IMAGE_FOLDER = "images"

# Create images folder
os.makedirs(IMAGE_FOLDER, exist_ok=True)


# ------------------------------------------------------------
# Fonts
# ------------------------------------------------------------

try:
    font_title = ImageFont.truetype("arial.ttf", 28)
    font_info = ImageFont.truetype("arial.ttf", 20)
except:
    font_title = ImageFont.load_default()
    font_info = ImageFont.load_default()


# ------------------------------------------------------------
# Generate image for every compound
# ------------------------------------------------------------

for i, row in dataset.iterrows():

    mol = row["Molecule"]

    # -----------------------------------------
    # Basic information
    # -----------------------------------------

    compound_id = i + 1

    name = str(row["Name"]) if pd.notna(row["Name"]) else "N/A"

    smiles = str(row["SMILES"]) if pd.notna(row["SMILES"]) else "N/A"

    alternative_smiles = (
        str(row["Alternative_SMILES"])
        if pd.notna(row["Alternative_SMILES"])
        else "N/A"
    )

    original_smiles = (
        str(row["Original_SMILES"])
        if pd.notna(row["Original_SMILES"])
        else "N/A"
    )

    bbb = str(row["BBB"]) if pd.notna(row["BBB"]) else "N/A"

    source = str(row["Source"]) if pd.notna(row["Source"]) else "N/A"

    no = str(row["No"]) if pd.notna(row["No"]) else "N/A"

    no_li = str(row["No-Li"]) if pd.notna(row["No-Li"]) else "N/A"

    reference = str(row["Ref"]) if pd.notna(row["Ref"]) else "N/A"


    # -----------------------------------------
    # Generate molecular structure
    # -----------------------------------------

    molecule_img = Draw.MolToImage(
        mol,
        size=(700, 500)
    ).convert("RGB")


    # -----------------------------------------
    # Create final canvas
    # -----------------------------------------

    canvas_width = 1000
    canvas_height = 1000

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)


    # -----------------------------------------
    # Title
    # -----------------------------------------

    draw.text(
        (40, 25),
        f"MOLECULE {compound_id:04d}",
        fill="black",
        font=font_title
    )


    # -----------------------------------------
    # Paste molecular structure
    # -----------------------------------------

    molecule_x = 150
    molecule_y = 70

    canvas.paste(
        molecule_img,
        (molecule_x, molecule_y)
    )


    # -----------------------------------------
    # Information section
    # -----------------------------------------

    y = 590

    information = [
        f"Compound ID : {compound_id:04d}",
        f"Name        : {name}",
        f"Source      : {source}",
        f"BBB Class   : {bbb}",
        f"No          : {no}",
        f"No-Li       : {no_li}",
        f"Reference   : {reference}",
        "",
        f"SMILES:",
        smiles,
        "",
        f"Alternative SMILES:",
        alternative_smiles,
        "",
        f"Original SMILES:",
        original_smiles
    ]


    for line in information:

        # Wrap very long lines
        max_chars = 85

        if len(line) > max_chars:

            chunks = [
                line[j:j + max_chars]
                for j in range(
                    0,
                    len(line),
                    max_chars
                )
            ]

            for chunk in chunks:

                draw.text(
                    (40, y),
                    chunk,
                    fill="black",
                    font=font_info
                )

                y += 28

        else:

            draw.text(
                (40, y),
                line,
                fill="black",
                font=font_info
            )

            y += 28


    # -----------------------------------------
    # Save image
    # -----------------------------------------

    filename = os.path.join(
        IMAGE_FOLDER,
        f"molecule_{compound_id:04d}.png"
    )

    canvas.save(
        filename
    )


print("\n" + "=" * 60)
print("DETAILED MOLECULAR IMAGES")
print("=" * 60)

print("Images generated :", len(dataset))
print("Folder            :", IMAGE_FOLDER)

# ============================================================
# SAVE CLEANED DATASET
# ============================================================

dataset.drop(
    columns=["Molecule"]
).to_csv(
    "Dataset/cleaned_bbb_dataset.csv",
    index=False
)

print("Saved: Dataset/cleaned_bbb_dataset.csv")