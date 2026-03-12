#!/usr/bin/env python3
"""
Generate 100K synthetic Indian farmer records with 384-dim embeddings.

Usage:
    pip install sentence-transformers numpy tqdm
    python tools/generate_dataset.py

Output:
    data/farmers_100k.json     — array of {id, fields..., vector:[384 floats]}
    data/farmers_100k.npy      — raw float32 vectors (100000 x 384)
    data/farmers_100k_meta.json — records WITHOUT vectors (for lighter inspection)
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────

NUM_RECORDS = 100_000
DIMS = 384
BATCH_SIZE = 512
SEED = 42

STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal",
]

SOIL_TYPES = [
    "Alluvial", "Black Cotton", "Red", "Laterite", "Sandy",
    "Clay Loam", "Saline", "Peaty", "Forest", "Desert",
]

SEASONS = ["Kharif", "Rabi", "Zaid"]

CROPS = {
    "Kharif": ["Rice", "Maize", "Jowar", "Bajra", "Cotton", "Sugarcane",
               "Soybean", "Groundnut", "Turmeric", "Jute", "Arhar Dal",
               "Moong", "Urad", "Sesame"],
    "Rabi":   ["Wheat", "Barley", "Mustard", "Chickpea", "Lentil",
               "Peas", "Linseed", "Safflower", "Coriander", "Cumin",
               "Fenugreek", "Sunflower"],
    "Zaid":   ["Watermelon", "Muskmelon", "Cucumber", "Bitter Gourd",
               "Pumpkin", "Moong", "Fodder Crops", "Vegetables"],
}

IRRIGATION_TYPES = [
    "Canal", "Tube Well", "Drip", "Sprinkler", "Rainfed",
    "Open Well", "Tank", "Lift Irrigation",
]

WEATHER_CONDITIONS = [
    "Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Heavy Rain",
    "Thunderstorm", "Fog", "Heatwave", "Cold Wave", "Normal",
    "Drought Conditions", "Flood Risk", "Hailstorm",
]

FERTILIZERS = [
    "Urea", "DAP", "MOP", "NPK 10-26-26", "NPK 20-20-0",
    "SSP", "Zinc Sulphate", "Vermicompost", "Farm Yard Manure",
    "Neem Cake", "Ammonium Sulphate", "Potash", "Bone Meal",
    "Bio-fertilizer", "Humic Acid",
]

WARNINGS = [
    "None", "Pest Alert: Bollworm", "Pest Alert: Stem Borer",
    "Pest Alert: Aphids", "Pest Alert: Whitefly",
    "Disease Alert: Blight", "Disease Alert: Rust",
    "Disease Alert: Wilt", "Disease Alert: Powdery Mildew",
    "Weather Warning: Unseasonal Rain", "Weather Warning: Frost",
    "Weather Warning: Heatwave", "Weather Warning: Cyclone",
    "Soil Advisory: Low Nitrogen", "Soil Advisory: High Salinity",
    "Soil Advisory: Low pH", "Water Scarcity Advisory",
    "Market Advisory: Price Drop Expected",
    "Market Advisory: Export Opportunity",
    "Harvest Window: Optimal in 3 days",
]

FIRST_NAMES = [
    "Ramesh", "Suresh", "Mahesh", "Ganesh", "Rajesh", "Mukesh",
    "Dinesh", "Naresh", "Kamlesh", "Pradeep", "Sunil", "Anil",
    "Vijay", "Sanjay", "Ajay", "Manoj", "Ashok", "Ravi",
    "Mohan", "Sohan", "Gopal", "Krishna", "Shyam", "Ram",
    "Lakshmi", "Savitri", "Anita", "Sunita", "Geeta", "Seema",
    "Rekha", "Meena", "Kamla", "Sita", "Parvati", "Durga",
    "Baldev", "Harpal", "Gurpreet", "Jaswinder", "Kuldeep",
    "Bhagwan", "Tulsi", "Nand", "Kishan", "Jagdish", "Pyare",
    "Chandra", "Surya", "Indra",
]

LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Singh", "Kumar", "Yadav",
    "Gupta", "Reddy", "Naidu", "Patil", "Deshmukh", "Joshi",
    "Chauhan", "Thakur", "Mishra", "Pandey", "Tiwari", "Dubey",
    "Rawat", "Nair", "Menon", "Pillai", "Mahato", "Mandal",
    "Das", "Bora", "Gogoi", "Choudhury", "Kaur", "Gill",
    "Sidhu", "Dhillon", "Rathore", "Shekhawat", "Meena",
    "Gowda", "Hegde", "Shetty", "Kulkarni", "Deshpande",
]


def generate_records(rng: random.Random) -> list[dict]:
    """Generate NUM_RECORDS synthetic farmer records."""
    records = []
    for i in tqdm(range(NUM_RECORDS), desc="Generating records"):
        state = rng.choice(STATES)
        season = rng.choice(SEASONS)
        crop = rng.choice(CROPS[season])
        num_fertilizers = rng.randint(1, 3)
        ferts = rng.sample(FERTILIZERS, num_fertilizers)
        area = round(rng.uniform(0.5, 50.0), 1)
        daily_water = round(rng.uniform(500, 15000), 0)  # litres
        yield_per_acre = round(rng.uniform(2.0, 45.0), 1)  # quintals

        rec = {
            "id": f"farmer_{i:06d}",
            "state": state,
            "farmerName": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            "soilType": rng.choice(SOIL_TYPES),
            "season": season,
            "crop": crop,
            "warnings": rng.choice(WARNINGS),
            "areaAcres": area,
            "irrigationType": rng.choice(IRRIGATION_TYPES),
            "yieldPerAcre": yield_per_acre,
            "weather": rng.choice(WEATHER_CONDITIONS),
            "dailyWaterLiters": daily_water,
            "fertilizers": ferts,
        }
        records.append(rec)
    return records


def record_to_text(rec: dict) -> str:
    """Convert a record to a natural-language sentence for embedding."""
    ferts = ", ".join(rec["fertilizers"])
    return (
        f"{rec['farmerName']} is a farmer in {rec['state']} growing {rec['crop']} "
        f"during the {rec['season']} season on {rec['areaAcres']} acres of "
        f"{rec['soilType']} soil. Irrigation: {rec['irrigationType']}. "
        f"Weather: {rec['weather']}. Yield: {rec['yieldPerAcre']} quintals/acre. "
        f"Daily water: {rec['dailyWaterLiters']:.0f} litres. "
        f"Fertilizers: {ferts}. Warning: {rec['warnings']}."
    )


def embed_records(records: list[dict]) -> np.ndarray:
    """Batch-embed all records using all-MiniLM-L6-v2."""
    from sentence_transformers import SentenceTransformer

    print(f"\nLoading all-MiniLM-L6-v2 model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [record_to_text(r) for r in records]
    print(f"Embedding {len(texts)} records in batches of {BATCH_SIZE}...")

    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def main():
    rng = random.Random(SEED)
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    # 1. Generate records
    records = generate_records(rng)

    # 2. Embed
    vectors = embed_records(records)
    assert vectors.shape == (NUM_RECORDS, DIMS), f"Expected ({NUM_RECORDS},{DIMS}), got {vectors.shape}"

    # 3. Save numpy vectors
    npy_path = out_dir / "farmers_100k.npy"
    np.save(npy_path, vectors)
    print(f"\nSaved vectors: {npy_path} ({vectors.nbytes / 1e6:.1f} MB)")

    # 4. Save metadata-only JSON (no vectors, lighter)
    meta_path = out_dir / "farmers_100k_meta.json"
    with open(meta_path, "w") as f:
        json.dump(records, f, separators=(",", ":"))
    print(f"Saved metadata: {meta_path} ({os.path.getsize(meta_path) / 1e6:.1f} MB)")

    # 5. Save full JSON (records + vectors)
    full_records = []
    for i, rec in enumerate(tqdm(records, desc="Attaching vectors")):
        rec_copy = dict(rec)
        rec_copy["vector"] = vectors[i].tolist()
        full_records.append(rec_copy)

    json_path = out_dir / "farmers_100k.json"
    with open(json_path, "w") as f:
        json.dump(full_records, f, separators=(",", ":"))
    size_mb = os.path.getsize(json_path) / 1e6
    print(f"Saved full JSON: {json_path} ({size_mb:.1f} MB)")

    print(f"\n✓ Done — {NUM_RECORDS} records, {DIMS} dims")


if __name__ == "__main__":
    main()
