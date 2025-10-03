import streamlit as st
import sqlite3
import imagehash
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os
import pandas as pd
import uuid

# ----------------------------
# Device & CLIP Setup
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_path = os.path.abspath("./clip_model_offline")  # Offline CLIP folder

model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True).to(device)
processor = CLIPProcessor.from_pretrained(clip_model_path, local_files_only=True)

# ----------------------------
# Database Setup
# ----------------------------
DB_FILE = "claims.db"
IMAGE_FOLDER = "./images"

os.makedirs(IMAGE_FOLDER, exist_ok=True)

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()

# Create table if not exists with damage_class column
c.execute("""
CREATE TABLE IF NOT EXISTS records (
    unique_image_id TEXT,
    customer_id TEXT,
    order_id TEXT,
    marketplace TEXT,
    description TEXT,
    damage_class TEXT,
    image_hash TEXT,
    embedding BLOB
)
""")
conn.commit()

# ----------------------------
# Utility Functions
# ----------------------------
def generate_unique_image_id():
    return str(uuid.uuid4())[:8]

def get_image_hash(image):
    return str(imagehash.phash(image))

def save_image(image, image_hash):
    """Save uploaded image to images folder"""
    path = os.path.join(IMAGE_FOLDER, f"{image_hash}.png")
    image.save(path)
    return path

def get_embedding(image, description):
    inputs = processor(text=[description], images=image, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        image_embeds = model.get_image_features(pixel_values)
        text_embeds = model.get_text_features(input_ids)

    combined = (0.4 * image_embeds + 0.6 * text_embeds).cpu().numpy().astype(np.float32)
    return combined.tobytes()

def cosine_similarity(a, b):
    a, b = np.frombuffer(a, dtype=np.float32), np.frombuffer(b, dtype=np.float32)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def check_duplicates(image, description):
    image_hash = get_image_hash(image)
    new_embedding = get_embedding(image, description)

    c.execute("SELECT * FROM records")
    all_records = c.fetchall()

    for rec in all_records:
        uid, _, _, _, _, _, stored_hash, stored_emb = rec

        if stored_hash == image_hash:
            return ("Exact Duplicate", uid)

        sim = cosine_similarity(new_embedding, stored_emb)
        if sim > 0.85:
            return ("Similar Image", uid)
        if sim > 0.65:
            return ("Same Narrative", uid)

    return ("No Duplicate", None)

def save_to_db(unique_image_id, customer_id, order_id, marketplace, description, damage_class, image):
    image_hash = get_image_hash(image)
    save_image(image, image_hash)
    embedding = get_embedding(image, description)
    c.execute(
        "INSERT INTO records VALUES (?,?,?,?,?,?,?,?)",
        (unique_image_id, customer_id, order_id, marketplace, description, damage_class, image_hash, embedding)
    )
    conn.commit()

# ----------------------------
# Streamlit UI
# ----------------------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to:", ["Submit Claim", "Database Viewer"])
st.title("Duplicate Image detection and Clustering")

# ----------------------------
# Submit Claim
# ----------------------------
if menu == "Submit Claim":
    customer_id_input = st.text_input("Customer ID")
    order_id_input = st.text_input("Order ID")
    marketplace_input = st.text_input("Marketplace")
    description_input = st.text_area("Image Description")

    damage_options = [
        "Burnt", "Spilled", "Broken", "Scratched", "Missing Parts",
        "Bent", "Malfunctioning", "Stained", "Packaging Damaged", "Expired", "Leaking", "Other"
    ]
    damage_input = st.selectbox("Damage Classification", damage_options)

    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file and st.button("Check for Duplicates"):
        image_input = Image.open(uploaded_file).convert("RGB")
        status, matched_uid = check_duplicates(image_input, description_input)

        if status == "No Duplicate":
            new_uid = generate_unique_image_id()
            save_to_db(new_uid, customer_id_input, order_id_input, marketplace_input,
                       description_input, damage_input, image_input)
            st.success(f"✅ No duplicate found. Stored under Unique Image ID: {new_uid}")
        else:
            save_to_db(matched_uid, customer_id_input, order_id_input, marketplace_input,
                       description_input, damage_input, image_input)
            st.warning(f"⚠️ {status} found! Claim linked to existing Unique Image ID: {matched_uid}")

# ----------------------------
# Database Viewer
# ----------------------------
elif menu == "Database Viewer":
    st.subheader("📂 Stored Claims in Database")

    c.execute("SELECT unique_image_id, customer_id, order_id, marketplace, description, damage_class, image_hash FROM records")
    rows = c.fetchall()

    if rows:
        df = pd.DataFrame(rows, columns=[
            "Unique Image ID", "Customer ID", "Order ID", "Marketplace", "Description", "Damage", "Image Hash"
        ])

        unique_ids = df["Unique Image ID"].unique()

        for uid in unique_ids:
            st.markdown(f"### Unique Image ID: {uid}")
            group = df[df["Unique Image ID"] == uid]

    # Header
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 3, 1, 1])
    col1.markdown("**Customer ID**")
    col2.markdown("**Order ID**")
    col3.markdown("**Marketplace**")
    col4.markdown("**Description**")
    col5.markdown("**Damage**")
    col6.markdown("**Image**")

    # Display records
    for _, row in group.iterrows():
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 3, 1, 1])
        col1.write(row["Customer ID"])
        col2.write(row["Order ID"])
        col3.write(row["Marketplace"])
        col4.write(row["Description"])
        col5.write(row["Damage"])

        image_path = os.path.join(IMAGE_FOLDER, f"{row['Image Hash']}.png")
        if os.path.exists(image_path):
            col6.image(image_path, width=60)
        else:
            col6.write("No image")
else:
    st.info("No records found in database.")