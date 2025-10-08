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

# =========================================================
# 1️⃣  CONFIG
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
DB_FILE = "claims.db"
IMAGE_FOLDER = "./images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# =========================================================
# 2️⃣  LOAD & CACHE MODEL
# =========================================================
@st.cache_resource
def load_clip_model():
    clip_model_path = os.path.abspath("./clip_model_offline")
    model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_path, local_files_only=True)
    model.eval()
    return model, processor

model, processor = load_clip_model()

# =========================================================
# 3️⃣  DATABASE SETUP
# =========================================================
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()
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
c.execute("CREATE INDEX IF NOT EXISTS idx_records_ids ON records(unique_image_id, customer_id, order_id)")
conn.commit()

# =========================================================
# 4️⃣  UTILITIES
# =========================================================
def generate_unique_image_id():
    return str(uuid.uuid4())[:8]

def get_image_hash(image):
    return imagehash.phash(image)

def save_image(image, image_hash):
    path = os.path.join(IMAGE_FOLDER, f"{image_hash}.png")
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    image.save(path)
    return path

def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

def get_embeddings(image, description):
    inputs = processor(text=[description], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_emb = model.get_image_features(inputs["pixel_values"].to(device))
        text_emb = model.get_text_features(inputs["input_ids"].to(device))

    image_emb = normalize(image_emb.cpu().numpy().astype(np.float32))
    text_emb = normalize(text_emb.cpu().numpy().astype(np.float32))
    combo_emb = normalize(0.4 * image_emb + 0.6 * text_emb)
    return image_emb.tobytes(), combo_emb.tobytes()

# =========================================================
# 5️⃣  DUPLICATE CHECK (HYBRID)
# =========================================================
def check_duplicates(image, description):
    img_hash = get_image_hash(image)
    new_img_emb, new_combo_emb = get_embeddings(image, description)
    new_img_emb = np.frombuffer(new_img_emb, dtype=np.float32)
    new_combo_emb = np.frombuffer(new_combo_emb, dtype=np.float32)

    c.execute("SELECT unique_image_id, image_hash, embedding FROM records")
    rows = c.fetchall()
    if not rows:
        return "No Duplicate", None

    stored_embs = np.array([np.frombuffer(r[2], dtype=np.float32) for r in rows])

    # Stage 1: perceptual hash (exact / near-duplicate)
    for uid, h, _ in rows:
        try:
            dist = img_hash - imagehash.hex_to_hash(h)
            if dist <= 5:
                return "Exact Duplicate", uid
        except Exception:
            continue

    # Stage 2: image-only similarity
    sims_img = np.dot(stored_embs, new_img_emb) / (
        np.linalg.norm(stored_embs, axis=1) * np.linalg.norm(new_img_emb)
    )
    best_idx = np.argmax(sims_img)
    if sims_img[best_idx] > 0.88:
        return "Similar Image", rows[best_idx][0]

    # Stage 3: combined image+text similarity
    sims_combo = np.dot(stored_embs, new_combo_emb) / (
        np.linalg.norm(stored_embs, axis=1) * np.linalg.norm(new_combo_emb)
    )
    best_idx = np.argmax(sims_combo)
    if sims_combo[best_idx] > 0.75:
        return "Same Narrative", rows[best_idx][0]

    return "No Duplicate", None

def save_to_db(uid, cust, order, market, desc, dmg, image):
    img_hash = get_image_hash(image)
    save_image(image, img_hash)
    _, combo = get_embeddings(image, desc)
    c.execute(
        "INSERT INTO records VALUES (?,?,?,?,?,?,?,?)",
        (uid, cust, order, market, desc, dmg, str(img_hash), combo)
    )
    conn.commit()

# =========================================================
# 6️⃣  STREAMLIT UI
# =========================================================
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to:", ["Submit Claim", "Database Viewer"])
st.title("🧭 Duplicate Image Detection & Clustering (Optimized)")

# ---------------------- Submit Claim ---------------------
if menu == "Submit Claim":
    cust = st.text_input("Customer ID")
    order = st.text_input("Order ID")
    market = st.text_input("Marketplace")
    desc = st.text_area("Image Description")
    dmg = st.selectbox(
        "Damage Classification",
        [
            "Burnt", "Spilled", "Broken", "Scratched", "Missing Parts", "Bent",
            "Malfunctioning", "Stained", "Packaging Damaged", "Expired", "Leaking", "Other"
        ],
    )
    upload = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if upload and st.button("🔍 Check for Duplicates"):
        img = Image.open(upload).convert("RGB")
        with st.spinner("Analyzing..."):
            status, match_uid = check_duplicates(img, desc)

        if status == "No Duplicate":
            uid = generate_unique_image_id()
            save_to_db(uid, cust, order, market, desc, dmg, img)
            st.success(f"✅ No duplicate found. Stored under UID: {uid}")
        else:
            save_to_db(match_uid, cust, order, market, desc, dmg, img)
            st.warning(f"⚠️ {status} found! Linked to existing UID: {match_uid}")

# ---------------------- Database Viewer ------------------
elif menu == "Database Viewer":
    st.subheader("📂 Stored Claims")

    search = st.text_input("🔍 Search (Customer ID, Order ID, Marketplace, Description, UID):")

    @st.cache_data(ttl=30)
    def load_db():
        c.execute("SELECT unique_image_id, customer_id, order_id, marketplace, description, damage_class, image_hash FROM records")
        rows = c.fetchall()
        return pd.DataFrame(rows, columns=["UID", "Customer", "Order", "Marketplace", "Description", "Damage", "Hash"])

    df = load_db()

    if df.empty:
        st.info("No records found.")
    else:
        if search.strip():
            s = search.lower()
            df = df[df.apply(lambda r: any(s in str(v).lower() for v in r.values), axis=1)]

        if df.empty:
            st.warning("No matching results.")
        else:
            page_size = 10
            pages = (len(df) - 1) // page_size + 1
            page = st.number_input("Page", 1, pages, 1)
            start, end = (page - 1) * page_size, page * page_size
            df_page = df.iloc[start:end]

            for uid in df_page["UID"].unique():
                st.markdown(f"### 🧩 UID: {uid}")
                sub = df_page[df_page["UID"] == uid]

                header_cols = st.columns([1, 1, 1, 3, 1, 1])
                headers = ["Customer", "Order", "Marketplace", "Description", "Damage", "Image"]
                for col, label in zip(header_cols, headers):
                    col.markdown(f"**{label}**")

                for _, row in sub.iterrows():
                    cols = st.columns([1, 1, 1, 3, 1, 1])
                    cols[0].write(row["Customer"])
                    cols[1].write(row["Order"])
                    cols[2].write(row["Marketplace"])
                    cols[3].write(row["Description"])
                    cols[4].write(row["Damage"])

                    img_path = os.path.join(IMAGE_FOLDER, f"{row['Hash']}.png")
                    if os.path.exists(img_path):
                        cols[5].image(img_path, width=60)
                    else:
                        cols[5].write("—")