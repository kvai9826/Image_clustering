# full_app.py
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
import io
from io import BytesIO

# --------------------------
# CONFIG / MODEL SETUP
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_PATH = os.path.abspath("./clip_model_offline")  # point this to your offline CLIP

model = CLIPModel.from_pretrained(CLIP_MODEL_PATH, local_files_only=True).to(DEVICE)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH, local_files_only=True)

# --------------------------
# DB / Folders
# --------------------------
DB_FILE = "claims4.db"
IMAGE_FOLDER = "./images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()

# columns (19)
c.execute("""
CREATE TABLE IF NOT EXISTS records (
    unique_image_id TEXT,
    sub_cluster_id TEXT,
    customer_id TEXT,
    order_id TEXT,
    ip_country_code TEXT,
    billing_country_code TEXT,
    shipping_country_code TEXT,
    credit_card_country_code TEXT,
    fast_lane INTEGER,
    isfba INTEGER,
    has_prime INTEGER,
    gl_code TEXT,
    payment_method TEXT,
    issuing_bank TEXT,
    description TEXT,
    damage_classification TEXT,
    chat_text TEXT,
    image_hash TEXT,
    embedding BLOB
)
""")
conn.commit()

# --------------------------
# UTILITIES
# --------------------------
def generate_unique_image_id():
    return str(uuid.uuid4())[:8]

def get_image_hash(pil_img: Image.Image):
    return str(imagehash.phash(pil_img))

def save_image(pil_img: Image.Image, image_hash: str):
    path = os.path.join(IMAGE_FOLDER, f"{image_hash}.png")
    pil_img.save(path)
    return path

def get_embedding(image: Image.Image, description: str):
    # safe truncation for text to avoid CLIP max length errors
    inputs = processor(text=[description], images=image, return_tensors="pt",
                       padding=True, truncation=True, max_length=77)
    pixel_values = inputs["pixel_values"].to(DEVICE)
    input_ids = inputs["input_ids"].to(DEVICE)
    with torch.no_grad():
        img_emb = model.get_image_features(pixel_values)
        txt_emb = model.get_text_features(input_ids)
    combined = (0.4 * img_emb + 0.6 * txt_emb).cpu().numpy().astype(np.float32)
    return combined.tobytes()

def safe_array_from_buffer(buf):
    # returns numpy array or None
    try:
        return np.frombuffer(buf, dtype=np.float32)
    except Exception:
        return None

def cosine_similarity(a_bytes, b_bytes):
    try:
        a = np.frombuffer(a_bytes, dtype=np.float32)
        b = np.frombuffer(b_bytes, dtype=np.float32)
        # protect against zero vectors
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return -1.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return -1.0

def get_chat_embedding(chat_text: str):
    if not chat_text or not str(chat_text).strip():
        return np.zeros(512, dtype=np.float32)
    # CLIP text encoder: truncate to safe length
    inputs = processor(text=[chat_text], return_tensors="pt", padding=True, truncation=True, max_length=77)
    input_ids = inputs["input_ids"].to(DEVICE)
    with torch.no_grad():
        emb = model.get_text_features(input_ids)
    return emb.cpu().numpy().flatten().astype(np.float32)

# --------------------------
# DUPLICATE CHECK
# --------------------------
def check_duplicates(image: Image.Image, description: str):
    image_hash = get_image_hash(image)
    new_emb = get_embedding(image, description)
    c.execute("SELECT unique_image_id, image_hash, embedding FROM records")
    all_records = c.fetchall()
    for uid, stored_hash, stored_emb in all_records:
        if stored_hash == image_hash:
            return ("Exact Duplicate", uid)
        # similarity
        sim = cosine_similarity(new_emb, stored_emb) if stored_emb is not None else -1.0
        if sim > 0.85:
            return ("Similar Image", uid)
        if sim > 0.65:
            return ("Same Narrative", uid)
    return ("No Duplicate", None)

# --------------------------
# SUBCLUSTERING (chat-aware)
# --------------------------
def compute_similarity(row_a, row_b, chat_weight=0.15):
    # compare fields as strings, robust to None/NaN
    def eq(a, b):
        return str(a) == str(b)
    t1_fields = ["billing_country_code", "shipping_country_code", "credit_card_country_code"]
    t2_fields = ["isfba", "has_prime", "fast_lane"]
    t3_fields = ["ip_country_code", "payment_method", "issuing_bank"]

    t1_score = sum(eq(row_a.get(f, ""), row_b.get(f, "")) for f in t1_fields) / len(t1_fields)
    t2_score = sum(eq(row_a.get(f, ""), row_b.get(f, "")) for f in t2_fields) / len(t2_fields)
    t3_score = sum(eq(row_a.get(f, ""), row_b.get(f, "")) for f in t3_fields) / len(t3_fields)

    # chat similarity (semantic)
    a_chat = get_chat_embedding(str(row_a.get("chat_text", "")))
    b_chat = get_chat_embedding(str(row_b.get("chat_text", "")))
    denom = (np.linalg.norm(a_chat) * np.linalg.norm(b_chat) + 1e-8)
    chat_sim = float(np.dot(a_chat, b_chat) / denom) if denom > 0 else 0.0

    base_weight = 1.0 - chat_weight
    # structured part is normalized combination
    structured_score = 0.45 * t1_score + 0.25 * t2_score + 0.15 * t3_score
    total = (base_weight * structured_score) + (chat_weight * chat_sim)
    return float(total)

def assign_subcluster(existing_df: pd.DataFrame, new_row: dict, chat_weight: float):
    # existing_df contains rows for the same main cluster (unique_image_id)
    subset = existing_df[existing_df["unique_image_id"] == new_row["unique_image_id"]]
    if subset.empty:
        return f"{new_row['unique_image_id']}_S0"
    best_sim, best_id = -1.0, None
    for _, row in subset.iterrows():
        sim = compute_similarity(row, new_row, chat_weight)
        if sim > best_sim:
            best_sim = sim
            best_id = row["sub_cluster_id"]
    if best_sim >= 0.8:
        return best_id
    else:
        existing = subset["sub_cluster_id"].dropna().unique()
        return f"{new_row['unique_image_id']}_S{len(existing)}"

def save_to_db(record_tuple):
    # record_tuple must have 19 elements in same order as table
    placeholders = ",".join(["?"]*19)
    c.execute(f"INSERT INTO records VALUES ({placeholders})", tuple(record_tuple))
    conn.commit()

# --------------------------
# STREAMLIT UI
# --------------------------
st.set_page_config(layout="wide")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["Submit Claim", "Database Viewer"])

st.title("Duplicate Detection & Chat-Aware Subclustering (full app)")

# --------------------------
# SUBMIT CLAIM PAGE
# --------------------------
if menu == "Submit Claim":
    st.header("Submit Claim")

    # Row 1: Customer ID, Order ID
    c1, c2 = st.columns(2)
    customer_id = c1.text_input("Customer ID")
    order_id = c2.text_input("Order ID")

    # Row 2: IP Country, Billing Country (same line)
    c3, c4 = st.columns(2)
    ip_country_code = c3.text_input("IP Country Code")
    billing_country_code = c4.text_input("Billing Country Code")

    # Row 3: Shipping Country, Credit Card Country
    c5, c6 = st.columns(2)
    shipping_country_code = c5.text_input("Shipping Country Code")
    credit_card_country_code = c6.text_input("Credit Card Country Code")

    # Row 4: Fast lane, isFBA, hasPrime (dropdown 0/1)
    c7, c8, c9 = st.columns(3)
    fast_lane = c7.selectbox("Fast Lane", [0, 1], index=0)
    isfba = c8.selectbox("Is FBA", [0, 1], index=0)
    has_prime = c9.selectbox("Has Prime", [0, 1], index=0)

    # Row 5: GL, Payment Method, Issuing Bank
    c10, c11, c12 = st.columns(3)
    gl_code = c10.text_input("GL Code")
    payment_method = c11.text_input("Payment Method")
    issuing_bank = c12.text_input("Issuing Bank")

    # Description, Chat, Damage
    description = st.text_area("Image Description", height=80)
    chat_text = st.text_area("Chat Conversation (optional)", height=120)

    damage_classification = st.selectbox(
        "Damage Classification",
        ["Burnt", "Spilled", "Broken", "Scratched", "Missing item",
         "Malfunctioning", "Stained", "Packaging Damaged",
         "Expired", "Leaking", "Other"]
    )

    # Image upload
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    # Chat weight slider
    chat_weight = st.slider("Chat weight in subclustering", 0.0, 0.3, 0.15, 0.01)

    if uploaded_file and st.button("Check for Duplicates and Save"):
        img = Image.open(uploaded_file).convert("RGB")
        status, matched_uid = check_duplicates(img, description)
        image_hash = get_image_hash(img)
        save_image(img, image_hash)
        embedding = get_embedding(img, description)

        uid = generate_unique_image_id() if status == "No Duplicate" else matched_uid
        existing_df = pd.read_sql_query("SELECT * FROM records", conn)

        new_row = {
            "unique_image_id": uid,
            "ip_country_code": ip_country_code,
            "billing_country_code": billing_country_code,
            "shipping_country_code": shipping_country_code,
            "credit_card_country_code": credit_card_country_code,
            "fast_lane": fast_lane,
            "isfba": isfba,
            "has_prime": has_prime,
            "payment_method": payment_method,
            "issuing_bank": issuing_bank,
            "chat_text": chat_text
        }

        sub_cluster_id = assign_subcluster(existing_df, new_row, chat_weight)

        record = (
            uid, sub_cluster_id, customer_id, order_id, ip_country_code, billing_country_code,
            shipping_country_code, credit_card_country_code, int(fast_lane), int(isfba), int(has_prime),
            gl_code, payment_method, issuing_bank, description, damage_classification,
            chat_text, image_hash, embedding
        )
        save_to_db(record)

        if status == "No Duplicate":
            st.success(f"Saved as new cluster {uid} / subcluster {sub_cluster_id}")
        else:
            st.warning(f"{status} -> linked to cluster {uid} (subcluster {sub_cluster_id})")

# --------------------------
# DATABASE VIEWER PAGE
# --------------------------
elif menu == "Database Viewer":
    st.header("Database Viewer — Subclusters as tables")

    # Load DB
    df = pd.read_sql_query("""
        SELECT unique_image_id, sub_cluster_id, customer_id, order_id,
               ip_country_code, billing_country_code, shipping_country_code,
               credit_card_country_code, fast_lane, isfba, has_prime,
               gl_code, payment_method, issuing_bank,
               description, damage_classification, chat_text, image_hash
        FROM records
    """, conn)

    if df.empty:
        st.info("No records")
    else:
        # top controls
        left, mid, right = st.columns([4, 1, 1])
        query = left.text_input("Search (any field or chat)")
        show_images = mid.toggle("Show Images", value=True)
        # global download
        csv_all = df.to_csv(index=False).encode("utf-8")
        right.download_button("📥 Download all CSV", csv_all, file_name="claims_all.csv", mime="text/csv")

        if query.strip():
            q = query.lower()
            df = df[df.apply(lambda r: any(q in str(v).lower() for v in r.values), axis=1)]

        # group by main cluster -> subcluster and render each subcluster as a dataframe
        for main_cluster in df["unique_image_id"].unique():
            st.subheader(f"Main Cluster: {main_cluster}")
            main_df = df[df["unique_image_id"] == main_cluster]

            for subcluster in main_df["sub_cluster_id"].unique():
                st.markdown(f"**Subcluster: {subcluster}**")
                sub_df = main_df[main_df["sub_cluster_id"] == subcluster].copy()

                # truncated chat
                sub_df["Chat Summary"] = sub_df["chat_text"].apply(
                    lambda x: (x[:100] + "...") if len(str(x)) > 100 else x
                )

                # prepare display dataframe
                display_df = sub_df[[
                    "customer_id", "order_id", "billing_country_code",
                    "issuing_bank", "damage_classification", "Chat Summary"
                ]].rename(columns={
                    "customer_id": "Customer ID",
                    "order_id": "Order ID",
                    "billing_country_code": "Billing Code",
                    "issuing_bank": "Bank",
                    "damage_classification": "Damage Type"
                }).reset_index(drop=True)

                # image handling: add a column with file paths (ImageColumn needs a path)
                if show_images:
                    def path_for(h):
                        p = os.path.join(IMAGE_FOLDER, f"{h}.png")
                        return p if os.path.exists(p) else None
                    display_df["Image"] = sub_df["image_hash"].apply(path_for)

                # per-subcluster download button
                csv_sub = sub_df.to_csv(index=False).encode("utf-8")
                dl_col, tbl_col = st.columns([1, 11])
                with dl_col:
                    st.download_button(
                        label="📥",
                        data=csv_sub,
                        file_name=f"{main_cluster}_{subcluster}.csv",
                        mime="text/csv",
                        help="Download this subcluster"
                    )

                # render table: try to use ImageColumn if available, otherwise render table without images
                try:
                    # streamlit column_config.ImageColumn is available in recent versions
                    if show_images and hasattr(st, "column_config"):
                        colconfig = {
                            "Image": st.column_config.ImageColumn("Image", help="thumbnail", width="small")
                        }
                        st.dataframe(display_df, use_container_width=True, hide_index=True, column_config=colconfig)
                    else:
                        # either user chose to hide images or ImageColumn not available
                        # drop Image column if present
                        if "Image" in display_df.columns:
                            st.dataframe(display_df.drop(columns=["Image"]), use_container_width=True, hide_index=True)
                        else:
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                except Exception:
                    # fallback: no column_config support
                    if "Image" in display_df.columns:
                        st.dataframe(display_df.drop(columns=["Image"]), use_container_width=True, hide_index=True)
                        if show_images:
                            st.write("Thumbnails for this subcluster:")
                            # show thumbnails as grid (simple fallback)
                            thumbs = []
                            for h in sub_df["image_hash"]:
                                p = os.path.join(IMAGE_FOLDER, f"{h}.png")
                                if os.path.exists(p):
                                    thumbs.append(Image.open(p).resize((80, 60)))
                                else:
                                    thumbs.append(None)
                            cols = st.columns(6)
                            idx = 0
                            for t in thumbs:
                                if t:
                                    cols[idx % 6].image(t, width=80)
                                idx += 1
                    else:
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

                st.markdown("---")