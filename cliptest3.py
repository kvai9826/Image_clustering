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
# CLIP MODEL SETUP
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_path = os.path.abspath("./clip_model_offline")

model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True).to(device)
processor = CLIPProcessor.from_pretrained(clip_model_path, local_files_only=True)

# =========================================================
# DATABASE SETUP
# =========================================================
DB_FILE = "claims.db"
IMAGE_FOLDER = "./images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()

# Create the table with all required columns
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
    image_hash TEXT,
    embedding BLOB
)
""")
conn.commit()

# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def generate_unique_image_id():
    return str(uuid.uuid4())[:8]

def get_image_hash(image):
    return str(imagehash.phash(image))

def save_image(image, image_hash):
    path = os.path.join(IMAGE_FOLDER, f"{image_hash}.png")
    image.save(path)
    return path

def get_embedding(image, description):
    inputs = processor(text=[description], images=image, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        img_emb = model.get_image_features(pixel_values)
        txt_emb = model.get_text_features(input_ids)
    combined = (0.4 * img_emb + 0.6 * txt_emb).cpu().numpy().astype(np.float32)
    return combined.tobytes()

def cosine_similarity(a, b):
    a, b = np.frombuffer(a, dtype=np.float32), np.frombuffer(b, dtype=np.float32)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def check_duplicates(image, description):
    image_hash = get_image_hash(image)
    new_emb = get_embedding(image, description)
    c.execute("SELECT unique_image_id, image_hash, embedding FROM records")
    all_records = c.fetchall()

    for uid, stored_hash, stored_emb in all_records:
        if stored_hash == image_hash:
            return ("Exact Duplicate", uid)
        sim = cosine_similarity(new_emb, stored_emb)
        if sim > 0.85:
            return ("Similar Image", uid)
        if sim > 0.65:
            return ("Same Narrative", uid)
    return ("No Duplicate", None)

# =========================================================
# WEIGHTED SUBCLUSTERING
# =========================================================
def compute_similarity(row_a, row_b):
    t1 = ["billing_country_code", "shipping_country_code", "credit_card_country_code"]
    t1_score = sum(row_a[f] == row_b[f] for f in t1) / len(t1)

    t2 = ["isfba", "has_prime", "fast_lane"]
    t2_score = sum(row_a[f] == row_b[f] for f in t2) / len(t2)

    t3 = ["ip_country_code", "payment_method", "issuing_bank"]
    t3_score = sum(row_a[f] == row_b[f] for f in t3) / len(t3)

    return 0.5 * t1_score + 0.3 * t2_score + 0.2 * t3_score

def assign_subcluster(df, new_row):
    subset = df[df["unique_image_id"] == new_row["unique_image_id"]]
    if subset.empty:
        return f"{new_row['unique_image_id']}_S0"

    best_sim, best_id = 0, None
    for _, row in subset.iterrows():
        sim = compute_similarity(row, new_row)
        if sim > best_sim:
            best_sim, best_id = sim, row["sub_cluster_id"]

    if best_sim >= 0.8:
        return best_id
    else:
        existing = subset["sub_cluster_id"].dropna().unique()
        num = len(existing)
        return f"{new_row['unique_image_id']}_S{num}"

# =========================================================
# AUTO-ADAPTIVE DB INSERT (fixes "wrong number of values" forever)
# =========================================================
def save_to_db(record_dict):
    """Automatically adapts to SQLite schema; never mismatched columns."""
    c.execute("PRAGMA table_info(records)")
    columns = [col[1] for col in c.fetchall()]
    values = [record_dict.get(col, None) for col in columns]

    placeholders = ",".join(["?"] * len(columns))
    query = f"INSERT INTO records VALUES ({placeholders})"

    try:
        c.execute(query, values)
        conn.commit()
    except Exception as e:
        st.error(f"❌ Database insert failed: {e}")

# =========================================================
# STREAMLIT UI
# =========================================================
st.sidebar.title("Menu")
menu = st.sidebar.radio("Go to:", ["Submit Claim", "Database Viewer"])
st.title("Duplicate Image Detection & Weighted Subclustering")

# ---------------------------------------------------------
# SUBMIT CLAIM
# ---------------------------------------------------------
if menu == "Submit Claim":
    st.subheader("Submit New Claim")

    # Row 1: Customer + Order
    col1, col2 = st.columns(2)
    cust_id = col1.text_input("Customer ID")
    order_id = col2.text_input("Order ID")

    # Row 2: IP + Billing
    col3, col4 = st.columns(2)
    ip_code = col3.text_input("IP Country Code")
    billing_code = col4.text_input("Billing Country Code")

    # Row 3: Shipping + Credit Card
    col5, col6 = st.columns(2)
    shipping_code = col5.text_input("Shipping Country Code")
    credit_card_code = col6.text_input("Credit Card Country Code")

    # Row 4: Fast Lane, IsFBA, Has Prime
    col7, col8, col9 = st.columns(3)
    fast_lane = col7.selectbox("Fast Lane", [0, 1])
    isfba = col8.selectbox("Is FBA", [0, 1])
    has_prime = col9.selectbox("Has Prime", [0, 1])

    # Row 5: GL, Payment Method, Issuing Bank
    col10, col11, col12 = st.columns(3)
    gl_code = col10.text_input("GL Code")
    payment_method = col11.text_input("Payment Method")
    issuing_bank = col12.text_input("Issuing Bank")

    # Damage Classification
    damage_classification = st.selectbox(
        "Damage Classification",
        [
            "Burnt", "Spilled", "Broken", "Scratched", "Missing item",
            "Malfunctioning", "Stained", "Packaging Damaged",
            "Expired", "Leaking", "Other"
        ]
    )

    description = st.text_area("Image Description", height=80)
    uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded and st.button("Check for Duplicates"):
        img = Image.open(uploaded).convert("RGB")
        status, matched_uid = check_duplicates(img, description)
        image_hash = get_image_hash(img)
        save_image(img, image_hash)
        embedding = get_embedding(img, description)

        uid = generate_unique_image_id() if status == "No Duplicate" else matched_uid
        df_existing = pd.read_sql_query("SELECT * FROM records", conn)

        new_row = {
            "unique_image_id": uid,
            "ip_country_code": ip_code,
            "billing_country_code": billing_code,
            "shipping_country_code": shipping_code,
            "credit_card_country_code": credit_card_code,
            "fast_lane": fast_lane,
            "isfba": isfba,
            "has_prime": has_prime,
            "payment_method": payment_method,
            "issuing_bank": issuing_bank
        }

        sub_cluster_id = assign_subcluster(df_existing, new_row)

        record = {
            "unique_image_id": uid,
            "sub_cluster_id": sub_cluster_id,
            "customer_id": cust_id,
            "order_id": order_id,
            "ip_country_code": ip_code,
            "billing_country_code": billing_code,
            "shipping_country_code": shipping_code,
            "credit_card_country_code": credit_card_code,
            "fast_lane": fast_lane,
            "isfba": isfba,
            "has_prime": has_prime,
            "gl_code": gl_code,
            "payment_method": payment_method,
            "issuing_bank": issuing_bank,
            "description": description,
            "damage_classification": damage_classification,
            "image_hash": image_hash,
            "embedding": embedding
        }

        save_to_db(record)

        if status == "No Duplicate":
            st.success(f"✅ New main cluster created. Stored under UID: {uid}")
        else:
            st.warning(f"⚠️ {status} found. Linked to existing cluster: {uid}")
        st.info(f"Assigned Subcluster: {sub_cluster_id}")

# ---------------------------------------------------------
# DATABASE VIEWER (Refined and Compact)
# ---------------------------------------------------------
elif menu == "Database Viewer":
    st.subheader("Database Viewer")

    df = pd.read_sql_query(
        "SELECT unique_image_id, sub_cluster_id, customer_id, order_id, "
        "ip_country_code, billing_country_code, shipping_country_code, "
        "credit_card_country_code, fast_lane, isfba, has_prime, gl_code, "
        "payment_method, issuing_bank, damage_classification, image_hash "
        "FROM records", conn)

    if df.empty:
        st.info("No records found.")
    else:
        df = df.replace({None: "", "None": "", np.nan: ""})

        # Top search and controls
        top1, top2, top3 = st.columns([3, 1, 0.8])
        query = top1.text_input("Search", placeholder="Search any field", label_visibility="collapsed")
        show_img = top2.toggle("Show Images", value=True)

        filtered_df = df.copy()
        if query.strip():
            q = query.lower()
            filtered_df = df[df.apply(lambda r: any(q in str(v).lower() for v in r.values), axis=1)]

        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        top3.download_button(
            label="Download",
            data=csv_data
            file_name="claims_database.csv",
            mime="text/csv",
            help="Download current filtered records"
        )

        if filtered_df.empty:
            st.warning("No matching results found.")
        else:
            for uid in filtered_df["unique_image_id"].unique():
                st.markdown(f"####  Main Cluster: `{uid}`")
                sub_df = filtered_df[filtered_df["unique_image_id"] == uid]

                for sc in sub_df["sub_cluster_id"].unique():
                    st.markdown(f"<div style='margin-top:-10px; font-size:14px; color:gray;'>Subcluster: {sc}</div>", unsafe_allow_html=True)
                    cluster_subset = sub_df[sub_df["sub_cluster_id"] == sc].copy()

                    # Add image thumbnail column if enabled
                    if show_img:
                        def get_thumb(hashv):
                            path = os.path.join(IMAGE_FOLDER, f"{hashv}.png")
                            if os.path.exists(path):
                                return f'<img src="data:image/png;base64,{base64.b64encode(open(path,"rb").read()).decode()}" width="50">'
                            return ""
                        import base64
                        cluster_subset["Image"] = cluster_subset["image_hash"].apply(get_thumb)

                    # Select concise columns for display
                    cols_to_show = [
                        "customer_id", "order_id", "ip_country_code",
                        "billing_country_code", "shipping_country_code",
                        "credit_card_country_code", "fast_lane", "isfba",
                        "has_prime", "payment_method", "issuing_bank",
                        "damage_classification"
                    ]
                    if show_img:
                        cols_to_show.append("Image")

                    styled = cluster_subset[cols_to_show].rename(columns={
                        "customer_id": "Customer",
                        "order_id": "Order",
                        "ip_country_code": "IP",
                        "billing_country_code": "Billing",
                        "shipping_country_code": "Ship",
                        "credit_card_country_code": "Card",
                        "fast_lane": "Fast",
                        "isfba": "FBA",
                        "has_prime": "Prime",
                        "payment_method": "Pay",
                        "issuing_bank": "Bank",
                        "damage_classification": "Damage"
                    })

                    st.markdown(styled.to_html(escape=False, index=False), unsafe_allow_html=True)
                    st.markdown("<hr style='margin-top:10px; margin-bottom:20px;'>", unsafe_allow_html=True)