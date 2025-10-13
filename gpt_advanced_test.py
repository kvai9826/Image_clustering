import os
import uuid
import sqlite3
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import faiss

# ----------------------------
# CONFIG
# ----------------------------
DB_FILE = "claims_final_v2.db"
IMAGE_FOLDER = "./images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_DIM = 512
SBERT_DIM = 384

# Weights and thresholds
W_CLIP = 0.6
W_TEXT = 0.4
THRESH_MAIN = 0.80
THRESH_NARRATIVE = 0.65
TOP_K = 10

# ----------------------------
# MODEL INITIALIZATION (Offline)
# ----------------------------
clip_model_path = os.path.abspath("./clip_model_offline")
clip = CLIPModel.from_pretrained(clip_model_path, local_files_only=True).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_path, local_files_only=True)
sbert = SentenceTransformer("./offline_sbert", device=device)

# ----------------------------
# DATABASE
# ----------------------------
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    item_name TEXT,
    description TEXT,
    damage_classification TEXT,
    chat_text TEXT,
    image_hash TEXT,
    image_embedding BLOB,
    text_embedding BLOB
)
""")
conn.commit()

# ----------------------------
# UTILS
# ----------------------------
def gen_uid():
    return f"C-{str(uuid.uuid4())[:8]}"

def save_image_file(image, image_hash):
    path = os.path.join(IMAGE_FOLDER, f"{image_hash}.png")
    image.save(path)
    return path

def phash_str(image):
    return str(imagehash.phash(image))

def get_clip_image_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip.get_image_features(inputs["pixel_values"])
    return emb.cpu().numpy().flatten().astype(np.float32)

def get_text_embedding(text):
    if not text:
        return np.zeros((SBERT_DIM,), dtype=np.float32)
    return sbert.encode([text], normalize_embeddings=True)[0].astype(np.float32)

def save_record_to_db(record):
    c.execute("PRAGMA table_info(records)")
    cols = [r[1] for r in c.fetchall()]
    vals = [record.get(col, None) for col in cols]
    placeholders = ",".join(["?"] * len(cols))
    c.execute(f"INSERT INTO records ({','.join(cols)}) VALUES ({placeholders})", vals)
    conn.commit()
    return c.lastrowid

# ----------------------------
# FAISS per-damage setup
# ----------------------------
faiss_indices = {}

def build_all_faiss_indices():
    global faiss_indices
    faiss_indices = {}
    c.execute("SELECT DISTINCT damage_classification FROM records")
    labels = [r[0] for r in c.fetchall() if r[0]]
    for dmg in labels:
        build_faiss_for_damage(dmg)

def build_faiss_for_damage(damage_label):
    c.execute("SELECT id, image_embedding FROM records WHERE damage_classification=?", (damage_label,))
    rows = c.fetchall()
    index = faiss.IndexFlatIP(CLIP_DIM)
    ids, vecs = [], []
    for rid, blob in rows:
        if not blob: continue
        vec = np.frombuffer(blob, dtype=np.float32).copy()
        if vec.size != CLIP_DIM: continue
        vec /= (np.linalg.norm(vec) + 1e-10)
        ids.append(rid)
        vecs.append(vec)
    if vecs:
        mat = np.vstack(vecs)
        index.add(mat)
        id_map = np.array(ids, dtype=np.int64)
    else:
        id_map = np.array([], dtype=np.int64)
    faiss_indices[damage_label] = (index, id_map)

build_all_faiss_indices()

# ----------------------------
# DUPLICATE CHECK
# ----------------------------
def cosine(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10))

def find_visual_candidates(img_emb, damage_label, k=TOP_K):
    if damage_label not in faiss_indices:
        return []
    index, id_map = faiss_indices[damage_label]
    if id_map.size == 0 or index.ntotal == 0:
        return []
    q = img_emb.copy() / (np.linalg.norm(img_emb) + 1e-10)
    D, I = index.search(q.reshape(1, -1), min(k, index.ntotal))
    return [(int(id_map[i]), float(D[0][j])) for j, i in enumerate(I[0]) if i >= 0]

def check_duplicate_weighted(image, description, damage_label):
    img_emb = get_clip_image_embedding(image)
    img_emb_norm = img_emb / (np.linalg.norm(img_emb) + 1e-10)
    img_hash = phash_str(image)

    c.execute("SELECT unique_image_id, id FROM records WHERE image_hash=?", (img_hash,))
    row = c.fetchone()
    if row:
        return ("Exact Duplicate", row[0], row[1], [])

    candidates = find_visual_candidates(img_emb_norm, damage_label)
    matches = []

    for db_id, clip_sim in candidates:
        c.execute("SELECT unique_image_id, description, text_embedding FROM records WHERE id=?", (db_id,))
        r = c.fetchone()
        if not r: continue
        uid, desc, text_blob = r
        if text_blob:
            stored_text_emb = np.frombuffer(text_blob, dtype=np.float32).copy()
        else:
            stored_text_emb = get_text_embedding((desc or "") + " " + (damage_label or ""))
        query_text_emb = get_text_embedding((description or "") + " " + (damage_label or ""))
        text_sim = cosine(query_text_emb, stored_text_emb)
        final_sim = W_CLIP * clip_sim + W_TEXT * text_sim
        matches.append((db_id, uid, clip_sim, text_sim, final_sim, desc))

    matches.sort(key=lambda x: x[4], reverse=True)
    if matches:
        best = matches[0]
        dbid, uid, cs, ts, fs, _ = best
        if fs >= THRESH_MAIN:
            return ("Same Main Cluster", uid, dbid, matches)
        if fs >= THRESH_NARRATIVE:
            return ("Candidate - Same Narrative", uid, dbid, matches)

    q_text_emb = get_text_embedding((description or "") + " " + (damage_label or ""))
    c.execute("SELECT id, unique_image_id, description FROM records WHERE damage_classification=?", (damage_label,))
    sem_matches = []
    for rid, uid, desc in c.fetchall():
        emb = get_text_embedding((desc or "") + " " + (damage_label or ""))
        t_sim = cosine(q_text_emb, emb)
        sem_matches.append((rid, uid, t_sim))
    sem_matches.sort(key=lambda x: x[2], reverse=True)
    if sem_matches and sem_matches[0][2] >= THRESH_MAIN:
        rid, uid, _ = sem_matches[0]
        return ("Same Narrative (Text)", uid, rid, sem_matches)
    if sem_matches and sem_matches[0][2] >= THRESH_NARRATIVE:
        rid, uid, _ = sem_matches[0]
        return ("Candidate - Narrative (Text)", uid, rid, sem_matches)

    return ("No Duplicate", None, None, [])

# ----------------------------
# SUBCLUSTER LOGIC
# ----------------------------
def metadata_similarity_count(a, b):
    keys = [
        "ip_country_code", "billing_country_code", "shipping_country_code",
        "credit_card_country_code", "payment_method", "fast_lane", "isfba", "has_prime"
    ]
    return sum(str(a.get(k, "")) == str(b.get(k, "")) for k in keys)

def assign_subcluster(df, meta, main_uid):
    subset = df[df["unique_image_id"] == main_uid]
    if subset.empty:
        return f"{main_uid}_S0"
    best_match = -1
    best_sub = None
    for _, r in subset.iterrows():
        m = metadata_similarity_count(r, meta)
        if m > best_match:
            best_match = m
            best_sub = r["sub_cluster_id"]
    if 8 - best_match >= 3:
        new_idx = len(subset["sub_cluster_id"].dropna().unique())
        return f"{main_uid}_S{new_idx}"
    return best_sub or f"{main_uid}_S0"

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(layout="wide")
st.title("Smart Duplicate Detection — CLIP + SBERT")

menu = st.sidebar.radio("Navigate", ["Submit Claim", "Database Viewer", "Rebuild Index"])

if menu == "Rebuild Index":
    st.info("Rebuilding FAISS indices...")
    build_all_faiss_indices()
    st.success("Rebuild complete.")
    st.stop()

# ----------------------------
# SUBMIT CLAIM
# ----------------------------
if menu == "Submit Claim":
    st.header("Submit Claim")

    c1, c2 = st.columns(2)
    cust_id = c1.text_input("Customer ID")
    order_id = c2.text_input("Order ID")

    c3, c4 = st.columns(2)
    ip_code = c3.text_input("IP Country Code")
    billing_code = c4.text_input("Billing Country Code")

    c5, c6 = st.columns(2)
    ship_code = c5.text_input("Shipping Country Code")
    card_code = c6.text_input("Credit Card Country Code")

    c7, c8, c9 = st.columns(3)
    fast_lane = c7.selectbox("Fast Lane", [0, 1])
    isfba = c8.selectbox("Is FBA", [0, 1])
    has_prime = c9.selectbox("Has Prime", [0, 1])

    c10, c11, c12 = st.columns(3)
    gl_code = c10.text_input("GL Code")
    payment = c11.text_input("Payment Method")
    bank = c12.text_input("Issuing Bank")

    item = st.text_input("Item Name")
    desc = st.text_area("Image Description", height=80)
    damage = st.selectbox("Damage Classification", [
        "Burnt", "Spilled", "Broken", "Scratched", "Missing item",
        "Malfunctioning", "Stained", "Packaging Damaged", "Expired", "Leaking", "Other"
    ])
    chat = st.text_area("Chat Conversation (optional)", height=100)
    uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded and st.button("Check & Save"):
        img = Image.open(uploaded).convert("RGB")
        status, matched_uid, matched_dbid, matches = check_duplicate_weighted(img, desc, damage)

        img_emb = get_clip_image_embedding(img)
        txt_emb = get_text_embedding((desc or "") + " " + (damage or ""))
        img_hash = phash_str(img)
        save_image_file(img, img_hash)

        if status in (
            "Exact Duplicate",
            "Same Main Cluster",
            "Same Narrative (Text)",
            "Candidate - Same Narrative",
            "Candidate - Narrative (Text)"
        ):
            uid = matched_uid
        else:
            uid = gen_uid()

        df_all = pd.read_sql_query("SELECT * FROM records", conn)
        new_meta = {
            "ip_country_code": ip_code,
            "billing_country_code": billing_code,
            "shipping_country_code": ship_code,
            "credit_card_country_code": card_code,
            "payment_method": payment,
            "fast_lane": fast_lane,
            "isfba": isfba,
            "has_prime": has_prime
        }
        sub_id = assign_subcluster(df_all, new_meta, uid)

        record = {
            "unique_image_id": uid,
            "sub_cluster_id": sub_id,
            "customer_id": cust_id,
            "order_id": order_id,
            "ip_country_code": ip_code,
            "billing_country_code": billing_code,
            "shipping_country_code": ship_code,
            "credit_card_country_code": card_code,
            "fast_lane": fast_lane,
            "isfba": isfba,
            "has_prime": has_prime,
            "gl_code": gl_code,
            "payment_method": payment,
            "issuing_bank": bank,
            "item_name": item,
            "description": desc,
            "damage_classification": damage,
            "chat_text": chat,
            "image_hash": img_hash,
            "image_embedding": img_emb.tobytes(),
            "text_embedding": txt_emb.tobytes()
        }

        save_record_to_db(record)
        build_faiss_for_damage(damage)

        # --- Simplified Result Summary ---
        if status == "Exact Duplicate":
            st.success(f"Exact duplicate found in Cluster `{uid}` and Subcluster `{sub_id}`.")
        elif status in (
            "Same Main Cluster",
            "Same Narrative (Text)",
            "Candidate - Same Narrative",
            "Candidate - Narrative (Text)"
        ):
            st.warning(f"Similar image or narrative found in Cluster `{uid}` and Subcluster `{sub_id}`.")
        else:
            st.info(f"No duplicates found. Creating a new Cluster `{uid}` with Subcluster `{sub_id}`.")

        if matches and len(matches[0]) >= 5:
            dbid, uid_m, cs, ts, fs, _ = matches[0]
            st.markdown("#### Similarity Scores (Top Match)")
            st.table(pd.DataFrame([{
                "Cluster": uid_m,
                "Visual": f"{cs:.2f}",
                "Text": f"{ts:.2f}",
                "Combined": f"{fs:.2f}"
            }]))

# ----------------------------
# DATABASE VIEWER
# ----------------------------
elif menu == "Database Viewer":
    st.title("Database Viewer — Table View")
    df = pd.read_sql_query("""
        SELECT id, unique_image_id, sub_cluster_id, customer_id, order_id,
               ip_country_code, billing_country_code, shipping_country_code,
               credit_card_country_code, fast_lane, isfba, has_prime,
               gl_code, payment_method, issuing_bank, item_name, description,
               damage_classification, chat_text, image_hash
        FROM records
    """, conn)

    if df.empty:
        st.info("No records found.")
    else:
        q_col, img_col, chat_col, dl_col = st.columns([3, 1, 1, 1])
        q = q_col.text_input("Search (any field or chat)")
        show_img = img_col.toggle("Show Images", True)
        show_chat = chat_col.toggle("Show Chat", False)

        csv = df.to_csv(index=False).encode("utf-8")
        dl_col.download_button("Download CSV", csv, "claims.csv")

        if q:
            ql = q.lower()
            df = df[df.apply(lambda r: ql in " ".join(map(str, r.values)).lower(), axis=1)]

        for uid in df["unique_image_id"].unique():
            st.markdown(f"### Main Cluster: `{uid}`")
            sub_df = df[df["unique_image_id"] == uid]
            for sc in sub_df["sub_cluster_id"].unique():
                st.caption(f"Subcluster: {sc}")
                sub = sub_df[sub_df["sub_cluster_id"] == sc].copy()
                sub_disp = sub.drop(columns=["unique_image_id", "sub_cluster_id", "image_hash"], errors="ignore")
                if not show_chat:
                    sub_disp = sub_disp.drop(columns=["chat_text"], errors="ignore")

                col_table, col_img = st.columns([5, 1.5])
                with col_table:
                    st.dataframe(sub_disp, use_container_width=True, hide_index=True)
                with col_img:
                    if show_img:
                        st.write("")
                        for _, r in sub.iterrows():
                            p = os.path.join(IMAGE_FOLDER, f"{r['image_hash']}.png")
                            if os.path.exists(p):
                                st.image(p, width=40)
                            else:
                                st.empty()