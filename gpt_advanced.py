# smart_claims_final.py
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
DB_FILE = "claims_final.db"
IMAGE_FOLDER = "./images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_DIM = 512
SBERT_DIM = 384

# similarity weights and thresholds
W_CLIP = 0.6
W_TEXT = 0.4
THRESH_MAIN = 0.80
THRESH_NARRATIVE = 0.65
TOP_K = 10  # how many visual candidates to retrieve per search (we filter by damage)

# ----------------------------
# MODELS (offline)
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

# Create table if not exists
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

def save_image_file(image: Image.Image, image_hash: str):
    path = os.path.join(IMAGE_FOLDER, f"{image_hash}.png")
    image.save(path)
    return path

def phash_str(image: Image.Image):
    return str(imagehash.phash(image))

def get_clip_image_embedding(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip.get_image_features(inputs["pixel_values"])
    return emb.cpu().numpy().flatten().astype(np.float32)

def get_text_embedding(text: str):
    if not text:
        return np.zeros((SBERT_DIM,), dtype=np.float32)
    e = sbert.encode([text], normalize_embeddings=True)[0]
    return e.astype(np.float32)

def save_record_to_db(record: dict):
    # automatically adapt to schema: prepare value list in column order
    c.execute("PRAGMA table_info(records)")
    cols = [r[1] for r in c.fetchall()]
    vals = [record.get(col, None) for col in cols]
    placeholders = ",".join(["?"] * len(cols))
    query = f"INSERT INTO records ({','.join(cols)}) VALUES ({placeholders})"
    c.execute(query, vals)
    conn.commit()
    return c.lastrowid

# ----------------------------
# FAISS indices per damage class
# ----------------------------
# structure: damage_label -> (faiss_index, np.array(id_map))
faiss_indices = {}  # damage -> (index, id_map)

def build_all_faiss_indices():
    """Build FAISS indices for each damage_class available in DB."""
    global faiss_indices
    faiss_indices = {}
    c.execute("SELECT DISTINCT damage_classification FROM records")
    damage_labels = [r[0] for r in c.fetchall() if r[0]]
    for dmg in damage_labels:
        build_faiss_for_damage(dmg)

def build_faiss_for_damage(damage_label: str):
    """Build FAISS index for a specific damage label."""
    c.execute("SELECT id, image_embedding FROM records WHERE damage_classification=? AND image_embedding IS NOT NULL", (damage_label,))
    rows = c.fetchall()
    index = faiss.IndexFlatIP(CLIP_DIM)
    ids = []
    vecs = []
    for rid, blob in rows:
        # blob may be None or read-only view, take a copy
        try:
            vec = np.frombuffer(blob, dtype=np.float32).copy()
        except Exception:
            continue
        if vec.size != CLIP_DIM:
            continue
        # normalize for inner-product (cosine with normalized vectors)
        norm = np.linalg.norm(vec) + 1e-10
        vec = vec / norm
        vecs.append(vec)
        ids.append(rid)
    if vecs:
        mat = np.vstack(vecs)
        index.add(mat)
        id_map = np.array(ids, dtype=np.int64)
    else:
        id_map = np.array([], dtype=np.int64)
    faiss_indices[damage_label] = (index, id_map)

# build indices at startup
build_all_faiss_indices()

# ----------------------------
# SIMILARITY & DUPLICATE LOGIC
# ----------------------------
def cosine(a, b):
    # a and b numpy arrays
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10))

def find_visual_candidates_by_damage(img_emb: np.ndarray, damage_label: str, k=TOP_K):
    """Return list of (db_id, clip_sim) candidates for given damage label."""
    if damage_label not in faiss_indices:
        return []
    index, id_map = faiss_indices[damage_label]
    if id_map.size == 0 or index.ntotal == 0:
        return []
    q = img_emb.copy()
    q = q / (np.linalg.norm(q) + 1e-10)
    D, I = index.search(q.reshape(1, -1), min(k, index.ntotal))
    results = []
    for sim, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        db_id = int(id_map[idx])
        results.append((db_id, float(sim)))
    return results

def check_duplicate_weighted(image: Image.Image, description: str, damage_label: str):
    """
    Phase 1: search visual candidates within same damage class (fast via FAISS)
    For each candidate compute clip_sim (from FAISS) and text_sim (SBERT(desc+damage vs stored desc+damage))
    final_sim = W_CLIP*clip_sim + W_TEXT*text_sim
    Decision made by thresholds.
    If no visual candidate yields final_sim >= THRESH_NARRATIVE, run Phase 2 semantic-only over descriptions within same damage class.
    """
    img_emb = get_clip_image_embedding(image)
    # ensure normalized copy for reuse
    img_emb_norm = img_emb / (np.linalg.norm(img_emb) + 1e-10)

    # quick exact-hash check
    img_hash = phash_str(image)
    c.execute("SELECT unique_image_id, id FROM records WHERE image_hash=?", (img_hash,))
    row = c.fetchone()
    if row:
        return ("Exact Duplicate", row[0], row[1], [])

    # Phase 1: visual candidates
    candidates = find_visual_candidates_by_damage(img_emb_norm, damage_label, k=TOP_K)
    matches = []
    for db_id, clip_sim in candidates:
        # fetch stored description & text embedding & unique id
        c.execute("SELECT unique_image_id, description, text_embedding FROM records WHERE id=?", (db_id,))
        r = c.fetchone()
        if not r:
            continue
        uid, stored_desc, stored_text_blob = r
        # compute text sim: if we have stored_text_embedding blob, use it; else compute on the fly
        if stored_text_blob:
            stored_text_emb = np.frombuffer(stored_text_blob, dtype=np.float32).copy()
        else:
            stored_text_emb = get_text_embedding((stored_desc or "") + " " + (damage_label or ""))
        query_text_emb = get_text_embedding((description or "") + " " + (damage_label or ""))
        text_sim = cosine(query_text_emb, stored_text_emb)
        final_sim = W_CLIP * clip_sim + W_TEXT * text_sim
        matches.append((db_id, uid, clip_sim, text_sim, final_sim, stored_desc))

    # sort by final_sim descending
    matches.sort(key=lambda x: x[4], reverse=True)

    # decide best match
    if matches:
        best = matches[0]
        db_id, uid, clip_sim, text_sim, final_sim, stored_desc = best
        if final_sim >= THRESH_MAIN:
            return ("Same Main Cluster", uid, db_id, matches)
        if final_sim >= THRESH_NARRATIVE:
            return ("Candidate - Same Narrative", uid, db_id, matches)

    # Phase 2: semantic fallback across text descriptions for same damage label
    # (only runs if Phase1 didn't return strong match)
    q_text_emb = get_text_embedding((description or "") + " " + (damage_label or ""))
    c.execute("SELECT id, unique_image_id, description FROM records WHERE damage_classification=?", (damage_label,))
    sem_matches = []
    for rid, uid, stored_desc in c.fetchall():
        stored_emb = get_text_embedding((stored_desc or "") + " " + (damage_label or ""))
        t_sim = cosine(q_text_emb, stored_emb)
        sem_matches.append((rid, uid, t_sim))
    sem_matches.sort(key=lambda x: x[2], reverse=True)
    if sem_matches and sem_matches[0][2] >= THRESH_MAIN:
        rid, uid, t_sim = sem_matches[0]
        return ("Same Narrative (Text)", uid, rid, sem_matches[:TOP_K])
    if sem_matches and sem_matches[0][2] >= THRESH_NARRATIVE:
        rid, uid, t_sim = sem_matches[0]
        return ("Candidate - Narrative (Text)", uid, rid, sem_matches[:TOP_K])

    return ("No Duplicate", None, None, [])

# ----------------------------
# SUBCLUSTER ASSIGNMENT
# ----------------------------
def metadata_similarity_count(row_a: dict, row_b: dict):
    keys = [
        "ip_country_code", "billing_country_code", "shipping_country_code",
        "credit_card_country_code", "payment_method", "fast_lane", "isfba", "has_prime"
    ]
    matches = 0
    for k in keys:
        if str(row_a.get(k, "")).strip() == str(row_b.get(k, "")).strip():
            matches += 1
    return matches  # number of matches

def assign_subcluster(existing_df: pd.DataFrame, new_meta: dict, main_uid: str):
    """
    existing_df: full db table as DataFrame
    new_meta: dict with metadata fields (ip..., billing..., fast_lane etc.)
    main_uid: unique_image_id for main cluster
    Strategy:
      - look at subset with unique_image_id == main_uid
      - if empty -> _S0
      - otherwise compute match counts, pick best match
      - if number_of_differences >= 3 -> create new subcluster
    """
    subset = existing_df[existing_df["unique_image_id"] == main_uid]
    if subset.empty:
        return f"{main_uid}_S0"
    best_match = -1
    best_sub = None
    for _, r in subset.iterrows():
        rdict = r.to_dict()
        matches = metadata_similarity_count(rdict, new_meta)
        if matches > best_match:
            best_match = matches
            best_sub = rdict.get("sub_cluster_id")
    # number of fields = 8 -> differences = 8 - matches
    differences = 8 - best_match
    if differences >= 3:
        existing_subs = subset["sub_cluster_id"].dropna().unique().tolist()
        new_index = len(existing_subs)
        return f"{main_uid}_S{new_index}"
    else:
        return best_sub or f"{main_uid}_S0"

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(layout="wide")
st.title("Smart Duplicate Detection — Image-first (CLIP) + Text (SBERT)")

menu = st.sidebar.radio("Go to:", ["Submit Claim", "Database Viewer", "Rebuild Index"])

if menu == "Rebuild Index":
    st.info("Rebuilding per-damage FAISS indices...")
    build_all_faiss_indices()
    st.success("Rebuild complete.")
    st.stop()

# ----------------------------
# Submit Claim screen
# ----------------------------
if menu == "Submit Claim":
    st.header("Submit Claim")

    # Row 1: Customer & Order
    c1, c2 = st.columns(2)
    customer_id = c1.text_input("Customer ID")
    order_id = c2.text_input("Order ID")

    # Row 2: IP + Billing
    c3, c4 = st.columns(2)
    ip_country = c3.text_input("IP Country Code")
    billing_country = c4.text_input("Billing Country Code")

    # Row 3: Ship + Card
    c5, c6 = st.columns(2)
    shipping_country = c5.text_input("Shipping Country Code")
    card_country = c6.text_input("Credit Card Country Code")

    # Row 4: fast, fba, prime
    c7, c8, c9 = st.columns(3)
    fast_lane = int(c7.selectbox("Fast Lane", [0, 1]))
    isfba = int(c8.selectbox("Is FBA", [0, 1]))
    has_prime = int(c9.selectbox("Has Prime", [0, 1]))

    # Row 5: GL, payment, bank
    c10, c11, c12 = st.columns(3)
    gl_code = c10.text_input("GL Code")
    payment_method = c11.text_input("Payment Method")
    issuing_bank = c12.text_input("Issuing Bank")

    item_name = st.text_input("Item Name")
    description = st.text_area("Image Description", height=80)
    damage = st.selectbox("Damage Classification", [
        "Burnt", "Spilled", "Broken", "Scratched", "Missing item",
        "Malfunctioning", "Stained", "Packaging Damaged", "Expired", "Leaking", "Other"
    ])
    chat_text = st.text_area("Chat (optional)", height=100)
    uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded and st.button("Check & Save"):
        img = Image.open(uploaded).convert("RGB")
        # 1) check duplicates using damage-class filtered FAISS + text refinement
        status, matched_uid, matched_dbid, matches = check_duplicate_weighted(img, description, damage)

        # 2) prepare embeddings and store
        img_emb = get_clip_image_embedding(img)
        text_emb = get_text_embedding((description or "") + " " + (damage or ""))
        img_hash = phash_str(img)
        save_image_file(img, img_hash)

        # pick uid
        if status in ("Exact Duplicate", "Same Main Cluster", "Candidate - Same Narrative", "Same Narrative (Text)", "Candidate - Narrative (Text)"):
            uid = matched_uid
        else:
            uid = gen_uid()

        # subcluster assignment
        df_all = pd.read_sql_query("SELECT * FROM records", conn)
        new_meta = {
            "ip_country_code": ip_country,
            "billing_country_code": billing_country,
            "shipping_country_code": shipping_country,
            "credit_card_country_code": card_country,
            "payment_method": payment_method,
            "fast_lane": fast_lane,
            "isfba": isfba,
            "has_prime": has_prime
        }
        sub_id = assign_subcluster(df_all, new_meta, uid)

        # build record dict to insert
        record = {
            "unique_image_id": uid,
            "sub_cluster_id": sub_id,
            "customer_id": customer_id,
            "order_id": order_id,
            "ip_country_code": ip_country,
            "billing_country_code": billing_country,
            "shipping_country_code": shipping_country,
            "credit_card_country_code": card_country,
            "fast_lane": fast_lane,
            "isfba": isfba,
            "has_prime": has_prime,
            "gl_code": gl_code,
            "payment_method": payment_method,
            "issuing_bank": issuing_bank,
            "item_name": item_name,
            "description": description,
            "damage_classification": damage,
            "chat_text": chat_text,
            "image_hash": img_hash,
            # store embeddings as bytes
            "image_embedding": img_emb.tobytes(),
            "text_embedding": text_emb.tobytes()
        }
        new_db_id = save_record_to_db(record)

        # update FAISS index for this damage label
        # append into in-memory index: rebuild for simplicity
        build_faiss_for_damage(damage)

        # show results
        if status in ("Exact Duplicate", "Same Main Cluster"):
            st.success(f"Linked to existing main cluster {uid} (match reason: {status})")
        elif status.startswith("Candidate"):
            st.warning(f"Candidate match found to cluster {uid} (reason: {status}) — review matches in CSV download")
        elif status.startswith("Same Narrative"):
            st.info(f"Matched by text/narrative to cluster {uid} (reason: {status})")
        else:
            st.success(f"Created new main cluster {uid}")

        st.info(f"Assigned subcluster: {sub_id}")
        # optional: show top matches for inspection
        if matches:
            match_table = []
            for m in matches[:10]:
                # matches contain tuples depending on stage -> standardize display
                if len(m) >= 5:
                    dbid, uid_m, clip_s, text_s, final_s, stored_desc = m
                    match_table.append({
                        "db_id": dbid, "cluster": uid_m,
                        "clip_sim": round(clip_s, 3),
                        "text_sim": round(text_s, 3),
                        "final_sim": round(final_s, 3),
                        "stored_desc": (stored_desc or "")[:80]
                    })
            if match_table:
                st.table(pd.DataFrame(match_table))


#--------DATABASE VEIWER-------------
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
        query_col, img_col, chat_col, dl_col = st.columns([3, 1, 1, 1])
        q = query_col.text_input("Search (any field or chat)")
        show_images = img_col.toggle("Show Images", value=True)
        show_chat = chat_col.toggle("Show Chat", value=False)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        dl_col.download_button("Download CSV", data=csv_bytes, file_name="claims.csv")

        if q:
            ql = q.lower()
            df = df[df.apply(lambda r: ql in " ".join(map(str, r.values)).lower(), axis=1)]

        for uid in df["unique_image_id"].unique():
            st.markdown(f"### Main Cluster: `{uid}`")
            sub_df = df[df["unique_image_id"] == uid]

            for sc in sub_df["sub_cluster_id"].unique():
                st.caption(f"Subcluster: {sc}")
                sub = sub_df[sub_df["sub_cluster_id"] == sc].copy()

                # Drop internal columns
                sub_display = sub.drop(
                    columns=["unique_image_id", "sub_cluster_id", "image_hash"], errors="ignore"
                )
                if not show_chat:
                    sub_display = sub_display.drop(columns=["chat_text"], errors="ignore")

                # Table on left, images on right
                col_table, col_img = st.columns([5, 1.5])

                with col_table:
                    st.dataframe(sub_display, use_container_width=True, hide_index=True)

                with col_img:
                    if show_images:
                        # Add top padding to align thumbnails with first row of dataframe
                        st.write("")  # Blank line acts as small spacer

                        # Display actual images for each record
                        for _, row in sub.iterrows():
                            img_path = os.path.join(IMAGE_FOLDER, f"{row['image_hash']}.png")

                            if os.path.exists(img_path):
                                st.image(img_path, width=40)
                            else:
                                # Maintain alignment for missing images
                                st.empty()
                    else:
                        st.empty()