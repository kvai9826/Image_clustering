# smart_claims_optimized.py
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
from typing import Tuple, List

# ----------------------------
# CONFIG
# ----------------------------
DB_FILE = "claims_final_refund.db"
IMAGE_FOLDER = "./images"
FAISS_FOLDER = "./faiss_indices"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(FAISS_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_DIM = 512
SBERT_DIM = 384

W_CLIP = 0.6
W_TEXT = 0.4
THRESH_MAIN = 0.80
THRESH_NARRATIVE = 0.65
TOP_K = 8  # number of visual candidates to retrieve
THUMB_WIDTH = 40  # thumbnail size in DB viewer
PAGINATION_SIZE = 20  # rows to show per cluster page

# ----------------------------
# DATABASE SETUP
# ----------------------------
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL;")  # reduce write-locking issues
c = conn.cursor()

# ensure expected table exists (keeps compatibility with your schema)
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
    value_usd REAL,
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
# MODEL LOADING (cached)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    clip_model_path = os.path.abspath("./clip_model_offline")
    # Load CLIP model + processor
    clip = CLIPModel.from_pretrained(clip_model_path, local_files_only=True).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path, local_files_only=True)
    # Load offline sentence transformer
    sbert = SentenceTransformer("./offline_sbert", device=device)
    return clip, clip_processor, sbert

clip, clip_processor, sbert = load_models()

# ----------------------------
# UTILS
# ----------------------------
def gen_uid() -> str:
    return f"C-{str(uuid.uuid4())[:8]}"

def phash_str(image: Image.Image) -> str:
    return str(imagehash.phash(image))

def save_image_file(image: Image.Image, image_hash: str) -> str:
    path = os.path.join(IMAGE_FOLDER, f"{image_hash}.png")
    image.save(path)
    return path

def get_clip_image_embedding(image: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip.get_image_features(inputs["pixel_values"])
    return emb.cpu().numpy().flatten().astype(np.float32)

@st.cache_data(show_spinner=False)
def get_text_embedding_cached(text: str) -> np.ndarray:
    if not text:
        return np.zeros((SBERT_DIM,), dtype=np.float32)
    e = sbert.encode([text], normalize_embeddings=True)[0]
    return e.astype(np.float32)

def bytes_to_vec(blob) -> np.ndarray:
    if blob is None:
        return None
    try:
        return np.frombuffer(blob, dtype=np.float32).copy()
    except Exception:
        return None

def save_record_to_db(record: dict) -> int:
    c.execute("PRAGMA table_info(records)")
    cols = [r[1] for r in c.fetchall()]
    vals = [record.get(col, None) for col in cols]
    placeholders = ",".join(["?"] * len(cols))
    c.execute(f"INSERT INTO records ({','.join(cols)}) VALUES ({placeholders})", vals)
    conn.commit()
    return c.lastrowid

# ----------------------------
# FAISS: per-damage incremental indices
# ----------------------------
# structure: damage_label -> {"index": faiss.IndexFlatIP, "id_map": np.ndarray}
faiss_store = {}

def faiss_index_path(damage_label: str):
    return os.path.join(FAISS_FOLDER, f"{damage_label}.index")

def faiss_idmap_path(damage_label: str):
    return os.path.join(FAISS_FOLDER, f"{damage_label}_ids.npy")

def build_faiss_for_damage(damage_label: str):
    """Load or build FAISS index for damage_label from DB (copy blobs)."""
    # If persisted index exists, try to load
    idx_path = faiss_index_path(damage_label)
    ids_path = faiss_idmap_path(damage_label)
    try:
        if os.path.exists(idx_path) and os.path.exists(ids_path):
            index = faiss.read_index(idx_path)
            id_map = np.load(ids_path, allow_pickle=False)
            faiss_store[damage_label] = {"index": index, "id_map": id_map}
            return
    except Exception:
        # If load fails, rebuild from DB below
        pass

    c.execute("SELECT id, image_embedding FROM records WHERE damage_classification=?", (damage_label,))
    rows = c.fetchall()
    vecs = []
    ids = []
    for rid, blob in rows:
        v = bytes_to_vec(blob)
        if v is None or v.size != CLIP_DIM:
            continue
        v = v / (np.linalg.norm(v) + 1e-10)
        vecs.append(v)
        ids.append(rid)
    if vecs:
        mat = np.vstack(vecs)
        index = faiss.IndexFlatIP(CLIP_DIM)
        index.add(mat)
        id_map = np.array(ids, dtype=np.int64)
    else:
        index = faiss.IndexFlatIP(CLIP_DIM)
        id_map = np.array([], dtype=np.int64)
    faiss_store[damage_label] = {"index": index, "id_map": id_map}
    # persist
    try:
        faiss.write_index(index, idx_path)
        np.save(ids_path, id_map)
    except Exception:
        pass

def build_all_faiss_indices():
    c.execute("SELECT DISTINCT damage_classification FROM records")
    labels = [r[0] for r in c.fetchall() if r[0]]
    for lbl in labels:
        build_faiss_for_damage(lbl)

# build indices at start (fast if persisted)
build_all_faiss_indices()

def add_vector_to_faiss(damage_label: str, vector: np.ndarray, db_id: int):
    """Append vector to the in-memory index and persist. Avoid full rebuild."""
    if damage_label not in faiss_store:
        build_faiss_for_damage(damage_label)
    store = faiss_store[damage_label]
    index = store["index"]
    id_map = store["id_map"]
    v = vector.copy()
    v = v / (np.linalg.norm(v) + 1e-10)
    # faiss IndexFlatIP supports add
    index.add(v.reshape(1, -1))
    id_map = np.append(id_map, np.array([db_id], dtype=np.int64))
    store["index"] = index
    store["id_map"] = id_map
    # persist
    try:
        faiss.write_index(index, faiss_index_path(damage_label))
        np.save(faiss_idmap_path(damage_label), id_map)
    except Exception:
        pass

# ----------------------------
# DUPLICATE CHECK (Phase 1 visual -> Phase 2 semantic)
# ----------------------------
def find_visual_candidates(img_emb: np.ndarray, damage_label: str, k: int = TOP_K) -> List[Tuple[int, float]]:
    if damage_label not in faiss_store:
        return []
    index = faiss_store[damage_label]["index"]
    id_map = faiss_store[damage_label]["id_map"]
    if id_map.size == 0 or index.ntotal == 0:
        return []
    q = img_emb.copy()
    q = q / (np.linalg.norm(q) + 1e-10)
    D, I = index.search(q.reshape(1, -1), min(k, int(index.ntotal)))
    results = []
    for pos, idx in enumerate(I[0]):
        if idx < 0:
            continue
        db_id = int(id_map[idx])
        sim = float(D[0][pos])
        results.append((db_id, sim))
    return results

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10))

def check_duplicate_weighted(image: Image.Image, description: str, damage_label: str):
    """
    - Phase 0 quick hash exact match
    - Phase 1: visual candidates within same damage class (FAISS) -> compute combined score (clip*W + text*W)
    - Phase 2: semantic-only fallback across descriptions within same damage class
    Returns tuple: (status_string, matched_cluster_uid_or_None, matched_db_id_or_None, matches_list)
    """
    img_emb = get_clip_image_embedding(image)
    img_emb_norm = img_emb / (np.linalg.norm(img_emb) + 1e-10)
    img_hash = phash_str(image)

    # Phase 0: exact hash quick check
    c.execute("SELECT unique_image_id, id, sub_cluster_id FROM records WHERE image_hash=?", (img_hash,))
    row = c.fetchone()
    if row:
        uid, dbid, scid = row
        return ("Exact Duplicate", uid, dbid, [{"dbid": dbid, "cluster": uid, "subcluster": scid}])

    # Phase 1: FAISS visual candidates filtered by damage
    candidates = find_visual_candidates(img_emb_norm, damage_label)
    matches = []
    if candidates:
        for db_id, clip_sim in candidates:
            c.execute("SELECT unique_image_id, description, text_embedding, sub_cluster_id FROM records WHERE id=?", (db_id,))
            r = c.fetchone()
            if not r:
                continue
            uid, stored_desc, text_blob, stored_sc = r
            if text_blob:
                stored_text_emb = bytes_to_vec(text_blob)
            else:
                stored_text_emb = get_text_embedding_cached((stored_desc or "") + " " + (damage_label or ""))
            query_text_emb = get_text_embedding_cached((description or "") + " " + (damage_label or ""))
            text_sim = cosine(query_text_emb, stored_text_emb)
            final_sim = W_CLIP * clip_sim + W_TEXT * text_sim
            matches.append((db_id, uid, stored_sc, clip_sim, text_sim, final_sim, stored_desc))

        # sort candidates by final combined similarity descending
        matches.sort(key=lambda x: x[5], reverse=True)

        best = matches[0]
        dbid, uid, scid, csim, tsim, fscore, _ = best
        if fscore >= THRESH_MAIN:
            return ("Similar image/Narrative found in Cluster", uid, dbid, [{"dbid": dbid, "cluster": uid, "subcluster": scid, "visual": csim, "text": tsim, "combined": fscore}])
        if fscore >= THRESH_NARRATIVE:
            return ("Similar image/Narrative found in Cluster", uid, dbid, [{"dbid": dbid, "cluster": uid, "subcluster": scid, "visual": csim, "text": tsim, "combined": fscore}])

    # Phase 2: semantic fallback searching descriptions under the same damage label
    query_text_emb = get_text_embedding_cached((description or "") + " " + (damage_label or ""))
    c.execute("SELECT id, unique_image_id, description, sub_cluster_id, text_embedding FROM records WHERE damage_classification=?", (damage_label,))
    sem_matches = []
    for rid, ruid, rdesc, rsc, rblob in c.fetchall():
        if rblob:
            r_emb = bytes_to_vec(rblob)
        else:
            r_emb = get_text_embedding_cached((rdesc or "") + " " + (damage_label or ""))
        t_sim = cosine(query_text_emb, r_emb)
        sem_matches.append((rid, ruid, rsc, t_sim, rdesc))
    sem_matches.sort(key=lambda x: x[3], reverse=True)
    if sem_matches:
        rid, ruid, rsc, t_sim, rdesc = sem_matches[0]
        if t_sim >= THRESH_MAIN:
            return ("Similar image/Narrative found in Cluster", ruid, rid, [{"dbid": rid, "cluster": ruid, "subcluster": rsc, "text": t_sim}])
        if t_sim >= THRESH_NARRATIVE:
            return ("Similar image/Narrative found in Cluster", ruid, rid, [{"dbid": rid, "cluster": ruid, "subcluster": rsc, "text": t_sim}])

    return ("No Duplicate", None, None, [])

# ----------------------------
# SUBCLUSTERING (metadata-based)
# ----------------------------
def metadata_similarity_count(a: dict, b: dict) -> int:
    keys = [
        "ip_country_code", "billing_country_code", "shipping_country_code",
        "credit_card_country_code", "payment_method", "fast_lane", "isfba", "has_prime"
    ]
    matches = 0
    for k in keys:
        if str(a.get(k, "")).strip() == str(b.get(k, "")).strip():
            matches += 1
    return matches

def assign_subcluster(existing_df: pd.DataFrame, new_meta: dict, main_uid: str) -> str:
    subset = existing_df[existing_df["unique_image_id"] == main_uid]
    if subset.empty:
        return f"{main_uid}_S0"
    best_match = -1
    best_sub = None
    for _, row in subset.iterrows():
        m = metadata_similarity_count(row.to_dict(), new_meta)
        if m > best_match:
            best_match = m
            best_sub = row["sub_cluster_id"]
    differences = 8 - best_match
    if differences >= 3:
        existing_subs = subset["sub_cluster_id"].dropna().unique().tolist()
        new_index = len(existing_subs)
        return f"{main_uid}_S{new_index}"
    return best_sub or f"{main_uid}_S0"

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(layout="wide")
st.title("Duplicate image identification and clustering")

menu = st.sidebar.radio("Go to:", ["Submit Claim", "Database Viewer"])


# ----------------------------
# SUBMIT CLAIM UI
# ----------------------------
if menu == "Submit Claim":
    st.header("Submit Claim")

    # Row 1: Account type, Customer + Order
    r1c1, r1c2= st.columns([2, 2])
    #account_type = r1c1.selectbox("Account Type", ["Normal", "Solicit", "Warn", "AOC enforced", "Abusive", "IVF", "Fraud"])
    customer_id = r1c1.text_input("Customer ID")
    order_id = r1c2.text_input("Order ID")

    # Row 2: IP + Billing
    c1, c2 = st.columns(2)
    ip_country = c1.text_input("IP Country Code")
    billing_country = c2.text_input("Billing Country Code")

    # Row 3: Ship + Card
    c3, c4 = st.columns(2)
    shipping_country = c3.text_input("Shipping Country Code")
    card_country = c4.text_input("Credit Card Country Code")

    # Row 4: flags
    c5, c6, c7 = st.columns(3)
    fast_lane = int(c5.selectbox("Fast Lane", [0, 1]))
    isfba = int(c6.selectbox("Is FBA", [0, 1]))
    has_prime = int(c7.selectbox("Has Prime", [0, 1]))

    # Row 5: GL, payment, bank
    c8, c9, c10 = st.columns(3)
    gl_code = c8.text_input("GL Code")
    payment_method = c9.text_input("Payment Method")
    issuing_bank = c10.text_input("Issuing Bank")

    item_name = st.text_input("Item Name")
    value_usd_input = st.text_input("Claim Value ($)", value="")
    try:
        value_usd = float(value_usd_input.strip()) if value_usd_input.strip() else None
    except Exception:
        st.warning("Please enter numeric Claim Value. Leaving blank will store NULL.")
        value_usd = None

    description = st.text_area("Image Description", height=80)
    damage = st.selectbox("Damage Classification", ["Burnt","Spilled","Broken","Scratched","Missing item","Malfunctioning","Stained","Packaging Damaged","Expired","Leaking","Other"])
    chat_text = st.text_area("Chat Conversation (optional)", height=100)
    uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if uploaded and st.button("Check & Save"):
        img = Image.open(uploaded).convert("RGB")

        # Check duplicates
        status, matched_uid, matched_dbid, matches = check_duplicate_weighted(img, description, damage)

        # Compute/store embeddings and image
        img_emb = get_clip_image_embedding(img)
        text_emb = get_text_embedding_cached((description or "") + " " + (damage or ""))

        image_hash = phash_str(img)
        save_image_file(img, image_hash)

        # Choose UID
        if status == "Exact Duplicate" and matched_uid:
            uid = matched_uid
        elif status.startswith("Similar") and matched_uid:
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
        subcluster_id = assign_subcluster(df_all, new_meta, uid)

        # Build record
        record = {
            "unique_image_id": uid,
            "sub_cluster_id": subcluster_id,
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
            "value_usd": value_usd,
            "description": description,
            "damage_classification": damage,
            "chat_text": chat_text,
            "image_hash": image_hash,
            "image_embedding": img_emb.tobytes(),
            "text_embedding": text_emb.tobytes()
        }
        new_db_id = save_record_to_db(record)

        # Add vector to FAISS incrementally
        try:
            add_vector_to_faiss(damage, img_emb, new_db_id)
        except Exception:
            # fallback to rebuild this damage index
            build_faiss_for_damage(damage)

        # Show simple, clear result messages (format requested)
        if status == "Exact Duplicate":
            # find subcluster for matched_dbid (if present)
            c.execute("SELECT sub_cluster_id FROM records WHERE id=?", (matched_dbid,))
            row = c.fetchone()
            matched_sub = row[0] if row else "N/A"
            st.success(f"Exact duplicate found in Cluster ({matched_uid}) and Subcluster ({matched_sub})")
        elif status.startswith("Similar"):
            # similar image/narrative found
            c.execute("SELECT sub_cluster_id FROM records WHERE id=?", (matched_dbid,))
            row = c.fetchone()
            matched_sub = row[0] if row else "N/A"
            st.warning(f"Similar image/Narrative found in Cluster ({matched_uid}) and Subcluster ({matched_sub})")
        else:
            st.info(f"No duplicates found. Creating a new Cluster ({uid}) with Subcluster ({subcluster_id})")

# ----------------------------
# DATABASE VIEWER
# ----------------------------
elif menu == "Database Viewer":
    st.title("Database Viewer")

    # Load database
    df = pd.read_sql_query("""
        SELECT id, unique_image_id, sub_cluster_id, customer_id, order_id,
               ip_country_code, billing_country_code, shipping_country_code,
               credit_card_country_code, fast_lane, isfba, has_prime,
               gl_code, payment_method, issuing_bank, item_name, value_usd, description,
               damage_classification, chat_text, image_hash
        FROM records
    """, conn)

    if df.empty:
        st.info("No records found.")
    else:
        # --------------------------------
        # SEARCH BAR + TOGGLES (aligned)
        # --------------------------------
        with st.container():
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                q = st.text_input("Search (any field or chat)", placeholder="Search by ID, item, or text...")
            with col2:
                show_img = st.toggle("Show Images", value=False)
            with col3:
                show_chat = st.toggle("Show Chat", value=False)

        # Filter by search term
        if q:
            ql = q.lower()
            df = df[df.apply(lambda r: ql in " ".join(map(str, r.values)).lower(), axis=1)]

        # Clean numeric values
        df["value_usd"] = pd.to_numeric(df["value_usd"], errors="coerce").fillna(0)

        # --------------------------------
        # REFUND STATUS LOGIC
        # --------------------------------
        df["refund_status"] = None
        df = df.sort_values(by="id", ascending=True)

        for cluster_id, group in df.groupby("unique_image_id"):
            min_id = group["id"].min()
            df.loc[df["id"] == min_id, "refund_status"] = "Approved"
            df.loc[(df["unique_image_id"] == cluster_id) & (df["id"] != min_id), "refund_status"] = "Denied"

        # --------------------------------
        # PAGINATION SETUP
        # --------------------------------
        ROWS_PER_PAGE = 100
        total_rows = len(df)
        total_pages = max(1, (total_rows - 1) // ROWS_PER_PAGE + 1)
        page = st.session_state.get("page", 1)
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * ROWS_PER_PAGE
        end_idx = start_idx + ROWS_PER_PAGE
        df_page = df.iloc[start_idx:end_idx]

        # --------------------------------
        # DISPLAY CLUSTERS (paged)
        # --------------------------------
        for uid in df_page["unique_image_id"].unique():
            cluster_df = df_page[df_page["unique_image_id"] == uid].copy()

            total_val = cluster_df["value_usd"].sum()
            sub_count = cluster_df["sub_cluster_id"].nunique()
            acc_count = len(cluster_df)
            savings = cluster_df.loc[cluster_df["refund_status"] == "Denied", "value_usd"].sum()

            # Cluster Header
            st.markdown(f"### **Cluster ID:** :green[{uid}]")
            st.text(
                f"Total Cluster Value: ${total_val:,.2f}   |   Subclusters: {sub_count}   |   "
                f"Accounts: {acc_count}   |   Savings: ${savings:,.2f}"
            )

            # Subclusters
            for sc in cluster_df["sub_cluster_id"].unique():
                sub = cluster_df[cluster_df["sub_cluster_id"] == sc].copy()
                sub_val = sub["value_usd"].sum()
                accs = len(sub)
                sub_sav = sub.loc[sub["refund_status"] == "Denied", "value_usd"].sum()

                st.markdown(f"#### **Subcluster ID:** :blue[{sc}]")
                st.text(f"Accounts: {accs}   |   Total: ${sub_val:,.2f}   |   Savings: ${sub_sav:,.2f}")

                # Prepare table
                sub_disp = sub.drop(columns=["unique_image_id", "sub_cluster_id", "image_hash"], errors="ignore")
                if not show_chat:
                    sub_disp = sub_disp.drop(columns=["chat_text"], errors="ignore")

                col_order = [
                    "refund_status", "customer_id", "order_id",
                    "ip_country_code", "billing_country_code", "shipping_country_code",
                    "credit_card_country_code", "fast_lane", "isfba", "has_prime",
                    "gl_code", "payment_method", "issuing_bank",
                    "item_name", "value_usd", "damage_classification",
                    "description", "chat_text"
                ]
                sub_disp = sub_disp[[c for c in col_order if c in sub_disp.columns]]

                def refund_color(v):
                    if v == "Approved":
                        return "color: green"
                    elif v == "Denied":
                        return "color: red"
                    return "color: black"

                col_table, col_img = st.columns([6, 1.5])
                with col_table:
                    style = sub_disp.style.map(refund_color, subset=["refund_status"])
                    if "chat_text" in sub_disp.columns:
                        style = style.set_properties(subset=["chat_text"], **{"white-space": "pre-wrap"})
                    st.dataframe(style, use_container_width=True, hide_index=True)

                with col_img:
                    if show_img:
                        for _, row in sub.iterrows():
                            img_path = os.path.join(IMAGE_FOLDER, f"{row['image_hash']}.png")
                            if os.path.exists(img_path):
                                st.image(img_path, width=55)
                            else:
                                st.write(" ")

            st.markdown("---")

        # --------------------------------
        # PAGINATION BUTTONS (BOTTOM CENTERED)
        # --------------------------------
        st.markdown("---")
        left_spacer, nav_col, right_spacer = st.columns([2, 4, 2])
        with nav_col:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("⬅️ Prev", key=f"prev_{page}") and page > 1:
                    st.session_state["page"] = page - 1
                    st.rerun()
            with c2:
                st.markdown(f"<div style='text-align:center;'>Page {page} of {total_pages}</div>", unsafe_allow_html=True)
            with c3:
                if st.button("Next ➡️", key=f"next_{page}") and page < total_pages:
                    st.session_state["page"] = page + 1
                    st.rerun()

        # Page info below navigation
        st.markdown("")
        st.caption(
            f"Showing {start_idx + 1:,}–{min(end_idx, total_rows):,} of {total_rows:,} records  "
            #f"(Page {page}/{total_pages})"
        )