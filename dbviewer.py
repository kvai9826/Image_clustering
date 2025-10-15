# ----------------------------
# DATABASE VIEWER
# ----------------------------
elif menu == "Database Viewer":
    st.title("Database Viewer")

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
        q_col, img_col, chat_col, dl_col = st.columns([3, 1, 1, 1])
        q = q_col.text_input("Search (any field or chat)")
        show_img = img_col.toggle("Show Images", True)
        show_chat = chat_col.toggle("Show Chat", False)

        csv = df.to_csv(index=False).encode("utf-8")
        dl_col.download_button("Download CSV", csv, "claims.csv")

        if q:
            ql = q.lower()
            df = df[df.apply(lambda r: ql in " ".join(map(str, r.values)).lower(), axis=1)]

        df["value_usd"] = pd.to_numeric(df["value_usd"], errors="coerce").fillna(0)

        # Refund Status Logic
        df["refund_status"] = None
        df = df.sort_values(by="id", ascending=True)

        for cluster_id, group in df.groupby("unique_image_id"):
            min_id = group["id"].min()
            df.loc[df["id"] == min_id, "refund_status"] = "Approved"
            df.loc[(df["unique_image_id"] == cluster_id) & (df["id"] != min_id), "refund_status"] = "Denied"

        for uid in df["unique_image_id"].unique():
            st.markdown(f"### Main Cluster: `{uid}`")
            cluster_df = df[df["unique_image_id"] == uid].copy()

            total_cluster_value = cluster_df["value_usd"].sum()
            total_subclusters = cluster_df["sub_cluster_id"].nunique()
            total_accounts = len(cluster_df)
            total_savings = cluster_df.loc[cluster_df["refund_status"] == "Denied", "value_usd"].sum()

            st.markdown(
                f"**Total Cluster Value:** ${total_cluster_value:,.2f}   |   "
                f"**Total Subclusters:** {total_subclusters}   |   "
                f"**Total Accounts:** {total_accounts}   |   "
                f"**Savings:** ${total_savings:,.2f}"
            )

            for sc in cluster_df["sub_cluster_id"].unique():
                sub = cluster_df[cluster_df["sub_cluster_id"] == sc].copy()
                total_value = sub["value_usd"].sum()
                account_count = len(sub)
                sub_savings = sub.loc[sub["refund_status"] == "Denied", "value_usd"].sum()

                st.caption(
                    f"Subcluster: {sc}  |  Accounts: {account_count}  |  "
                    f"Total: ${total_value:,.2f}  |  Savings: ${sub_savings:,.2f}"
                )

                sub_disp = sub.drop(
                    columns=["unique_image_id", "sub_cluster_id", "image_hash"],
                    errors="ignore"
                )
                if not show_chat:
                    sub_disp = sub_disp.drop(columns=["chat_text"], errors="ignore")

                # Display Refund Status prominently
                sub_disp = sub_disp[[
                    "refund_status", "customer_id", "order_id", "item_name", "value_usd",
                    "damage_classification", "payment_method", "shipping_country_code", "description"
                ] + [col for col in sub_disp.columns if col not in [
                    "refund_status", "customer_id", "order_id", "item_name", "value_usd",
                    "damage_classification", "payment_method", "shipping_country_code", "description"
                ]]]

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