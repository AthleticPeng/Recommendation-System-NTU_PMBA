import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
# streamlit run rec_app.py
# ==== 模擬資料 ====
data = {
    'user_id': [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5,
                6, 6, 6, 6, 6, 6,
                5, 4, 3, 2],
    'item_id': [1, 7, 2, 3, 1, 4, 5, 2, 3, 5, 6, 1, 5, 7, 2, 3,
                6, 8, 2, 5, 3, 4,
                1, 6, 1, 6],
    'rating':  [5, 4, 3, 4, 2, 5, 4, 4, 4, 3, 5, 5, 3, 4, 5, 2,
                2, 4, 4, 2, 5, 3,
                5, 4, 5, 3]
}

user_names = {1: '小明', 2: '小美', 3: '小安', 4: '小天', 5: '小林', 6: '小艾'}
item_names = {
    1: '輝達之道', 2: '失控的焦慮世代', 3: '逆思維', 4: '原始碼',
    5: '凝視太陽：面對死亡恐懼', 6: '思考的藝術',
    7: '一次讀懂養生精髓', 8: '東京自由行終極指南'
}

# ==== 準備資料 ====
df = pd.DataFrame(data)
user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# ==== Streamlit App ====
st.title("📚 User-Based vs Item-Based 協同過濾推薦系統")
st.write("可以切換推薦方式並選擇使用者，查看推薦結果")

user_display = {v: k for k, v in user_names.items()}
selected_user_name = st.selectbox("選擇一位使用者：", list(user_display.keys()))
selected_user_id = user_display[selected_user_name]

method = st.radio("選擇推薦方法：", ["Item-Based Filtering", "User-Based Filtering"])

# ==== Item-based Filtering ====
if method == "Item-Based Filtering":
    target_ratings = user_item_matrix.loc[selected_user_id]
    liked_items = target_ratings[target_ratings >= 4].index.tolist()

    if not liked_items:
        st.warning("此用戶目前沒有評分 ≥ 4 的書，無法進行推薦。")
        st.stop()

    st.markdown(f"### 👍 {selected_user_name} 喜歡的書（評分 ≥ 4）：")
    for i in liked_items:
        st.write(f"- {item_names[i]}")

    item_similarity_matrix = pd.DataFrame(
        cosine_similarity(user_item_matrix.T),
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    similar_items = []
    top_k_per_item = st.slider("每本喜歡的書找幾本相似書？", 1, 5, 3)

    for item_id in liked_items:
        similar = item_similarity_matrix[item_id].drop(index=item_id)
        top_sim = similar.sort_values(ascending=False).head(top_k_per_item)
        st.markdown(f"#### 與《{item_names[item_id]}》相似的書：")
        for sim_item_id, score in top_sim.items():
            st.write(f"→ {item_names[sim_item_id]}（相似度：{score:.2f}）")
            similar_items.append(sim_item_id)

    recommend_counts = Counter(similar_items)
    already_read = target_ratings[target_ratings > 0].index.tolist()
    filtered = {i: c for i, c in recommend_counts.items() if i not in already_read}

    top_n = st.slider("顯示推薦幾本書：", 1, 5, 2)
    final_recommend = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n]

    st.markdown("### 🎯 最終推薦書籍：")
    if not final_recommend:
        st.info("無可推薦的新書（可能已經全看過了 😅）")
    else:
        for i, count in final_recommend:
            st.success(f"{item_names[i]}（被推薦次數：{count}）")

# ==== User-based Filtering ====
else:
    def pearson_correlation(user1, user2, matrix):
        common = (matrix.loc[user1] > 0) & (matrix.loc[user2] > 0)
        if common.sum() == 0:
            return 0
        u1 = matrix.loc[user1, common]
        u2 = matrix.loc[user2, common]
        u1_mean = u1.mean()
        u2_mean = u2.mean()
        num = ((u1 - u1_mean) * (u2 - u2_mean)).sum()
        denom = np.sqrt(((u1 - u1_mean)**2).sum()) * np.sqrt(((u2 - u2_mean)**2).sum())
        return num / denom if denom != 0 else 0

    similarities = {
        other: pearson_correlation(selected_user_id, other, user_item_matrix)
        for other in user_item_matrix.index if other != selected_user_id
    }
    top2 = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:2]
    ref_users = [u for u, _ in top2]
    ref_sims = [s for _, s in top2]

    st.markdown("### 👥 相似使用者：")
    for u, s in top2:
        st.write(f"{user_names[u]}（相似度：{s:.4f}）")

    target_ratings = user_item_matrix.loc[selected_user_id]
    weighted_sum = pd.Series(0.0, index=user_item_matrix.columns)
    for uid, sim in zip(ref_users, ref_sims):
        weighted_sum += user_item_matrix.loc[uid] * sim

    unrated = target_ratings[target_ratings == 0].index
    predictions = weighted_sum[unrated].sort_values(ascending=False).head(3)

    st.markdown("### 🎯 最終推薦書籍（加總分數最高）：")
    for item_id, score in predictions.items():
        st.success(f"{item_names[item_id]}（加總分數：{score:.2f}）")