# 📚 協同過濾推薦系統 (User-Based & Item-Based Filtering)

這是一個使用 [Streamlit](https://streamlit.io/) 架設的推薦系統 Demo，採用兩種協同過濾方法：
使用網址：https://recommendation-system-ntupmba-7mucgjjeabbtrtk4gk3hvc.streamlit.app/
- **User-Based Filtering**（使用者為中心）
- **Item-Based Filtering**（物品為中心）

資料為模擬書籍評分資料集，並以互動介面展示推薦結果。

---

## 🚀 功能介紹

- ✅ **切換推薦方式**：使用者可自由選擇使用者導向或物品導向推薦
- ✅ **即時推薦**：基於目前使用者評分自動產生推薦書籍
- ✅ **相似度演算法**：
  - User-Based 使用 **皮爾森相關係數**（Pearson Correlation）
  - Item-Based 使用 **餘弦相似度**（Cosine Similarity）
- ✅ **評分矩陣可視化**：NaN 代表尚未評分

---

## 🧪 模擬資料

- 6 位使用者（小明、小美、小安、小天、小林、小艾）
- 8 本書籍（例如《輝達之道》、《逆思維》、《東京自由行終極指南》等）
- 每位使用者針對不同書籍給予 1～5 分評分
