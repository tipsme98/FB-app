import streamlit as st
import pandas as pd
import os
import json
import base64
from datetime import datetime

# 嘗試載入機器學習套件 (加入防呆機制，避免缺少 requirements.txt 時直接當機)
try:
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    HAS_AI_MODULES = True
except ImportError:
    HAS_AI_MODULES = False

# ==========================================
# 1. 初始化設定與資料庫 Schema
# ==========================================
st.set_page_config(page_title="足球博彩精算系統", page_icon="⚽", layout="wide")

# 完整維度資料庫欄位定義
DB_COLUMNS = [
    'ID', 'Date', 'Status', 
    # 賽前基礎維度
    'League', 'Match', 'Home_Team', 'Away_Team', 'Home_Rating', 'Away_Rating', 'Home_Form', 'Away_Form',
    # 投注與賠率維度
    'Bet_Type', 'Selection', 'Initial_Line', 'Initial_Odds', 'Stake', 'Odds_History',
    # 即場動態進階維度 (In-Play)
    'InPlay_Minute', 'Home_DA', 'Away_DA', 'Home_SoT', 'Away_SoT', 'Home_SoFF', 'Away_SoFF',
    'Home_Red', 'Away_Red', 'Home_Sub', 'Away_Sub', 'Home_Possession', 'Away_Possession',
    # 賽果與結算維度
    'Home_Goal', 'Away_Goal', 'Home_Corner', 'Away_Corner', 
    'Result_Label', 'Profit', 'Unit_Profit', 'Payout'
]

LOG_COLUMNS = ['ID', 'Date', 'Match', 'Analysis_Content', 'Confidence_Level']

def load_db(filename, columns):
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            for col in columns:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception:
            return pd.DataFrame(columns=columns)
    else:
        return pd.DataFrame(columns=columns)

def save_db(df, filename):
    df.to_csv(filename, index=False)

# ==========================================
# 2. 核心邏輯與輔助函式
# ==========================================
def calculate_settlement(bet_type, selection, line, odds, stake, h, a):
    """5 態精算邏輯 (全贏、贏半、走盤、輸半、全輸)"""
    diff = 0.0
    if bet_type == '讓球':
        if selection == 'Home':
            diff = h + line - a
        elif selection == 'Away':
            diff = a + line - h
    elif bet_type == '大小':
        total_goals = h + a
        if selection == 'Over':
            diff = total_goals - line
        elif selection == 'Under':
            diff = line - total_goals

    diff = round(diff, 2)

    if diff >= 0.5:
        res_label, profit = "✅ 全贏", stake * (odds - 1)
        payout = stake + profit
    elif diff == 0.25:
        res_label, profit = "🟢 贏半", stake * (odds - 1) / 2
        payout = stake + profit
    elif diff == 0.0:
        res_label, profit, payout = "⚪ 走盤退本", 0.0, stake
    elif diff == -0.25:
        res_label, profit, payout = "🔴 輸半 (退回半本)", -stake / 2, stake / 2
    else:
        res_label, profit, payout = "❌ 全輸", -stake, 0.0

    unit_profit = profit / 100.0 
    return round(profit, 2), round(payout, 2), round(unit_profit, 2), res_label, diff

def get_html_link(df, title):
    if df.empty:
        return "<p style='color: gray;'>目前尚無數據可供獨立開啟</p>"
    html_table = df.to_html(classes='table table-striped', index=False)
    full_html = f"""
    <html><head><meta charset="utf-8"><title>{title}</title>
    <style>
        body {{ font-family: 'Microsoft JhengHei', sans-serif; padding: 20px; background-color: #f9f9f9; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 13px; background-color: white; white-space: nowrap; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #007BFF; color: white; position: sticky; top: 0; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #ddd; }}
    </style></head><body><h2>{title}</h2>{html_table}</body></html>
    """
    b64 = base64.b64encode(full_html.encode('utf-8')).decode()
    return f'<a href="data:text/html;base64,{b64}" target="_blank" style="text-decoration: none; display: inline-block; padding: 8px 16px; background-color: #28a745; color: white; border-radius: 5px; font-weight: bold; font-size: 14px; margin-top: 10px;">🌐 在獨立新頁面開啟 {title} (HTML)</a>'

@st.dialog("🔍 全維度數據庫即時線上預覽", width="large")
def show_database_dialog(db_file, log_file):
    df_db = load_db(db_file, DB_COLUMNS)
    df_log = load_db(log_file, LOG_COLUMNS)

    tab1, tab2 = st.tabs(["📋 投注與動態紀錄", "📊 賽前分析日誌"])
    
    with tab1:
        st.caption(f"📂 當前讀取檔案: `{db_file}`")
        st.dataframe(df_db, use_container_width=True, hide_index=True, height=450)
        st.markdown(get_html_link(df_db, "📋 投注與動態紀錄全覽"), unsafe_allow_html=True)
        
    with tab2:
        st.caption(f"📂 當前讀取檔案: `{log_file}`")
        st.dataframe(df_log, use_container_width=True, hide_index=True, height=450)
        st.markdown(get_html_link(df_log, "📊 賽前分析日誌全覽"), unsafe_allow_html=True)

# ==========================================
# 3. 主程式 UI 與分頁
# ==========================================
def main():
    st.title("⚽ 足球博彩精算系統 (AI 輔助升級版)")
    
    st.sidebar.header("⚙️ 系統設定")
    mode = st.sidebar.radio("運作模式選擇", ["🧪 測試模式", "🟢 真實模式"])
    
    db_file = "football_betting_db_test.csv" if mode == "🧪 測試模式" else "football_betting_db.csv"
    log_file = "football_analysis_log_test.csv" if mode == "🧪 測試模式" else "football_analysis_log.csv"
    
    st.session_state.df_db = load_db(db_file, DB_COLUMNS)
    st.session_state.df_log = load_db(log_file, LOG_COLUMNS)

    st.sidebar.divider()
    if st.sidebar.button("🔍 開啟/預覽數據庫視窗", type="primary", use_container_width=True):
        show_database_dialog(db_file, log_file)
    
    # 加入 Tab 5: 機器學習預測
    t_pre, t_inplay, t_settle, t_log, t_ai = st.tabs(["📝 賽前建檔", "⏱️ 即場動態", "⚖️ 賽果結算", "📓 分析日誌", "🤖 AI 賽事預測"])

    # --- Tab 1: 賽前建檔 ---
    with t_pre:
        with st.form("pre_match_form"):
            st.subheader("1. 賽事基礎維度")
            col1, col2, col3 = st.columns(3)
            league = col1.text_input("賽事類別 (例: 英超)")
            home_team = col2.text_input("主隊名稱")
            away_team = col3.text_input("客隊名稱")
            
            col4, col5 = st.columns(2)
            home_rating = col4.selectbox("主隊評級", ["S", "A", "B", "C", "D"])
            away_rating = col5.selectbox("客隊評級", ["S", "A", "B", "C", "D"])
            
            home_form = col4.text_input("主隊近況 (例: 3W1D1L)")
            away_form = col5.text_input("客隊近況 (例: 2W2D1L)")
            
            st.subheader("2. 投注與初始盤口維度")
            c_bet1, c_bet2, c_bet3 = st.columns(3)
            bet_type = c_bet1.selectbox("玩法類型", ["讓球", "大小", "角球"])
            selection = c_bet2.selectbox("投注選項", ["Home", "Away", "Over", "Under"])
            line = c_bet3.number_input("初始盤口線 (Line)", value=0.0, step=0.25)
            
            odds = c_bet1.number_input("初始賠率", min_value=1.01, value=1.90, step=0.01)
            stake = c_bet2.number_input("投注本金", min_value=10.0, value=100.0, step=10.0)
            
            match_name = f"{home_team} vs {away_team}"
            
            if st.form_submit_button("💾 建立初始盤紀錄") and home_team and away_team:
                new_id = f"B{datetime.now().strftime('%Y%m%d%H%M%S')}"
                now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
                initial_odds_history = [{"time": now_str, "type": "Initial", "line": line, "odds": odds}]
                
                new_record = {
                    'ID': new_id, 'Date': now_str, 'Status': 'Pending',
                    'League': league, 'Match': match_name, 'Home_Team': home_team, 'Away_Team': away_team,
                    'Home_Rating': home_rating, 'Away_Rating': away_rating, 'Home_Form': home_form, 'Away_Form': away_form,
                    'Bet_Type': bet_type, 'Selection': selection, 'Initial_Line': line, 'Initial_Odds': odds, 'Stake': stake,
                    'Odds_History': json.dumps(initial_odds_history, ensure_ascii=False)
                }
                
                new_df = pd.DataFrame([new_record])
                st.session_state.df_db = pd.concat([st.session_state.df_db, new_df], ignore_index=True)
                save_db(st.session_state.df_db, db_file)
                st.success(f"✅ 賽前紀錄建立成功！單號：{new_id}")

    # --- Tab 2: 即場動態 ---
    with t_inplay:
        pending_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Pending']
        if pending_df.empty:
            st.info("尚無未結算的賽事可供追蹤。")
        else:
            select_idx = st.selectbox("選擇要追蹤動態的賽事", pending_df.index, format_func=lambda i: pending_df.loc[i, 'Match'])
            row = pending_df.loc[select_idx]
            
            with st.form("inplay_form"):
                st.write(f"### ⚡ 即場更新: {row['Match']}")
                minute = st.number_input("當下比賽時間 (分鐘)", min_value=0, max_value=120, value=45)
                
                i_c1, i_c2, i_c3 = st.columns(3)
                h_da = i_c1.number_input("主-危險進攻", min_value=0, value=int(row['Home_DA']) if pd.notna(row['Home_DA']) else 0)
                a_da = i_c2.number_input("客-危險進攻", min_value=0, value=int(row['Away_DA']) if pd.notna(row['Away_DA']) else 0)
                h_pos = i_c3.slider("主隊控球率 (%)", 0, 100, 50)
                
                new_line = st.number_input("最新盤口線", value=float(row['Initial_Line']), step=0.25)
                new_odds = st.number_input("最新賠率", value=float(row['Initial_Odds']), step=0.01)
                
                if st.form_submit_button("🔄 更新動態與賠率"):
                    try:
                        history_list = json.loads(row['Odds_History']) if pd.notna(row['Odds_History']) else []
                    except:
                        history_list = []
                    history_list.append({"time": f"{minute}'", "type": "Live", "line": new_line, "odds": new_odds})
                    
                    st.session_state.df_db.loc[select_idx, 'InPlay_Minute'] = minute
                    st.session_state.df_db.loc[select_idx, 'Home_DA'] = h_da
                    st.session_state.df_db.loc[select_idx, 'Away_DA'] = a_da
                    st.session_state.df_db.loc[select_idx, 'Home_Possession'] = h_pos
                    st.session_state.df_db.loc[select_idx, 'Away_Possession'] = 100 - h_pos
                    st.session_state.df_db.loc[select_idx, 'Odds_History'] = json.dumps(history_list, ensure_ascii=False)
                    save_db(st.session_state.df_db, db_file)
                    st.success("✅ 即場數據更新成功！")

    # --- Tab 3: 賽果結算 ---
    with t_settle:
        pending_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Pending']
        if pending_df.empty:
            st.info("🎉 目前沒有待結算的投注單。")
        else:
            settle_idx = st.selectbox("選擇結算單", pending_df.index, format_func=lambda i: pending_df.loc[i, 'Match'])
            s_row = pending_df.loc[settle_idx]
            
            c1, c2 = st.columns(2)
            h_goal = c1.number_input("🏠 主隊進球數", min_value=0, value=0)
            a_goal = c2.number_input("✈️ 客隊進球數", min_value=0, value=0)
            
            if st.button("⚖️ 執行全維度結算", type="primary"):
                profit, payout, u_profit, res_label, diff = calculate_settlement(
                    s_row['Bet_Type'], s_row['Selection'], s_row['Initial_Line'], s_row['Initial_Odds'], s_row['Stake'], h_goal, a_goal
                )
                
                st.session_state.df_db.loc[settle_idx, 'Home_Goal'] = h_goal
                st.session_state.df_db.loc[settle_idx, 'Away_Goal'] = a_goal
                st.session_state.df_db.loc[settle_idx, 'Profit'] = profit
                st.session_state.df_db.loc[settle_idx, 'Unit_Profit'] = u_profit
                st.session_state.df_db.loc[settle_idx, 'Payout'] = payout
                st.session_state.df_db.loc[settle_idx, 'Result_Label'] = res_label
                st.session_state.df_db.loc[settle_idx, 'Status'] = 'Settled'
                
                save_db(st.session_state.df_db, db_file)
                st.success("✅ 結算成功！賽果已記錄並可供 AI 訓練使用。")

    # --- Tab 4: 分析日誌 ---
    with t_log:
        with st.form("log_form"):
            log_match = st.text_input("關聯賽事 (例: 曼聯 vs 阿仙奴)")
            log_content = st.text_area("賽前深度分析內容")
            log_conf = st.slider("信心指數 (1-10)", 1, 10, 7)
            if st.form_submit_button("📝 儲存日誌"):
                new_log = {'ID': f"L{datetime.now().strftime('%Y%m%d%H%M%S')}", 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Match': log_match, 'Analysis_Content': log_content, 'Confidence_Level': log_conf}
                st.session_state.df_log = pd.concat([st.session_state.df_log, pd.DataFrame([new_log])], ignore_index=True)
                save_db(st.session_state.df_log, log_file)
                st.success("✅ 日誌已儲存")

    # --- Tab 5: AI 機器學習預測 ---
    with t_ai:
        st.header("🤖 AI 賽事勝負預測 (基於 Random Forest)")
        st.write("此模組會自動讀取您過往「已結算」的賽事歷史，根據實力評級與賠率，預測新賽事的勝負機率。")
        
        if not HAS_AI_MODULES:
            st.error("🚨 **系統偵測到缺少機器學習套件！**")
            st.warning("請務必在 GitHub 專案根目錄新增 `requirements.txt` 檔案，並寫入 `scikit-learn` 與 `numpy`，然後重新啟動 Streamlit App。")
        else:
            df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
            
            if len(df_settled) < 5:
                st.warning(f"目前只有 {len(df_settled)} 筆結算數據。AI 需要至少 5 筆結算數據才能進行初步訓練，請先在「賽果結算」完成更多賽事。")
            else:
                try:
                    # 1. 數據前處理
                    rating_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
                    df_settled['H_Rating_Num'] = df_settled['Home_Rating'].map(rating_map).fillna(3)
                    df_settled['A_Rating_Num'] = df_settled['Away_Rating'].map(rating_map).fillna(3)
                    df_settled['Initial_Line'] = pd.to_numeric(df_settled['Initial_Line'], errors='coerce').fillna(0)
                    df_settled['Initial_Odds'] = pd.to_numeric(df_settled['Initial_Odds'], errors='coerce').fillna(1.9)
                    df_settled['Goal_Diff'] = pd.to_numeric(df_settled['Home_Goal'], errors='coerce').fillna(0) - pd.to_numeric(df_settled['Away_Goal'], errors='coerce').fillna(0)
                    
                    # 標籤定義：1 (主勝), 0 (和局), -1 (客勝)
                    df_settled['Target'] = df_settled['Goal_Diff'].apply(lambda x: 1 if x > 0 else (0 if x == 0 else -1))
                    
                    features = ['H_Rating_Num', 'A_Rating_Num', 'Initial_Line', 'Initial_Odds']
                    X = df_settled[features]
                    y = df_settled['Target']
                    
                    # 2. 訓練模型
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    st.success(f"✅ AI 模型已成功利用 {len(df_settled)} 筆您的專屬歷史數據完成訓練！")
                    
                    # 3. 預測介面
                    st.divider()
                    st.subheader("🔮 預測即將到來的賽事")
                    col_ai1, col_ai2 = st.columns(2)
                    
                    pred_h_rating = col_ai1.selectbox("預測賽事 - 主隊評級", ["S", "A", "B", "C", "D"], key="ai_hr")
                    pred_a_rating = col_ai2.selectbox("預測賽事 - 客隊評級", ["S", "A", "B", "C", "D"], key="ai_ar")
                    pred_line = col_ai1.number_input("預測賽事 - 初始盤口", value=0.0, step=0.25)
                    pred_odds = col_ai2.number_input("預測賽事 - 初始賠率", value=1.90, step=0.01)
                    
                    if st.button("🧠 執行 AI 預測", type="primary"):
                        X_new = pd.DataFrame({
                            'H_Rating_Num': [rating_map.get(pred_h_rating, 3)],
                            'A_Rating_Num': [rating_map.get(pred_a_rating, 3)],
                            'Initial_Line': [pred_line],
                            'Initial_Odds': [pred_odds]
                        })
                        
                        prediction = model.predict(X_new)[0]
                        probabilities = model.predict_proba(X_new)[0]
                        classes = model.classes_
                        
                        # 整理機率輸出
                        prob_dict = {c: p for c, p in zip(classes, probabilities)}
                        p_home = prob_dict.get(1, 0.0)
                        p_draw = prob_dict.get(0, 0.0)
                        p_away = prob_dict.get(-1, 0.0)
                        
                        st.markdown("### 📊 AI 預測結果")
                        if prediction == 1:
                            st.info(f"🏆 AI 傾向判定：**主隊勝出 (Home Win)**")
                        elif prediction == 0:
                            st.info(f"🤝 AI 傾向判定：**和局 (Draw)**")
                        else:
                            st.info(f"✈️ AI 傾向判定：**客隊勝出 (Away Win)**")
                            
                        c_p1, c_p2, c_p3 = st.columns(3)
                        c_p1.metric("主勝機率", f"{p_home*100:.1f}%")
                        c_p2.metric("和局機率", f"{p_draw*100:.1f}%")
                        c_p3.metric("客勝機率", f"{p_away*100:.1f}%")
                        st.caption("備註：AI 預測準確度將隨著您在「賽果結算」輸入的歷史資料量增加而持續進化。")
                        
                except Exception as e:
                    st.error(f"資料處理或模型訓練發生錯誤：{e}")

if __name__ == "__main__":
    main()
