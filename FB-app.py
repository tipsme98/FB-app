import streamlit as st
import pandas as pd
import os
import json
import base64
from datetime import datetime

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
            # 確保舊檔案兼容新欄位
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

    # 假設平注單位為 $100 計算 Unit Profit
    unit_profit = profit / 100.0 
    return round(profit, 2), round(payout, 2), round(unit_profit, 2), res_label, diff

def get_html_link(df, title):
    """將 DataFrame 轉為 HTML 格式，並封裝成新分頁開啟的 Base64 連結"""
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
    </style></head><body><h2>{title} <span>(全維度預覽)</span></h2>{html_table}</body></html>
    """
    b64 = base64.b64encode(full_html.encode('utf-8')).decode()
    return f'<a href="data:text/html;base64,{b64}" target="_blank" style="text-decoration: none; display: inline-block; padding: 8px 16px; background-color: #28a745; color: white; border-radius: 5px; font-weight: bold; font-size: 14px; margin-top: 10px;">🌐 在獨立新頁面開啟 {title} (HTML)</a>'

# ==========================================
# 3. 視窗預覽邏輯 (Dialog)
# ==========================================
@st.dialog("🔍 全維度數據庫即時線上預覽", width="large")
def show_database_dialog(db_file, log_file):
    df_db = load_db(db_file, DB_COLUMNS)
    df_log = load_db(log_file, LOG_COLUMNS)

    tab1, tab2 = st.tabs(["📋 投注與動態紀錄", "📊 賽前分析日誌"])
    
    with tab1:
        st.caption(f"📂 當前讀取檔案: `{db_file}` (支援左右滑動、點擊標題排序、右上角放大及搜尋)")
        col1, col2, col3 = st.columns(3)
        col1.metric("總記錄筆數", f"{len(df_db)} 筆")
        total_pl = pd.to_numeric(df_db["Profit"], errors="coerce").sum()
        col2.metric("累積總盈虧", f"${total_pl:,.2f}", delta=f"${total_pl:,.2f}", delta_color="normal" if total_pl >= 0 else "inverse")
        total_bet = pd.to_numeric(df_db["Stake"], errors="coerce").sum()
        col3.metric("累積投注額", f"${total_bet:,.2f}")
        
        st.dataframe(df_db, use_container_width=True, hide_index=True, height=450)
        st.markdown(get_html_link(df_db, "📋 投注與動態紀錄全覽"), unsafe_allow_html=True)
        
    with tab2:
        st.caption(f"📂 當前讀取檔案: `{log_file}`")
        st.dataframe(df_log, use_container_width=True, hide_index=True, height=450)
        st.markdown(get_html_link(df_log, "📊 賽前分析日誌全覽"), unsafe_allow_html=True)

# ==========================================
# 4. 主程式 UI 與側邊欄
# ==========================================
def main():
    st.title("⚽ 足球博彩精算系統 (全維度架構)")
    
    # 側邊欄：模式切換與預覽
    st.sidebar.header("⚙️ 系統設定")
    mode = st.sidebar.radio("運作模式選擇", ["🧪 測試模式", "🟢 真實模式"])
    
    db_file = "football_betting_db_test.csv" if mode == "🧪 測試模式" else "football_betting_db.csv"
    log_file = "football_analysis_log_test.csv" if mode == "🧪 測試模式" else "football_analysis_log.csv"
    
    st.session_state.df_db = load_db(db_file, DB_COLUMNS)
    st.session_state.df_log = load_db(log_file, LOG_COLUMNS)

    st.sidebar.divider()
    if st.sidebar.button("📥 輸出 Excel 數據庫", use_container_width=True):
        st.sidebar.success("✅ 提示：如有需要請擴充至 xlsx 匯出套件。")
    if st.sidebar.button("🔍 開啟/預覽數據庫視窗", type="primary", use_container_width=True):
        show_database_dialog(db_file, log_file)

    st.write(f"當前運行於：**{mode}**")
    
    # 分頁設計
    t_pre, t_inplay, t_settle, t_log = st.tabs(["📝 賽前建檔與投注", "⏱️ 即場動態與賠率", "⚖️ 賽果結算", "📓 撰寫分析日誌"])

    # ------------------------------------------
    # Tab 1: 賽前建檔與投注 (Pre-Match)
    # ------------------------------------------
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
            submit_pre = st.form_submit_button("💾 建立初始盤紀錄")
            
            if submit_pre and home_team and away_team:
                new_id = f"B{datetime.now().strftime('%Y%m%d%H%M%S')}"
                now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
                
                # 初始化 JSON 格式的歷史賠率追蹤
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

    # ------------------------------------------
    # Tab 2: 即場動態與賠率追蹤 (In-Play Stats)
    # ------------------------------------------
    with t_inplay:
        pending_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Pending']
        if pending_df.empty:
            st.info("尚無未結算的賽事可供追蹤。")
        else:
            select_idx = st.selectbox("選擇要追蹤動態的賽事", pending_df.index, format_func=lambda i: pending_df.loc[i, 'Match'])
            row = pending_df.loc[select_idx]
            
            st.write(f"### ⚡ 即場更新: {row['Match']}")
            with st.form("inplay_form"):
                st.write("**A. 攻守與紀律數據**")
                minute = st.number_input("當下比賽時間 (分鐘)", min_value=0, max_value=120, value=45)
                
                i_c1, i_c2, i_c3 = st.columns(3)
                h_da = i_c1.number_input("主-危險進攻", min_value=0, step=1, value=int(row['Home_DA']) if pd.notna(row['Home_DA']) else 0)
                h_sot = i_c2.number_input("主-射正", min_value=0, step=1, value=int(row['Home_SoT']) if pd.notna(row['Home_SoT']) else 0)
                h_soff = i_c3.number_input("主-射偏", min_value=0, step=1, value=int(row['Home_SoFF']) if pd.notna(row['Home_SoFF']) else 0)
                
                a_da = i_c1.number_input("客-危險進攻", min_value=0, step=1, value=int(row['Away_DA']) if pd.notna(row['Away_DA']) else 0)
                a_sot = i_c2.number_input("客-射正", min_value=0, step=1, value=int(row['Away_SoT']) if pd.notna(row['Away_SoT']) else 0)
                a_soff = i_c3.number_input("客-射偏", min_value=0, step=1, value=int(row['Away_SoFF']) if pd.notna(row['Away_SoFF']) else 0)
                
                h_red = i_c1.number_input("主-紅牌數", min_value=0, max_value=5, step=1, value=0)
                a_red = i_c2.number_input("客-紅牌數", min_value=0, max_value=5, step=1, value=0)
                h_pos = i_c3.slider("主隊控球率 (%)", 0, 100, 50)
                
                st.write("**B. 賠率與盤口變化紀錄 (Odds History)**")
                new_line = st.number_input("最新盤口線", value=float(row['Initial_Line']), step=0.25)
                new_odds = st.number_input("最新賠率", value=float(row['Initial_Odds']), step=0.01)
                
                if st.form_submit_button("🔄 更新即場動態與賠率"):
                    # 讀取舊有 JSON，加上新時間點
                    try:
                        history_list = json.loads(row['Odds_History']) if pd.notna(row['Odds_History']) else []
                    except:
                        history_list = []
                    
                    history_list.append({
                        "time": f"{minute}'", "type": "Live", "line": new_line, "odds": new_odds
                    })
                    
                    st.session_state.df_db.loc[select_idx, 'InPlay_Minute'] = minute
                    st.session_state.df_db.loc[select_idx, 'Home_DA'] = h_da
                    st.session_state.df_db.loc[select_idx, 'Away_DA'] = a_da
                    st.session_state.df_db.loc[select_idx, 'Home_SoT'] = h_sot
                    st.session_state.df_db.loc[select_idx, 'Away_SoT'] = a_sot
                    st.session_state.df_db.loc[select_idx, 'Home_SoFF'] = h_soff
                    st.session_state.df_db.loc[select_idx, 'Away_SoFF'] = a_soff
                    st.session_state.df_db.loc[select_idx, 'Home_Red'] = h_red
                    st.session_state.df_db.loc[select_idx, 'Away_Red'] = a_red
                    st.session_state.df_db.loc[select_idx, 'Home_Possession'] = h_pos
                    st.session_state.df_db.loc[select_idx, 'Away_Possession'] = 100 - h_pos
                    st.session_state.df_db.loc[select_idx, 'Odds_History'] = json.dumps(history_list, ensure_ascii=False)
                    
                    save_db(st.session_state.df_db, db_file)
                    st.success("✅ 即場數據已寫入資料庫！")

    # ------------------------------------------
    # Tab 3: 賽果結算 (Post-Match)
    # ------------------------------------------
    with t_settle:
        pending_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Pending']
        if pending_df.empty:
            st.info("🎉 目前沒有待結算的投注單。")
        else:
            settle_idx = st.selectbox("選擇結算單", pending_df.index, format_func=lambda i: pending_df.loc[i, 'Match'])
            s_row = pending_df.loc[settle_idx]
            
            st.write(f"### 🏁 結算: {s_row['Match']} ({s_row['Bet_Type']} | {s_row['Selection']})")
            c1, c2 = st.columns(2)
            h_goal = c1.number_input("🏠 主隊進球數", min_value=0, value=0)
            a_goal = c2.number_input("✈️ 客隊進球數", min_value=0, value=0)
            h_cor = c1.number_input("🚩 主隊角球數", min_value=0, value=0)
            a_cor = c2.number_input("🚩 客隊角球數", min_value=0, value=0)
            
            if st.button("⚖️ 執行全維度結算", type="primary"):
                profit, payout, u_profit, res_label, diff = calculate_settlement(
                    s_row['Bet_Type'], s_row['Selection'], s_row['Initial_Line'], s_row['Initial_Odds'], s_row['Stake'], h_goal, a_goal
                )
                
                st.session_state.df_db.loc[settle_idx, 'Home_Goal'] = h_goal
                st.session_state.df_db.loc[settle_idx, 'Away_Goal'] = a_goal
                st.session_state.df_db.loc[settle_idx, 'Home_Corner'] = h_cor
                st.session_state.df_db.loc[settle_idx, 'Away_Corner'] = a_cor
                st.session_state.df_db.loc[settle_idx, 'Profit'] = profit
                st.session_state.df_db.loc[settle_idx, 'Unit_Profit'] = u_profit
                st.session_state.df_db.loc[settle_idx, 'Payout'] = payout
                st.session_state.df_db.loc[settle_idx, 'Result_Label'] = res_label
                st.session_state.df_db.loc[settle_idx, 'Status'] = 'Settled'
                
                save_db(st.session_state.df_db, db_file)
                st.success("✅ 結算成功！賽果與盈虧已持久化至資料庫。")
                st.write(f"**結果:** {res_label} | **淨利潤:** ${profit} \vert{} **單位盈虧:** {u_profit}U \vert{} **退回總額:** ${payout}")

    # ------------------------------------------
    # Tab 4: 賽前分析日誌 (Analysis Log)
    # ------------------------------------------
    with t_log:
        st.subheader("撰寫與儲存分析日誌")
        with st.form("log_form"):
            log_match = st.text_input("關聯賽事 (例: 曼聯 vs 阿仙奴)")
            log_content = st.text_area("賽前深度分析內容 (傷停、戰意、盤路等)")
            log_conf = st.slider("信心指數 (1-10)", 1, 10, 7)
            
            if st.form_submit_button("📝 儲存日誌"):
                new_log = {
                    'ID': f"L{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'Match': log_match, 'Analysis_Content': log_content, 'Confidence_Level': log_conf
                }
                new_log_df = pd.DataFrame([new_log])
                st.session_state.df_log = pd.concat([st.session_state.df_log, new_log_df], ignore_index=True)
                save_db(st.session_state.df_log, log_file)
                st.success("✅ 分析日誌已成功寫入 `football_analysis_log.csv`")

if __name__ == "__main__":
    main()
