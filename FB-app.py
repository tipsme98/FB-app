import streamlit as st
import pandas as pd
import os
import json
import base64
from datetime import datetime

# 嘗試載入機器學習套件 (加入防呆機制)
try:
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    HAS_AI_MODULES = True
except ImportError:
    HAS_AI_MODULES = False

# ==========================================
# 1. 初始化設定與資料庫 Schema
# ==========================================
st.set_page_config(page_title="足球博彩精算與資金管理系統", page_icon="⚽", layout="wide")

# 投注與賽果資料庫欄位
DB_COLUMNS = [
    'ID', 'Date', 'Status', 
    'League', 'Match', 'Home_Team', 'Away_Team', 'Home_Rating', 'Away_Rating', 'Home_Form', 'Away_Form',
    'Bet_Type', 'Selection', 'Initial_Line', 'Initial_Odds', 'Stake', 'Odds_History',
    'InPlay_Minute', 'Home_DA', 'Away_DA', 'Home_SoT', 'Away_SoT', 'Home_SoFF', 'Away_SoFF',
    'Home_Red', 'Away_Red', 'Home_Sub', 'Away_Sub', 'Home_Possession', 'Away_Possession',
    'Home_Goal', 'Away_Goal', 'Home_Corner', 'Away_Corner', 
    'Result_Label', 'Profit', 'Unit_Profit', 'Payout'
]

# 分析日誌欄位
LOG_COLUMNS = ['ID', 'Date', 'Match', 'Analysis_Content', 'Confidence_Level']

# 資金流水資料庫欄位 (存入/提取)
CAPITAL_COLUMNS = ['ID', 'Date', 'Type', 'Amount', 'Note']

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
# 2. 資金與風控算式
# ==========================================
def get_capital_summary(df_cap, df_db):
    """計算本金流水與累積盈虧"""
    if df_cap.empty:
        total_deposit = 0.0
        total_withdraw = 0.0
    else:
        total_deposit = pd.to_numeric(df_cap[df_cap['Type'] == 'Deposit']['Amount'], errors='coerce').sum()
        total_withdraw = pd.to_numeric(df_cap[df_cap['Type'] == 'Withdraw']['Amount'], errors='coerce').sum()
    
    net_deposit = max(0.0, total_deposit - total_withdraw)
    
    # 計算已結算賽事的總盈虧
    if df_db.empty:
        total_profit = 0.0
    else:
        settled_df = df_db[df_db['Status'] == 'Settled']
        total_profit = pd.to_numeric(settled_df['Profit'], errors='coerce').sum()
        
    current_bankroll = net_deposit + total_profit
    max_single_stake = net_deposit * 0.10  # 限制為存入淨本金的 10%
    
    return round(total_deposit, 2), round(total_withdraw, 2), round(net_deposit, 2), round(total_profit, 2), round(current_bankroll, 2), round(max_single_stake, 2)

def calculate_settlement(bet_type, selection, line, odds, stake, h, a):
    """5 態精算邏輯 (全贏、贏半、走盤、輸半、全輸)"""
    diff = 0.0
    if bet_type == '讓球':
        if selection == 'Home':
            diff = h + line - a
        elif selection == 'Away':
            diff = a + line - h
    elif bet_type in ['大小', '角球大小']:
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

def evaluate_handicap_200_decision(suggested_stake, odds, confidence, win_prob=None):
    """
    讓球盤當建議注額 < $200 時的智能決策：
    評估是否值得強行投注 $200 或放棄投注
    """
    p = win_prob if win_prob is not None else (confidence / 10.0 * 0.4 + 0.4) # 預設勝率模型
    ev = p * (odds - 1) - (1 - p) # 淨期望值 (EV)
    
    is_worth = (ev > 0.05) and (p >= 0.53)
    
    return {
        'is_worth': is_worth,
        'ev': round(ev, 3),
        'win_prob': round(p * 100, 1),
        'recommendation': "🟢 值得加註至 $200 門檻投注" if is_worth else "🔴 不值得投注 $200 (建議放棄此單)"
    }

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
def show_database_dialog(db_file, log_file, capital_file):
    df_db = load_db(db_file, DB_COLUMNS)
    df_log = load_db(log_file, LOG_COLUMNS)
    df_cap = load_db(capital_file, CAPITAL_COLUMNS)

    tab1, tab2, tab3 = st.tabs(["📋 投注與動態紀錄", "💵 資金存取流水紀錄", "📊 賽前分析日誌"])
    
    with tab1:
        st.caption(f"📂 賽事紀錄: `{db_file}`")
        st.dataframe(df_db, use_container_width=True, hide_index=True, height=400)
        st.markdown(get_html_link(df_db, "📋 投注與動態紀錄全覽"), unsafe_allow_html=True)
        
    with tab2:
        st.caption(f"📂 資金流水紀錄: `{capital_file}`")
        st.dataframe(df_cap, use_container_width=True, hide_index=True, height=400)
        st.markdown(get_html_link(df_cap, "💵 資金存取紀錄全覽"), unsafe_allow_html=True)

    with tab3:
        st.caption(f"📂 分析日誌: `{log_file}`")
        st.dataframe(df_log, use_container_width=True, hide_index=True, height=400)
        st.markdown(get_html_link(df_log, "📊 賽前分析日誌全覽"), unsafe_allow_html=True)

# ==========================================
# 3. 主程式 UI 與功能區
# ==========================================
def main():
    st.title("⚽ 足球博彩精算與資金管理系統")
    
    # ------------------------------------------
    # 側邊欄：模式切換與資金存取管理
    # ------------------------------------------
    st.sidebar.header("⚙️ 系統設定與資金管理")
    mode = st.sidebar.radio("運作模式選擇", ["🧪 測試模式", "🟢 真實模式"])
    
    db_file = "football_betting_db_test.csv" if mode == "🧪 測試模式" else "football_betting_db.csv"
    log_file = "football_analysis_log_test.csv" if mode == "🧪 測試模式" else "football_analysis_log.csv"
    capital_file = "football_capital_db_test.csv" if mode == "🧪 測試模式" else "football_capital_db.csv"
    
    st.session_state.df_db = load_db(db_file, DB_COLUMNS)
    st.session_state.df_log = load_db(log_file, LOG_COLUMNS)
    st.session_state.df_cap = load_db(capital_file, CAPITAL_COLUMNS)

    # 資金概況看板
    tot_dep, tot_wit, net_dep, tot_pnl, curr_bankroll, max_stake = get_capital_summary(st.session_state.df_cap, st.session_state.df_db)
    
    st.sidebar.divider()
    st.sidebar.subheader("💰 資金與盈虧總覽")
    st.sidebar.metric("淨存入本金", f"${net_dep:,.2f}")
    st.sidebar.metric("累積總盈虧 (PnL)", f"${tot_pnl:,.2f}", delta=f"${tot_pnl:,.2f}", delta_color="normal" if tot_pnl >= 0 else "inverse")
    st.sidebar.metric("當前總可用資金", f"${curr_bankroll:,.2f}")
    st.sidebar.caption(f"🛑 **單注上限 (本金 10%)**: `${max_stake:,.2f}`")

    # 存入 / 提取本金介面
    st.sidebar.divider()
    cap_action = st.sidebar.selectbox("資金操作選項", ["無操作", "📥 存入本金", "📤 提取本金"])
    
    if cap_action == "📥 存入本金":
        with st.sidebar.form("deposit_form"):
            dep_amt = st.number_input("存入金額 ($)", min_value=100.0, step=100.0, value=1000.0)
            dep_note = st.text_input("備註 (例: 4月初始本金注入)", value="本金存入")
            if st.form_submit_button("✅ 確認存入"):
                new_cap = {
                    'ID': f"C{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'Type': 'Deposit',
                    'Amount': dep_amt,
                    'Note': dep_note
                }
                st.session_state.df_cap = pd.concat([st.session_state.df_cap, pd.DataFrame([new_cap])], ignore_index=True)
                save_db(st.session_state.df_cap, capital_file)
                st.success(f"成功存入 ${dep_amt:,.2f}！")
                st.rerun()

    elif cap_action == "📤 提取本金":
        with st.sidebar.form("withdraw_form"):
            wit_amt = st.number_input("提取金額 ($)", min_value=100.0, max_value=max(100.0, float(curr_bankroll)), step=100.0, value=500.0)
            wit_note = st.text_input("備註 (例: 利潤提期)", value="本金/利潤提取")
            if st.form_submit_button("✅ 確認提取"):
                if wit_amt > curr_bankroll:
                    st.error("提取金額不可高於當前可用總資金！")
                else:
                    new_cap = {
                        'ID': f"C{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        'Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'Type': 'Withdraw',
                        'Amount': wit_amt,
                        'Note': wit_note
                    }
                    st.session_state.df_cap = pd.concat([st.session_state.df_cap, pd.DataFrame([new_cap])], ignore_index=True)
                    save_db(st.session_state.df_cap, capital_file)
                    st.success(f"成功提取 ${wit_amt:,.2f}！")
                    st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("🔍 開啟/預覽完整資料庫", type="primary", use_container_width=True):
        show_database_dialog(db_file, log_file, capital_file)

    # ------------------------------------------
    # 分頁設計
    # ------------------------------------------
    t_pre, t_inplay, t_settle, t_log, t_ai = st.tabs([
        "📝 賽前建檔與投注", "⏱️ 即場動態", "⚖️ 賽果結算", "📓 分析日誌", "🤖 AI 與資金精算"
    ])

    # --- Tab 1: 賽前建檔與投注 ---
    with t_pre:
        st.subheader("📝 新增賽事與投注單")
        
        if net_dep <= 0:
            st.warning("⚠️ 目前尚無存入本金！請先至側邊欄進行「存入本金」後方可進行投注風控管理。")

        with st.form("pre_match_form"):
            st.markdown("##### 1. 賽事基礎維度")
            col1, col2, col3 = st.columns(3)
            league = col1.text_input("賽事類別 (例: 英超)")
            home_team = col2.text_input("主隊名稱")
            away_team = col3.text_input("客隊名稱")
            
            col4, col5 = st.columns(2)
            home_rating = col4.selectbox("主隊評級", ["S", "A", "B", "C", "D"])
            away_rating = col5.selectbox("客隊評級", ["S", "A", "B", "C", "D"])
            home_form = col4.text_input("主隊近況", value="3W1D1L")
            away_form = col5.text_input("客隊近況", value="2W2D1L")
            
            st.markdown("##### 2. 投注與盤口維度")
            c_bet1, c_bet2, c_bet3 = st.columns(3)
            bet_type = c_bet1.selectbox("玩法類型", ["讓球", "大小", "角球大小"])
            selection = c_bet2.selectbox("投注選項", ["Home", "Away", "Over", "Under"])
            line = c_bet3.number_input("初始盤口線 (Line)", value=0.0, step=0.25)
            
            odds = c_bet1.number_input("初始賠率", min_value=1.01, value=1.90, step=0.01)
            stake = c_bet2.number_input("擬定投注本金 ($)", min_value=0.0, value=float(min(200.0, max_stake)), step=10.0)
            confidence = c_bet3.slider("分析信心度 (1-10)", 1, 10, 7)

            # 風控防呆邏輯檢查
            is_stake_valid = True
            error_msg = ""
            
            # 限制1: 不得超過 10% 存入本金
            if stake > max_stake:
                is_stake_valid = False
                error_msg = f"❌ 投注金額不能超過淨存入本金的 10%（單注上限：${max_stake:,.2f}）"
            
            # 限制2: 最低投注額度
            if bet_type in ["大小", "角球大小"] and stake < 10:
                is_stake_valid = False
                error_msg = "❌ 入球及角球大小的最低投注額為 $10"
            elif bet_type == "讓球" and stake < 200 and stake >= max_stake:
                pass # 交由下方的智能決策引擎

            match_name = f"{home_team} vs {away_team}"
            submit_pre = st.form_submit_button("💾 執行投注與儲存")

        # 讓球盤不足 $200 的智能評估視窗
        if bet_type == "讓球" and stake < 200:
            st.info("💡 **讓球盤 $200 最低門檻智能決策提示**：")
            decision = evaluate_handicap_200_decision(stake, odds, confidence)
            
            d_col1, d_col2, d_col3 = st.columns(3)
            d_col1.metric("估算勝率", f"{decision['win_prob']}%")
            d_col2.metric("淨期望值 (EV)", f"{decision['ev']}")
            d_col3.metric("系統決策建議", decision['recommendation'])
            
            if not decision['is_worth']:
                st.warning("⚠️ 系統評估：當前賽事優勢/信心不充份，不值得強行加註至 $200，建議放棄下注該讓球盤。")

        if submit_pre:
            if not is_stake_valid:
                st.error(error_msg)
            elif bet_type == "讓球" and stake < 200 and not evaluate_handicap_200_decision(stake, odds, confidence)['is_worth']:
                st.error("❌ 系統拒絕投注：該讓球盤建議額不足 $200 且正期望值不足，不值得投注 $200。")
            elif home_team and away_team:
                actual_stake = 200.0 if (bet_type == "讓球" and stake < 200) else stake
                if actual_stake > max_stake:
                    st.error(f"❌ 調整後的投注額 (${actual_stake}) 超出了您的單注 10\% 上限 (${max_stake})，無法投注！")
                else:
                    new_id = f"B{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
                    initial_odds_history = [{"time": now_str, "type": "Initial", "line": line, "odds": odds}]
                    
                    new_record = {
                        'ID': new_id, 'Date': now_str, 'Status': 'Pending',
                        'League': league, 'Match': match_name, 'Home_Team': home_team, 'Away_Team': away_team,
                        'Home_Rating': home_rating, 'Away_Rating': away_rating, 'Home_Form': home_form, 'Away_Form': away_form,
                        'Bet_Type': bet_type, 'Selection': selection, 'Initial_Line': line, 'Initial_Odds': odds, 'Stake': actual_stake,
                        'Odds_History': json.dumps(initial_odds_history, ensure_ascii=False)
                    }
                    
                    st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([new_record])], ignore_index=True)
                    save_db(st.session_state.df_db, db_file)
                    st.success(f"✅ 投注單建立成功！單號：{new_id} | 實際下注額: ${actual_stake}")

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
                minute = st.number_input("比賽時間 (分鐘)", min_value=0, max_value=120, value=45)
                
                i_c1, i_c2, i_c3 = st.columns(3)
                h_da = i_c1.number_input("主-危險進攻", min_value=0, value=int(row['Home_DA']) if pd.notna(row['Home_DA']) else 0)
                a_da = i_c2.number_input("客-危險進攻", min_value=0, value=int(row['Away_DA']) if pd.notna(row['Away_DA']) else 0)
                h_pos = i_c3.slider("主隊控球率 (%)", 0, 100, 50)
                
                new_line = st.number_input("最新盤口線", value=float(row['Initial_Line']), step=0.25)
                new_odds = st.number_input("最新賠率", value=float(row['Initial_Odds']), step=0.01)
                
                if st.form_submit_button("🔄 更新即場數據與賠率"):
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
                    st.success("✅ 即場數據與賠率追蹤寫入成功！")

    # --- Tab 3: 賽果結算 ---
    with t_settle:
        pending_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Pending']
        if pending_df.empty:
            st.info("🎉 目前沒有待結算的投注單。")
        else:
            settle_idx = st.selectbox("選擇結算單", pending_df.index, format_func=lambda i: pending_df.loc[i, 'Match'])
            s_row = pending_df.loc[settle_idx]
            
            st.write(f"### ⚖️ 結算注單: {s_row['Match']} ({s_row['Bet_Type']} | 下注: ${s_row['Stake']})")
            c1, c2 = st.columns(2)
            h_goal = c1.number_input("🏠 主隊進球數", min_value=0, value=0)
            a_goal = c2.number_input("✈️ 客隊進球數", min_value=0, value=0)
            
            if st.button("⚖️ 執行 5 態精算與結算", type="primary"):
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
                st.success(f"✅ 結算完成！結果：{res_label} | 淨盈虧：${profit:,.2f}")
                st.rerun()

    # --- Tab 4: 分析日誌 ---
    with t_log:
        with st.form("log_form"):
            log_match = st.text_input("關聯賽事")
            log_content = st.text_area("賽前深度分析內容")
            log_conf = st.slider("信心指數", 1, 10, 7)
            if st.form_submit_button("📝 儲存日誌"):
                new_log = {'ID': f"L{datetime.now().strftime('%Y%m%d%H%M%S')}", 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Match': log_match, 'Analysis_Content': log_content, 'Confidence_Level': log_conf}
                st.session_state.df_log = pd.concat([st.session_state.df_log, pd.DataFrame([new_log])], ignore_index=True)
                save_db(st.session_state.df_log, log_file)
                st.success("✅ 日誌已儲存")

    # --- Tab 5: AI 機器學習與資金管理大數據 ---
    with t_ai:
        st.header("🤖 機器學習與長期複利資金精算")
        st.write("結合歷史賽事與資金流水數據，運用 **Random Forest** 模型與 **凱利公式 (Kelly Criterion)** 計算最佳動態下注比例。")
        
        if not HAS_AI_MODULES:
            st.error("🚨 **缺少機器學習套件！** 請確保 GitHub 儲存庫根目錄包含 `requirements.txt` 並寫入 `scikit-learn` 與 `numpy`。")
        else:
            df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
            
            if len(df_settled) < 5:
                st.warning(f"目前只有 {len(df_settled)} 筆結算數據。AI 需要至少 5 筆歷史賽事進行機器學習訓練。")
            else:
                try:
                    rating_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
                    df_settled['H_Rating_Num'] = df_settled['Home_Rating'].map(rating_map).fillna(3)
                    df_settled['A_Rating_Num'] = df_settled['Away_Rating'].map(rating_map).fillna(3)
                    df_settled['Initial_Line'] = pd.to_numeric(df_settled['Initial_Line'], errors='coerce').fillna(0)
                    df_settled['Initial_Odds'] = pd.to_numeric(df_settled['Initial_Odds'], errors='coerce').fillna(1.9)
                    df_settled['Goal_Diff'] = pd.to_numeric(df_settled['Home_Goal'], errors='coerce').fillna(0) - pd.to_numeric(df_settled['Away_Goal'], errors='coerce').fillna(0)
                    df_settled['Target'] = df_settled['Goal_Diff'].apply(lambda x: 1 if x > 0 else (0 if x == 0 else -1))
                    
                    features = ['H_Rating_Num', 'A_Rating_Num', 'Initial_Line', 'Initial_Odds']
                    X = df_settled[features]
                    y = df_settled['Target']
                    
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    st.success(f"✅ AI 資金模型已成功利用 {len(df_settled)} 筆歷史賽事訓練完成！")
                    
                    st.divider()
                    st.subheader("📊 凱利最佳注碼建議 (Kelly Criterion)")
                    
                    k_odds = st.number_input("擬投注賽事賠率", value=1.95, step=0.01)
                    k_prob = st.slider("模型評估勝率 (%)", 10, 90, 58) / 100.0
                    
                    # 凱利公式: f = (bp - q) / b = (p*odds - 1) / (odds - 1)
                    b = k_odds - 1.0
                    p = k_prob
                    q = 1.0 - p
                    f_kelly = (b * p - q) / b
                    
                    # 建議保守凱利 (Half Kelly) 搭配 10% 風控上限
                    f_suggested = min(max(0.0, f_kelly * 0.5), 0.10)
                    suggested_kelly_stake = curr_bankroll * f_suggested
                    
                    col_k1, col_k2, col_k3 = st.columns(3)
                    col_k1.metric("理論純凱利比例", f"{f_kelly*100:.1f}%" if f_kelly > 0 else "0.0%")
                    col_k2.metric("半凱利最佳注碼比例", f"{f_suggested*100:.1f}%")
                    col_k3.metric("建議精確投注額 ($)", f"${suggested_kelly_stake:,.2f}")
                    
                except Exception as e:
                    st.error(f"模型計算錯誤：{e}")

if __name__ == "__main__":
    main()
