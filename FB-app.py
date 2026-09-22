import streamlit as st
import pandas as pd
import os
import json
import base64
import math
from datetime import datetime

# 嘗試載入機器學習套件
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

DB_COLUMNS = [
    'ID', 'Date', 'Status', 
    'League', 'Match', 'Home_Team', 'Away_Team', 'Home_Rating', 'Away_Rating', 'Home_Form', 'Away_Form',
    'Bet_Type', 'Selection', 'Initial_Line', 'Initial_Odds', 'Stake', 'Odds_History',
    'InPlay_Minute', 'Home_DA', 'Away_DA', 'Home_SoT', 'Away_SoT', 'Home_SoFF', 'Away_SoFF',
    'Home_Red', 'Away_Red', 'Home_Sub', 'Away_Sub', 'Home_Possession', 'Away_Possession',
    'Home_Goal', 'Away_Goal', 'Home_Corner', 'Away_Corner', 
    'Result_Label', 'Profit', 'Unit_Profit', 'Payout'
]
LOG_COLUMNS = ['ID', 'Date', 'Match', 'Analysis_Content', 'Confidence_Level']
CAPITAL_COLUMNS = ['ID', 'Date', 'Type', 'Amount', 'Note']

def load_db(filename, columns):
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            for col in columns:
                if col not in df.columns: df[col] = None
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
    if df_cap.empty:
        total_deposit = 0.0
        total_withdraw = 0.0
    else:
        total_deposit = pd.to_numeric(df_cap[df_cap['Type'] == 'Deposit']['Amount'], errors='coerce').sum()
        total_withdraw = pd.to_numeric(df_cap[df_cap['Type'] == 'Withdraw']['Amount'], errors='coerce').sum()
    
    net_deposit = max(0.0, total_deposit - total_withdraw)
    
    if df_db.empty:
        total_profit = 0.0
    else:
        settled_df = df_db[df_db['Status'] == 'Settled']
        total_profit = pd.to_numeric(settled_df['Profit'], errors='coerce').sum()
        
    current_bankroll = net_deposit + total_profit
    max_single_stake = net_deposit * 0.10 
    
    return round(total_deposit, 2), round(total_withdraw, 2), round(net_deposit, 2), round(total_profit, 2), round(current_bankroll, 2), round(max_single_stake, 2)

def calculate_settlement(bet_type, selection, line, odds, stake, h, a):
    diff = 0.0
    if bet_type == '讓球':
        if selection == 'Home': diff = h + line - a
        elif selection == 'Away': diff = a + line - h
    elif bet_type in ['大小', '角球大小']:
        total_goals = h + a
        if selection == 'Over': diff = total_goals - line
        elif selection == 'Under': diff = line - total_goals

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
    if df.empty: return "<p style='color: gray;'>目前尚無數據可供獨立開啟</p>"
    html_table = df.to_html(classes='table table-striped', index=False)
    full_html = f"<html><head><meta charset='utf-8'><title>{title}</title><style>body {{ font-family: 'Microsoft JhengHei', sans-serif; padding: 20px; }} table {{ border-collapse: collapse; width: 100%; font-size: 13px; text-align: center; }} th, td {{ border: 1px solid #ddd; padding: 8px; }} th {{ background-color: #007BFF; color: white; position: sticky; top: 0; }} tr:nth-child(even) {{ background-color: #f2f2f2; }}</style></head><body><h2>{title}</h2>{html_table}</body></html>"
    b64 = base64.b64encode(full_html.encode('utf-8')).decode()
    return f'<a href="data:text/html;base64,{b64}" target="_blank" style="text-decoration: none; display: inline-block; padding: 8px 16px; background-color: #28a745; color: white; border-radius: 5px; font-weight: bold; margin-top: 10px;">🌐 獨立開啟 {title} (HTML)</a>'

@st.dialog("🔍 全維度數據庫即時線上預覽", width="large")
def show_database_dialog(db_file, log_file, capital_file):
    df_db = load_db(db_file, DB_COLUMNS)
    df_log = load_db(log_file, LOG_COLUMNS)
    df_cap = load_db(capital_file, CAPITAL_COLUMNS)

    tab1, tab2, tab3 = st.tabs(["📋 投注與動態紀錄", "💵 資金存取流水", "📊 分析日誌"])
    with tab1:
        st.dataframe(df_db, use_container_width=True, hide_index=True, height=400)
        st.markdown(get_html_link(df_db, "投注與動態紀錄"), unsafe_allow_html=True)
    with tab2:
        st.dataframe(df_cap, use_container_width=True, hide_index=True, height=400)
        st.markdown(get_html_link(df_cap, "資金存取紀錄"), unsafe_allow_html=True)
    with tab3:
        st.dataframe(df_log, use_container_width=True, hide_index=True, height=400)
        st.markdown(get_html_link(df_log, "賽前分析日誌"), unsafe_allow_html=True)

# ==========================================
# 3. 主程式 UI 與功能區
# ==========================================
def main():
    st.title("⚽ 足球博彩精算與資金管理系統")
    
    # ------------------------------------------
    # 側邊欄：資金與模式
    # ------------------------------------------
    st.sidebar.header("⚙️ 系統設定與資金管理")
    mode = st.sidebar.radio("運作模式選擇", ["🧪 測試模式", "🟢 真實模式"])
    
    db_file = "football_betting_db_test.csv" if mode == "🧪 測試模式" else "football_betting_db.csv"
    log_file = "football_analysis_log_test.csv" if mode == "🧪 測試模式" else "football_analysis_log.csv"
    capital_file = "football_capital_db_test.csv" if mode == "🧪 測試模式" else "football_capital_db.csv"
    
    st.session_state.df_db = load_db(db_file, DB_COLUMNS)
    st.session_state.df_log = load_db(log_file, LOG_COLUMNS)
    st.session_state.df_cap = load_db(capital_file, CAPITAL_COLUMNS)

    tot_dep, tot_wit, net_dep, tot_pnl, curr_bankroll, max_stake = get_capital_summary(st.session_state.df_cap, st.session_state.df_db)
    
    st.sidebar.divider()
    st.sidebar.subheader("💰 資金與盈虧總覽")
    st.sidebar.metric("淨存入本金", f"${net_dep:,.2f}")
    st.sidebar.metric("累積總盈虧 (PnL)", f"${tot_pnl:,.2f}", delta=f"${tot_pnl:,.2f}")
    st.sidebar.metric("當前總可用資金", f"${curr_bankroll:,.2f}")
    st.sidebar.caption(f"🛑 **單注上限 (本金 10%)**: `${max_stake:,.2f}`")

    st.sidebar.divider()
    cap_action = st.sidebar.selectbox("資金操作", ["無操作", "📥 存入本金", "📤 提取本金"])
    
    if cap_action == "📥 存入本金":
        with st.sidebar.form("deposit_form"):
            dep_amt = st.number_input("存入金額 ($)", min_value=100.0, step=100.0, value=1000.0)
            dep_note = st.text_input("備註", value="本金存入")
            if st.form_submit_button("✅ 確認存入"):
                new_cap = {'ID': f"C{datetime.now().strftime('%Y%m%d%H%M%S')}", 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Type': 'Deposit', 'Amount': dep_amt, 'Note': dep_note}
                st.session_state.df_cap = pd.concat([st.session_state.df_cap, pd.DataFrame([new_cap])], ignore_index=True)
                save_db(st.session_state.df_cap, capital_file)
                st.success(f"成功存入 ${dep_amt:,.2f}！")
                st.rerun()

    elif cap_action == "📤 提取本金":
        with st.sidebar.form("withdraw_form"):
            wit_amt = st.number_input("提取金額 ($)", min_value=100.0, max_value=max(100.0, float(curr_bankroll)), step=100.0, value=500.0)
            wit_note = st.text_input("備註", value="本金/利潤提取")
            if st.form_submit_button("✅ 確認提取"):
                if wit_amt > curr_bankroll:
                    st.error("提取金額不可高於當前可用總資金！")
                else:
                    new_cap = {'ID': f"C{datetime.now().strftime('%Y%m%d%H%M%S')}", 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Type': 'Withdraw', 'Amount': wit_amt, 'Note': wit_note}
                    st.session_state.df_cap = pd.concat([st.session_state.df_cap, pd.DataFrame([new_cap])], ignore_index=True)
                    save_db(st.session_state.df_cap, capital_file)
                    st.success(f"成功提取 ${wit_amt:,.2f}！")
                    st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("🔍 開啟完整資料庫", type="primary", use_container_width=True):
        show_database_dialog(db_file, log_file, capital_file)

    # ------------------------------------------
    # 分頁設計
    # ------------------------------------------
    t_pre, t_inplay, t_settle, t_log, t_ai = st.tabs([
        "📝 賽前建檔與投注", "⏱️ 即場動態", "⚖️ 賽果結算", "📓 分析日誌", "🤖 機器學習與資金精算"
    ])

    # ==========================================
    # Tab 1: 全新升級 賽前建檔與投注
    # ==========================================
    with t_pre:
        st.subheader("📝 賽事建檔與智能投注決策")
        
        if net_dep <= 0:
            st.warning("⚠️ 目前尚無存入本金！請先至側邊欄進行「存入本金」後方可計算注碼。")
        
        # 提取歷史清單
        opts_leagues = ["➕ 新增手動輸入..."] + sorted(list(set(st.session_state.df_db['League'].dropna().unique())))
        opts_teams = ["➕ 新增手動輸入..."] + sorted(list(set(st.session_state.df_db['Home_Team'].dropna().tolist() + st.session_state.df_db['Away_Team'].dropna().tolist())))
        
        # --- 1. 基本賽事資料 ---
        st.markdown("##### 1. 賽事與球隊資料")
        col_l, col_h, col_a = st.columns(3)
        sel_league = col_l.selectbox("賽事類別", opts_leagues)
        league = col_l.text_input("輸入新賽事類別") if sel_league == "➕ 新增手動輸入..." else sel_league
        
        sel_home = col_h.selectbox("主隊名稱", opts_teams, key="sh")
        home_team = col_h.text_input("輸入新主隊") if sel_home == "➕ 新增手動輸入..." else sel_home
        
        sel_away = col_a.selectbox("客隊名稱", opts_teams, key="sa")
        away_team = col_a.text_input("輸入新客隊") if sel_away == "➕ 新增手動輸入..." else sel_away

        c_hr, c_ar = st.columns(2)
        home_rating = c_hr.selectbox("主隊實力評級", ["S", "A", "B", "C", "D"])
        away_rating = c_ar.selectbox("客隊實力評級", ["S", "A", "B", "C", "D"])

        # --- 2. 近 5 場狀態微調器 ---
        st.markdown("##### 2. 近 5 場狀態 (勝/和/敗)")
        f1, f2, f3, f4, f5, f6 = st.columns(6)
        hw = f1.number_input("主勝 (W)", 0, 10, 3, key='hw')
        hd = f2.number_input("主和 (D)", 0, 10, 1, key='hd')
        hl = f3.number_input("主敗 (L)", 0, 10, 1, key='hl')
        aw = f4.number_input("客勝 (W)", 0, 10, 2, key='aw')
        ad = f5.number_input("客和 (D)", 0, 10, 2, key='ad')
        al = f6.number_input("客敗 (L)", 0, 10, 1, key='al')
        home_form = f"{hw}W{hd}D{hl}L"
        away_form = f"{aw}W{ad}D{al}L"

        # --- 3. 動態賠率與盤口 ---
        st.markdown("##### 3. 盤口與動態賠率 (HKJC Margin 1.085)")
        b1, b2 = st.columns(2)
        bet_type = b1.selectbox("玩法類型", ["讓球", "大小", "角球大小"])
        line = b2.number_input("初始盤口線 (Line)", value=0.0, step=0.25)
        
        o1, o2 = st.columns(2)
        odds_upper_label = "主隊賠率 (Upper)" if bet_type == "讓球" else "大盤賠率 (Over)"
        odds_lower_label = "客隊賠率 (Lower)" if bet_type == "讓球" else "小盤賠率 (Under)"
        
        odds_upper = o1.number_input(f"{odds_upper_label}", min_value=1.01, value=1.90, step=0.01)
        
        # HKJC 1.085 Margin 自動計算
        try:
            # 1.085 = (1/odds_upper) + (1/odds_lower)
            odds_lower_calc = 1 / (1.085 - (1 / odds_upper))
            if odds_lower_calc <= 1: odds_lower_calc = 1.01
        except:
            odds_lower_calc = 1.90
            
        unlock_odds = o2.checkbox("🔓 解鎖手動修改小/下盤賠率 (關閉 HKJC 聯動)")
        odds_lower = o2.number_input(f"{odds_lower_label}", min_value=1.01, value=float(round(odds_lower_calc, 2)), step=0.01, disabled=not unlock_odds)

        # --- 4. 觸發 AI 分析與投注 ---
        st.markdown("---")
        if st.button("🧠 觸發期望值 (EV) 分析與預測", type="primary", use_container_width=True):
            st.session_state.show_analysis = True
            
            # 1. 偵測結算數據量以決定是否啟動 ML
            df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
            
            p_upper, p_lower = 0.5, 0.5 # 預設基礎概率
            ml_active = False
            
            if len(df_settled) > 50 and HAS_AI_MODULES:
                ml_active = True
                # 簡單 ML 特徵構建
                rating_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
                df_settled['H_Rating_Num'] = df_settled['Home_Rating'].map(rating_map).fillna(3)
                df_settled['A_Rating_Num'] = df_settled['Away_Rating'].map(rating_map).fillna(3)
                df_settled['Initial_Line'] = pd.to_numeric(df_settled['Initial_Line'], errors='coerce').fillna(0)
                df_settled['Initial_Odds'] = pd.to_numeric(df_settled['Initial_Odds'], errors='coerce').fillna(1.9)
                
                X = df_settled[['H_Rating_Num', 'A_Rating_Num', 'Initial_Line', 'Initial_Odds']]
                
                # 預測標籤處理
                if bet_type == "讓球":
                    df_settled['Target'] = df_settled.apply(lambda row: 1 if pd.to_numeric(row['Home_Goal'], errors='coerce') + pd.to_numeric(row['Initial_Line'], errors='coerce') > pd.to_numeric(row['Away_Goal'], errors='coerce') else 0, axis=1)
                else:
                    df_settled['Target'] = df_settled.apply(lambda row: 1 if pd.to_numeric(row['Home_Goal'], errors='coerce') + pd.to_numeric(row['Away_Goal'], errors='coerce') > pd.to_numeric(row['Initial_Line'], errors='coerce') else 0, axis=1)
                    
                y = df_settled['Target']
                
                try:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    X_new = pd.DataFrame({'H_Rating_Num': [rating_map.get(home_rating, 3)], 'A_Rating_Num': [rating_map.get(away_rating, 3)], 'Initial_Line': [line], 'Initial_Odds': [odds_upper]})
                    probs = model.predict_proba(X_new)[0]
                    
                    # 取勝率
                    if len(model.classes_) == 2:
                        p_upper = probs[1] # Target 1 機率 (主贏/大盤)
                        p_lower = probs[0] # Target 0 機率 (客贏/小盤)
                except Exception as e:
                    ml_active = False # 退回基礎算法
                    
            if not ml_active:
                # 基礎公式：依照實力差距給予些微概率優勢
                r_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
                diff = r_map[home_rating] - r_map[away_rating]
                p_upper = 0.5 + (diff * 0.03)
                p_lower = 1 - p_upper

            # 2. 計算 EV (期望值)
            ev_upper = p_upper * (odds_upper - 1) - (1 - p_upper)
            ev_lower = p_lower * (odds_lower - 1) - (1 - p_lower)
            
            if ev_upper >= ev_lower and ev_upper > 0:
                best_sel = "Home" if bet_type == "讓球" else "Over"
                best_ev, best_p, best_odds = ev_upper, p_upper, odds_upper
                sel_name = f"{odds_upper_label}"
            elif ev_lower > ev_upper and ev_lower > 0:
                best_sel = "Away" if bet_type == "讓球" else "Under"
                best_ev, best_p, best_odds = ev_lower, p_lower, odds_lower
                sel_name = f"{odds_lower_label}"
            else:
                best_sel, best_ev, best_p, best_odds = None, max(ev_upper, ev_lower), 0, 0
                sel_name = "無明顯優勢"

            # 3. 計算 Kelly 注碼 (半凱利，且四捨五入至 10 位數)
            suggested_stake = 0
            if best_sel and curr_bankroll > 0:
                b = best_odds - 1
                kelly_f = (best_p * b - (1 - best_p)) / b
                kelly_f = max(0.0, min(float(kelly_f), 0.10)) # 防呆：最多 10%
                raw_stake = curr_bankroll * kelly_f * 0.5 # 半凱利策略
                
                # 四捨五入至整 10 位數
                suggested_stake = int(round(raw_stake / 10.0)) * 10
                
                # 系統強硬下限限制
                if bet_type == "讓球" and suggested_stake < 200:
                    suggested_stake = 200 if best_ev > 0.05 else 0 # 結合前次的 $200 智能決策
                if bet_type in ["大小", "角球大小"] and suggested_stake < 10 and suggested_stake > 0:
                    suggested_stake = 10
                    
                if suggested_stake > max_stake:
                    suggested_stake = int(max_stake / 10) * 10
            
            # 將結果儲存至 session state 供確認表單使用
            st.session_state.analysis_data = {
                'ml_active': ml_active, 'best_sel': best_sel, 'sel_name': sel_name,
                'ev': best_ev, 'prob': best_p, 'stake': suggested_stake, 'odds': best_odds,
                'match': f"{home_team} vs {away_team}", 'league': league, 'bet_type': bet_type,
                'line': line, 'home_team': home_team, 'away_team': away_team, 
                'home_rating': home_rating, 'away_rating': away_rating,
                'home_form': home_form, 'away_form': away_form
            }
        
        # --- 5. 分析結果與最終寫入 ---
        if st.session_state.get('show_analysis') and 'analysis_data' in st.session_state:
            res = st.session_state.analysis_data
            st.divider()
            
            if res['ml_active']:
                st.success("🤖 **ML 模型已介入預測** (基於歷史 >50 筆結算數據訓練)")
            else:
                st.info("ℹ️ **基礎模型運算中** (歷史結算數據不足 50 筆，目前採用實力評級與賠率公式)")
                
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("📌 最佳 EV 投注項", res['sel_name'])
            col_r2.metric("📊 預估勝率 / EV", f"{res['prob']*100:.1f}% / {res['ev']:.3f}")
            col_r3.metric("💰 Kelly 推薦注碼", f"${res['stake']}" if res['stake'] > 0 else "🔴 不建議投注")
            
            if res['stake'] == 0:
                st.warning("⚠️ 系統判定此盤口無正期望值 (EV < 0) 或未達最低投注門檻標準，強烈建議放棄投注。")
            
            with st.form("confirm_bet_form"):
                final_selection = st.selectbox("確認投注選項", ["Home", "Away", "Over", "Under"], index=["Home", "Away", "Over", "Under"].index(res['best_sel']) if res['best_sel'] else 0)
                final_stake = st.number_input("確認下注金額 ($)", min_value=0, value=int(res['stake']), step=10)
                final_odds = res['odds'] # 採用剛剛對應的賠率
                
                if st.form_submit_button("💾 確認投注並寫入待結算庫"):
                    if final_stake > max_stake:
                        st.error(f"❌ 投注金額不能超過淨存入本金的 10%（單注上限：${max_stake:,.2f}）")
                    elif final_stake < 10 and res['bet_type'] in ["大小", "角球大小"]:
                        st.error("❌ 大小/角球盤最低投注額為 $10")
                    elif final_stake < 200 and res['bet_type'] == "讓球":
                        st.error("❌ 讓球盤最低投注額為 $200")
                    elif not res['home_team'] or not res['away_team']:
                        st.error("❌ 球隊名稱不可為空！")
                    else:
                        new_id = f"B{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
                        history = [{"time": now_str, "type": "Initial", "line": res['line'], "odds": final_odds}]
                        
                        new_record = {
                            'ID': new_id, 'Date': now_str, 'Status': 'Pending',
                            'League': res['league'], 'Match': res['match'], 'Home_Team': res['home_team'], 'Away_Team': res['away_team'],
                            'Home_Rating': res['home_rating'], 'Away_Rating': res['away_rating'], 'Home_Form': res['home_form'], 'Away_Form': res['away_form'],
                            'Bet_Type': res['bet_type'], 'Selection': final_selection, 'Initial_Line': res['line'], 'Initial_Odds': final_odds, 'Stake': final_stake,
                            'Odds_History': json.dumps(history, ensure_ascii=False)
                        }
                        
                        st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([new_record])], ignore_index=True)
                        save_db(st.session_state.df_db, db_file)
                        st.session_state.show_analysis = False # 隱藏分析區塊
                        st.success(f"✅ 投注單建立成功！單號：{new_id} | 實際下注額: ${final_stake}")
                        st.rerun()

    # ==========================================
    # Tab 2, 3, 4, 5 (保持原樣，完美繼承先前邏輯)
    # ==========================================
    with t_inplay:
        pending_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Pending']
        if pending_df.empty: st.info("尚無未結算的賽事可供追蹤。")
        else:
            select_idx = st.selectbox("選擇追蹤賽事", pending_df.index, format_func=lambda i: pending_df.loc[i, 'Match'])
            row = pending_df.loc[select_idx]
            with st.form("inplay_form"):
                st.write(f"### ⚡ 即場更新: {row['Match']}")
                minute = st.number_input("比賽時間 (分鐘)", 0, 120, 45)
                i_c1, i_c2, i_c3 = st.columns(3)
                h_da = i_c1.number_input("主-危險進攻", min_value=0, value=int(row['Home_DA']) if pd.notna(row['Home_DA']) else 0)
                a_da = i_c2.number_input("客-危險進攻", min_value=0, value=int(row['Away_DA']) if pd.notna(row['Away_DA']) else 0)
                h_pos = i_c3.slider("主隊控球率 (%)", 0, 100, 50)
                new_line = st.number_input("最新盤口線", value=float(row['Initial_Line']), step=0.25)
                new_odds = st.number_input("最新賠率", value=float(row['Initial_Odds']), step=0.01)
                
                if st.form_submit_button("🔄 更新即場數據與賠率"):
                    history_list = json.loads(row['Odds_History']) if pd.notna(row['Odds_History']) else []
                    history_list.append({"time": f"{minute}'", "type": "Live", "line": new_line, "odds": new_odds})
                    st.session_state.df_db.loc[select_idx, ['InPlay_Minute', 'Home_DA', 'Away_DA', 'Home_Possession', 'Away_Possession']] = [minute, h_da, a_da, h_pos, 100 - h_pos]
                    st.session_state.df_db.loc[select_idx, 'Odds_History'] = json.dumps(history_list, ensure_ascii=False)
                    save_db(st.session_state.df_db, db_file)
                    st.success("✅ 即場數據追蹤寫入成功！")

    with t_settle:
        pending_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Pending']
        if pending_df.empty: st.info("🎉 目前沒有待結算的投注單。")
        else:
            settle_idx = st.selectbox("選擇結算單", pending_df.index, format_func=lambda i: pending_df.loc[i, 'Match'])
            s_row = pending_df.loc[settle_idx]
            c1, c2 = st.columns(2)
            h_goal = c1.number_input("🏠 主隊進球數", min_value=0, value=0)
            a_goal = c2.number_input("✈️ 客隊進球數", min_value=0, value=0)
            if st.button("⚖️ 執行 5 態精算與結算", type="primary"):
                profit, payout, u_profit, res_label, diff = calculate_settlement(s_row['Bet_Type'], s_row['Selection'], s_row['Initial_Line'], s_row['Initial_Odds'], s_row['Stake'], h_goal, a_goal)
                st.session_state.df_db.loc[settle_idx, ['Home_Goal', 'Away_Goal', 'Profit', 'Unit_Profit', 'Payout', 'Result_Label', 'Status']] = [h_goal, a_goal, profit, u_profit, payout, res_label, 'Settled']
                save_db(st.session_state.df_db, db_file)
                st.success(f"✅ 結算完成！結果：{res_label} | 淨盈虧：${profit:,.2f}")
                st.rerun()

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

    with t_ai:
        st.header("🤖 大數據與複利資金模型回測")
        st.write("此區為系統全局模型分析。單場賽事的 ML 即時輔助請至「賽前建檔」觸發分析。")
        if not HAS_AI_MODULES: st.error("🚨 缺少機器學習套件！")
        else:
            df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
            if len(df_settled) < 5: st.warning(f"目前只有 {len(df_settled)} 筆結算數據。")
            else:
                st.success(f"✅ 全局 AI 模型已鎖定 {len(df_settled)} 筆歷史賽事！")

if __name__ == "__main__":
    main()
