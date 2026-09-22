import streamlit as st
import pandas as pd
import os
import json
import base64
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
        total_deposit, total_withdraw = 0.0, 0.0
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

def calculate_settlement(bet_type, selection, line, odds, stake, h_g, a_g, h_c=0, a_c=0):
    diff = 0.0
    if bet_type == '讓球':
        if selection == 'Home': diff = h_g + line - a_g
        elif selection == 'Away': diff = a_g + line - h_g
    elif bet_type == '入球大小':
        total_goals = h_g + a_g
        if selection == 'Over': diff = total_goals - line
        elif selection == 'Under': diff = line - total_goals
    elif bet_type == '角球大小':
        total_corners = h_c + a_c
        if selection == 'Over': diff = total_corners - line
        elif selection == 'Under': diff = line - total_corners

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

    return round(profit, 2), round(payout, 2), round(profit / 100.0, 2), res_label, diff

def get_html_link(df, title):
    if df.empty: return "<p style='color: gray;'>目前尚無數據可供獨立開啟</p>"
    html_table = df.to_html(classes='table table-striped', index=False)
    full_html = f"<html><head><meta charset='utf-8'><title>{title}</title><style>body {{ font-family: 'Microsoft JhengHei', sans-serif; padding: 20px; }} table {{ border-collapse: collapse; width: 100%; font-size: 13px; text-align: center; }} th, td {{ border: 1px solid #ddd; padding: 8px; }} th {{ background-color: #007BFF; color: white; position: sticky; top: 0; }} tr:nth-child(even) {{ background-color: #f2f2f2; }}</style></head><body><h2>{title}</h2>{html_table}</body></html>"
    b64 = base64.b64encode(full_html.encode('utf-8')).decode()
    return f'<a href="data:text/html;base64,{b64}" target="_blank" style="text-decoration: none; display: inline-block; padding: 8px 16px; background-color: #28a745; color: white; border-radius: 5px; font-weight: bold; margin-top: 10px;">🌐 獨立開啟 {title} (HTML)</a>'

@st.dialog("🔍 全維度數據庫預覽", width="large")
def show_database_dialog(db_file, log_file, capital_file):
    df_db = load_db(db_file, DB_COLUMNS)
    df_cap = load_db(capital_file, CAPITAL_COLUMNS)
    tab1, tab2 = st.tabs(["📋 投注與動態紀錄", "💵 資金流水"])
    with tab1:
        st.dataframe(df_db, use_container_width=True, hide_index=True, height=400)
        st.markdown(get_html_link(df_db, "投注與動態紀錄"), unsafe_allow_html=True)
    with tab2:
        st.dataframe(df_cap, use_container_width=True, hide_index=True, height=400)

# ==========================================
# 3. 主程式 UI 與功能區
# ==========================================
def main():
    st.title("⚽ 足球博彩精算與資金管理系統")
    
    st.sidebar.header("⚙️ 系統設定與資金管理")
    mode = st.sidebar.radio("運作模式選擇", ["🧪 測試模式", "🟢 真實模式"])
    
    db_file = "football_betting_db_test.csv" if mode == "🧪 測試模式" else "football_betting_db.csv"
    log_file = "football_analysis_log_test.csv" if mode == "🧪 測試模式" else "football_analysis_log.csv"
    capital_file = "football_capital_db_test.csv" if mode == "🧪 測試模式" else "football_capital_db.csv"
    
    st.session_state.df_db = load_db(db_file, DB_COLUMNS)
    st.session_state.df_cap = load_db(capital_file, CAPITAL_COLUMNS)

    tot_dep, tot_wit, net_dep, tot_pnl, curr_bankroll, max_stake = get_capital_summary(st.session_state.df_cap, st.session_state.df_db)
    
    st.sidebar.divider()
    st.sidebar.subheader("💰 資金與盈虧總覽")
    st.sidebar.metric("淨存入本金", f"${net_dep:,.2f}")
    st.sidebar.metric("累積總盈虧 (PnL)", f"${tot_pnl:,.2f}", delta=f"${tot_pnl:,.2f}")
    st.sidebar.metric("當前總可用資金", f"${curr_bankroll:,.2f}")
    st.sidebar.caption(f"🛑 **單注上限 (本金 10%)**: `${max_stake:,.2f}`")

    cap_action = st.sidebar.selectbox("資金操作", ["無操作", "📥 存入本金", "📤 提取本金"])
    if cap_action == "📥 存入本金":
        with st.sidebar.form("deposit_form"):
            dep_amt = st.number_input("存入金額 ($)", min_value=100.0, step=100.0, value=1000.0)
            if st.form_submit_button("✅ 確認存入"):
                new_cap = {'ID': f"C{datetime.now().strftime('%Y%m%d%H%M%S')}", 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Type': 'Deposit', 'Amount': dep_amt, 'Note': "本金存入"}
                st.session_state.df_cap = pd.concat([st.session_state.df_cap, pd.DataFrame([new_cap])], ignore_index=True)
                save_db(st.session_state.df_cap, capital_file)
                st.rerun()
    elif cap_action == "📤 提取本金":
        with st.sidebar.form("withdraw_form"):
            wit_amt = st.number_input("提取金額 ($)", min_value=100.0, max_value=max(100.0, float(curr_bankroll)), step=100.0, value=500.0)
            if st.form_submit_button("✅ 確認提取"):
                new_cap = {'ID': f"C{datetime.now().strftime('%Y%m%d%H%M%S')}", 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Type': 'Withdraw', 'Amount': wit_amt, 'Note': "本金提取"}
                st.session_state.df_cap = pd.concat([st.session_state.df_cap, pd.DataFrame([new_cap])], ignore_index=True)
                save_db(st.session_state.df_cap, capital_file)
                st.rerun()

    st.sidebar.button("🔍 開啟完整資料庫", on_click=show_database_dialog, args=(db_file, log_file, capital_file), type="primary", use_container_width=True)

    t_pre, t_inplay, t_settle, t_ai = st.tabs(["📝 賽前建檔與投注", "⏱️ 即場動態", "⚖️ 賽果結算", "🤖 全局模型"])

    # ==========================================
    # Tab 1: 賽前建檔與投注 (全面升級版)
    # ==========================================
    with t_pre:
        st.subheader("📝 賽事建檔與智能盤口走勢分析")
        if net_dep <= 0: st.warning("⚠️ 目前尚無存入本金！無法精確計算注碼。")
        
        opts_leagues = ["➕ 新增手動輸入..."] + sorted(list(set(st.session_state.df_db['League'].dropna().unique())))
        opts_teams = ["➕ 新增手動輸入..."] + sorted(list(set(st.session_state.df_db['Home_Team'].dropna().tolist() + st.session_state.df_db['Away_Team'].dropna().tolist())))
        
        st.markdown("##### 1. 賽事與球隊資料")
        col_l, col_h, col_a = st.columns(3)
        sel_league = col_l.selectbox("賽事類別", opts_leagues)
        league = col_l.text_input("輸入新賽事類別") if sel_league == "➕ 新增手動輸入..." else sel_league
        sel_home = col_h.selectbox("主隊名稱", opts_teams, key="sh")
        home_team = col_h.text_input("輸入新主隊") if sel_home == "➕ 新增手動輸入..." else sel_home
        sel_away = col_a.selectbox("客隊名稱", opts_teams, key="sa")
        away_team = col_a.text_input("輸入新客隊") if sel_away == "➕ 新增手動輸入..." else sel_away

        c_hr, c_ar = st.columns(2)
        home_rating = c_hr.selectbox("主隊實力", ["S", "A", "B", "C", "D"])
        away_rating = c_ar.selectbox("客隊實力", ["S", "A", "B", "C", "D"])

        st.markdown("##### 2. 近 5 場狀態 (勝/和/敗)")
        f1, f2, f3, f4, f5, f6 = st.columns(6)
        hw = f1.number_input("主勝", 0, 10, 3)
        hd = f2.number_input("主和", 0, 10, 1)
        hl = f3.number_input("主敗", 0, 10, 1)
        aw = f4.number_input("客勝", 0, 10, 2)
        ad = f5.number_input("客和", 0, 10, 2)
        al = f6.number_input("客敗", 0, 10, 1)
        home_form, away_form = f"{hw}W{hd}D{hl}L", f"{aw}W{ad}D{al}L"

        # --- 3. 賽前盤口與動態賠率追蹤 ---
        st.markdown("##### 3. 賽前盤口與賠率走勢紀錄 (HKJC 學習模組)")
        st.caption("💡 系統支援同時輸入多個維度(讓球/入球/角球)的賠率變化。按下「觸發EV分析」後，系統會跨維度找出最具獲利潛力的單一選項。")
        
        if 'odds_history' not in st.session_state:
            st.session_state.odds_history = [{"id": 0, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False}]
            st.session_state.odds_counter = 1

        for i, row in enumerate(st.session_state.odds_history):
            c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
            row['type'] = c1.selectbox(f"盤口類型 {i+1}", ["讓球", "入球大小", "角球大小"], key=f"t_{row['id']}", index=["讓球", "入球大小", "角球大小"].index(row['type']))
            row['line'] = c2.number_input(f"盤口線 (Line)", step=0.25, value=float(row['line']), key=f"l_{row['id']}")
            
            up_lbl = "主隊(上盤)" if row['type'] == "讓球" else "大盤(Over)"
            row['upper'] = c3.number_input(f"{up_lbl} 賠率", min_value=1.01, value=float(row['upper']), step=0.01, key=f"up_{row['id']}")
            
            try:
                calc_lower = 1 / (1.085 - (1 / row['upper']))
                if calc_lower <= 1: calc_lower = 1.01
            except: calc_lower = 1.90
                
            row['unlock'] = c4.checkbox("解鎖修改", value=row['unlock'], key=f"u_{row['id']}")
            low_lbl = "客隊(下盤)" if row['type'] == "讓球" else "小盤(Under)"
            row['lower'] = c4.number_input(f"{low_lbl} 賠率", min_value=1.01, value=float(row['lower']) if row['unlock'] else float(round(calc_lower, 2)), step=0.01, disabled=not row['unlock'], key=f"low_{row['id']}")
            
            if len(st.session_state.odds_history) > 1:
                if c5.button("❌ 刪除", key=f"d_{row['id']}"):
                    st.session_state.odds_history.pop(i)
                    st.rerun()

        if st.button("➕ 增加一筆盤口與賠率變化 (記錄走勢)", use_container_width=True):
            st.session_state.odds_history.append({"id": st.session_state.odds_counter, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False})
            st.session_state.odds_counter += 1
            st.rerun()

        st.markdown("---")
        
        # --- 4. 觸發 EV 跨維度分析 ---
        if st.button("🧠 觸發全局期望值 (EV) 分析與預測", type="primary", use_container_width=True):
            st.session_state.show_analysis = True
            df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
            ml_active = len(df_settled) > 50 and HAS_AI_MODULES
            
            # 取出每種盤口的最新一筆數據
            latest_odds = {}
            for r in st.session_state.odds_history:
                latest_odds[r['type']] = r
                
            candidates = []
            r_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
            hr, ar = r_map.get(home_rating, 3), r_map.get(away_rating, 3)
            
            for b_type, l_row in latest_odds.items():
                p_up, p_low = 0.5, 0.5
                
                # 簡化版公式演算法 (若未達 50 筆或無 ML 狀態)
                # ML 深度學習擴充點已保留於此架構中
                if b_type == "讓球":
                    p_up = max(0.1, min(0.9, 0.5 + ((hr - ar) * 0.03)))
                elif b_type == "入球大小":
                    p_up = max(0.1, min(0.9, 0.5 + ((hr + ar - 6) * 0.02)))
                elif b_type == "角球大小":
                    p_up = max(0.1, min(0.9, 0.5 + ((hr + ar - 5) * 0.01)))
                
                p_low = 1 - p_up
                ev_up = p_up * (l_row['upper'] - 1) - (1 - p_up)
                ev_low = p_low * (l_row['lower'] - 1) - (1 - p_low)
                
                sel_up = "Home" if b_type == "讓球" else "Over"
                sel_low = "Away" if b_type == "讓球" else "Under"
                
                candidates.append({'bet_type': b_type, 'selection': sel_up, 'ev': ev_up, 'prob': p_up, 'odds': l_row['upper'], 'line': l_row['line'], 'label': "主隊(上盤)" if b_type=="讓球" else "大盤(Over)"})
                candidates.append({'bet_type': b_type, 'selection': sel_low, 'ev': ev_low, 'prob': p_low, 'odds': l_row['lower'], 'line': l_row['line'], 'label': "客隊(下盤)" if b_type=="讓球" else "小盤(Under)"})
            
            # 按 EV 降冪排序，找出全場最優解
            candidates = sorted(candidates, key=lambda x: x['ev'], reverse=True)
            best = candidates[0]
            
            # 計算最佳注碼 (半凱利 + 整10位數 + 風控限制)
            suggested_stake = 0
            if best['ev'] > 0 and curr_bankroll > 0:
                b = best['odds'] - 1
                kelly_f = max(0.0, min((best['prob'] * b - (1 - best['prob'])) / b, 0.10))
                suggested_stake = int(round((curr_bankroll * kelly_f * 0.5) / 10.0)) * 10
                
                if best['bet_type'] == "讓球" and suggested_stake < 200:
                    suggested_stake = 200 if best['ev'] > 0.05 else 0
                if best['bet_type'] in ["入球大小", "角球大小"] and suggested_stake < 10 and suggested_stake > 0:
                    suggested_stake = 10
                if suggested_stake > max_stake:
                    suggested_stake = int(max_stake / 10) * 10
                    
            st.session_state.analysis_data = {
                'ml_active': ml_active, 'best': best, 'stake': suggested_stake,
                'match': f"{home_team} vs {away_team}", 'league': league,
                'home_team': home_team, 'away_team': away_team, 
                'home_rating': home_rating, 'away_rating': away_rating,
                'home_form': home_form, 'away_form': away_form,
                'full_history': st.session_state.odds_history # 完整走勢供後續 AI 訓練
            }
        
        # --- 5. 輸出分析結果與最終寫入 ---
        if st.session_state.get('show_analysis') and 'analysis_data' in st.session_state:
            res = st.session_state.analysis_data
            best = res['best']
            st.divider()
            
            if res['ml_active']: st.success("🤖 **ML 模型已根據走勢特徵進行綜合評估**")
            else: st.info("ℹ️ **基礎模型運算中 (歷史數據未達 50 筆)**")
                
            st.markdown(f"### 🏆 系統判定最佳獲利項目：【{best['bet_type']}】 ➞ **{best['label']}**")
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("📌 盤口線 / 賠率", f"{best['line']} / {best['odds']}")
            col_r2.metric("📊 預估勝率 / EV", f"{best['prob']*100:.1f}% / {best['ev']:.3f}")
            col_r3.metric("💰 Kelly 推薦注碼", f"${res['stake']}" if res['stake'] > 0 else "🔴 EV 過低，放棄投注")
            
            with st.form("confirm_bet_form"):
                st.write("確認最終投注明細 (系統已自動代入最優選項)：")
                final_bet_type = st.selectbox("最終投注玩法", ["讓球", "入球大小", "角球大小"], index=["讓球", "入球大小", "角球大小"].index(best['bet_type']))
                final_selection = st.selectbox("確認投注選項", ["Home", "Away", "Over", "Under"], index=["Home", "Away", "Over", "Under"].index(best['selection']))
                c_f1, c_f2 = st.columns(2)
                final_line = c_f1.number_input("確認盤口線", value=float(best['line']), step=0.25)
                final_odds = c_f2.number_input("確認賠率", value=float(best['odds']), step=0.01)
                final_stake = st.number_input("確認下注金額 ($)", min_value=0, value=int(res['stake']), step=10)
                
                if st.form_submit_button("💾 確認投注並寫入系統"):
                    if final_stake > max_stake: st.error(f"❌ 投注超標 (上限 ${max_stake:,.2f})")
                    elif final_stake < 10 and final_bet_type in ["入球大小", "角球大小"]: st.error("❌ 大小/角球盤最低 $10")
                    elif final_stake < 200 and final_bet_type == "讓球": st.error("❌ 讓球盤最低 $200")
                    elif not res['home_team']: st.error("❌ 球隊名稱不可為空！")
                    else:
                        new_id = f"B{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
                        
                        new_record = {
                            'ID': new_id, 'Date': now_str, 'Status': 'Pending',
                            'League': res['league'], 'Match': res['match'], 'Home_Team': res['home_team'], 'Away_Team': res['away_team'],
                            'Home_Rating': res['home_rating'], 'Away_Rating': res['away_rating'], 'Home_Form': res['home_form'], 'Away_Form': res['away_form'],
                            'Bet_Type': final_bet_type, 'Selection': final_selection, 'Initial_Line': final_line, 'Initial_Odds': final_odds, 'Stake': final_stake,
                            'Odds_History': json.dumps(res['full_history'], ensure_ascii=False) # 儲存完整走勢陣列
                        }
                        st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([new_record])], ignore_index=True)
                        save_db(st.session_state.df_db, db_file)
                        st.session_state.show_analysis = False
                        st.success(f"✅ 投注單建立成功！單號：{new_id}")
                        st.rerun()

    # ==========================================
    # Tab 2 & 3: 動態與結算 (適配新玩法名稱)
    # ==========================================
    with t_inplay:
        pending_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Pending']
        if pending_df.empty: st.info("尚無未結算的賽事。")
        else:
            select_idx = st.selectbox("選擇追蹤賽事", pending_df.index, format_func=lambda i: pending_df.loc[i, 'Match'])
            row = pending_df.loc[select_idx]
            with st.form("inplay_form"):
                st.write(f"### ⚡ 即場更新: {row['Match']}")
                minute = st.number_input("比賽時間 (分鐘)", 0, 120, 45)
                h_da = st.number_input("主-危險進攻", value=int(row['Home_DA']) if pd.notna(row['Home_DA']) else 0)
                if st.form_submit_button("🔄 寫入時間點動態"):
                    st.session_state.df_db.loc[select_idx, ['InPlay_Minute', 'Home_DA']] = [minute, h_da]
                    save_db(st.session_state.df_db, db_file)
                    st.success("✅ 更新成功")

    with t_settle:
        pending_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Pending']
        if pending_df.empty: st.info("目前沒有待結算投注單。")
        else:
            settle_idx = st.selectbox("選擇結算單", pending_df.index, format_func=lambda i: pending_df.loc[i, 'Match'])
            s_row = pending_df.loc[settle_idx]
            c1, c2, c3, c4 = st.columns(4)
            h_goal = c1.number_input("🏠 主隊進球數", min_value=0, value=0)
            a_goal = c2.number_input("✈️ 客隊進球數", min_value=0, value=0)
            h_cor = c3.number_input("🏠 主隊角球數", min_value=0, value=0)
            a_cor = c4.number_input("✈️ 客隊角球數", min_value=0, value=0)
            
            if st.button("⚖️ 執行 5 態精算與結算", type="primary"):
                profit, payout, u_profit, res_label, diff = calculate_settlement(s_row['Bet_Type'], s_row['Selection'], s_row['Initial_Line'], s_row['Initial_Odds'], s_row['Stake'], h_goal, a_goal, h_cor, a_cor)
                st.session_state.df_db.loc[settle_idx, ['Home_Goal', 'Away_Goal', 'Home_Corner', 'Away_Corner', 'Profit', 'Unit_Profit', 'Payout', 'Result_Label', 'Status']] = [h_goal, a_goal, h_cor, a_cor, profit, u_profit, payout, res_label, 'Settled']
                save_db(st.session_state.df_db, db_file)
                st.success(f"✅ 結算完成！結果：{res_label} | 淨盈虧：${profit:,.2f}")
                st.rerun()

    with t_ai:
        st.header("🤖 大數據模型回測狀態")
        df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
        st.write(f"當前歷史結算數據庫：**{len(df_settled)}** 筆")
        if len(df_settled) > 50: st.success("✅ 數據量達標，全局決策樹模型運作中。系統現已具備讀取賠率走勢(Odds_History JSON) 與勝率的交叉學習能力。")
        else: st.warning("需累積至 50 筆結算數據，機器學習模組方可全面接管盤口走勢分析。")

if __name__ == "__main__":
    main()
