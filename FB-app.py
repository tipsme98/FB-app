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
                if col not in df.columns: 
                    df[col] = pd.Series(dtype='object')
            return df
        except Exception:
            return pd.DataFrame(columns=columns)
    else:
        return pd.DataFrame(columns=columns)

def save_db(df, filename):
    df.to_csv(filename, index=False)

# ==========================================
# 2. 資金、風控與累計算式 (State Rebuilding Engine)
# ==========================================
def recalculate_bankroll_from_scratch(df_cap, df_db):
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
        elif selection == 'Away': diff = a_g - line - h_g
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

def display_cumulative_metrics(df):
    total_profit = pd.to_numeric(df['Profit'], errors='coerce').sum()
    total_unit_profit = pd.to_numeric(df['Unit_Profit'], errors='coerce').sum()
    total_payout = pd.to_numeric(df['Payout'], errors='coerce').sum()
    
    st.markdown("### 📊 數據庫累計總額看板 (Cumulative Summary)")
    m1, m2, m3 = st.columns(3)
    
    m1.metric("累積淨盈虧 (Total Profit)", f"${total_profit:,.2f}", delta=f"{total_profit:,.2f}")
    m2.metric("累積單位平注盈虧 (Total Unit Profit)", f"{total_unit_profit:,.2f} U", delta=f"{total_unit_profit:,.2f}")
    m3.metric("累積派彩總額 (Total Payout)", f"${total_payout:,.2f}")
    st.divider()

@st.dialog("🔍 全維度數據庫預覽", width="large")
def show_database_dialog(db_file, log_file, capital_file):
    df_db = load_db(db_file, DB_COLUMNS)
    df_cap = load_db(capital_file, CAPITAL_COLUMNS)
    
    display_cumulative_metrics(df_db)
    
    tab1, tab2 = st.tabs(["📋 投注與動態紀錄", "💵 系統資金流水"])
    with tab1:
        st.dataframe(df_db, use_container_width=True, hide_index=True, height=400)
        st.markdown(get_html_link(df_db, "投注與動態紀錄"), unsafe_allow_html=True)
    with tab2:
        if not df_cap.empty:
            disp_df = df_cap.copy()
            disp_df['Amount'] = pd.to_numeric(disp_df['Amount'], errors='coerce').fillna(0)
            
            # 將金額轉為帶正負號的值：存入為負，提取為正
            disp_df['Signed_Amount'] = disp_df.apply(
                lambda row: -row['Amount'] if row['Type'] == 'Deposit' else row['Amount'], axis=1
            )
            
            total_sum = disp_df['Signed_Amount'].sum()
            
            # 新增總和列
            total_row = pd.DataFrame([{
                'ID': '總和 (Total)',
                'Date': '',
                'Type': '',
                'Amount': 0, 
                'Note': '',
                'Signed_Amount': total_sum
            }])
            disp_df = pd.concat([disp_df, total_row], ignore_index=True)
            
            # 格式化顯示字串 (加入 +/- 符號與千分位)
            def format_money(x):
                if pd.isna(x): return ""
                if x > 0: res = f"+{x:,.2f}"
                elif x < 0: res = f"{x:,.2f}"
                else: res = "0"
                return res.replace(".00", "")
                
            disp_df['Amount'] = disp_df['Signed_Amount'].apply(format_money)
            disp_df = disp_df.drop(columns=['Signed_Amount'])
            
            # 樣式渲染：正數(提取)綠色，負數(存入)紅色
            def color_amount(val):
                val_str = str(val)
                if val_str.startswith('+'): return 'color: #00cc66;'
                elif val_str.startswith('-'): return 'color: #ff4444;'
                return ''
            
            if hasattr(disp_df.style, 'map'):
                styled_df = disp_df.style.map(color_amount, subset=['Amount'])
            else:
                styled_df = disp_df.style.applymap(color_amount, subset=['Amount'])
                
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)
        else:
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

    tot_dep, tot_wit, net_dep, tot_pnl, curr_bankroll, max_stake = recalculate_bankroll_from_scratch(st.session_state.df_cap, st.session_state.df_db)
    
    st.sidebar.divider()
    st.sidebar.subheader("💰 系統本金與盈虧總覽")
    st.sidebar.metric("系統淨存入本金", f"${net_dep:,.2f}")
    st.sidebar.metric("累積總盈虧 (PnL)", f"${tot_pnl:,.2f}", delta=f"${tot_pnl:,.2f}")
    st.sidebar.metric("當前總可用資金", f"${curr_bankroll:,.2f}")
    st.sidebar.caption(f"🛑 **單注上限 (本金 10%)**: `${max_stake:,.2f}`")

    cap_action = st.sidebar.selectbox("系統內部本金操作", ["無操作", "📥 存入本金", "📤 提取本金"])
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

    with t_pre:
        st.subheader("📝 賽事建檔與智能盤口走勢分析")
        if net_dep <= 0: st.warning("⚠️ 目前系統內部尚無存入本金！無法精確計算注碼。")
        
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

        st.markdown("##### 3. 賽前盤口與賠率走勢紀錄")
        if 'odds_history' not in st.session_state:
            st.session_state.odds_history = [{"id": 0, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False}]
            st.session_state.odds_counter = 1

        for i, row in enumerate(st.session_state.odds_history):
            c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
            row['type'] = c1.selectbox(f"盤口類型 {i+1}", ["讓球", "入球大小", "角球大小"], key=f"t_{row['id']}", index=["讓球", "入球大小", "角球大小"].index(row['type']))
            row['line'] = c2.number_input(f"盤口線 (Line)", step=0.25, value=float(row['line']), key=f"l_{row['id']}")
            up_lbl = "主隊" if row['type'] == "讓球" else "大盤(Over)"
            row['upper'] = c3.number_input(f"{up_lbl} 賠率", min_value=1.01, value=float(row['upper']), step=0.01, key=f"up_{row['id']}")
            try:
                calc_lower = round(1 / (1.085 - (1 / row['upper'])), 2)
                if calc_lower <= 1: calc_lower = 1.01
            except: 
                calc_lower = 1.90
            row['unlock'] = c4.checkbox("🔓 手動更改", value=row['unlock'], key=f"u_{row['id']}")
            low_key = f"low_{row['id']}"
            if not row['unlock']:
                st.session_state[low_key] = calc_lower
                row['lower'] = calc_lower
            low_lbl = "客隊" if row['type'] == "讓球" else "小盤(Under)"
            row['lower'] = c4.number_input(f"{low_lbl} 賠率", min_value=1.01, step=0.01, disabled=not row['unlock'], key=low_key)
            if len(st.session_state.odds_history) > 1:
                if c5.button("❌", key=f"d_{row['id']}"):
                    st.session_state.odds_history.pop(i)
                    st.rerun()

        if st.button("➕ 增加一筆盤口與賠率變化", use_container_width=True):
            st.session_state.odds_history.append({"id": st.session_state.odds_counter, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False})
            st.session_state.odds_counter += 1
            st.rerun()

        st.markdown("---")
        
        if st.button("🧠 觸發全局期望值 (EV) 分析與預測", type="primary", use_container_width=True):
            st.session_state.show_analysis = True
            df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
            ml_active = len(df_settled) > 50 and HAS_AI_MODULES
            latest_odds = {}
            for r in st.session_state.odds_history: latest_odds[r['type']] = r
            candidates = []
            r_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
            hr, ar = r_map.get(home_rating, 3), r_map.get(away_rating, 3)
            
            for b_type, l_row in latest_odds.items():
                p_up, p_low = 0.5, 0.5
                line_val = float(l_row['line'])
                if b_type == "讓球": p_up = max(0.1, min(0.9, 0.5 + ((hr - ar) * 0.03)))
                elif b_type == "入球大小": p_up = max(0.1, min(0.9, 0.5 + ((hr + ar - 6) * 0.02)))
                elif b_type == "角球大小": p_up = max(0.1, min(0.9, 0.5 + ((hr + ar - 5) * 0.01)))
                p_low = 1 - p_up
                ev_up = p_up * (l_row['upper'] - 1) - (1 - p_up)
                ev_low = p_low * (l_row['lower'] - 1) - (1 - p_low)
                if b_type == "讓球":
                    home_is_upper = (line_val <= 0)
                    label_home = "主隊(上盤)" if home_is_upper else "主隊(下盤)"
                    label_away = "客隊(下盤)" if home_is_upper else "客隊(上盤)"
                    candidates.append({'bet_type': b_type, 'selection': 'Home', 'ev': ev_up, 'prob': p_up, 'odds': l_row['upper'], 'line': line_val, 'label': label_home})
                    candidates.append({'bet_type': b_type, 'selection': 'Away', 'ev': ev_low, 'prob': p_low, 'odds': l_row['lower'], 'line': line_val, 'label': label_away})
                else:
                    candidates.append({'bet_type': b_type, 'selection': 'Over', 'ev': ev_up, 'prob': p_up, 'odds': l_row['upper'], 'line': line_val, 'label': "大盤(Over)"})
                    candidates.append({'bet_type': b_type, 'selection': 'Under', 'ev': ev_low, 'prob': p_low, 'odds': l_row['lower'], 'line': line_val, 'label': "小盤(Under)"})
            
            candidates = sorted(candidates, key=lambda x: x['ev'], reverse=True)
            best = candidates[0]
            suggested_stake = 0
            if best['ev'] > 0 and curr_bankroll > 0:
                b = best['odds'] - 1
                kelly_f = max(0.0, min((best['prob'] * b - (1 - best['prob'])) / b, 0.10))
                suggested_stake = max(10.0, min(float(max_stake), float(round((curr_bankroll * (kelly_f * 0.5)) / 10) * 10)))

            st.session_state.analysis_result = {
                'candidates': candidates, 'best': best, 'ml_active': ml_active, 'stake': suggested_stake,
                'home_form': home_form, 'away_form': away_form, 'home_rating': home_rating, 'away_rating': away_rating
            }
            
        if st.session_state.get('show_analysis', False):
            res = st.session_state.analysis_result
            best = res['best']
            st.success("✅ EV 運算完成！")
            c1, c2, c3 = st.columns(3)
            c1.metric("💡 首選推薦", f"{best['bet_type']} - {best['label']}")
            c2.metric("🎯 預期勝率", f"{best['prob']*100:.1f}%")
            c3.metric("📊 EV", f"{best['ev']:.3f}", delta="正期望值" if best['ev']>0 else "負期望值", delta_color="normal" if best['ev']>0 else "inverse")
            st.info(f"**建議注碼**：`${res['stake']:,.2f}`")

            with st.form("bet_form"):
                bc1, bc2, bc3 = st.columns(3)
                final_btype = bc1.selectbox("最終投注項目", [c['bet_type'] for c in res['candidates']], index=0)
                final_sel = bc2.selectbox("最終投注方向", ["Home", "Away", "Over", "Under"], index=["Home", "Away", "Over", "Under"].index(best['selection']))
                final_stake = bc3.number_input("實際下注金額 ($)", min_value=10.0, step=10.0, value=float(res['stake'] if res['stake'] > 0 else 50.0))
                final_row = next((r for r in st.session_state.odds_history if r['type'] == final_btype), st.session_state.odds_history[-1])
                line = float(final_row['line'])
                odds = float(final_row['upper']) if final_sel in ["Home", "Over"] else float(final_row['lower'])
                
                if st.form_submit_button("✅ 確定投注並寫入資料庫"):
                    new_id = f"B{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    new_record = {
                        'ID': new_id, 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Status': 'Open',
                        'League': league, 'Match': f"{home_team} vs {away_team}", 'Home_Team': home_team, 'Away_Team': away_team,
                        'Home_Rating': res['home_rating'], 'Away_Rating': res['away_rating'], 'Home_Form': res['home_form'], 'Away_Form': res['away_form'],
                        'Bet_Type': final_btype, 'Selection': final_sel, 'Initial_Line': line, 'Initial_Odds': odds, 'Stake': final_stake,
                        'Odds_History': json.dumps(st.session_state.odds_history, ensure_ascii=False)
                    }
                    st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([new_record])], ignore_index=True)
                    save_db(st.session_state.df_db, db_file)
                    st.session_state.odds_history = [{"id": 0, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False}] 
                    st.session_state.show_analysis = False
                    st.toast("✅ 投注紀錄寫入成功！", icon="📝")
                    st.rerun()

    with t_inplay:
        st.info("⏱️ 即場數據更新區")
        pending_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
        if not pending_df.empty:
            select_idx = st.selectbox("選擇追蹤賽事", pending_df.index, format_func=lambda i: pending_df.loc[i, 'Match'])
            row = pending_df.loc[select_idx]
            with st.form("inplay_form"):
                minute = st.number_input("比賽時間 (分鐘)", 0, 120, 45)
                h_da = st.number_input("主-危險進攻", value=int(row['Home_DA']) if pd.notna(row['Home_DA']) else 0)
                if st.form_submit_button("🔄 寫入更新"):
                    st.session_state.df_db.loc[select_idx, ['InPlay_Minute', 'Home_DA']] = [minute, h_da]
                    save_db(st.session_state.df_db, db_file)
                    st.success("✅ 更新成功")
    
    with t_settle:
        st.subheader("⚖️ 賽事結算區")
        display_cumulative_metrics(st.session_state.df_db)
        
        st.markdown("#### ⏳ 待結算注單")
        open_bets = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
        if not open_bets.empty:
            for idx, row in open_bets.iterrows():
                with st.expander(f"📌 {row['Match']} - {row['Bet_Type']} ({row['Selection']}) | 盤口: {row['Initial_Line']}"):
                    with st.form(f"settle_form_{row['ID']}"):
                        col1, col2 = st.columns(2)
                        h_g = col1.number_input("全場主隊入球數", min_value=0, value=0)
                        a_g = col2.number_input("全場客隊入球數", min_value=0, value=0)
                        h_c, a_c = 0, 0
                        if row['Bet_Type'] == '角球大小':
                            h_c = col1.number_input("全場主隊角球", min_value=0, value=0)
                            a_c = col2.number_input("全場客隊角球", min_value=0, value=0)
                        
                        if st.form_submit_button("確認賽果並結算"):
                            prof, payout, u_prof, lbl, diff = calculate_settlement(
                                row['Bet_Type'], row['Selection'], float(row['Initial_Line']), 
                                float(row['Initial_Odds']), float(row['Stake']), h_g, a_g, h_c, a_c
                            )
                            for col in ['Result_Label', 'Status']:
                                if st.session_state.df_db[col].dtype != 'object': st.session_state.df_db[col] = st.session_state.df_db[col].astype('object')
                            st.session_state.df_db.loc[idx, ['Home_Goal', 'Away_Goal', 'Home_Corner', 'Away_Corner', 'Result_Label', 'Profit', 'Unit_Profit', 'Payout', 'Status']] = [h_g, a_g, h_c, a_c, lbl, prof, u_prof, payout, 'Settled']
                            save_db(st.session_state.df_db, db_file)
                            st.success(f"結算完成！結果：{lbl} | 淨利：${prof:.2f}")
                            st.rerun()
        st.divider()
        st.subheader("↩️ 撤銷已結算紀錄")
        settled_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
        if not settled_df.empty:
            recent_settled = settled_df.tail(5).iloc[::-1]
            options_dict = {f"[{r['Date']}] {r['Match']} | 結算: {r['Result_Label']} | 盈虧: {'+$' if pd.to_numeric(r['Profit'], errors='coerce')>=0 else '-$'}{abs(pd.to_numeric(r['Profit'], errors='coerce')):,.2f}": r['ID'] for _, r in recent_settled.iterrows()}
            selected_option = st.selectbox("選擇欲撤銷的歷史結算紀錄 (最近 5 筆)", list(options_dict.keys()))
            with st.form("rollback_form"):
                if st.form_submit_button("🔴 刪除並回滾所選紀錄", type="primary", use_container_width=True) and selected_option:
                    st.session_state.df_db = st.session_state.df_db[st.session_state.df_db['ID'] != options_dict[selected_option]].reset_index(drop=True)
                    recalculate_bankroll_from_scratch(st.session_state.df_cap, st.session_state.df_db)
                    save_db(st.session_state.df_db, db_file)
                    st.toast("已移除紀錄並完成本金全局回滾！", icon="🔄")
                    st.rerun()

    with t_ai:
        st.header("🤖 全局預測模型")
        df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
        st.write(f"當前歷史結算數據：**{len(df_settled)}** 筆")

if __name__ == "__main__":
    main()
