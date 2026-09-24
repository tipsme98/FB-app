import streamlit as st
import pandas as pd
import os
import json
import base64
import re
import io
from datetime import datetime

# ==========================================
# 0. 嘗試載入依賴套件 (AI與雲端資料庫)
# ==========================================
try:
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    HAS_AI_MODULES = True
except ImportError:
    HAS_AI_MODULES = False
    np = None

try:
    from sqlalchemy import create_engine
    import sqlalchemy
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# ==========================================
# 1. 初始化設定與資料庫 Schema
# ==========================================
st.set_page_config(page_title="足球博彩精算與資金管理系統", page_icon="⚽", layout="wide")

DB_COLUMNS = [
    'ID', 'Date', 'Status', 
    'Tournament_Name', 'Tournament_Category', 'Match', 'Home_Team', 'Away_Team', 
    'Home_Rating', 'Away_Rating', 'Home_Form', 'Away_Form',
    'Bet_Type', 'Selection', 'Initial_Line', 'Initial_Odds', 'Stake', 'Odds_History',
    'InPlay_Minute', 'Home_DA', 'Away_DA', 'Home_SoT', 'Away_SoT', 'Home_SoFF', 'Away_SoFF',
    'Home_Red', 'Away_Red', 'Home_Sub', 'Away_Sub', 'Home_Possession', 'Away_Possession',
    'Home_Goal', 'Away_Goal', 'Home_Corner', 'Away_Corner', 
    'Home_Goal_Conversion', 'Away_Goal_Conversion', 'Home_Firepower', 'Away_Firepower',
    'Result_Label', 'Profit', 'Unit_Profit', 'Payout'
]
LOG_COLUMNS = ['ID', 'Date', 'Match', 'Analysis_Content', 'Confidence_Level']
CAPITAL_COLUMNS = ['ID', 'Date', 'Type', 'Amount', 'Note']
CATEGORY_OPTIONS = ["國內聯賽 (Domestic League)", "國際聯賽 (International League)", "國際盃賽 (Cup)", "國內盃賽 (Domestic Cup)", "友誼賽 (Friendly)"]

def load_db(filename, columns, table_name):
    string_cols = [
        'ID', 'Date', 'Status', 'Tournament_Name', 'Tournament_Category', 
        'Match', 'Home_Team', 'Away_Team', 'Home_Rating', 'Away_Rating', 
        'Home_Form', 'Away_Form', 'Bet_Type', 'Selection', 'Odds_History', 'Result_Label',
        'Type', 'Note'
    ]
    
    # 策略 A: 嘗試從雲端 PostgreSQL 資料庫載入 (確保跨裝置同步)
    if HAS_SQLALCHEMY and "DB_URL" in st.secrets:
        try:
            engine = create_engine(st.secrets["DB_URL"])
            df = pd.read_sql_table(table_name, engine)
            
            for col in columns:
                if col not in df.columns: 
                    df[col] = pd.Series(dtype='object')
            for col in string_cols:
                if col in df.columns:
                    df[col] = df[col].astype('object')
                    
            return df[columns]
        except ValueError:
            # 資料表尚未建立 (第一次執行)，無縫降級到建立空 DataFrame
            pass
        except Exception as e:
            st.sidebar.error(f"⚠️ 雲端資料庫讀取異常，切換至本地模式: {e}")

    # 策略 B: 備用本地 CSV 載入 (原有機制)
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            if 'League' in df.columns and 'Tournament_Name' not in df.columns:
                df['Tournament_Name'] = df['League']
                df['Tournament_Category'] = CATEGORY_OPTIONS[0]
            
            for col in columns:
                if col not in df.columns: 
                    df[col] = pd.Series(dtype='object')
            for col in string_cols:
                if col in df.columns:
                    df[col] = df[col].astype('object')
                    
            return df[columns]
        except Exception:
            pass
            
    # 策略 C: 全新建立
    df = pd.DataFrame(columns=columns)
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype('object')
    return df

def save_db(df, filename, table_name):
    # 策略 A: 嘗試優先寫入雲端資料庫
    if HAS_SQLALCHEMY and "DB_URL" in st.secrets:
        try:
            engine = create_engine(st.secrets["DB_URL"])
            # 確保物件型態能正確寫入 SQL
            df_to_db = df.copy()
            for col in df_to_db.columns:
                if df_to_db[col].dtype == 'object':
                    df_to_db[col] = df_to_db[col].apply(lambda x: str(x) if pd.notna(x) else None)
            
            df_to_db.to_sql(table_name, engine, if_exists='replace', index=False)
        except Exception as e:
            st.sidebar.error(f"⚠️ 雲端資料庫寫入失敗: {e}")

    # 策略 B: 同步備份至本地 CSV
    df.to_csv(filename, index=False)

# ==========================================
# 2. 資金、風控與累計算式
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
        open_stake = 0.0
    else:
        settled_df = df_db[df_db['Status'] == 'Settled']
        open_df = df_db[df_db['Status'] == 'Open']
        total_profit = pd.to_numeric(settled_df['Profit'], errors='coerce').sum()
        open_stake = pd.to_numeric(open_df['Stake'], errors='coerce').sum()
        
    current_bankroll = net_deposit + total_profit - open_stake
    max_single_stake = (net_deposit + total_profit) * 0.10 
    
    return round(total_deposit, 2), round(total_withdraw, 2), round(net_deposit, 2), round(total_profit, 2), round(current_bankroll, 2), round(max_single_stake, 2)

def calculate_settlement(bet_type, selection, line, odds, stake, h_g, a_g, h_c=0, a_c=0):
    diff = 0.0
    clean_btype = bet_type.replace(" (即場)", "")
    
    if clean_btype == '讓球':
        if selection == 'Home': diff = h_g + line - a_g
        elif selection == 'Away': diff = a_g - line - h_g
    elif clean_btype == '入球大小':
        total_goals = h_g + a_g
        if selection == 'Over': diff = total_goals - line
        elif selection == 'Under': diff = line - total_goals
    elif clean_btype == '角球大小':
        total_corners = h_c + a_c
        if selection == 'Over': diff = total_corners - line
        elif selection == 'Under': diff = line - total_corners

    diff = round(diff, 2)
    
    if diff >= 0.5:
        res_label = "✅ 全贏"
        profit = stake * (odds - 1)
        payout = stake + profit
        unit_profit = round(odds - 1.0, 2)
    elif diff == 0.25:
        res_label = "🟢 贏半"
        profit = stake * (odds - 1) / 2
        payout = stake + profit
        unit_profit = round((odds - 1.0) / 2.0, 2)
    elif diff == 0.0:
        res_label = "⚪ 走盤退本"
        profit = 0.0
        payout = stake
        unit_profit = 0.0
    elif diff == -0.25:
        res_label = "🔴 輸半 (退回半本)"
        profit = -stake / 2
        payout = stake / 2
        unit_profit = -0.50
    else:
        res_label = "❌ 全輸"
        profit = -stake
        payout = 0.0
        unit_profit = -1.00

    return round(profit, 2), round(payout, 2), unit_profit, res_label, diff

def display_cumulative_metrics(df):
    total_profit = pd.to_numeric(df['Profit'], errors='coerce').sum()
    total_unit_profit = pd.to_numeric(df['Unit_Profit'], errors='coerce').sum()
    total_payout = pd.to_numeric(df['Payout'], errors='coerce').sum()
    
    st.markdown("### 📊 數據庫累計總額看板 (Cumulative Summary)")
    m1, m2, m3 = st.columns(3)
    m1.metric("累積淨盈虧 (Total Profit)", f"${total_profit:,.2f}", delta=f"{total_profit:,.2f}")
    m2.metric("累積單位平注盈虧 (Total Unit Profit)", f"{total_unit_profit:,.2f} U", delta=f"{total_unit_profit:,.2f} U")
    m3.metric("累積派彩總額 (Total Payout)", f"${total_payout:,.2f}")
    st.divider()

# ==========================================
# 3. 三維度 ML 與 EV 分析引擎
# ==========================================
def extract_form_points(form_str):
    try:
        w = int(re.search(r'(\d+)W', str(form_str)).group(1))
        d = int(re.search(r'(\d+)D', str(form_str)).group(1))
        return w * 3 + d * 1
    except:
        return 0

def prepare_ml_dataset(df, rating_map):
    X, y = [], []
    for _, r in df.iterrows():
        try:
            hr = rating_map.get(r.get('Home_Rating', 'C'), 3)
            ar = rating_map.get(r.get('Away_Rating', 'C'), 3)
            hf = extract_form_points(r.get('Home_Form', '0W0D0L'))
            af = extract_form_points(r.get('Away_Form', '0W0D0L'))
            line = float(r.get('Initial_Line', 0))
            odds = float(r.get('Initial_Odds', 1.90))
            X.append([hr, ar, hf, af, line, odds])
            y.append(1 if float(r.get('Unit_Profit', 0)) > 0 else 0)
        except:
            continue
    return np.array(X) if len(X) > 0 else None, np.array(y) if len(y) > 0 else None

def evaluate_dimension(df_subset, dim_name, candidates_base, rating_map, h_data):
    n_samples = len(df_subset)
    if n_samples < 15:
        return {'dim': dim_name, 'valid': False, 'msg': f"樣本數不足 ({n_samples} < 15場)", 'roi': 0, 'acc': 0, 'n': n_samples}
    
    total_stake = pd.to_numeric(df_subset['Stake'], errors='coerce').sum()
    total_profit = pd.to_numeric(df_subset['Profit'], errors='coerce').sum()
    roi = (total_profit / total_stake) if total_stake > 0 else 0
    wins = len(df_subset[pd.to_numeric(df_subset['Unit_Profit'], errors='coerce') > 0])
    acc = wins / n_samples if n_samples > 0 else 0

    candidates = [c.copy() for c in candidates_base]
    
    model_success = False
    if HAS_AI_MODULES:
        X, y = prepare_ml_dataset(df_subset, rating_map)
        if X is not None and len(np.unique(y)) > 1:
            try:
                clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
                clf.fit(X, y)
                for c in candidates:
                    x_input = np.array([[h_data['hr'], h_data['ar'], h_data['hf'], h_data['af'], c['line'], c['odds']]])
                    prob = clf.predict_proba(x_input)[0][1]
                    c['prob'] = (c['base_prob'] * 0.4) + (prob * 0.6)
                model_success = True
            except Exception:
                pass

    if not model_success:
        for c in candidates:
            shift = (acc - 0.5) * 0.2 + (roi * 0.1)
            c['prob'] = max(0.05, min(0.95, c['base_prob'] + shift))

    for c in candidates:
        c['ev'] = c['prob'] * (c['odds'] - 1) - (1 - c['prob'])

    candidates = sorted(candidates, key=lambda x: x['ev'], reverse=True)
    score = (roi * 0.7) + (acc * 0.3)
    
    return {
        'dim': dim_name, 'valid': True, 'msg': "運算成功", 'roi': roi, 'acc': acc, 
        'n': n_samples, 'candidates': candidates, 'best': candidates[0], 'score': score
    }

# ==========================================
# 4. 資金流水 HTML 構建
# ==========================================
def build_capital_flow_html(df_cap):
    if df_cap.empty:
        empty_html = '<div style="text-align:center; padding: 20px; color: gray;"><p>目前尚無資金流水紀錄。</p></div>'
        return empty_html, 0.0, "0", "#888888", " (無紀錄)"
    
    rows_html = []
    total_amount = 0.0
    
    for _, r in df_cap.iterrows():
        c_id = str(r.get('ID', ''))
        c_date = str(r.get('Date', ''))
        c_type_raw = str(r.get('Type', ''))
        
        try:
            amt_raw = float(r.get('Amount', 0.0))
        except (ValueError, TypeError):
            amt_raw = 0.0
            
        c_note = str(r.get('Note', '')) if pd.notna(r.get('Note')) else ''
        
        if 'Deposit' in c_type_raw or '存入' in c_type_raw:
            c_type_disp = "存入本金 (Deposit)"
            signed_amt = -amt_raw
            amt_formatted = f"-{int(amt_raw) if amt_raw.is_integer() else amt_raw:g}"
            color = "#ff4d4d" 
        else:
            c_type_disp = "提取本金 (Withdraw)"
            signed_amt = amt_raw
            amt_formatted = f"+{int(amt_raw) if amt_raw.is_integer() else amt_raw:g}"
            color = "#28a745" 
            
        total_amount += signed_amt
        
        rows_html.append(
            f'<tr>'
            f'<td style="padding: 10px; border: 1px solid #444;">{c_id}</td>'
            f'<td style="padding: 10px; border: 1px solid #444;">{c_date}</td>'
            f'<td style="padding: 10px; border: 1px solid #444;">{c_type_disp}</td>'
            f'<td style="padding: 10px; border: 1px solid #444; color: {color}; font-weight: bold; text-align: right; font-size: 1.05em;">{amt_formatted}</td>'
            f'<td style="padding: 10px; border: 1px solid #444;">{c_note}</td>'
            f'</tr>'
        )
    
    if total_amount < 0:
        tot_color = "#ff4d4d"
        abs_tot = abs(total_amount)
        tot_str = f"-{int(abs_tot) if abs_tot.is_integer() else abs_tot:g}"
        tot_label = f" (代表淨存入 ${abs_tot:,.2f})"
    elif total_amount > 0:
        tot_color = "#28a745"
        tot_str = f"+{int(total_amount) if total_amount.is_integer() else total_amount:g}"
        tot_label = f" (代表淨提取 ${total_amount:,.2f})"
    else:
        tot_color = "#888888"
        tot_str = "0"
        tot_label = " (收支平衡)"
        
    summary_row_html = (
        f'<tr style="background-color: rgba(128, 128, 128, 0.2); font-weight: bold; border-top: 2px solid #888;">'
        f'<td colspan="3" style="padding: 12px; border: 1px solid #444; text-align: right; font-size: 1.05em;">金額總和 (Total Amount Sum):</td>'
        f'<td style="padding: 12px; border: 1px solid #444; color: {tot_color}; font-weight: bold; font-size: 1.25em; text-align: right;">{tot_str}</td>'
        f'<td style="padding: 12px; border: 1px solid #444; color: {tot_color}; font-weight: bold; font-size: 0.95em;">{tot_label}</td>'
        f'</tr>'
    )
    
    rows_str = "".join(rows_html)
    table_html = (
        f'<div style="width: 100%; overflow-x: auto; margin-top: 10px;">'
        f'<table style="width: 100%; border-collapse: collapse; font-family: system-ui, -apple-system, sans-serif; font-size: 14px;">'
        f'<thead>'
        f'<tr style="background-color: rgba(128, 128, 128, 0.3); text-align: left;">'
        f'<th style="padding: 10px; border: 1px solid #444;">流水號 (ID)</th>'
        f'<th style="padding: 10px; border: 1px solid #444;">日期 (Date)</th>'
        f'<th style="padding: 10px; border: 1px solid #444;">類型 (Type)</th>'
        f'<th style="padding: 10px; border: 1px solid #444; text-align: right;">金額 (Amount)</th>'
        f'<th style="padding: 10px; border: 1px solid #444;">備註 (Note)</th>'
        f'</tr>'
        f'</thead>'
        f'<tbody>'
        f'{rows_str}'
        f'{summary_row_html}'
        f'</tbody>'
        f'</table>'
        f'</div>'
    )
    return table_html, total_amount, tot_str, tot_color, tot_label

@st.dialog("📊 數據庫即時線上預覽", width="large")
def preview_db_dialog(df_db, df_cap):
    tab_bets, tab_capital = st.tabs(["⚽ 投注紀錄數據庫", "💰 系統資金流水"])
    
    with tab_bets:
        st.write("您可以在下方表格中自由滑動、點擊欄位排序，或使用關鍵字搜尋特定賽事。")
        search_query = st.text_input("🔍 關鍵字搜尋 (例如: 球隊名稱、盤口)", "", key="search_bets")
        
        if search_query:
            mask = df_db.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1)
            show_df = df_db[mask].copy()
        else:
            show_df = df_db.copy()

        total_profit = pd.to_numeric(show_df['Profit'], errors='coerce').sum()
        total_unit = pd.to_numeric(show_df['Unit_Profit'], errors='coerce').sum()
        total_payout = pd.to_numeric(show_df['Payout'], errors='coerce').sum()

        summary_data = {col: None for col in show_df.columns}
        if 'ID' in summary_data: summary_data['ID'] = "TOTAL (總計)"
        if 'Profit' in summary_data: summary_data['Profit'] = round(total_profit, 2)
        if 'Unit_Profit' in summary_data: summary_data['Unit_Profit'] = round(total_unit, 2)
        if 'Payout' in summary_data: summary_data['Payout'] = round(total_payout, 2)

        summary_row = pd.DataFrame([summary_data])
        show_df_with_summary = pd.concat([show_df, summary_row], ignore_index=True)

        st.dataframe(show_df_with_summary, use_container_width=True)
        
        excel_data = io.BytesIO()
        try:
            show_df_with_summary.to_excel(excel_data, index=False)
            st.download_button("📥 點擊下載投注紀錄 Excel 報表", excel_data.getvalue(), "football_betting_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="btn_down_bets_xlsx")
        except:
            st.download_button("📥 點擊下載投注紀錄報表 (CSV)", show_df_with_summary.to_csv(index=False).encode('utf-8'), "football_betting_report.csv", "text/csv", key="btn_down_bets_csv")

    with tab_capital:
        st.subheader("💰 系統資金流水帳目 (Capital Flow Ledger)")
        table_html, total_amount, tot_str, tot_color, tot_label = build_capital_flow_html(df_cap)
        st.markdown(table_html, unsafe_allow_html=True)
        
        df_cap_exp = df_cap.copy()
        cap_summary = {col: None for col in df_cap_exp.columns}
        if 'ID' in cap_summary: cap_summary['ID'] = "TOTAL (總計)"
        if 'Amount' in cap_summary: cap_summary['Amount'] = round(total_amount, 2)
        if 'Note' in cap_summary: cap_summary['Note'] = tot_label.strip(" ()")
        df_cap_exp = pd.concat([df_cap_exp, pd.DataFrame([cap_summary])], ignore_index=True)

        excel_cap = io.BytesIO()
        try:
            df_cap_exp.to_excel(excel_cap, index=False)
            st.download_button("📥 點擊下載資金流水 Excel 報表", excel_cap.getvalue(), "capital_flow_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="btn_down_cap_xlsx")
        except:
            st.download_button("📥 點擊下載資金流水報表 (CSV)", df_cap_exp.to_csv(index=False).encode('utf-8'), "capital_flow_report.csv", "text/csv", key="btn_down_cap_csv")

def render_odds_section(odds_history_state, prefix="pre"):
    for i, row in enumerate(odds_history_state):
        r_id = row['id']
        up_key, type_key, unlock_key = f"{prefix}_up_{r_id}", f"{prefix}_t_{r_id}", f"{prefix}_u_{r_id}"
        margin_key, low_key, line_key = f"{prefix}_m_{r_id}", f"{prefix}_low_{r_id}", f"{prefix}_l_{r_id}"

        if type_key in st.session_state:
            old_type, new_type = row['type'], st.session_state[type_key]
            if old_type != new_type:
                row['type'] = new_type
                if new_type == "入球大小": row['line'] = 2.5
                elif new_type == "角球大小": row['line'] = 9.5
                elif new_type == "讓球": row['line'] = 0.0
                st.session_state[line_key] = float(row['line'])
                
        if line_key in st.session_state: row['line'] = st.session_state[line_key]
        if up_key in st.session_state: row['upper'] = st.session_state[up_key]
        if unlock_key in st.session_state: row['unlock'] = st.session_state[unlock_key]
        if margin_key in st.session_state: row['margin'] = st.session_state[margin_key]
        if low_key in st.session_state and row['unlock']: row['lower'] = st.session_state[low_key]

        if not row['unlock']:
            m_val, u_val = float(row.get('margin', 1.085)), float(row.get('upper', 1.90))
            try:
                calc_lower = round(1 / (m_val - (1 / u_val)), 2)
                row['lower'] = calc_lower if calc_lower > 1 else 1.01
            except:
                row['lower'] = 1.90
            st.session_state[low_key] = float(row['lower'])
        else:
            try:
                row['margin'] = (1 / float(row.get('upper', 1.90))) + (1 / float(row.get('lower', 1.90)))
            except:
                row['margin'] = 1.085
            st.session_state[margin_key] = float(row['margin'])

        c1, c2, c3, c4, c5, c6 = st.columns([2, 1.5, 1.5, 2, 1.5, 1])
        type_idx = ["讓球", "入球大小", "角球大小"].index(row['type']) if row['type'] in ["讓球", "入球大小", "角球大小"] else 0
        row['type'] = c1.selectbox(f"盤口類型 {i+1}", ["讓球", "入球大小", "角球大小"], key=type_key, index=type_idx)
        
        line_step = 1.0 if row['type'] == "角球大小" else 0.25
        row['line'] = c2.number_input("盤口線", step=line_step, value=float(row['line']), key=line_key)
        
        up_lbl, low_lbl = ("主隊", "客隊") if row['type'] == "讓球" else ("大盤(Over)", "小盤(Under)")
        row['upper'] = c3.number_input(f"{up_lbl} 賠率", min_value=1.01, value=float(row['upper']), step=0.01, key=up_key)
        row['unlock'] = c4.checkbox("🔓 解鎖", value=row.get('unlock', False), key=unlock_key)
        
        if not row['unlock']:
            row['margin'] = c4.number_input("抽水(Margin)", min_value=1.00, value=float(row.get('margin', 1.085)), step=0.005, format="%.3f", key=margin_key)
            margin_pct = (row['margin'] - 1) * 100
            c4.caption(f"抽水率: **{margin_pct:.2f}%**")
            row['lower'] = c5.number_input(f"{low_lbl} 賠率", value=float(row['lower']), disabled=True, key=low_key)
        else:
            row['lower'] = c5.number_input(f"{low_lbl} 賠率", min_value=1.01, step=0.01, value=float(row['lower']), key=low_key)
            margin_pct = (row['margin'] - 1) * 100
            c4.caption(f"隱含抽水: **{row['margin']:.3f} ({margin_pct:.2f}%)**")

        if len(odds_history_state) > 1:
            if c6.button("❌", key=f"{prefix}_d_{r_id}"):
                odds_history_state.pop(i)
                st.rerun()

    ac1, ac2, ac3 = st.columns(3)
    max_id = max([r['id'] for r in odds_history_state]) if odds_history_state else 0
    if ac1.button("➕ 讓球盤 (0.0)", key=f"{prefix}_add_hand", use_container_width=True):
        odds_history_state.append({"id": max_id+1, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}); st.rerun()
    if ac2.button("➕ 入球大小 (2.5)", key=f"{prefix}_add_goal", use_container_width=True):
        odds_history_state.append({"id": max_id+1, "type": "入球大小", "line": 2.5, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}); st.rerun()
    if ac3.button("➕ 角球大小 (9.5)", key=f"{prefix}_add_corn", use_container_width=True):
        odds_history_state.append({"id": max_id+1, "type": "角球大小", "line": 9.5, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}); st.rerun()

# ==========================================
# 5. 主程式 UI 
# ==========================================
def main():
    st.title("⚽ 足球博彩精算與資金管理系統")
    
    st.sidebar.header("⚙️ 系統設定與資金管理")
    mode = st.sidebar.radio("運作模式選擇", ["🧪 測試模式", "🟢 真實模式"])
    
    # 宣告檔案與資料表名稱
    db_file = "football_betting_db_test.csv" if mode == "🧪 測試模式" else "football_betting_db.csv"
    capital_file = "football_capital_db_test.csv" if mode == "🧪 測試模式" else "football_capital_db.csv"
    db_table = "football_bets_test" if mode == "🧪 測試模式" else "football_bets"
    cap_table = "football_cap_test" if mode == "🧪 測試模式" else "football_cap"
    
    # 雲端同步狀態顯示
    st.sidebar.divider()
    st.sidebar.subheader("☁️ 雲端同步狀態")
    if HAS_SQLALCHEMY and "DB_URL" in st.secrets:
        st.sidebar.success("🟢 已連線至雲端資料庫！跨裝置資料將自動同步，不再遺失。")
    else:
        st.sidebar.error("🔴 尚未連線雲端資料庫 (僅本地暫存)")
        st.sidebar.caption("提示: 您的環境會在休眠時清空資料。請於 App 佈署後台的 Secrets 加上 `DB_URL` 啟用永久雲端存檔。")
    
    # 載入資料庫
    st.session_state.df_db = load_db(db_file, DB_COLUMNS, db_table)
    st.session_state.df_cap = load_db(capital_file, CAPITAL_COLUMNS, cap_table)

    tot_dep, tot_wit, net_dep, tot_pnl, curr_bankroll, max_stake = recalculate_bankroll_from_scratch(st.session_state.df_cap, st.session_state.df_db)
    
    st.sidebar.divider()
    st.sidebar.subheader("💰 系統本金與盈虧總覽")
    st.sidebar.metric("總存入本金", f"${tot_dep:,.2f}")
    st.sidebar.metric("總提取本金", f"${tot_wit:,.2f}")
    st.sidebar.metric("累積總盈虧 (PnL)", f"${tot_pnl:,.2f}", delta=f"${tot_pnl:,.2f}")
    st.sidebar.metric("當前總可用資金 (Bankroll)", f"${curr_bankroll:,.2f}")
    st.sidebar.caption(f"🛑 **單注上限 (動態資金 10%)**: `${max_stake:,.2f}`")

    with st.sidebar.expander("💸 資金存提管理"):
        cap_action = st.radio("動作", ["Deposit (存入本金)", "Withdraw (提取本金)"])
        cap_amount = st.number_input("金額 ($)", min_value=1.0, value=1000.0, step=100.0)
        cap_note = st.text_input("備註 (選填)")
        
        if st.button("確認寫入資金紀錄"):
            new_cap_record = {
                'ID': f"C{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'Type': 'Deposit' if 'Deposit' in cap_action else 'Withdraw',
                'Amount': float(cap_amount),
                'Note': cap_note
            }
            st.session_state.df_cap = pd.concat([st.session_state.df_cap, pd.DataFrame([new_cap_record])], ignore_index=True)
            save_db(st.session_state.df_cap, capital_file, cap_table)
            st.toast("✅ 資金紀錄雲端寫入成功！系統本金已自動重構。", icon="💰")
            st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("🔍 數據庫即時線上預覽", use_container_width=True):
        preview_db_dialog(st.session_state.df_db, st.session_state.df_cap)

    t_pre, t_inplay, t_settle, t_ai = st.tabs(["📝 賽前建檔與投注", "⏱️ 即場賽事與預測", "⚖️ 賽果結算與管理", "🤖 全局模型"])

    with t_pre:
        st.subheader("📝 賽事建檔與智能盤口走勢分析")
        if curr_bankroll <= 0: st.warning("⚠️ 目前系統可用資金不足！無法精確計算建議注碼。請先至側邊欄存入本金。")
        
        opts_tournaments = ["➕ 新增手動輸入..."] + sorted(list(set(st.session_state.df_db['Tournament_Name'].dropna().unique())))
        opts_teams = ["➕ 新增手動輸入..."] + sorted(list(set(st.session_state.df_db['Home_Team'].dropna().tolist() + st.session_state.df_db['Away_Team'].dropna().tolist())))
        
        st.markdown("##### 1. 賽事與球隊資料")
        col_t, col_c = st.columns(2)
        sel_tournament = col_t.selectbox("賽事名稱 (Tournament Name)", opts_tournaments)
        tournament_name = col_t.text_input("輸入新賽事名稱") if sel_tournament == "➕ 新增手動輸入..." else sel_tournament
        
        default_cat_idx = 0
        if sel_tournament != "➕ 新增手動輸入...":
            match_rows = st.session_state.df_db[st.session_state.df_db['Tournament_Name'] == tournament_name]
            if not match_rows.empty:
                last_cat = match_rows.iloc[-1]['Tournament_Category']
                if last_cat in CATEGORY_OPTIONS:
                    default_cat_idx = CATEGORY_OPTIONS.index(last_cat)
                    
        tournament_category = col_c.selectbox("賽事分類 (Tournament Category)", CATEGORY_OPTIONS, index=default_cat_idx)

        col_h, col_a = st.columns(2)
        sel_home = col_h.selectbox("主隊名稱", opts_teams, key="sh")
        home_team = col_h.text_input("輸入新主隊") if sel_home == "➕ 新增手動輸入..." else sel_home
        sel_away = col_a.selectbox("客隊名稱", opts_teams, key="sa")
        away_team = col_a.text_input("輸入新客隊") if sel_away == "➕ 新增手動輸入..." else sel_away

        c_hr, c_ar = st.columns(2)
        home_rating = c_hr.selectbox("主隊實力", ["S", "A", "B", "C", "D"])
        away_rating = c_ar.selectbox("客隊實力", ["S", "A", "B", "C", "D"])

        st.markdown("##### 2. 近 5 場狀態 (勝/和/敗)")
        f1, f2, f3, f4, f5, f6 = st.columns(6)
        home_form = f"{f1.number_input('主勝',0,10,3)}W{f2.number_input('主和',0,10,1)}D{f3.number_input('主敗',0,10,1)}L"
        away_form = f"{f4.number_input('客勝',0,10,2)}W{f5.number_input('客和',0,10,2)}D{f6.number_input('客敗',0,10,1)}L"

        st.markdown("##### 3. 賽前盤口與賠率走勢紀錄 (JSON結構儲存)")
        if 'odds_history' not in st.session_state:
            st.session_state.odds_history = [{"id": 0, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}]
        render_odds_section(st.session_state.odds_history, "pre")
        
        st.markdown("---")
        
        if st.button("🚀 賽前數據分析執行", type="primary", use_container_width=True):
            st.session_state.show_analysis = True
            df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
            
            rating_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
            hr_val = rating_map.get(home_rating, 3)
            ar_val = rating_map.get(away_rating, 3)
            
            candidates_base = []
            for r in st.session_state.odds_history:
                b_type, line_val = r['type'], float(r['line'])
                if b_type == "讓球":
                    p_up = max(0.1, min(0.9, 0.5 + ((hr_val - ar_val) * 0.03)))
                    label_h, label_a = ("主隊(上盤)", "客隊(下盤)") if line_val <= 0 else ("主隊(下盤)", "客隊(上盤)")
                    candidates_base.extend([
                        {'bet_type': b_type, 'selection': 'Home', 'base_prob': p_up, 'odds': float(r['upper']), 'line': line_val, 'label': label_h},
                        {'bet_type': b_type, 'selection': 'Away', 'base_prob': 1-p_up, 'odds': float(r['lower']), 'line': line_val, 'label': label_a}
                    ])
                else:
                    p_up = max(0.1, min(0.9, 0.5 + ((hr_val + ar_val - (6 if b_type=="入球大小" else 5)) * (0.02 if b_type=="入球大小" else 0.01))))
                    candidates_base.extend([
                        {'bet_type': b_type, 'selection': 'Over', 'base_prob': p_up, 'odds': float(r['upper']), 'line': line_val, 'label': "大盤(Over)"},
                        {'bet_type': b_type, 'selection': 'Under', 'base_prob': 1-p_up, 'odds': float(r['lower']), 'line': line_val, 'label': "小盤(Under)"}
                    ])

            h_data = {'hr': hr_val, 'ar': ar_val, 'hf': extract_form_points(home_form), 'af': extract_form_points(away_form)}
            
            df_micro = df_settled[df_settled['Tournament_Name'] == tournament_name]
            df_meso = df_settled[df_settled['Tournament_Category'] == tournament_category]
            df_macro = df_settled
            
            res_micro = evaluate_dimension(df_micro, "微觀 - 賽事名稱", candidates_base, rating_map, h_data)
            res_meso = evaluate_dimension(df_meso, "中觀 - 賽事分類", candidates_base, rating_map, h_data)
            res_macro = evaluate_dimension(df_macro, "宏觀 - 總數據", candidates_base, rating_map, h_data)
            
            valid_res = [r for r in [res_micro, res_meso, res_macro] if r['valid']]
            best_model = max(valid_res, key=lambda x: x['score']) if valid_res else res_macro
            if not valid_res: best_model['msg'] = "所有維度樣本數不足，降級為純基礎期望值運算。"

            best_bet = best_model['best'] if 'best' in best_model else candidates_base[0]
            suggested_stake = 0
            if 'ev' in best_bet and best_bet['ev'] > 0 and curr_bankroll > 0:
                b = best_bet['odds'] - 1
                kelly = max(0.0, min((best_bet['prob'] * b - (1 - best_bet['prob'])) / b, 0.10))
                suggested_stake = max(10.0, min(float(max_stake), float(round((curr_bankroll * (kelly * 0.5)) / 10) * 10)))

            st.session_state.analysis_result = {
                'micro': res_micro, 'meso': res_meso, 'macro': res_macro, 'best_model': best_model,
                'best_bet': best_bet, 'stake': suggested_stake, 't_name': tournament_name, 't_cat': tournament_category
            }
            
        if st.session_state.get('show_analysis', False):
            res = st.session_state.analysis_result
            st.success("✅ 三維度數據分析與 EV 運算完成！")
            
            c1, c2, c3 = st.columns(3)
            for col, r, title in zip([c1, c2, c3], [res['micro'], res['meso'], res['macro']], ["A. 微觀 (賽事名稱)", "B. 中觀 (賽事分類)", "C. 宏觀 (全局數據)"]):
                with col.container(border=True):
                    st.markdown(f"**{title}**")
                    if r['valid']:
                        st.write(f"樣本數: `{r['n']}` 場")
                        st.write(f"回測 ROI: `{r['roi']*100:.1f}%`")
                        st.write(f"歷史勝率: `{r['acc']*100:.1f}%`")
                    else:
                        st.warning(r['msg'])

            bm = res['best_model']
            bb = res['best_bet']
            dim_label_map = {"微觀 - 賽事名稱": "微觀", "中觀 - 賽事分類": "中觀", "宏觀 - 總數據": "宏觀"}
            dim_short = dim_label_map.get(bm['dim'], "宏觀")

            st.markdown(f"### 🧠 AI 預測模型推薦")
            st.info(f"系統分析顯示，針對『{res['t_name']}』，採用『{bm['dim']}』級別的模型進行運算，其歷史準確率與 EV 獲利期望值最高，故本次投注策略依據此模型生成。")
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("💡 首選推薦", f"{bb['bet_type']} - {bb['label']}")
            mc2.metric(f"🎯 預期勝率 ({dim_short}修正)", f"{bb.get('prob', bb.get('base_prob',0))*100:.1f}%")
            mc3.metric("📊 修正 EV", f"{bb.get('ev', 0):.3f}")
            st.markdown(f"**建議注碼**：`${res['stake']:,.2f}`")

            with st.form("bet_form"):
                bc1, bc2, bc3 = st.columns(3)
                final_btype = bc1.selectbox("最終投注項目", [c['bet_type'] for c in bm.get('candidates', [bb])], index=0)
                final_sel = bc2.selectbox("最終投注方向", ["Home", "Away", "Over", "Under"], index=["Home", "Away", "Over", "Under"].index(bb['selection']))
                final_stake = bc3.number_input("實際下注金額 ($)", min_value=10.0, step=10.0, value=float(res['stake'] if res['stake'] > 0 else 50.0))
                
                final_row = next((r for r in st.session_state.odds_history if r['type'] == final_btype), st.session_state.odds_history[-1])
                line, odds = float(final_row['line']), float(final_row['upper']) if final_sel in ["Home", "Over"] else float(final_row['lower'])
                
                if st.form_submit_button("✅ 確定投注並寫入雲端資料庫"):
                    new_id = f"B{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    new_record = {
                        'ID': new_id, 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Status': 'Open',
                        'Tournament_Name': res['t_name'], 'Tournament_Category': res['t_cat'], 
                        'Match': f"{home_team} vs {away_team}", 'Home_Team': home_team, 'Away_Team': away_team,
                        'Home_Rating': home_rating, 'Away_Rating': away_rating, 'Home_Form': home_form, 'Away_Form': away_form,
                        'Bet_Type': final_btype, 'Selection': final_sel, 'Initial_Line': line, 'Initial_Odds': odds, 'Stake': final_stake,
                        'Odds_History': json.dumps(st.session_state.odds_history, ensure_ascii=False)
                    }
                    st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([new_record])], ignore_index=True)
                    save_db(st.session_state.df_db, db_file, db_table)
                    st.session_state.odds_history = [{"id": 0, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}] 
                    st.session_state.show_analysis = False
                    st.toast("✅ 投注紀錄雲端寫入成功！", icon="📝")
                    st.rerun()

    with t_inplay:
        st.subheader("⏱️ 即場賽事實時更新與智慧火力分析")
        
        if 'inplay_odds_history' not in st.session_state:
            st.session_state.inplay_odds_history = [
                {"id": 0, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085},
                {"id": 1, "type": "入球大小", "line": 2.5, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085},
                {"id": 2, "type": "角球大小", "line": 9.5, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}
            ]

        pending_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
        if pending_df.empty:
            st.info("目前沒有待結算的進行中賽事 (Status='Open')。請先於「📝 賽前建檔」建立賽事。")
        else:
            unique_matches = pending_df.drop_duplicates(subset=['Match']).reset_index(drop=True)
            selected_match_name = st.selectbox("📌 請選擇正在進行中的賽事", unique_matches['Match'])
            row = pending_df[pending_df['Match'] == selected_match_name].iloc[0]

            def get_val(r, col, default=0):
                val = r.get(col)
                if pd.isna(val) or val == "": return default
                return int(float(val))

            st.markdown("##### 1. 實時數據輸入與自動效率計算")
            minute = st.number_input("比賽進行時間 (分鐘)", min_value=0, max_value=120, value=get_val(row, 'InPlay_Minute', 45))
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 🏠 主隊實時數據")
                h_g = st.number_input("主隊入球", min_value=0, value=get_val(row, 'Home_Goal'))
                h_c = st.number_input("主隊角球", min_value=0, value=get_val(row, 'Home_Corner'))
                h_da = st.number_input("主隊危險進攻 (DA)", min_value=0, value=get_val(row, 'Home_DA'))
                h_sot = st.number_input("主隊射正 (SoT)", min_value=0, value=get_val(row, 'Home_SoT'))
                h_soff = st.number_input("主隊射偏 (SoFF)", min_value=0, value=get_val(row, 'Home_SoFF'))
                h_red = st.number_input("主隊紅牌", min_value=0, value=get_val(row, 'Home_Red'))
                h_sub = st.number_input("主隊換人", min_value=0, value=get_val(row, 'Home_Sub'))
                
                h_poss = st.number_input("主隊控球率 (%)", min_value=0, max_value=100, value=get_val(row, 'Home_Possession', 50))
                a_poss = max(0, 100 - h_poss)
                
                h_conv = (h_g / h_sot * 100) if h_sot > 0 else 0.0
                h_fire = ((h_sot + h_soff) / h_da * 100) if h_da > 0 else 0.0
                
                st.markdown(f"> 🎯 **主隊得分率**: `{h_conv:.1f}%` ({h_g}進球 / {h_sot}射正)")
                st.markdown(f"> ⚡ **主隊進攻火力**: `{h_fire:.1f}%` ({h_sot+h_soff}射門 / {h_da}危險進攻)")
                
            with c2:
                st.markdown("##### ✈️ 客隊實時數據")
                a_g = st.number_input("客隊入球", min_value=0, value=get_val(row, 'Away_Goal'))
                a_c = st.number_input("客隊角球", min_value=0, value=get_val(row, 'Away_Corner'))
                a_da = st.number_input("客隊危險進攻 (DA)", min_value=0, value=get_val(row, 'Away_DA'))
                a_sot = st.number_input("客隊射正 (SoT)", min_value=0, value=get_val(row, 'Away_SoT'))
                a_soff = st.number_input("客隊射偏 (SoFF)", min_value=0, value=get_val(row, 'Away_SoFF'))
                a_red = st.number_input("客隊紅牌", min_value=0, value=get_val(row, 'Away_Red'))
                a_sub = st.number_input("客隊換人", min_value=0, value=get_val(row, 'Away_Sub'))
                
                st.number_input("客隊控球率 (%) [自動計算]", min_value=0, max_value=100, value=a_poss, disabled=True)
                
                a_conv = (a_g / a_sot * 100) if a_sot > 0 else 0.0
                a_fire = ((a_sot + a_soff) / a_da * 100) if a_da > 0 else 0.0
                
                st.markdown(f"> 🎯 **客隊得分率**: `{a_conv:.1f}%` ({a_g}進球 / {a_sot}射正)")
                st.markdown(f"> ⚡ **客隊進攻火力**: `{a_fire:.1f}%` ({a_sot+a_soff}射門 / {a_da}危險進攻)")

            if st.button("🔄 僅保存賽事實時數據與分析指標", use_container_width=True):
                match_mask = st.session_state.df_db['Match'] == selected_match_name
                st.session_state.df_db.loc[match_mask, 'InPlay_Minute'] = minute
                st.session_state.df_db.loc[match_mask, 'Home_Goal'] = h_g
                st.session_state.df_db.loc[match_mask, 'Away_Goal'] = a_g
                st.session_state.df_db.loc[match_mask, 'Home_Corner'] = h_c
                st.session_state.df_db.loc[match_mask, 'Away_Corner'] = a_c
                st.session_state.df_db.loc[match_mask, 'Home_DA'] = h_da
                st.session_state.df_db.loc[match_mask, 'Away_DA'] = a_da
                st.session_state.df_db.loc[match_mask, 'Home_SoT'] = h_sot
                st.session_state.df_db.loc[match_mask, 'Away_SoT'] = a_sot
                st.session_state.df_db.loc[match_mask, 'Home_SoFF'] = h_soff
                st.session_state.df_db.loc[match_mask, 'Away_SoFF'] = a_soff
                st.session_state.df_db.loc[match_mask, 'Home_Red'] = h_red
                st.session_state.df_db.loc[match_mask, 'Away_Red'] = a_red
                st.session_state.df_db.loc[match_mask, 'Home_Sub'] = h_sub
                st.session_state.df_db.loc[match_mask, 'Away_Sub'] = a_sub
                st.session_state.df_db.loc[match_mask, 'Home_Possession'] = h_poss
                st.session_state.df_db.loc[match_mask, 'Away_Possession'] = a_poss
                st.session_state.df_db.loc[match_mask, 'Home_Goal_Conversion'] = h_conv
                st.session_state.df_db.loc[match_mask, 'Away_Goal_Conversion'] = a_conv
                st.session_state.df_db.loc[match_mask, 'Home_Firepower'] = h_fire
                st.session_state.df_db.loc[match_mask, 'Away_Firepower'] = a_fire
                save_db(st.session_state.df_db, db_file, db_table)
                st.success("✅ 實時數據雲端儲存成功！")
                st.rerun()

            st.divider()
            st.markdown("##### 2. 即場盤口與賠率計算 (手動/自動抽水)")
            render_odds_section(st.session_state.inplay_odds_history, "inplay")

            if st.button("🚀 結合火力與剩餘時間計算 EV 智能推薦", type="primary", use_container_width=True):
                st.session_state.show_inplay_analysis = True
                
                safe_min = max(1, minute)
                rem_time = max(1, 90 - minute)
                
                h_atk = (h_da * (max(10.0, h_fire) / 100.0) * 0.7 + h_sot * (max(10.0, h_conv) / 100.0 + 1) * 2.0) / safe_min + (h_poss / 100 * 0.5)
                a_atk = (a_da * (max(10.0, a_fire) / 100.0) * 0.7 + a_sot * (max(10.0, a_conv) / 100.0 + 1) * 2.0) / safe_min + (a_poss / 100 * 0.5)
                
                candidates = []
                for r in st.session_state.inplay_odds_history:
                    b_type, line = r['type'], float(r['line'])
                    u_odds, l_odds = float(r['upper']), float(r['lower'])
                    
                    if b_type == "讓球":
                        base_p = 0.5 + (h_atk - a_atk) * 0.15 - (h_red - a_red) * 0.20
                        base_p = max(0.1, min(0.9, base_p))
                        lbl_h, lbl_a = ("主隊(上盤)", "客隊(下盤)") if line <= 0 else ("主隊(下盤)", "客隊(上盤)")
                        candidates.extend([
                            {'bet_type': b_type, 'selection': 'Home', 'prob': base_p, 'odds': u_odds, 'line': line, 'label': lbl_h},
                            {'bet_type': b_type, 'selection': 'Away', 'prob': 1-base_p, 'odds': l_odds, 'line': line, 'label': lbl_a}
                        ])
                    elif b_type == "入球大小":
                        intensity = (h_atk + a_atk) * (rem_time / 90.0 + 0.5)
                        base_p_over = 0.5 + (intensity - 1.2) * 0.25
                        base_p_over = max(0.1, min(0.9, base_p_over))
                        candidates.extend([
                            {'bet_type': b_type, 'selection': 'Over', 'prob': base_p_over, 'odds': u_odds, 'line': line, 'label': "大盤(Over)"},
                            {'bet_type': b_type, 'selection': 'Under', 'prob': 1-base_p_over, 'odds': l_odds, 'line': line, 'label': "小盤(Under)"}
                        ])
                    elif b_type == "角球大小":
                        corner_intensity = ((h_da + a_da) / safe_min) * ((h_fire + a_fire) / 200.0 + 0.5)
                        base_p_over = 0.5 + (corner_intensity - 0.9) * 0.3
                        base_p_over = max(0.1, min(0.9, base_p_over))
                        candidates.extend([
                            {'bet_type': b_type, 'selection': 'Over', 'prob': base_p_over, 'odds': u_odds, 'line': line, 'label': "大盤(Over)"},
                            {'bet_type': b_type, 'selection': 'Under', 'prob': 1-base_p_over, 'odds': l_odds, 'line': line, 'label': "小盤(Under)"}
                        ])

                for c in candidates:
                    c['ev'] = c['prob'] * (c['odds'] - 1) - (1 - c['prob'])
                
                candidates = sorted(candidates, key=lambda x: x['ev'], reverse=True)
                best_bet = candidates[0] if candidates else None
                
                suggested_stake = 0
                if best_bet and best_bet['ev'] > 0 and curr_bankroll > 0:
                    b = best_bet['odds'] - 1
                    kelly = max(0.0, min((best_bet['prob'] * b - (1 - best_bet['prob'])) / b, 0.10))
                    suggested_stake = max(10.0, min(float(max_stake), float(round((curr_bankroll * (kelly * 0.5)) / 10) * 10)))
                
                st.session_state.inplay_analysis_result = {
                    'candidates': candidates, 'best_bet': best_bet,
                    'stake': suggested_stake, 'match_row': row.to_dict()
                }

            if st.session_state.get('show_inplay_analysis', False):
                res = st.session_state.inplay_analysis_result
                best_bet = res['best_bet']
                match_info = res['match_row']
                
                st.success("✅ 結合火力與剩餘時間的即場 EV 精算完成！")
                if best_bet:
                    st.info("系統已成功納入進攻火力效率與得分率，為您挑選出最佳價值的即場盤口：")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("💡 首選推薦", f"{best_bet['bet_type']} - {best_bet['label']}")
                    mc2.metric(f"🎯 動態勝率預測", f"{best_bet['prob']*100:.1f}%")
                    mc3.metric("📊 即場 EV", f"{best_bet['ev']:.3f}")
                    st.markdown(f"**建議即場注碼**：`${res['stake']:,.2f}`")
                    
                    with st.form("inplay_bet_form"):
                        bc1, bc2, bc3 = st.columns(3)
                        final_btype = bc1.selectbox("最終投注項目", [c['bet_type'] for c in res['candidates']], index=0)
                        final_sel = bc2.selectbox("最終投注方向", ["Home", "Away", "Over", "Under"], index=["Home", "Away", "Over", "Under"].index(best_bet['selection']))
                        final_stake = bc3.number_input("實際下注金額 ($)", min_value=10.0, step=10.0, value=float(res['stake'] if res['stake'] > 0 else 50.0))
                        
                        if st.form_submit_button("✅ 確認即場投注並扣除本金"):
                            final_row = next((r for r in st.session_state.inplay_odds_history if r['type'] == final_btype), st.session_state.inplay_odds_history[-1])
                            line = float(final_row['line'])
                            odds = float(final_row['upper']) if final_sel in ["Home", "Over"] else float(final_row['lower'])
                            
                            new_id = f"B{datetime.now().strftime('%Y%m%d%H%M%S')}_INPLAY"
                            
                            new_record = match_info.copy()
                            new_record.update({
                                'ID': new_id, 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                'Bet_Type': f"{final_btype} (即場)", 'Selection': final_sel,
                                'Initial_Line': line, 'Initial_Odds': odds, 'Stake': final_stake,
                                'Status': 'Open', 'Odds_History': json.dumps(st.session_state.inplay_odds_history, ensure_ascii=False),
                                'InPlay_Minute': minute, 'Home_Goal': h_g, 'Away_Goal': a_g, 'Home_Corner': h_c, 'Away_Corner': a_c,
                                'Home_DA': h_da, 'Away_DA': a_da, 'Home_SoT': h_sot, 'Away_SoT': a_sot,
                                'Home_SoFF': h_soff, 'Away_SoFF': a_soff, 'Home_Red': h_red, 'Away_Red': a_red,
                                'Home_Sub': h_sub, 'Away_Sub': a_sub, 'Home_Possession': h_poss, 'Away_Possession': a_poss,
                                'Home_Goal_Conversion': h_conv, 'Away_Goal_Conversion': a_conv, 
                                'Home_Firepower': h_fire, 'Away_Firepower': a_fire,
                                'Result_Label': '', 'Profit': 0, 'Unit_Profit': 0, 'Payout': 0
                            })
                            
                            match_mask = st.session_state.df_db['Match'] == match_info['Match']
                            st.session_state.df_db.loc[match_mask, 'InPlay_Minute'] = minute
                            st.session_state.df_db.loc[match_mask, 'Home_Goal'] = h_g
                            st.session_state.df_db.loc[match_mask, 'Away_Goal'] = a_g
                            st.session_state.df_db.loc[match_mask, 'Home_Corner'] = h_c
                            st.session_state.df_db.loc[match_mask, 'Away_Corner'] = a_c
                            st.session_state.df_db.loc[match_mask, 'Home_DA'] = h_da
                            st.session_state.df_db.loc[match_mask, 'Away_DA'] = a_da
                            st.session_state.df_db.loc[match_mask, 'Home_SoT'] = h_sot
                            st.session_state.df_db.loc[match_mask, 'Away_SoT'] = a_sot
                            st.session_state.df_db.loc[match_mask, 'Home_SoFF'] = h_soff
                            st.session_state.df_db.loc[match_mask, 'Away_SoFF'] = a_soff
                            st.session_state.df_db.loc[match_mask, 'Home_Red'] = h_red
                            st.session_state.df_db.loc[match_mask, 'Away_Red'] = a_red
                            st.session_state.df_db.loc[match_mask, 'Home_Sub'] = h_sub
                            st.session_state.df_db.loc[match_mask, 'Away_Sub'] = a_sub
                            st.session_state.df_db.loc[match_mask, 'Home_Possession'] = h_poss
                            st.session_state.df_db.loc[match_mask, 'Away_Possession'] = a_poss
                            st.session_state.df_db.loc[match_mask, 'Home_Goal_Conversion'] = h_conv
                            st.session_state.df_db.loc[match_mask, 'Away_Goal_Conversion'] = a_conv
                            st.session_state.df_db.loc[match_mask, 'Home_Firepower'] = h_fire
                            st.session_state.df_db.loc[match_mask, 'Away_Firepower'] = a_fire

                            st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([new_record])], ignore_index=True)
                            save_db(st.session_state.df_db, db_file, db_table)
                            
                            st.session_state.show_inplay_analysis = False
                            st.toast("✅ 即場注單雲端寫入成功！", icon="📝")
                            st.rerun()
                else:
                    st.warning("⚠️ 目前該場賽事並無明顯具備 EV 價值的即場盤口推薦。")

    with t_settle:
        st.subheader("⚖️ 賽果結算與資料庫維護")
        display_cumulative_metrics(st.session_state.df_db)
        open_bets = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
        if not open_bets.empty:
            for idx, row in open_bets.iterrows():
                with st.expander(f"📌 {row['Match']} - {row['Bet_Type']} ({row['Selection']}) | 盤口: {row['Initial_Line']}"):
                    with st.form(f"settle_form_{row['ID']}"):
                        st.markdown("##### ⚽ 全場賽果輸入 (入球與角球)")
                        col1, col2 = st.columns(2)
                        h_g = col1.number_input("全場主隊入球數", min_value=0, value=int(row.get('Home_Goal', 0)) if pd.notna(row.get('Home_Goal')) else 0, key=f"hg_{row['ID']}")
                        a_g = col2.number_input("全場客隊入球數", min_value=0, value=int(row.get('Away_Goal', 0)) if pd.notna(row.get('Away_Goal')) else 0, key=f"ag_{row['ID']}")
                        
                        col3, col4 = st.columns(2)
                        h_c = col3.number_input("全場主隊角球數", min_value=0, value=int(row.get('Home_Corner', 0)) if pd.notna(row.get('Home_Corner')) else 0, key=f"hc_{row['ID']}")
                        a_c = col4.number_input("全場客隊角球數", min_value=0, value=int(row.get('Away_Corner', 0)) if pd.notna(row.get('Away_Corner')) else 0, key=f"ac_{row['ID']}")
                        
                        if st.form_submit_button("確認賽果並雲端結算"):
                            prof, payout, u_prof, lbl, diff = calculate_settlement(
                                row['Bet_Type'], row['Selection'], float(row['Initial_Line']), 
                                float(row['Initial_Odds']), float(row['Stake']), h_g, a_g, h_c, a_c
                            )
                            st.session_state.df_db['Result_Label'] = st.session_state.df_db['Result_Label'].astype('object')
                            st.session_state.df_db['Status'] = st.session_state.df_db['Status'].astype('object')
                            
                            st.session_state.df_db.loc[idx, 'Home_Goal'] = h_g
                            st.session_state.df_db.loc[idx, 'Away_Goal'] = a_g
                            st.session_state.df_db.loc[idx, 'Home_Corner'] = h_c
                            st.session_state.df_db.loc[idx, 'Away_Corner'] = a_c
                            st.session_state.df_db.loc[idx, 'Result_Label'] = str(lbl)
                            st.session_state.df_db.loc[idx, 'Profit'] = float(prof)
                            st.session_state.df_db.loc[idx, 'Unit_Profit'] = float(u_prof)
                            st.session_state.df_db.loc[idx, 'Payout'] = float(payout)
                            st.session_state.df_db.loc[idx, 'Status'] = 'Settled'
                            
                            save_db(st.session_state.df_db, db_file, db_table)
                            st.success(f"結算完成並同步雲端！結果：{lbl} | 單位盈虧：{u_prof:+.2f} U")
                            st.rerun()
                            
        st.markdown("---")
        st.subheader("⚠️ 撤銷與回滾中心 (Settlement Rollback)")
        st.info("若發生結算錯誤，您可在此刪除錯誤的結算紀錄。系統會自動重構出絕對精準的資金池。")
        
        settled_bets = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].tail(5)
        
        if not settled_bets.empty:
            rollback_options = []
            for _, r in settled_bets.iterrows():
                rollback_options.append(f"{r['ID']} | [{r['Date']}] {r['Match']} | 結算狀態: {r['Result_Label']} | 盈虧: ${r['Profit']}")
                
            sel_rollback = st.selectbox("請選擇要刪除並回滾的最近結算紀錄：", rollback_options)
            
            if st.button("🗑️ 刪除並回滾所選的結算紀錄", type="primary"):
                rollback_id = sel_rollback.split(" | ")[0]
                
                st.session_state.df_db = st.session_state.df_db[st.session_state.df_db['ID'] != rollback_id].reset_index(drop=True)
                save_db(st.session_state.df_db, db_file, db_table)
                
                st.success("✅ 已成功移除錯誤紀錄，雲端本金已重構完成回滾！")
                st.rerun()
        else:
            st.write("目前沒有可供撤銷的已結算紀錄。")

    with t_ai:
        st.header("🤖 全局預測模型監控")
        df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
        st.write(f"當前可供訓練的歷史結算數據：**{len(df_settled)}** 筆")
        st.write(f"雲端資料庫模組狀態 (SQLAlchemy): **{'🟢 已啟用' if HAS_SQLALCHEMY else '🔴 未載入'}**")
        st.write(f"機器學習模組狀態 (Scikit-Learn): **{'🟢 已啟用' if HAS_AI_MODULES else '🔴 未偵測到，使用啟發式算法'}**")

if __name__ == "__main__":
    main()
