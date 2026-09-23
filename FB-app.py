import streamlit as st
import pandas as pd
import os
import json
import base64
import re
import io
from datetime import datetime

# 嘗試載入機器學習套件
try:
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    HAS_AI_MODULES = True
except ImportError:
    HAS_AI_MODULES = False
    np = None

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

def load_db(filename, columns):
    string_cols = [
        'ID', 'Date', 'Status', 'Tournament_Name', 'Tournament_Category', 
        'Match', 'Home_Team', 'Away_Team', 'Home_Rating', 'Away_Rating', 
        'Home_Form', 'Away_Form', 'Bet_Type', 'Selection', 'Odds_History', 'Result_Label',
        'Type', 'Note'
    ]
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
            df = pd.DataFrame(columns=columns)
            for col in string_cols:
                if col in df.columns:
                    df[col] = df[col].astype('object')
            return df
    else:
        df = pd.DataFrame(columns=columns)
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype('object')
        return df

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
    
    # 5態派彩精算
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
# 4. 資金流水 HTML 構建與預覽彈窗
# ==========================================
def build_capital_flow_html(df_cap):
    if df_cap.empty:
        empty_html = """
        <div style="text-align:center; padding: 20px; color: gray;">
            <p>目前尚無資金流水紀錄。</p>
        </div>
        """
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
            color = "#ff4d4d"  # 紅色代表存入 (-)
        else:
            c_type_disp = "提取本金 (Withdraw)"
            signed_amt = amt_raw
            amt_formatted = f"+{int(amt_raw) if amt_raw.is_integer() else amt_raw:g}"
            color = "#28a745"  # 綠色代表提取 (+)
            
        total_amount += signed_amt
        
        rows_html.append(f"""
        <tr>
            <td style="padding: 10px; border: 1px solid #444;">{c_id}</td>
            <td style="padding: 10px; border: 1px solid #444;">{c_date}</td>
            <td style="padding: 10px; border: 1px solid #444;">{c_type_disp}</td>
            <td style="padding: 10px; border: 1px solid #444; color: {color}; font-weight: bold; text-align: right; font-size: 1.05em;">{amt_formatted}</td>
            <td style="padding: 10px; border: 1px solid #444;">{c_note}</td>
        </tr>
        """)
    
    if total_amount < 0:
        tot_color = "#ff4d4d"  # 紅色
        abs_tot = abs(total_amount)
        tot_str = f"-{int(abs_tot) if abs_tot.is_integer() else abs_tot:g}"
        tot_label = f" (代表淨存入 ${abs_tot:,.2f})"
    elif total_amount > 0:
        tot_color = "#28a745"  # 綠色
        tot_str = f"+{int(total_amount) if total_amount.is_integer() else total_amount:g}"
        tot_label = f" (代表淨提取 ${total_amount:,.2f})"
    else:
        tot_color = "#888888"
        tot_str = "0"
        tot_label = " (收支平衡)"
        
    summary_row_html = f"""
    <tr style="background-color: rgba(128, 128, 128, 0.2); font-weight: bold; border-top: 2px solid #888;">
        <td colspan="3" style="padding: 12px; border: 1px solid #444; text-align: right; font-size: 1.05em;">金額總和 (Total Amount Sum):</td>
        <td style="padding: 12px; border: 1px solid #444; color: {tot_color}; font-weight: bold; font-size: 1.25em; text-align: right;">{tot_str}</td>
        <td style="padding: 12px; border: 1px solid #444; color: {tot_color}; font-weight: bold; font-size: 0.95em;">{tot_label}</td>
    </tr>
    """
    
    table_html = f"""
    <div style="width: 100%; overflow-x: auto; margin-top: 10px;">
        <table style="width: 100%; border-collapse: collapse; font-family: system-ui, -apple-system, sans-serif; font-size: 14px;">
            <thead>
                <tr style="background-color: rgba(128, 128, 128, 0.3); text-align: left;">
                    <th style="padding: 10px; border: 1px solid #444;">流水號 (ID)</th>
                    <th style="padding: 10px; border: 1px solid #444;">日期 (Date)</th>
                    <th style="padding: 10px; border: 1px solid #444;">類型 (Type)</th>
                    <th style="padding: 10px; border: 1px solid #444; text-align: right;">金額 (Amount)</th>
                    <th style="padding: 10px; border: 1px solid #444;">備註 (Note)</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
                {summary_row_html}
            </tbody>
        </table>
    </div>
    """
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

        # ---------------------------------------------------------
        # 新增 Profit, Unit_Profit, Payout 總計列
        # ---------------------------------------------------------
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

        # 渲染帶有總和的 DataFrame
        st.dataframe(show_df_with_summary, use_container_width=True)
        
        # 匯出獨立 Excel 報表
        excel_data = io.BytesIO()
        try:
            show_df_with_summary.to_excel(excel_data, index=False)
            st.download_button(
                label="📥 點擊下載投注紀錄 Excel 報表",
                data=excel_data.getvalue(),
                file_name="football_betting_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="btn_down_bets_xlsx"
            )
        except Exception:
            # 防呆機制：若雲端環境缺少 openpyxl/xlsxwriter，自動降級下載 CSV 以防當機
            csv_data = show_df_with_summary.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 點擊下載投注紀錄報表 (自動降級為 CSV)",
                data=csv_data,
                file_name="football_betting_report.csv",
                mime="text/csv",
                key="btn_down_bets_csv"
            )

    with tab_capital:
        st.subheader("💰 系統資金流水帳目 (Capital Flow Ledger)")
        st.write("記錄所有資金存入與提取之金額。存入金額為紅色 (`-`)，提取金額為綠色 (`+`)。Amount 欄位最後附有總和數值與說明。")
        
        table_html, total_amount, tot_str, tot_color, tot_label = build_capital_flow_html(df_cap)
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("---")
        
        # 將資金流水同步匯出為 Excel 報表 (亦含總計列)
        df_cap_exp = df_cap.copy()
        cap_summary = {col: None for col in df_cap_exp.columns}
        if 'ID' in cap_summary: cap_summary['ID'] = "TOTAL (總計)"
        if 'Amount' in cap_summary: cap_summary['Amount'] = round(total_amount, 2)
        if 'Note' in cap_summary: cap_summary['Note'] = tot_label.strip(" ()")
        df_cap_exp = pd.concat([df_cap_exp, pd.DataFrame([cap_summary])], ignore_index=True)

        excel_cap = io.BytesIO()
        try:
            df_cap_exp.to_excel(excel_cap, index=False)
            st.download_button(
                label="📥 點擊下載資金流水 Excel 報表",
                data=excel_cap.getvalue(),
                file_name="capital_flow_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="btn_down_cap_xlsx"
            )
        except Exception:
            csv_cap = df_cap_exp.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 點擊下載資金流水報表 (自動降級為 CSV)",
                data=csv_cap,
                file_name="capital_flow_report.csv",
                mime="text/csv",
                key="btn_down_cap_csv"
            )

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
            row['margin'] = c4.number_input("抽水(Margin)", min_value=1.00, value=float(row.get('margin', 1.085)), step=0.005, format你尚未提供目前的 `FB-app.py` 程式碼。請將你現有的完整程式碼貼上來，我才能基於你的原始邏輯進行精準修改，並提供可以直接貼入 GitHub 覆蓋執行的全新完整版本。

針對你的兩項需求，我將會在拿到程式碼後進行以下核心調整：

1. **Excel 報表下載**：利用 `io.BytesIO()` 搭配 `pandas.ExcelWriter`（通常使用 `xlsxwriter` 或 `openpyxl` 引擎），將原本生成 HTML 的邏輯替換為匯出 `.xlsx` 格式的 `st.download_button`。
2. **底部總和列**：在渲染數據庫預覽表格（`st.dataframe` 或 `st.table`）前，複製一份顯示專用的 DataFrame，並使用 Pandas 的 `.loc` 或 `pd.concat` 在表格最下方新增一列「總和 (Total)」，專門計算並填入 `Profit`、`Unit_Profit` 及 `Payout` 的加總數值。

請提供你目前的 `FB-app.py`，我會立刻為你處理。
