import streamlit as st
import pandas as pd
import os
import json
import base64
import re
import io
import requests
from datetime import datetime, timedelta, timezone

# ==========================================
# 0. 嘗試載入依賴套件 (AI與雲端資料庫) & 全局時區設定
# ==========================================
HKT = timezone(timedelta(hours=8))

def get_hkt_now():
    return datetime.now(HKT)

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
st.set_page_config(page_title="Actuarial and fund management system by Dr. EdwinPro", page_icon="⚽", layout="wide")

DB_COLUMNS = [
    'ID', 'Date', 'Status', 
    'Tournament_Name', 'Tournament_Category', 'Match', 'Home_Team', 'Away_Team', 
    'Home_Rating', 'Away_Rating', 'Home_Form', 'Away_Form',
    'Bet_Type', 'Selection', 'Initial_Line', 'Initial_Odds', 
    'System_Stake', 'User_Stake', 'Odds_History',
    'InPlay_Minute', 'Home_DA', 'Away_DA', 'Home_SoT', 'Away_SoT', 'Home_SoFF', 'Away_SoFF',
    'Home_Red', 'Away_Red', 'Home_Sub', 'Away_Sub', 'Home_Possession', 'Away_Possession',
    'Home_Goal', 'Away_Goal', 'Home_Corner', 'Away_Corner', 
    'Home_Goal_Conversion', 'Away_Goal_Conversion', 'Home_Firepower', 'Away_Firepower',
    'Result_Label', 'System_Profit', 'User_Profit', 'Unit_Profit', 'System_Payout', 'User_Payout'
]
CAPITAL_COLUMNS = ['ID', 'Date', 'Type', 'Account', 'Amount', 'Note']
CATEGORY_OPTIONS = ["國內聯賽 (Domestic League)", "國際聯賽 (International League)", "國際盃賽 (Cup)", "國內盃賽 (Domestic Cup)", "友誼賽 (Friendly)"]

# GitHub API 讀取與寫入輔助函式
def load_db_github(repo, path, token):
    timestamp = int(get_hkt_now().timestamp())
    url = f"https://api.github.com/repos/{repo}/contents/{path}?t={timestamp}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3.raw",
        "Cache-Control": "no-cache"
    }
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return pd.read_csv(io.StringIO(res.text))
    return None

def save_db_github(df, repo, path, token):
    timestamp = int(get_hkt_now().timestamp())
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Cache-Control": "no-cache"
    }
    res_get = requests.get(f"{url}?t={timestamp}", headers=headers)
    sha = res_get.json().get("sha") if res_get.status_code == 200 else None
    
    csv_content = df.to_csv(index=False)
    content_b64 = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
    
    payload = {
        "message": f"Auto-update {path} via Streamlit App [{get_hkt_now().strftime('%Y-%m-%d %H:%M:%S')}]",
        "content": content_b64
    }
    if sha:
        payload["sha"] = sha
        
    res = requests.put(url, json=payload, headers=headers)
    return res.status_code in [200, 201]

def process_legacy_columns(df):
    if 'Stake' in df.columns and 'System_Stake' not in df.columns:
        df['System_Stake'] = df['Stake']
        df['User_Stake'] = df['Stake']
    if 'Profit' in df.columns and 'System_Profit' not in df.columns:
        df['System_Profit'] = df['Profit']
        df['User_Profit'] = df['Profit']
    if 'Payout' in df.columns and 'System_Payout' not in df.columns:
        df['System_Payout'] = df['Payout']
        df['User_Payout'] = df['Payout']
    return df

def enforce_columns(df, columns):
    if 'Account' in columns and 'Account' not in df.columns:
        df['Account'] = 'Both'
    for col in columns:
        if col not in df.columns: 
            df[col] = pd.Series(dtype='object')
    return df[columns]

def load_db(filename, columns, table_name, force_cloud=False):
    string_cols = [
        'ID', 'Date', 'Status', 'Tournament_Name', 'Tournament_Category', 
        'Match', 'Home_Team', 'Away_Team', 'Home_Rating', 'Away_Rating', 
        'Home_Form', 'Away_Form', 'Bet_Type', 'Selection', 'Odds_History', 'Result_Label',
        'Type', 'Account', 'Note'
    ]
    
    df = pd.DataFrame()
    if HAS_SQLALCHEMY and "DB_URL" in st.secrets and st.secrets["DB_URL"]:
        try:
            engine = create_engine(st.secrets["DB_URL"])
            df = pd.read_sql_table(table_name, engine)
            df = process_legacy_columns(df)
        except Exception:
            pass

    if df.empty and "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets:
        try:
            gh_df = load_db_github(st.secrets["GITHUB_REPO"], filename, st.secrets["GITHUB_TOKEN"])
            if gh_df is not None and not gh_df.empty:
                df = process_legacy_columns(gh_df)
        except Exception:
            pass

    if df.empty and os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            if 'League' in df.columns and 'Tournament_Name' not in df.columns:
                df['Tournament_Name'] = df['League']
                df['Tournament_Category'] = CATEGORY_OPTIONS[0]
            df = process_legacy_columns(df)
        except Exception:
            pass
            
    if df.empty:
        df = pd.DataFrame(columns=columns)
        
    df = enforce_columns(df, columns)
    for col in string_cols:
        if col in df.columns: df[col] = df[col].astype('object')
        
    return df

def save_db(df, filename, table_name):
    if HAS_SQLALCHEMY and "DB_URL" in st.secrets and st.secrets["DB_URL"]:
        try:
            engine = create_engine(st.secrets["DB_URL"])
            df_to_db = df.copy()
            for col in df_to_db.columns:
                if df_to_db[col].dtype == 'object':
                    df_to_db[col] = df_to_db[col].apply(lambda x: str(x) if pd.notna(x) else None)
            df_to_db.to_sql(table_name, engine, if_exists='replace', index=False)
        except Exception as e:
            st.error(f"SQL 資料庫儲存失敗: {e}")

    if "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets:
        try:
            save_db_github(df, st.secrets["GITHUB_REPO"], filename, st.secrets["GITHUB_TOKEN"])
        except Exception as e:
            st.error(f"GitHub 雲端同步失敗: {e}")

    try:
        df.to_csv(filename, index=False)
    except Exception:
        pass

# ==========================================
# 2. 資金、風控與累計算式 (分離系統與真實資金)
# ==========================================
def recalculate_bankroll_from_scratch(df_cap, df_db):
    if df_cap.empty:
        sys_dep = sys_wit = usr_dep = usr_wit = 0.0
    else:
        sys_cap = df_cap[df_cap['Account'].isin(['System', 'Both'])]
        usr_cap = df_cap[df_cap['Account'].isin(['User', 'Both'])]

        sys_dep = pd.to_numeric(sys_cap[sys_cap['Type'] == 'Deposit']['Amount'], errors='coerce').sum()
        sys_wit = pd.to_numeric(sys_cap[sys_cap['Type'] == 'Withdraw']['Amount'], errors='coerce').sum()
        
        usr_dep = pd.to_numeric(usr_cap[usr_cap['Type'] == 'Deposit']['Amount'], errors='coerce').sum()
        usr_wit = pd.to_numeric(usr_cap[usr_cap['Type'] == 'Withdraw']['Amount'], errors='coerce').sum()
    
    sys_net = max(0.0, sys_dep - sys_wit)
    usr_net = max(0.0, usr_dep - usr_wit)
    
    if df_db.empty:
        sys_profit = user_profit = 0.0
        sys_open = user_open = 0.0
    else:
        settled_df = df_db[df_db['Status'] == 'Settled']
        open_df = df_db[df_db['Status'] == 'Open']
        
        sys_profit = pd.to_numeric(settled_df['System_Profit'], errors='coerce').sum()
        user_profit = pd.to_numeric(settled_df['User_Profit'], errors='coerce').sum()
        sys_open = pd.to_numeric(open_df['System_Stake'], errors='coerce').sum()
        user_open = pd.to_numeric(open_df['User_Stake'], errors='coerce').sum()
        
    sys_bankroll = sys_net + sys_profit - sys_open
    user_bankroll = usr_net + user_profit - user_open
    
    sys_max_stake = (sys_net + sys_profit) * 0.10 
    user_max_stake = (usr_net + user_profit) * 0.10 
    
    return (
        round(sys_dep, 2), round(sys_wit, 2), round(sys_net, 2), round(sys_profit, 2), round(sys_bankroll, 2), round(sys_max_stake, 2),
        round(usr_dep, 2), round(usr_wit, 2), round(usr_net, 2), round(user_profit, 2), round(user_bankroll, 2), round(user_max_stake, 2)
    )

def calculate_settlement(bet_type, selection, line, odds, sys_stake, user_stake, h_g, a_g, h_c=0, a_c=0):
    diff = 0.0
    clean_btype = bet_type.replace(" (即場)", "").strip()
    
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
        sys_profit = sys_stake * (odds - 1)
        user_profit = user_stake * (odds - 1)
        sys_payout = sys_stake + sys_profit
        user_payout = user_stake + user_profit
        unit_profit = round(odds - 1.0, 2)
    elif diff == 0.25:
        res_label = "🟢 贏半"
        sys_profit = sys_stake * (odds - 1) / 2
        user_profit = user_stake * (odds - 1) / 2
        sys_payout = sys_stake + sys_profit
        user_payout = user_stake + user_profit
        unit_profit = round((odds - 1.0) / 2.0, 2)
    elif diff == 0.0:
        res_label = "⚪ 走盤退本"
        sys_profit = user_profit = 0.0
        sys_payout = sys_stake
        user_payout = user_stake
        unit_profit = 0.0
    elif diff == -0.25:
        res_label = "🔴 輸半 (退回半本)"
        sys_profit = -sys_stake / 2
        user_profit = -user_stake / 2
        sys_payout = sys_stake / 2
        user_payout = user_stake / 2
        unit_profit = -0.50
    else:
        res_label = "❌ 全輸"
        sys_profit = -sys_stake
        user_profit = -user_stake
        sys_payout = user_payout = 0.0
        unit_profit = -1.00

    return (
        round(sys_profit, 2), round(user_profit, 2), 
        round(sys_payout, 2), round(user_payout, 2), 
        unit_profit, res_label, diff
    )

def display_cumulative_metrics(df):
    sys_profit = pd.to_numeric(df['System_Profit'], errors='coerce').sum()
    user_profit = pd.to_numeric(df['User_Profit'], errors='coerce').sum()
    user_payout = pd.to_numeric(df['User_Payout'], errors='coerce').sum()
    
    st.markdown("### 📊 數據庫累計總額看板 (Cumulative Summary)")
    m1, m2, m3 = st.columns(3)
    m1.metric("系統累積淨盈虧 (System Profit)", f"${sys_profit:,.2f}", delta=f"${sys_profit:,.2f}")
    m2.metric("用家真實淨盈虧 (User Profit)", f"${user_profit:,.2f}", delta=f"${user_profit:,.2f}")
    m3.metric("用家派彩總額 (User Payout)", f"${user_payout:,.2f}")
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

def parse_form_str(f_str):
    try:
        w = int(re.search(r'(\d+)W', str(f_str)).group(1))
    except:
        w = 0
    try:
        d = int(re.search(r'(\d+)D', str(f_str)).group(1))
    except:
        d = 0
    try:
        l = int(re.search(r'(\d+)L', str(f_str)).group(1))
    except:
        l = 0
    return w, d, l

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
    valid = n_samples >= 15
    msg = "運算成功" if valid else f"樣本數不足 ({n_samples} < 15場)"
    
    if valid:
        total_sys_stake = pd.to_numeric(df_subset['System_Stake'], errors='coerce').sum()
        total_sys_profit = pd.to_numeric(df_subset['System_Profit'], errors='coerce').sum()
        roi = (total_sys_profit / total_sys_stake) if total_sys_stake > 0 else 0
        wins = len(df_subset[pd.to_numeric(df_subset['Unit_Profit'], errors='coerce') > 0])
        acc = wins / n_samples if n_samples > 0 else 0
    else:
        roi = 0
        acc = 0

    candidates = [c.copy() for c in candidates_base]
    
    model_success = False
    if valid and HAS_AI_MODULES:
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
            shift = ((acc - 0.5) * 0.2 + (roi * 0.1)) if valid else 0
            c['prob'] = max(0.05, min(0.95, c['base_prob'] + shift))

    for c in candidates:
        c['ev'] = c['prob'] * (c['odds'] - 1) - (1 - c['prob'])

    candidates = sorted(candidates, key=lambda x: x['ev'], reverse=True)
    score = (roi * 0.7) + (acc * 0.3) if valid else -1
    
    return {
        'dim': dim_name, 'valid': valid, 'msg': msg, 'roi': roi, 'acc': acc, 
        'n': n_samples, 'candidates': candidates, 'best': candidates[0], 'score': score
    }

# ==========================================
# 4. 注單載入與修改與即場賠率連動邏輯
# ==========================================
def load_bet_to_edit(bet_id):
    match_df = st.session_state.df_db[st.session_state.df_db['ID'] == bet_id]
    if match_df.empty:
        return
    row = match_df.iloc[0]
    st.session_state.editing_bet_id = str(bet_id)
    
    st.session_state.edit_t_name = str(row.get('Tournament_Name', '')) if pd.notna(row.get('Tournament_Name')) else ''
    st.session_state.edit_t_cat = str(row.get('Tournament_Category', CATEGORY_OPTIONS[0])) if pd.notna(row.get('Tournament_Category')) else CATEGORY_OPTIONS[0]
    st.session_state.edit_h_team = str(row.get('Home_Team', '')) if pd.notna(row.get('Home_Team')) else ''
    st.session_state.edit_a_team = str(row.get('Away_Team', '')) if pd.notna(row.get('Away_Team')) else ''
    st.session_state.edit_h_rating = str(row.get('Home_Rating', 'C')) if pd.notna(row.get('Home_Rating')) else 'C'
    st.session_state.edit_a_rating = str(row.get('Away_Rating', 'C')) if pd.notna(row.get('Away_Rating')) else 'C'
    
    hw, hd, hl = parse_form_str(row.get('Home_Form', '3W1D1L'))
    aw, ad, al = parse_form_str(row.get('Away_Form', '2W2D1L'))
    st.session_state.edit_hw, st.session_state.edit_hd, st.session_state.edit_hl = hw, hd, hl
    st.session_state.edit_aw, st.session_state.edit_ad, st.session_state.edit_al = aw, ad, al
    
    try:
        oh = json.loads(str(row.get('Odds_History', '[]')))
        if isinstance(oh, list) and len(oh) > 0:
            st.session_state.odds_history = oh
    except:
        pass
        
    st.session_state.edit_sys_stake = float(row.get('System_Stake', 0.0)) if pd.notna(row.get('System_Stake')) else 0.0
    st.session_state.edit_user_stake = float(row.get('User_Stake', 0.0)) if pd.notna(row.get('User_Stake')) else 0.0
    st.session_state.edit_bet_type = str(row.get('Bet_Type', '')).replace(" (即場)", "").strip() if pd.notna(row.get('Bet_Type')) else ''
    st.session_state.edit_selection = str(row.get('Selection', 'Home')) if pd.notna(row.get('Selection')) else 'Home'
    st.session_state.edit_line = float(row.get('Initial_Line', 0.0)) if pd.notna(row.get('Initial_Line')) else 0.0

def clear_edit_mode():
    st.session_state.editing_bet_id = None
    keys_to_del = [k for k in st.session_state.keys() if k.startswith('edit_')]
    for k in keys_to_del:
        del st.session_state[k]

def on_inplay_upper_change(pfx):
    try:
        up = st.session_state[f"{pfx}_up"]
        if up > 1.0:
            st.session_state[f"{pfx}_lw"] = round(1 / (1.085 - 1/up), 2)
            st.session_state[f"{pfx}_margin"] = 1.085
    except ZeroDivisionError:
        pass

def on_inplay_lower_change(pfx):
    try:
        up = st.session_state[f"{pfx}_up"]
        lw = st.session_state[f"{pfx}_lw"]
        if up > 0 and lw > 0:
            st.session_state[f"{pfx}_margin"] = round((1/up) + (1/lw), 3)
    except ZeroDivisionError:
        pass

# ==========================================
# 5. UI組件與系統邏輯
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
        except:
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
            f'<tr><td style="padding: 10px; border: 1px solid #444;">{c_id}</td>'
            f'<td style="padding: 10px; border: 1px solid #444;">{c_date}</td>'
            f'<td style="padding: 10px; border: 1px solid #444;">{c_type_disp}</td>'
            f'<td style="padding: 10px; border: 1px solid #444; color: {color}; font-weight: bold; text-align: right;">{amt_formatted}</td>'
            f'<td style="padding: 10px; border: 1px solid #444;">{c_note}</td></tr>'
        )
    
    if total_amount < 0:
        tot_color, abs_tot = "#ff4d4d", abs(total_amount)
        tot_str = f"-{int(abs_tot) if abs_tot.is_integer() else abs_tot:g}"
        tot_label = f" (代表淨存入 ${abs_tot:,.2f})"
    elif total_amount > 0:
        tot_color = "#28a745"
        tot_str = f"+{int(total_amount) if total_amount.is_integer() else total_amount:g}"
        tot_label = f" (代表淨提取 ${total_amount:,.2f})"
    else:
        tot_color, tot_str, tot_label = "#888888", "0", " (收支平衡)"
        
    summary_row_html = (
        f'<tr style="background-color: rgba(128, 128, 128, 0.2); font-weight: bold; border-top: 2px solid #888;">'
        f'<td colspan="3" style="padding: 12px; border: 1px solid #444; text-align: right;">金額總和:</td>'
        f'<td style="padding: 12px; border: 1px solid #444; color: {tot_color}; text-align: right;">{tot_str}</td>'
        f'<td style="padding: 12px; border: 1px solid #444; color: {tot_color};">{tot_label}</td></tr>'
    )
    
    table_html = (
        f'<div style="width: 100%; overflow-x: auto; margin-top: 10px;">'
        f'<table style="width: 100%; border-collapse: collapse; font-size: 14px;">'
        f'<thead><tr style="background-color: rgba(128, 128, 128, 0.3); text-align: left;">'
        f'<th style="padding: 10px; border: 1px solid #444;">流水號 (ID)</th>'
        f'<th style="padding: 10px; border: 1px solid #444;">日期 (Date)</th>'
        f'<th style="padding: 10px; border: 1px solid #444;">類型 (Type)</th>'
        f'<th style="padding: 10px; border: 1px solid #444; text-align: right;">金額 (Amount)</th>'
        f'<th style="padding: 10px; border: 1px solid #444;">備註 (Note)</th>'
        f'</tr></thead><tbody>{"".join(rows_html)}{summary_row_html}</tbody></table></div>'
    )
    return table_html, total_amount, tot_str, tot_color, tot_label

@st.dialog("📊 數據庫即時線上預覽與管理", width="large")
def preview_db_dialog(df_db, df_cap, db_file, capital_file, db_table, cap_table):
    tab_bets, tab_capital, tab_manage = st.tabs(["⚽ 投注紀錄預覽", "💰 資金流水預覽", "🗑️ 數據清理與還原"])
    
    with tab_bets:
        search_query = st.text_input("🔍 關鍵字搜尋", "", key="search_bets")
        show_df = df_db.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1) if search_query else df_db.copy()
        if type(show_df) == pd.Series: show_df = df_db[show_df].copy()

        summary_data = {col: None for col in show_df.columns}
        summary_data.update({'ID': "TOTAL (總計)", 'System_Profit': round(pd.to_numeric(show_df['System_Profit'], errors='coerce').sum(), 2),
                             'User_Profit': round(pd.to_numeric(show_df['User_Profit'], errors='coerce').sum(), 2),
                             'Unit_Profit': round(pd.to_numeric(show_df['Unit_Profit'], errors='coerce').sum(), 2),
                             'User_Payout': round(pd.to_numeric(show_df['User_Payout'], errors='coerce').sum(), 2)})

        show_df_with_summary = pd.concat([show_df, pd.DataFrame([summary_data])], ignore_index=True)
        st.dataframe(show_df_with_summary, use_container_width=True)
        
        try:
            excel_data = io.BytesIO()
            show_df_with_summary.to_excel(excel_data, index=False)
            st.download_button("📥 下載 Excel 報表", excel_data.getvalue(), "football_betting.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except:
            st.download_button("📥 下載 CSV 報表", show_df_with_summary.to_csv(index=False).encode('utf-8-sig'), "football_betting.csv", "text/csv")

    with tab_capital:
        cap_tab1, cap_tab2 = st.tabs(["🤖 系統資金流水", "👤 用家真實資金流水"])
        with cap_tab1:
            table_html_sys, _, _, _, _ = build_capital_flow_html(df_cap[df_cap['Account'].isin(['System', 'Both'])])
            st.markdown(table_html_sys, unsafe_allow_html=True)
        with cap_tab2:
            table_html_usr, _, _, _, _ = build_capital_flow_html(df_cap[df_cap['Account'].isin(['User', 'Both'])])
            st.markdown(table_html_usr, unsafe_allow_html=True)

    with tab_manage:
        del_mode = st.radio("選擇要清理的資料表", ["⚽ 投注紀錄", "💰 資金流水"])
        df_target, target_name = (df_db, 'bets') if del_mode == "⚽ 投注紀錄" else (df_cap, 'cap')
        opts = [f"{r['ID']} | {r['Date']}" for _, r in df_target.iterrows()]
        
        selected_to_delete = st.multiselect("選擇要刪除的紀錄:", opts)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑 刪除選中紀錄", use_container_width=True) and selected_to_delete:
                ids_to_delete = [x.split(" | ")[0] for x in selected_to_delete]
                df_to_delete = df_target[df_target['ID'].isin(ids_to_delete)]
                st.session_state.undo_stack.append({'id': f"U{get_hkt_now().strftime('%Y%m%d%H%M%S%f')}", 'target': target_name, 'data': df_to_delete.copy(), 'timestamp': get_hkt_now().strftime('%Y-%m-%d %H:%M:%S')})
                if target_name == 'bets':
                    st.session_state.df_db = df_db[~df_db['ID'].isin(ids_to_delete)]
                    save_db(st.session_state.df_db, db_file, db_table)
                else:
                    st.session_state.df_cap = df_cap[~df_cap['ID'].isin(ids_to_delete)]
                    save_db(st.session_state.df_cap, capital_file, cap_table)
                st.rerun()
        with col2:
            if st.button("⚠️ 確認清空全部", type="primary", use_container_width=True):
                if target_name == 'bets':
                    st.session_state.df_db = pd.DataFrame(columns=DB_COLUMNS)
                    save_db(st.session_state.df_db, db_file, db_table)
                else:
                    st.session_state.df_cap = pd.DataFrame(columns=CAPITAL_COLUMNS)
                    save_db(st.session_state.df_cap, capital_file, cap_table)
                st.rerun()

def parse_dt_for_comparison(dt_str, fallback_idx):
    if not dt_str or not str(dt_str).strip(): return (datetime.min, fallback_idx)
    s = str(dt_str).strip()
    for fmt in ["%Y-%m-%d %H:%M", "%m-%d %H:%M", "%d-%m %H:%M", "%Y/%m/%d %H:%M", "%m/%d %H:%M"]:
        try: return (datetime.strptime(s, fmt), fallback_idx)
        except: pass
    return (datetime.min, fallback_idx)

def parse_single_type_text(text, bet_type):
    if not text or not text.strip(): return []
    lines_raw = [l.strip() for l in text.split('\n') if l.strip()]
    default_line = 2.5 if bet_type == "入球大小" else (9.5 if bet_type == "角球大小" else 0.0)
    
    def parse_line_val(s):
        s = str(s).replace('球', '').replace('+', '').replace('[', '').replace(']', '').strip()
        if '/' in s:
            try: return (float(s.split('/')[0]) + float(s.split('/')[1])) / 2.0
            except: return 0.0
        try: return float(s)
        except: return 0.0

    def parse_odds_val(s):
        try:
            val = float(re.sub(r'[^\d\.]', '', str(s)))
            return val if val >= 1.01 else None
        except: return None

    blocks, current_block = [], {"datetime": "", "lines_content": []}
    dt_pattern = re.compile(r'(\d{1,4}[-/.]\d{1,2}(?:[-/.]\d{1,4})?\s*\d{1,2}:\d{2})')
    
    for l in lines_raw:
        m = dt_pattern.search(l)
        if m:
            if current_block["datetime"] or current_block["lines_content"]: blocks.append(current_block)
            rem = l.replace(m.group(1), '').strip()
            current_block = {"datetime": m.group(1), "lines_content": [rem] if rem else []}
        else:
            current_block["lines_content"].append(l)
    if current_block["datetime"] or current_block["lines_content"]: blocks.append(current_block)
        
    parsed_items, active_line = [], default_line
    for block in blocks:
        dt = block["datetime"]
        line_idx, found_line_val = -1, None
        
        for idx, l_item in enumerate(block["lines_content"]):
            bm = re.search(r'\[(.*?)\]', l_item)
            gm = re.search(r'([+-]?\d+(?:\.\d+)?(?:/[+-]?\d+(?:\.\d+)?)?)\s*球', l_item)
            if bm: found_line_val, line_idx = parse_line_val(bm.group(1)), idx; break
            elif gm: found_line_val, line_idx = parse_line_val(gm.group(1)), idx; break
        
        if found_line_val is not None: active_line = found_line_val
        
        upper_lines = block["lines_content"][:line_idx] if line_idx != -1 else block["lines_content"]
        lower_lines = block["lines_content"][line_idx:] if line_idx != -1 else []
            
        up_list = [parse_odds_val(n) for l in upper_lines for n in re.findall(r'\b\d+(?:\.\d+)?\b', l) if parse_odds_val(n) is not None and abs(parse_odds_val(n) - active_line) > 1e-4]
        lw_list = [parse_odds_val(n) for l in lower_lines for n in re.findall(r'\b\d+(?:\.\d+)?\b', re.sub(r'\[.*?\]', '', l)) if parse_odds_val(n) is not None and abs(parse_odds_val(n) - active_line) > 1e-4]
        
        up = up_list[-1] if up_list else 1.90
        lw = lw_list[-1] if lw_list else 1.90
            
        if up == 1.90 or lw == 1.90:
            all_nums = [parse_odds_val(n) for l in block["lines_content"] for n in re.findall(r'\b\d+(?:\.\d+)?\b', re.sub(r'\[.*?\]', '', l)) if parse_odds_val(n) is not None and abs(parse_odds_val(n) - active_line) > 1e-4]
            if len(all_nums) >= 2: up, lw = all_nums[0], all_nums[1]
            elif len(all_nums) == 1: up = lw = all_nums[0]
                
        margin = round((1/up) + (1/lw), 3) if (up > 0 and lw > 0) else 1.085
        parsed_items.append({"type": bet_type, "record_time": dt, "line": active_line, "upper": up, "lower": lw, "unlock": True, "margin": margin})
        
    return parsed_items

def render_odds_section(odds_history_state, prefix="pre"):
    col_hd, col_ou, col_cr = st.columns(3)
    raw_hd = col_hd.text_area("⚽ 讓球 貼上區", height=130, key=f"{prefix}_paste_hd")
    raw_ou = col_ou.text_area("⚽ 入球大小 貼上區", height=130, key=f"{prefix}_paste_ou")
    raw_cr = col_cr.text_area("⚽ 角球大小 貼上區", height=130, key=f"{prefix}_paste_cr")
        
    parsed_all = parse_single_type_text(raw_hd, "讓球") + parse_single_type_text(raw_ou, "入球大小") + parse_single_type_text(raw_cr, "角球大小")
    
    if raw_hd.strip() or raw_ou.strip() or raw_cr.strip():
        odds_history_state.clear()
        if parsed_all:
            for idx, item in enumerate(parsed_all): item["id"] = idx; odds_history_state.append(item)
        else: odds_history_state.append({"id": 0, "type": "讓球", "record_time": "", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": True, "margin": 1.085})
    elif not odds_history_state:
        odds_history_state.append({"id": 0, "type": "讓球", "record_time": "", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": True, "margin": 1.085})
        
    df = pd.DataFrame(odds_history_state)
    for col, default in [('type', '讓球'), ('record_time', ''), ('line', 0.0), ('upper', 1.90), ('lower', 1.90)]:
        if col not in df.columns: df[col] = default
            
    df['margin_disp'] = df.apply(lambda r: f"{((1/r.upper + 1/r.lower)-1)*100:.2f}% ({(1/r.upper + 1/r.lower):.3f})" if r.upper>0 and r.lower>0 else "8.50% (1.085)", axis=1)
    df_display = df[['record_time', 'type', 'line', 'upper', 'lower', 'margin_disp']].copy()
    df_display.columns = ['📅日期及時間', '盤口類型', '盤口線', '主隊/大盤賠率', '客隊/小盤賠率', '盤口抽水 (百分比)']
    
    edited_df = st.data_editor(df_display, num_rows="dynamic", use_container_width=True, key=f"{prefix}_odds_editor")
    
    new_history = []
    for i, row in edited_df.iterrows():
        try:
            up = float(row["主隊/大盤賠率"]) if pd.notna(row["主隊/大盤賠率"]) else 1.90
            lw = float(row["客隊/小盤賠率"]) if pd.notna(row["客隊/小盤賠率"]) else 1.90
            margin = (1/up) + (1/lw) if (up > 0 and lw > 0) else 1.085
        except: up, lw, margin = 1.90, 1.90, 1.085
        new_history.append({"id": i, "type": str(row["盤口類型"]), "record_time": str(row["📅日期及時間"]), "line": float(row["盤口線"]), "upper": up, "lower": lw, "unlock": True, "margin": round(margin, 3)})
    odds_history_state.clear()
    odds_history_state.extend(new_history)

def calc_suggested_stake(cand, sys_bankroll, sys_max_stake):
    suggested_stake = raw_stake = 0.0
    if cand.get('ev', 0) > 0 and sys_bankroll > 0:
        b = cand.get('odds', 1.90) - 1
        prob = cand.get('prob', cand.get('base_prob', 0.5))
        if b > 0:
            kelly = max(0.0, min((prob * b - (1 - prob)) / b, 0.10))
            raw_stake = (sys_bankroll * (kelly * 0.5))
            suggested_stake = min(float(sys_max_stake), float(round(raw_stake / 10) * 10))
            if "讓球" in cand.get('bet_type', ''):
                if 0 < suggested_stake < 200:
                    suggested_stake = 200.0 if cand.get('ev', 0) >= 0.03 and prob >= 0.50 else 0.0
            else:
                suggested_stake = max(10.0, suggested_stake)
    return suggested_stake, raw_stake

# ==========================================
# 6. 主程式 UI
# ==========================================
def main():
    st.title("⚽ Actuarial and fund management system by Dr. EdwinPro")
    
    if 'undo_stack' not in st.session_state: st.session_state.undo_stack = []
    if 'editing_bet_id' not in st.session_state: st.session_state.editing_bet_id = None
    if 'last_bet_id' not in st.session_state: st.session_state.last_bet_id = None
        
    st.sidebar.header("⚙ 系統設定與資金管理")
    db_file, capital_file, db_table, cap_table = "football_betting_db.csv", "football_capital_db.csv", "football_bets", "football_cap"

    if st.sidebar.button("🔄 即時從雲端同步最新數據", use_container_width=True):
        st.session_state.df_db = load_db(db_file, DB_COLUMNS, db_table, force_cloud=True)
        st.session_state.df_cap = load_db(capital_file, CAPITAL_COLUMNS, cap_table, force_cloud=True)
        st.toast("✅ 數據已與雲端同步！", icon="🔄")
        st.rerun()

    if 'df_db' not in st.session_state: st.session_state.df_db = load_db(db_file, DB_COLUMNS, db_table)
    if 'df_cap' not in st.session_state: st.session_state.df_cap = load_db(capital_file, CAPITAL_COLUMNS, cap_table)

    (sys_dep, sys_wit, sys_net, sys_pnl, sys_bankroll, sys_max_stake, 
     usr_dep, usr_wit, usr_net, usr_pnl, usr_bankroll, usr_max_stake) = recalculate_bankroll_from_scratch(st.session_state.df_cap, st.session_state.df_db)
    
    st.sidebar.metric("系統當前可用資金 (Bankroll)", f"${sys_bankroll:,.2f}")
    st.sidebar.metric("用家當前真實可用資金 (Bankroll)", f"${usr_bankroll:,.2f}")

    with st.sidebar.expander("💸 資金存提管理"):
        cap_account = st.radio("目標帳戶 (Account)", ["🤖 系統本金 (System)", "👤 用家本金 (User)", "🔄 兩者同步 (Both)"], index=2)
        cap_action = st.radio("動作", ["Deposit (存入本金)", "Withdraw (提取本金)"])
        cap_amount = st.number_input("金額 ($)", min_value=1.0, value=1000.0, step=100.0)
        if st.button("確認寫入資金紀錄"):
            acc_val = 'System' if "System" in cap_account else ('User' if "User" in cap_account else 'Both')
            new_cap_record = {'ID': f"C{get_hkt_now().strftime('%Y%m%d%H%M%S')}", 'Date': get_hkt_now().strftime('%Y-%m-%d %H:%M'), 'Type': 'Deposit' if 'Deposit' in cap_action else 'Withdraw', 'Account': acc_val, 'Amount': float(cap_amount), 'Note': ''}
            st.session_state.df_cap = pd.concat([st.session_state.df_cap, pd.DataFrame([new_cap_record])], ignore_index=True)
            save_db(st.session_state.df_cap, capital_file, cap_table)
            st.rerun()

    t_pre, t_inplay, t_settle = st.tabs(["📝 賽前建檔與投注", "⏱️ 即場賽事與預測", "⚖ 賽果結算與管理"])

    with t_pre:
        st.subheader("📝 賽事建檔與智能盤口走勢分析")
        
        is_editing = bool(st.session_state.editing_bet_id)
        opts_tournaments = ["➕ 新增手動輸入..."] + sorted(list(set(st.session_state.df_db['Tournament_Name'].dropna().unique())))
        opts_teams = ["➕ 新增手動輸入..."] + sorted(list(set(st.session_state.df_db['Home_Team'].dropna().tolist() + st.session_state.df_db['Away_Team'].dropna().tolist())))
        
        c_t, c_c = st.columns(2)
        default_tourn = st.session_state.get('edit_t_name', '') if is_editing else ''
        sel_tournament = c_t.selectbox("賽事名稱", opts_tournaments, index=opts_tournaments.index(default_tourn) if default_tourn in opts_tournaments else 0)
        tournament_name = c_t.text_input("輸入新賽事名稱") if sel_tournament == "➕ 新增手動輸入..." else sel_tournament
        tournament_category = c_c.selectbox("賽事分類", CATEGORY_OPTIONS)

        c_h, c_a = st.columns(2)
        default_h_team = st.session_state.get('edit_h_team', '') if is_editing else ''
        sel_home = c_h.selectbox("主隊名稱", opts_teams, index=opts_teams.index(default_h_team) if default_h_team in opts_teams else 0)
        home_team = c_h.text_input("輸入新主隊") if sel_home == "➕ 新增手動輸入..." else sel_home

        default_a_team = st.session_state.get('edit_a_team', '') if is_editing else ''
        sel_away = c_a.selectbox("客隊名稱", opts_teams, index=opts_teams.index(default_a_team) if default_a_team in opts_teams else 0)
        away_team = c_a.text_input("輸入新客隊") if sel_away == "➕ 新增手動輸入..." else sel_away

        c_hr, c_ar = st.columns(2)
        home_rating = c_hr.selectbox("主隊實力", ["S", "A", "B", "C", "D"], index=3)
        away_rating = c_ar.selectbox("客隊實力", ["S", "A", "B", "C", "D"], index=3)

        f1, f2, f3, f4, f5, f6 = st.columns(6)
        home_form = f"{f1.number_input('主勝',0,10,3)}W{f2.number_input('主和',0,10,1)}D{f3.number_input('主敗',0,10,1)}L"
        away_form = f"{f4.number_input('客勝',0,10,2)}W{f5.number_input('客和',0,10,2)}D{f6.number_input('客敗',0,10,1)}L"

        st.markdown("##### 3. 賽前盤口與賠率走勢紀錄 (JSON結構儲存)")
        if 'odds_history' not in st.session_state: st.session_state.odds_history = [{"id": 0, "type": "讓球", "record_time": "", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}]
        render_odds_section(st.session_state.odds_history, "pre")
        
        if st.button("🚀 賽前數據分析執行", type="primary", use_container_width=True):
            st.session_state.show_analysis = True
            rating_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
            hr_val = rating_map.get(home_rating, 3)
            ar_val = rating_map.get(away_rating, 3)
            
            candidates_base = []
            for idx, r in enumerate(st.session_state.odds_history):
                b_type, line_val = r['type'], float(r['line'])
                if b_type == "讓球":
                    p_up = max(0.1, min(0.9, 0.5 + ((hr_val - ar_val) * 0.03)))
                    candidates_base.extend([
                        {'bet_type': b_type, 'selection': 'Home', 'base_prob': p_up, 'odds': float(r['upper']), 'line': line_val, 'label': f"{line_val:g}主隊(上/下)"},
                        {'bet_type': b_type, 'selection': 'Away', 'base_prob': 1-p_up, 'odds': float(r['lower']), 'line': line_val, 'label': f"{line_val:g}客隊(上/下)"}
                    ])
                else:
                    p_up = max(0.1, min(0.9, 0.5 + ((hr_val + ar_val - (6 if b_type=="入球大小" else 5)) * (0.02 if b_type=="入球大小" else 0.01))))
                    candidates_base.extend([
                        {'bet_type': b_type, 'selection': 'Over', 'base_prob': p_up, 'odds': float(r['upper']), 'line': line_val, 'label': f"{line_val:g}大盤"},
                        {'bet_type': b_type, 'selection': 'Under', 'base_prob': 1-p_up, 'odds': float(r['lower']), 'line': line_val, 'label': f"{line_val:g}小盤"}
                    ])

            h_data = {'hr': hr_val, 'ar': ar_val, 'hf': extract_form_points(home_form), 'af': extract_form_points(away_form)}
            df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
            
            res_micro = evaluate_dimension(df_settled[df_settled['Tournament_Name'] == tournament_name], "微觀", candidates_base, rating_map, h_data)
            res_meso = evaluate_dimension(df_settled[df_settled['Tournament_Category'] == tournament_category], "中觀", candidates_base, rating_map, h_data)
            res_macro = evaluate_dimension(df_settled, "宏觀", candidates_base, rating_map, h_data)
            
            valid_res = [r for r in [res_micro, res_meso, res_macro] if r['valid']]
            best_model = max(valid_res, key=lambda x: x['score']) if valid_res else res_macro
            
            st.session_state.analysis_result = {
                'micro': res_micro, 'meso': res_meso, 'macro': res_macro, 'best_model': best_model,
                'best_bet': best_model.get('best', candidates_base[0]), 't_name': tournament_name, 't_cat': tournament_category
            }

        if st.session_state.get('show_analysis', False):
            res = st.session_state.analysis_result
            st.success("✅ 數據分析完成！推薦首選: " + res['best_bet']['label'])
            
            cand_list = res['best_model'].get('candidates', [])
            all_options = []
            option_map = {}
            for c in cand_list:
                opt_str = f"【{c['bet_type']}】{c['label']} @ {c['odds']}"
                if opt_str not in option_map:
                    all_options.append(opt_str)
                    option_map[opt_str] = c

            c1, c2 = st.columns(2)
            sys_selected = c1.multiselect("系統投注", all_options, default=[all_options[0]] if all_options else [])
            sys_stakes = {sel: c1.number_input(f"系統金額: {sel}", value=calc_suggested_stake(option_map[sel], sys_bankroll, sys_max_stake)[0], disabled=True) for sel in sys_selected}
            
            usr_selected = c2.multiselect("用家投注", all_options)
            usr_stakes = {sel: c2.number_input(f"用家金額: {sel}", min_value=0.0, step=10.0, value=100.0) for sel in usr_selected}

            if st.button("✅ 確定投注", type="primary"):
                all_keys = set(sys_selected + usr_selected)
                new_records = []
                for i, sel in enumerate(all_keys):
                    cand = option_map[sel]
                    new_records.append({
                        'ID': f"B{get_hkt_now().strftime('%Y%m%d%H%M%S')}{i}", 'Date': get_hkt_now().strftime('%Y-%m-%d %H:%M'), 'Status': 'Open',
                        'Tournament_Name': res['t_name'], 'Tournament_Category': res['t_cat'], 'Match': f"{home_team} vs {away_team}",
                        'Home_Team': home_team, 'Away_Team': away_team, 'Home_Rating': home_rating, 'Away_Rating': away_rating,
                        'Home_Form': home_form, 'Away_Form': away_form, 'Bet_Type': cand['bet_type'], 'Selection': cand['selection'],
                        'Initial_Line': cand['line'], 'Initial_Odds': cand['odds'], 'System_Stake': sys_stakes.get(sel, 0), 'User_Stake': usr_stakes.get(sel, 0),
                        'Odds_History': json.dumps(st.session_state.odds_history, ensure_ascii=False)
                    })
                st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame(new_records)], ignore_index=True)
                save_db(st.session_state.df_db, db_file, db_table)
                st.rerun()

    with t_inplay:
        st.subheader("⏱ 即場賽事實時更新與智慧火力分析")
        
        open_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
        match_groups = (open_df if not open_df.empty else st.session_state.df_db).groupby(['Tournament_Name', 'Match']).size().reset_index(name='Bet_Count')
        match_options = [f"{r['Tournament_Name']} | {r['Match']}" for _, r in match_groups.iterrows()]
        
        sel_match_str = st.selectbox("⚽ 請選擇要更新即場數據的比賽場次：", match_options, key="inplay_match_select")

        if sel_match_str:
            selected_tourn, selected_match = sel_match_str.split(" | ")
            matching_bets = st.session_state.df_db[(st.session_state.df_db['Tournament_Name'] == selected_tourn) & (st.session_state.df_db['Match'] == selected_match)]
            last_row = matching_bets.iloc[-1]
            
            st.markdown("##### 1. 實時賽況與進攻數據輸入")
            col_time, col_ip1, col_ip2, col_ip3 = st.columns([2, 3, 3, 3])
            ip_minute = col_time.number_input("比賽時間", 0, 120, int(float(last_row.get('InPlay_Minute', 45))), key="ip_minute")
            h_g = col_ip1.number_input("主隊入球", 0, 50, int(float(last_row.get('Home_Goal', 0))), key="ip_hg")
            a_g = col_ip1.number_input("客隊入球", 0, 50, int(float(last_row.get('Away_Goal', 0))), key="ip_ag")
            h_c = col_ip2.number_input("主隊角球", 0, 50, int(float(last_row.get('Home_Corner', 0))), key="ip_hc")
            a_c = col_ip2.number_input("客隊角球", 0, 50, int(float(last_row.get('Away_Corner', 0))), key="ip_ac")
            
            c1, c2, c3, c4 = st.columns(4)
            h_da = c1.number_input("主隊危險進攻", 0, 200, int(float(last_row.get('Home_DA', 0))), key="ip_hda")
            a_da = c1.number_input("客隊危險進攻", 0, 200, int(float(last_row.get('Away_DA', 0))), key="ip_ada")
            h_sot = c2.number_input("主隊射正", 0, 50, int(float(last_row.get('Home_SoT', 0))), key="ip_hsot")
            a_sot = c2.number_input("客隊射正", 0, 50, int(float(last_row.get('Away_SoT', 0))), key="ip_asot")
            h_soff = c3.number_input("主隊射偏", 0, 50, int(float(last_row.get('Home_SoFF', 0))), key="ip_hsoff")
            a_soff = c3.number_input("客隊射偏", 0, 50, int(float(last_row.get('Away_SoFF', 0))), key="ip_asoff")
            h_poss = c4.number_input("主隊控球率 (%)", 0, 100, int(float(last_row.get('Home_Possession', 50))), key="ip_hposs")
            a_poss = 100 - h_poss

            # 計算即時指標
            h_conversion = (h_g / (h_sot + h_soff) * 100.0) if (h_sot + h_soff) > 0 else 0.0
            a_conversion = (a_g / (a_sot + a_soff) * 100.0) if (a_sot + a_soff) > 0 else 0.0
            h_firepower = ((h_sot + h_soff) / h_da * 100.0) if h_da > 0 else 0.0
            a_firepower = ((a_sot + a_soff) / a_da * 100.0) if a_da > 0 else 0.0
            h_efficiency = (h_firepower / h_poss * 100.0) if h_poss > 0 else 0.0
            a_efficiency = (a_firepower / a_poss * 100.0) if a_poss > 0 else 0.0
            h_corner_efficiency = (h_c / h_da * 100.0) if h_da > 0 else 0.0
            a_corner_efficiency = (a_c / a_da * 100.0) if a_da > 0 else 0.0

            st.markdown("##### 2. 自動計算實時進攻效率與指標看板")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown("**🏠 主隊 (Home Team)**")
                st.metric("射球命中率", f"{h_conversion:.2f}%")
                st.metric("進攻火力", f"{h_firepower:.2f}%")
                st.metric("實際進攻效率", f"{h_efficiency:.2f}%")
                st.metric("進攻產生角球效率", f"{h_corner_efficiency:.2f}%", help="角球數量 / 危險進攻數量 * 100%")
            with m_col2:
                st.markdown("**✈️ 客隊 (Away Team)**")
                st.metric("射球命中率", f"{a_conversion:.2f}%")
                st.metric("進攻火力", f"{a_firepower:.2f}%")
                st.metric("實際進攻效率", f"{a_efficiency:.2f}%")
                st.metric("進攻產生角球效率", f"{a_corner_efficiency:.2f}%", help="角球數量 / 危險進攻數量 * 100%")

            if st.button("💾 儲存實時賽況至此比賽", type="primary"):
                mask = (st.session_state.df_db['Tournament_Name'] == selected_tourn) & (st.session_state.df_db['Match'] == selected_match)
                st.session_state.df_db.loc[mask, 'InPlay_Minute'] = ip_minute
                st.session_state.df_db.loc[mask, 'Home_Goal'], st.session_state.df_db.loc[mask, 'Away_Goal'] = h_g, a_g
                st.session_state.df_db.loc[mask, 'Home_Corner'], st.session_state.df_db.loc[mask, 'Away_Corner'] = h_c, a_c
                st.session_state.df_db.loc[mask, 'Home_DA'], st.session_state.df_db.loc[mask, 'Away_DA'] = h_da, a_da
                save_db(st.session_state.df_db, db_file, db_table)
                st.session_state.show_inplay_analysis = False
                st.rerun()

            st.divider()
            
            # --- 3. 即場投注 (In-Play Betting) 新功能模組 ---
            st.markdown("##### 3. 即場投注 (In-Play Betting)")
            st.caption("💡 更改大/上賠率會連動算出小/下賠率。手動更改小/下賠率則會直接覆蓋，並重新計算抽水。")

            for pfx in ["inplay_hd", "inplay_ou", "inplay_cr"]:
                if f"{pfx}_line" not in st.session_state: st.session_state[f"{pfx}_line"] = 0.0
                if f"{pfx}_up" not in st.session_state: st.session_state[f"{pfx}_up"] = 1.90
                if f"{pfx}_lw" not in st.session_state: st.session_state[f"{pfx}_lw"] = 1.90
                if f"{pfx}_margin" not in st.session_state: st.session_state[f"{pfx}_margin"] = 1.085

            inplay_odds_data = []
            labels = [("讓球", "inplay_hd"), ("入球大小", "inplay_ou"), ("角球大小", "inplay_cr")]
            
            for label, pfx in labels:
                st.markdown(f"**⚽ {label} 盤口**")
                cc1, cc2, cc3, cc4 = st.columns(4)
                line_val = cc1.number_input("盤口線", value=st.session_state[f"{pfx}_line"], key=f"{pfx}_line", step=0.5, format="%.2f")
                up_val = cc2.number_input("大/上賠率", min_value=1.01, value=st.session_state[f"{pfx}_up"], key=f"{pfx}_up", step=0.01, on_change=on_inplay_upper_change, args=(pfx,))
                lw_val = cc3.number_input("小/下賠率", min_value=1.01, value=st.session_state[f"{pfx}_lw"], key=f"{pfx}_lw", step=0.01, on_change=on_inplay_lower_change, args=(pfx,))
                margin_val = st.session_state.get(f"{pfx}_margin", 1.085)
                
                cc4.markdown(f"<div style='margin-top: 32px; color: #888; font-weight: bold;'>抽水: {(margin_val - 1.0) * 100.0:.2f}% ({margin_val:.3f})</div>", unsafe_allow_html=True)
                
                inplay_odds_data.append({"bet_type": label, "line": line_val, "upper": up_val, "lower": lw_val, "margin": margin_val})

            if st.button("🚀 即場數據分析執行", type="primary", use_container_width=True):
                st.session_state.show_inplay_analysis = True
                rating_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
                hr_val = rating_map.get(last_row.get('Home_Rating', 'C'), 3)
                ar_val = rating_map.get(last_row.get('Away_Rating', 'C'), 3)
                
                time_factor = max(0.01, (90 - ip_minute) / 90.0)
                attack_diff = (h_firepower - a_firepower) * 0.005
                eff_diff = (h_efficiency - a_efficiency) * 0.002
                total_attack = (h_firepower + a_firepower) * 0.005
                
                candidates_base_ip = []
                for item in inplay_odds_data:
                    b_type = item['bet_type']
                    if b_type == "讓球":
                        p_up = max(0.1, min(0.9, 0.5 + ((hr_val - ar_val) * 0.02) + attack_diff + eff_diff))
                        candidates_base_ip.extend([
                            {'bet_type': b_type, 'selection': 'Home', 'base_prob': p_up, 'odds': item['upper'], 'line': item['line'], 'label': f"{item['line']:g}主隊(即場)"},
                            {'bet_type': b_type, 'selection': 'Away', 'base_prob': 1-p_up, 'odds': item['lower'], 'line': item['line'], 'label': f"{item['line']:g}客隊(即場)"}
                        ])
                    else:
                        base_over = 0.5 + total_attack - ((1 - time_factor) * 0.2)
                        p_up = max(0.1, min(0.9, base_over))
                        candidates_base_ip.extend([
                            {'bet_type': b_type, 'selection': 'Over', 'base_prob': p_up, 'odds': item['upper'], 'line': item['line'], 'label': f"{item['line']:g}大盤(即場)"},
                            {'bet_type': b_type, 'selection': 'Under', 'base_prob': 1-p_up, 'odds': item['lower'], 'line': item['line'], 'label': f"{item['line']:g}小盤(即場)"}
                        ])
                        
                h_data_ip = {'hr': hr_val, 'ar': ar_val, 'hf': extract_form_points(last_row.get('Home_Form')), 'af': extract_form_points(last_row.get('Away_Form'))}
                df_settled_ip = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
                
                res_macro_ip = evaluate_dimension(df_settled_ip, "宏觀 - 總數據", candidates_base_ip, rating_map, h_data_ip)
                best_model_ip = res_macro_ip
                
                st.session_state.inplay_analysis_result = {
                    'best_model': best_model_ip, 'best_bet': best_model_ip.get('best', candidates_base_ip[0]),
                    't_name': selected_tourn, 't_cat': last_row['Tournament_Category'],
                    'home_team': last_row['Home_Team'], 'away_team': last_row['Away_Team'],
                    'h_rating': last_row.get('Home_Rating'), 'a_rating': last_row.get('Away_Rating'),
                    'h_form': last_row.get('Home_Form'), 'a_form': last_row.get('Away_Form')
                }

            if st.session_state.get('show_inplay_analysis', False):
                res_ip = st.session_state.inplay_analysis_result
                st.success(f"✅ 即場三維度數據分析與 EV 運算完成！推薦首選: {res_ip['best_bet']['label']}")
                
                cand_list_ip = res_ip['best_model'].get('candidates', [])
                all_options_ip, option_map_ip = [], {}
                for c in cand_list_ip:
                    opt_str = f"【{c['bet_type']}】{c['label']} @ {c['odds']}"
                    if opt_str not in option_map_ip:
                        all_options_ip.append(opt_str)
                        option_map_ip[opt_str] = c

                col_sys_ip, col_usr_ip = st.columns(2)
                sys_selected_ip = col_sys_ip.multiselect("選擇系統即場投注", all_options_ip, default=[all_options_ip[0]] if all_options_ip else [])
                sys_stakes_ip = {sel: col_sys_ip.number_input(f"系統建議金額: {sel}", value=calc_suggested_stake(option_map_ip[sel], sys_bankroll, sys_max_stake)[0], disabled=True) for sel in sys_selected_ip}
                
                usr_selected_ip = col_usr_ip.multiselect("選擇用家即場投注", all_options_ip)
                usr_stakes_ip = {sel: col_usr_ip.number_input(f"用家金額: {sel}", min_value=0.0, step=10.0, value=100.0) for sel in usr_selected_ip}

                if st.button("✅ 確定即場投注並寫入雲端資料庫", type="primary", use_container_width=True):
                    all_keys_ip = set(sys_selected_ip + usr_selected_ip)
                    new_records_ip = []
                    for i, sel in enumerate(all_keys_ip):
                        cand = option_map_ip[sel]
                        new_records_ip.append({
                            'ID': f"B{get_hkt_now().strftime('%Y%m%d%H%M%S')}{i}_IP", 'Date': get_hkt_now().strftime('%Y-%m-%d %H:%M'), 'Status': 'Open',
                            'Tournament_Name': res_ip['t_name'], 'Tournament_Category': res_ip['t_cat'], 
                            'Match': f"{res_ip['home_team']} vs {res_ip['away_team']}", 'Home_Team': res_ip['home_team'], 'Away_Team': res_ip['away_team'],
                            'Home_Rating': res_ip['h_rating'], 'Away_Rating': res_ip['a_rating'], 'Home_Form': res_ip['h_form'], 'Away_Form': res_ip['a_form'],
                            'Bet_Type': f"{cand['bet_type']} (即場)", 'Selection': cand['selection'], 'Initial_Line': cand['line'], 'Initial_Odds': cand['odds'], 
                            'System_Stake': sys_stakes_ip.get(sel, 0.0), 'User_Stake': usr_stakes_ip.get(sel, 0.0),
                            'Odds_History': json.dumps([{"id":0, "type": cand['bet_type'], "record_time": get_hkt_now().strftime('%m-%d %H:%M'), "line": cand['line'], "upper": cand['odds'], "lower": cand['odds'], "unlock": False, "margin": 1.085}], ensure_ascii=False)
                        })
                    st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame(new_records_ip)], ignore_index=True)
                    save_db(st.session_state.df_db, db_file, db_table)
                    st.session_state.show_inplay_analysis = False
                    st.rerun()

    with t_settle:
        st.subheader("⚖️ 賽果結算與派彩管理")
        display_cumulative_metrics(st.session_state.df_db)
        
        open_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
        if open_df.empty:
            st.info("目前沒有待結算的注單。")
        else:
            st.write("請選擇要結算的比賽並輸入最終賽果 (系統將自動結算該場比賽的所有注單)")

if __name__ == "__main__":
    main()
