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

# GitHub API 讀取與寫入輔助函式 (加入時間戳記防快取)
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
    """處理舊資料庫欄位轉移防呆"""
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
    # 優先從雲端 SQL 資料庫讀取
    if HAS_SQLALCHEMY and "DB_URL" in st.secrets and st.secrets["DB_URL"]:
        try:
            engine = create_engine(st.secrets["DB_URL"])
            df = pd.read_sql_table(table_name, engine)
            df = process_legacy_columns(df)
        except Exception:
            pass

    # 其次從 GitHub 雲端倉庫讀取
    if df.empty and "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets:
        try:
            gh_df = load_db_github(st.secrets["GITHUB_REPO"], filename, st.secrets["GITHUB_TOKEN"])
            if gh_df is not None and not gh_df.empty:
                df = process_legacy_columns(gh_df)
        except Exception:
            pass

    # 最後退回本地 CSV (僅作為備用)
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
    # 1. 寫入 SQL 雲端資料庫
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

    # 2. 寫入 GitHub 雲端倉庫
    if "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets:
        try:
            save_db_github(df, st.secrets["GITHUB_REPO"], filename, st.secrets["GITHUB_TOKEN"])
        except Exception as e:
            st.error(f"GitHub 雲端同步失敗: {e}")

    # 3. 寫入本地 CSV
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
    """解析近況字串，例如 '3W1D1L'"""
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
# 4. 注單載入與修改輔助邏輯 (新增/覆蓋)
# ==========================================
def load_bet_to_edit(bet_id):
    """將雲端數據庫中的注單載入至 session_state 供使用者重新修改"""
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
            st.session_state.inplay_odds_history = oh
    except:
        pass
        
    st.session_state.edit_inplay_minute = int(float(row.get('InPlay_Minute', 45))) if pd.notna(row.get('InPlay_Minute')) else 45
    st.session_state.edit_h_g = int(float(row.get('Home_Goal', 0))) if pd.notna(row.get('Home_Goal')) else 0
    st.session_state.edit_a_g = int(float(row.get('Away_Goal', 0))) if pd.notna(row.get('Away_Goal')) else 0
    st.session_state.edit_h_c = int(float(row.get('Home_Corner', 0))) if pd.notna(row.get('Home_Corner')) else 0
    st.session_state.edit_a_c = int(float(row.get('Away_Corner', 0))) if pd.notna(row.get('Away_Corner')) else 0
    st.session_state.edit_h_da = int(float(row.get('Home_DA', 0))) if pd.notna(row.get('Home_DA')) else 0
    st.session_state.edit_a_da = int(float(row.get('Away_DA', 0))) if pd.notna(row.get('Away_DA')) else 0
    st.session_state.edit_h_sot = int(float(row.get('Home_SoT', 0))) if pd.notna(row.get('Home_SoT')) else 0
    st.session_state.edit_a_sot = int(float(row.get('Away_SoT', 0))) if pd.notna(row.get('Away_SoT')) else 0
    st.session_state.edit_h_soff = int(float(row.get('Home_SoFF', 0))) if pd.notna(row.get('Home_SoFF')) else 0
    st.session_state.edit_a_soff = int(float(row.get('Away_SoFF', 0))) if pd.notna(row.get('Away_SoFF')) else 0
    st.session_state.edit_h_red = int(float(row.get('Home_Red', 0))) if pd.notna(row.get('Home_Red')) else 0
    st.session_state.edit_a_red = int(float(row.get('Away_Red', 0))) if pd.notna(row.get('Away_Red')) else 0
    st.session_state.edit_h_sub = int(float(row.get('Home_Sub', 0))) if pd.notna(row.get('Home_Sub')) else 0
    st.session_state.edit_a_sub = int(float(row.get('Away_Sub', 0))) if pd.notna(row.get('Away_Sub')) else 0
    st.session_state.edit_h_poss = int(float(row.get('Home_Possession', 50))) if pd.notna(row.get('Home_Possession')) else 50
    
    st.session_state.edit_sys_stake = float(row.get('System_Stake', 0.0)) if pd.notna(row.get('System_Stake')) else 0.0
    st.session_state.edit_user_stake = float(row.get('User_Stake', 0.0)) if pd.notna(row.get('User_Stake')) else 0.0
    st.session_state.edit_bet_type = str(row.get('Bet_Type', '')).replace(" (即場)", "").strip() if pd.notna(row.get('Bet_Type')) else ''
    st.session_state.edit_selection = str(row.get('Selection', 'Home')) if pd.notna(row.get('Selection')) else 'Home'
    st.session_state.edit_line = float(row.get('Initial_Line', 0.0)) if pd.notna(row.get('Initial_Line')) else 0.0

def clear_edit_mode():
    """清除修改模式狀態"""
    st.session_state.editing_bet_id = None
    keys_to_del = [k for k in st.session_state.keys() if k.startswith('edit_')]
    for k in keys_to_del:
        del st.session_state[k]

# ==========================================
# 5. 資金流水 HTML 構建
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

@st.dialog("📊 數據庫即時線上預覽與管理", width="large")
def preview_db_dialog(df_db, df_cap, db_file, capital_file, db_table, cap_table):
    tab_bets, tab_capital, tab_manage = st.tabs(["⚽ 投注紀錄預覽", "💰 資金流水預覽", "🗑️ 數據清理與還原"])
    
    with tab_bets:
        st.write("您可以在下方表格中自由滑動、點擊欄位排序，或使用關鍵字搜尋特定賽事。")
        search_query = st.text_input("🔍 關鍵字搜尋 (例如: 球隊名稱、盤口)", "", key="search_bets")
        
        if search_query:
            mask = df_db.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1)
            show_df = df_db[mask].copy()
        else:
            show_df = df_db.copy()

        total_sys_profit = pd.to_numeric(show_df['System_Profit'], errors='coerce').sum()
        total_user_profit = pd.to_numeric(show_df['User_Profit'], errors='coerce').sum()
        total_unit = pd.to_numeric(show_df['Unit_Profit'], errors='coerce').sum()
        total_user_payout = pd.to_numeric(show_df['User_Payout'], errors='coerce').sum()

        summary_data = {col: None for col in show_df.columns}
        if 'ID' in summary_data: summary_data['ID'] = "TOTAL (總計)"
        if 'System_Profit' in summary_data: summary_data['System_Profit'] = round(total_sys_profit, 2)
        if 'User_Profit' in summary_data: summary_data['User_Profit'] = round(total_user_profit, 2)
        if 'Unit_Profit' in summary_data: summary_data['Unit_Profit'] = round(total_unit, 2)
        if 'User_Payout' in summary_data: summary_data['User_Payout'] = round(total_user_payout, 2)

        summary_row = pd.DataFrame([summary_data])
        show_df_with_summary = pd.concat([show_df, summary_row], ignore_index=True)

        st.dataframe(show_df_with_summary, use_container_width=True)
        
        excel_data = io.BytesIO()
        try:
            show_df_with_summary.to_excel(excel_data, index=False)
            st.download_button("📥 點擊下載投注紀錄 Excel 報表", excel_data.getvalue(), "football_betting_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="btn_down_bets_xlsx")
        except:
            st.download_button("📥 點擊下載投注紀錄報表 (CSV)", show_df_with_summary.to_csv(index=False).encode('utf-8-sig'), "football_betting_report.csv", "text/csv", key="btn_down_bets_csv")

    with tab_capital:
        st.subheader("💰 資金流水帳目 (Capital Flow Ledger)")
        cap_tab1, cap_tab2 = st.tabs(["🤖 系統資金流水", "👤 用家真實資金流水"])
        
        sys_cap = df_cap[df_cap['Account'].isin(['System', 'Both'])]
        usr_cap = df_cap[df_cap['Account'].isin(['User', 'Both'])]
        
        with cap_tab1:
            st.markdown("##### 🤖 系統資金流水")
            table_html_sys, tot_amt_sys, tot_str_sys, tot_color_sys, tot_label_sys = build_capital_flow_html(sys_cap)
            st.markdown(table_html_sys, unsafe_allow_html=True)
            
            sys_cap_exp = sys_cap.copy()
            sys_cap_summary = {col: None for col in sys_cap_exp.columns}
            if 'ID' in sys_cap_summary: sys_cap_summary['ID'] = "TOTAL (總計)"
            if 'Amount' in sys_cap_summary: sys_cap_summary['Amount'] = round(tot_amt_sys, 2)
            if 'Note' in sys_cap_summary: sys_cap_summary['Note'] = tot_label_sys.strip(" ()")
            sys_cap_exp = pd.concat([sys_cap_exp, pd.DataFrame([sys_cap_summary])], ignore_index=True)
            
            st.download_button("📥 下載系統資金報表 (CSV)", sys_cap_exp.to_csv(index=False).encode('utf-8-sig'), "system_capital_flow.csv", "text/csv", key="btn_down_sys_cap_csv")
            
        with cap_tab2:
            st.markdown("##### 👤 用家真實資金流水")
            table_html_usr, tot_amt_usr, tot_str_usr, tot_color_usr, tot_label_usr = build_capital_flow_html(usr_cap)
            st.markdown(table_html_usr, unsafe_allow_html=True)
            
            usr_cap_exp = usr_cap.copy()
            usr_cap_summary = {col: None for col in usr_cap_exp.columns}
            if 'ID' in usr_cap_summary: usr_cap_summary['ID'] = "TOTAL (總計)"
            if 'Amount' in usr_cap_summary: usr_cap_summary['Amount'] = round(tot_amt_usr, 2)
            if 'Note' in usr_cap_summary: usr_cap_summary['Note'] = tot_label_usr.strip(" ()")
            usr_cap_exp = pd.concat([usr_cap_exp, pd.DataFrame([usr_cap_summary])], ignore_index=True)
            
            st.download_button("📥 下載用家資金報表 (CSV)", usr_cap_exp.to_csv(index=False).encode('utf-8-sig'), "user_capital_flow.csv", "text/csv", key="btn_down_usr_cap_csv")

    with tab_manage:
        st.subheader("1. 批量刪除與一鍵清除")
        del_mode = st.radio("選擇要清理的資料表", ["⚽ 投注紀錄", "💰 資金流水"])
        
        if del_mode == "⚽ 投注紀錄":
            df_target = df_db
            target_name = 'bets'
            opts = [f"{r['ID']} | {r['Date']} | {r.get('Match', '')}" for _, r in df_target.iterrows()]
        else:
            df_target = df_cap
            target_name = 'cap'
            opts = [f"{r['ID']} | {r['Date']} | {r.get('Account', 'Both')} | {r.get('Type', '')} | ${r.get('Amount', 0)}" for _, r in df_target.iterrows()]
            
        selected_to_delete = st.multiselect("選擇要刪除的紀錄 (可多選):", opts)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑 刪除選中紀錄", use_container_width=True):
                if selected_to_delete:
                    ids_to_delete = [x.split(" | ")[0] for x in selected_to_delete]
                    df_to_delete = df_target[df_target['ID'].isin(ids_to_delete)]
                    
                    st.session_state.undo_stack.append({
                        'id': f"U{get_hkt_now().strftime('%Y%m%d%H%M%S%f')}",
                        'target': target_name,
                        'data': df_to_delete.copy(),
                        'timestamp': get_hkt_now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    st.session_state.undo_stack = st.session_state.undo_stack[-10:]
                    
                    if target_name == 'bets':
                        st.session_state.df_db = df_db[~df_db['ID'].isin(ids_to_delete)]
                        save_db(st.session_state.df_db, db_file, db_table)
                    else:
                        st.session_state.df_cap = df_cap[~df_cap['ID'].isin(ids_to_delete)]
                        save_db(st.session_state.df_cap, capital_file, cap_table)
                        
                    st.rerun()
                else:
                    st.warning("請先選擇要刪除的紀錄。")
                    
        with col2:
            with st.expander("💣 一鍵清除全部資料 (危險操作)"):
                st.warning(f"確認要清空所有 **{del_mode}** 嗎？")
                st.caption("此操作會將當前資料表所有紀錄移除，點擊下方確認執行。")
                if st.button("⚠️ 確認清空全部", type="primary", use_container_width=True):
                    if not df_target.empty:
                        st.session_state.undo_stack.append({
                            'id': f"U{get_hkt_now().strftime('%Y%m%d%H%M%S%f')}",
                            'target': target_name,
                            'data': df_target.copy(),
                            'timestamp': get_hkt_now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        st.session_state.undo_stack = st.session_state.undo_stack[-10:]
                        
                        if target_name == 'bets':
                            st.session_state.df_db = pd.DataFrame(columns=DB_COLUMNS)
                            save_db(st.session_state.df_db, db_file, db_table)
                        else:
                            st.session_state.df_cap = pd.DataFrame(columns=CAPITAL_COLUMNS)
                            save_db(st.session_state.df_cap, capital_file, cap_table)
                        st.rerun()
                        
        st.divider()
        st.subheader("2. ↩ 狀態重置 (Undo 復原中心)")
        if not st.session_state.undo_stack:
            st.info("目前沒有可還原的刪除紀錄。")
        else:
            st.write(f"目前系統為您保留最近 **{len(st.session_state.undo_stack)}** 次的刪除操作供隨時復原。")
            undo_options = []
            
            for idx, action in enumerate(reversed(st.session_state.undo_stack)):
                t_label = "⚽ 投注紀錄" if action['target'] == 'bets' else "💰 資金流水"
                real_idx = len(st.session_state.undo_stack) - 1 - idx
                opt_str = f"[{action['timestamp']}] 刪除了 {len(action['data'])} 筆 {t_label}"
                undo_options.append((real_idx, opt_str))
                
            sel_undo = st.selectbox("請選擇要復原的刪除紀錄：", undo_options, format_func=lambda x: x[1])
            
            if st.button("↩️ 復原所選的刪除紀錄 (Undo)", type="primary"):
                real_idx = sel_undo[0]
                action = st.session_state.undo_stack.pop(real_idx)
                
                if action['target'] == 'bets':
                    st.session_state.df_db = pd.concat([st.session_state.df_db, action['data']], ignore_index=True)
                    save_db(st.session_state.df_db, db_file, db_table)
                else:
                    st.session_state.df_cap = pd.concat([st.session_state.df_cap, action['data']], ignore_index=True)
                    save_db(st.session_state.df_cap, capital_file, cap_table)
                    
                st.toast(f"✅ 已成功復原 {len(action['data'])} 筆資料！系統資金池已自動重構。", icon="↩️")
                st.rerun()

def parse_single_type_text(text, bet_type):
    """基於時間區塊、上下盤精準分隔與重複去重的解析器"""
    if not text or not text.strip():
        return []
    
    lines_raw = [l.strip() for l in text.split('\n') if l.strip()]
    default_line = 2.5 if bet_type == "入球大小" else (9.5 if bet_type == "角球大小" else 0.0)
    
    def parse_line_val(s):
        s = str(s).replace('球', '').replace('+', '').replace('[', '').replace(']', '').strip()
        if '/' in s:
            try:
                parts = s.split('/')
                return (float(parts[0]) + float(parts[1])) / 2.0
            except:
                return 0.0
        try:
            return float(s)
        except:
            return 0.0

    def parse_odds_val(s):
        try:
            clean = re.sub(r'[^\d\.]', '', str(s))
            val = float(clean)
            return val if val >= 1.01 else None
        except:
            return None

    dt_pattern = re.compile(r'(\d{1,4}[-/.]\d{1,2}(?:[-/.]\d{1,4})?\s*\d{1,2}:\d{2})')
    
    blocks = []
    current_block = {"datetime": "", "lines_content": []}
    
    for l in lines_raw:
        m = dt_pattern.search(l)
        if m:
            if current_block["datetime"] or current_block["lines_content"]:
                blocks.append(current_block)
            dt_str = m.group(1)
            remaining = l.replace(dt_str, '').strip()
            current_block = {"datetime": dt_str, "lines_content": [remaining] if remaining else []}
        else:
            current_block["lines_content"].append(l)
            
    if current_block["datetime"] or current_block["lines_content"]:
        blocks.append(current_block)
        
    parsed_items = []
    active_line = default_line
    
    for block in blocks:
        dt = block["datetime"]
        block_lines = block["lines_content"]
        
        line_idx = -1
        found_line_val = None
        
        for idx, l_item in enumerate(block_lines):
            bracket_match = re.search(r'\[(.*?)\]', l_item)
            if bracket_match:
                found_line_val = parse_line_val(bracket_match.group(1))
                line_idx = idx
                break
            else:
                goal_match = re.search(r'([+-]?\d+(?:\.\d+)?(?:/[+-]?\d+(?:\.\d+)?)?)\s*球', l_item)
                if goal_match:
                    found_line_val = parse_line_val(goal_match.group(1))
                    line_idx = idx
                    break
        
        if found_line_val is not None:
            active_line = found_line_val
        
        if line_idx == -1:
            upper_lines = block_lines
            lower_lines = []
        else:
            upper_lines = block_lines[:line_idx]
            lower_lines = block_lines[line_idx:]
            
        upper_odds_list = []
        for l_item in upper_lines:
            for n_str in re.findall(r'\b\d+(?:\.\d+)?\b', l_item):
                val = parse_odds_val(n_str)
                if val is not None and abs(val - active_line) > 1e-4:
                    upper_odds_list.append(val)
                    
        lower_odds_list = []
        for l_item in lower_lines:
            cleaned_l_item = re.sub(r'\[.*?\]', '', l_item)
            for n_str in re.findall(r'\b\d+(?:\.\d+)?\b', cleaned_l_item):
                val = parse_odds_val(n_str)
                if val is not None and abs(val - active_line) > 1e-4:
                    lower_odds_list.append(val)
                    
        unique_upper = []
        for u in upper_odds_list:
            if not any(abs(u - existing) < 1e-4 for existing in unique_upper):
                unique_upper.append(u)
                
        unique_lower = []
        for l_val in lower_odds_list:
            if not any(abs(l_val - existing) < 1e-4 for existing in unique_lower):
                unique_lower.append(l_val)
                
        if len(unique_upper) > 0:
            up = unique_upper[-1]
        elif len(upper_odds_list) > 0:
            up = upper_odds_list[-1]
        else:
            up = 1.90
            
        if len(unique_lower) > 0:
            lw = unique_lower[-1]
        elif len(lower_odds_list) > 0:
            lw = lower_odds_list[-1]
        else:
            lw = 1.90
            
        if up == 1.90 or lw == 1.90:
            all_nums = []
            for l_item in block_lines:
                clean_item = re.sub(r'\[.*?\]', '', l_item)
                for n_str in re.findall(r'\b\d+(?:\.\d+)?\b', clean_item):
                    v = parse_odds_val(n_str)
                    if v is not None and abs(v - active_line) > 1e-4:
                        if not any(abs(v - x) < 1e-4 for x in all_nums):
                            all_nums.append(v)
            if len(all_nums) >= 2:
                up, lw = all_nums[0], all_nums[1]
            elif len(all_nums) == 1:
                up = lw = all_nums[0]
                
        margin = round((1/up) + (1/lw), 3) if (up > 0 and lw > 0) else 1.085
        
        parsed_items.append({
            "type": bet_type,
            "record_time": dt,
            "line": active_line,
            "upper": up,
            "lower": lw,
            "unlock": True,
            "margin": margin
        })
        
    return parsed_items

def calc_margin_str(row):
    """計算每一個盤口的抽水與百分比"""
    try:
        up = float(row.get('upper', 1.90))
        lw = float(row.get('lower', 1.90))
        if up > 0 and lw > 0:
            margin = (1.0 / up) + (1.0 / lw)
            pct = (margin - 1.0) * 100.0
            return f"{pct:.2f}% ({margin:.3f})"
        return "8.50% (1.085)"
    except:
        return "8.50% (1.085)"

def render_odds_section(odds_history_state, prefix="pre"):
    st.markdown("💡 **智能解析與動態同步區：** 請分別貼上各盤口數據（包含跨行的日期及時間、盤口、整數或小數賠率）。系統會自動依時間區塊與上下盤位置智慧解析並自動去除重複賠率。")
    
    col_hd, col_ou, col_cr = st.columns(3)
    with col_hd:
        raw_hd = st.text_area("⚽ 讓球 貼上區", height=130, key=f"{prefix}_paste_hd", placeholder="例如:\n2\n01-10 23:40\n1.88\n[0/+0.5]\n1.88")
    with col_ou:
        raw_ou = st.text_area("⚽ 入球大小 貼上區", height=130, key=f"{prefix}_paste_ou", placeholder="例如:\n01-10 21:05\n2.5\n[2.5]\n2")
    with col_cr:
        raw_cr = st.text_area("⚽ 角球大小 貼上區", height=130, key=f"{prefix}_paste_cr", placeholder="例如:\n18-09 07:11 2.05\n2.05\n[10.5] 1.68\n1.68")
        
    parsed_hd = parse_single_type_text(raw_hd, "讓球")
    parsed_ou = parse_single_type_text(raw_ou, "入球大小")
    parsed_cr = parse_single_type_text(raw_cr, "角球大小")
    
    parsed_all = parsed_hd + parsed_ou + parsed_cr
    
    if raw_hd.strip() or raw_ou.strip() or raw_cr.strip():
        odds_history_state.clear()
        if parsed_all:
            for idx, item in enumerate(parsed_all):
                item["id"] = idx
                odds_history_state.append(item)
        else:
            odds_history_state.append({
                "id": 0, "type": "讓球", "record_time": "", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": True, "margin": 1.085
            })
    elif not odds_history_state:
        odds_history_state.append({
            "id": 0, "type": "讓球", "record_time": "", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": True, "margin": 1.085
        })
        
    df = pd.DataFrame(odds_history_state)
    for col, default in [('type', '讓球'), ('record_time', ''), ('line', 0.0), ('upper', 1.90), ('lower', 1.90)]:
        if col not in df.columns:
            df[col] = default
            
    df['margin_disp'] = df.apply(calc_margin_str, axis=1)
    df_display = df[['record_time', 'type', 'line', 'upper', 'lower', 'margin_disp']].copy()
    df_display.columns = ['📅日期及時間', '盤口類型', '盤口線', '主隊/大盤賠率', '客隊/小盤賠率', '盤口抽水 (百分比)']
    
    st.markdown("##### 📝 盤口與賠率走勢表 (自動計算抽水與百分比，亦可直接刪除或修改資料)")
    edited_df = st.data_editor(
        df_display,
        num_rows="dynamic",
        column_config={
            "📅日期及時間": st.column_config.TextColumn("📅日期及時間", required=False),
            "盤口類型": st.column_config.SelectboxColumn("盤口類型", options=["讓球", "入球大小", "角球大小"], required=True),
            "盤口線": st.column_config.NumberColumn("盤口線", format="%.2f", required=True),
            "主隊/大盤賠率": st.column_config.NumberColumn("主隊/大盤賠率", min_value=1.01, format="%.2f", required=True),
            "客隊/小盤賠率": st.column_config.NumberColumn("客隊/小盤賠率", min_value=1.01, format="%.2f", required=True),
            "盤口抽水 (百分比)": st.column_config.TextColumn("盤口抽水 (百分比)", disabled=True)
        },
        use_container_width=True,
        key=f"{prefix}_odds_editor"
    )
    
    new_history = []
    new_idx = 0
    for i, row in edited_df.iterrows():
        try:
            up = float(row["主隊/大盤賠率"]) if pd.notna(row["主隊/大盤賠率"]) else 1.90
            lw = float(row["客隊/小盤賠率"]) if pd.notna(row["客隊/小盤賠率"]) else 1.90
            margin = (1/up) + (1/lw) if (up > 0 and lw > 0) else 1.085
        except:
            up, lw, margin = 1.90, 1.90, 1.085
            
        new_history.append({
            "id": new_idx,
            "type": str(row["盤口類型"]) if pd.notna(row["盤口類型"]) else "讓球",
            "record_time": str(row["📅日期及時間"]) if pd.notna(row["📅日期及時間"]) else "",
            "line": float(row["盤口線"]) if pd.notna(row["盤口線"]) else 0.0,
            "upper": up,
            "lower": lw,
            "unlock": True,
            "margin": round(margin, 3)
        })
        new_idx += 1
        
    odds_history_state.clear()
    odds_history_state.extend(new_history)

def calc_suggested_stake(cand, sys_bankroll, sys_max_stake):
    suggested_stake = 0.0
    raw_stake = 0.0
    if cand.get('ev', 0) > 0 and sys_bankroll > 0:
        b = cand.get('odds', 1.90) - 1
        prob = cand.get('prob', cand.get('base_prob', 0.5))
        if b > 0:
            kelly = max(0.0, min((prob * b - (1 - prob)) / b, 0.10))
            raw_stake = (sys_bankroll * (kelly * 0.5))
            suggested_stake = min(float(sys_max_stake), float(round(raw_stake / 10) * 10))
            
            if "讓球" in cand.get('bet_type', ''):
                if 0 < suggested_stake < 200:
                    if cand.get('ev', 0) >= 0.03 and prob >= 0.50:
                        suggested_stake = 200.0
                    else:
                        suggested_stake = 0.0
            else:
                suggested_stake = max(10.0, suggested_stake)
    return suggested_stake, raw_stake

# ==========================================
# 6. 主程式 UI 
# ==========================================
def main():
    st.title("⚽ Actuarial and fund management system by Dr. EdwinPro")
    
    if 'undo_stack' not in st.session_state:
        st.session_state.undo_stack = []
    if 'editing_bet_id' not in st.session_state:
        st.session_state.editing_bet_id = None
    if 'last_bet_id' not in st.session_state:
        st.session_state.last_bet_id = None
        
    st.sidebar.header("⚙ 系統設定與資金管理")
    
    db_file = "football_betting_db.csv"
    capital_file = "football_capital_db.csv"
    db_table = "football_bets"
    cap_table = "football_cap"

    # --- 數據雲端同步狀態提示與手動刷新 ---
    has_sql = HAS_SQLALCHEMY and "DB_URL" in st.secrets and st.secrets["DB_URL"]
    has_gh = "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets
    
    st.sidebar.subheader("☁️ 數據同步狀態看板")
    if has_sql:
        st.sidebar.success("✅ 已連接 PostgreSQL/SQL 雲端資料庫")
    elif has_gh:
        st.sidebar.success(f"✅ 已連接 GitHub 倉庫同步 (`{st.secrets['GITHUB_REPO']}`)")
    else:
        st.sidebar.warning("⚠️ 未偵測到 Secrets！資料僅存於臨時容器。如需永久儲存，請至 Streamlit 設定 GITHUB_TOKEN 與 GITHUB_REPO。")

    if st.sidebar.button("🔄 即時從雲端同步最新數據", use_container_width=True):
        st.session_state.df_db = load_db(db_file, DB_COLUMNS, db_table, force_cloud=True)
        st.session_state.df_cap = load_db(capital_file, CAPITAL_COLUMNS, cap_table, force_cloud=True)
        st.toast("✅ 數據已與雲端同步！", icon="🔄")
        st.rerun()

    if 'df_db' not in st.session_state:
        st.session_state.df_db = load_db(db_file, DB_COLUMNS, db_table)
    if 'df_cap' not in st.session_state:
        st.session_state.df_cap = load_db(capital_file, CAPITAL_COLUMNS, cap_table)

    (sys_dep, sys_wit, sys_net, sys_pnl, sys_bankroll, sys_max_stake, 
     usr_dep, usr_wit, usr_net, usr_pnl, usr_bankroll, usr_max_stake) = recalculate_bankroll_from_scratch(st.session_state.df_cap, st.session_state.df_db)
    
    # --- 系統本金區塊 ---
    st.sidebar.divider()
    st.sidebar.subheader("🤖 系統本金與盈虧總覽 (System)")
    st.sidebar.caption("主要供機器學習與策略檢驗使用")
    st.sidebar.metric("系統總存入本金", f"${sys_dep:,.2f}")
    st.sidebar.metric("系統總提取本金", f"${sys_wit:,.2f}")
    st.sidebar.metric("系統累積總盈虧 (PnL)", f"${sys_pnl:,.2f}", delta=f"${sys_pnl:,.2f}")
    st.sidebar.metric("系統當前可用資金 (Bankroll)", f"${sys_bankroll:,.2f}")
    st.sidebar.caption(f"🛑 **系統單注上限 (動態資金 10%)**: `${sys_max_stake:,.2f}`")

    # --- 用家真實本金區塊 ---
    st.sidebar.divider()
    st.sidebar.subheader("👤 用家真實本金與盈虧總覽 (User Actual)")
    st.sidebar.caption("供用家作真實資金管理及記錄參考")
    st.sidebar.metric("用家總存入本金", f"${usr_dep:,.2f}")
    st.sidebar.metric("用家總提取本金", f"${usr_wit:,.2f}")
    st.sidebar.metric("用家真實累積總盈虧 (PnL)", f"${usr_pnl:,.2f}", delta=f"${usr_pnl:,.2f}")
    st.sidebar.metric("用家當前真實可用資金 (Bankroll)", f"${usr_bankroll:,.2f}")

    with st.sidebar.expander("💸 資金存提管理"):
        cap_account = st.radio("目標帳戶 (Account)", ["🤖 系統本金 (System)", "👤 用家本金 (User)", "🔄 兩者同步 (Both)"], index=2)
        cap_action = st.radio("動作", ["Deposit (存入本金)", "Withdraw (提取本金)"])
        cap_amount = st.number_input("金額 ($)", min_value=1.0, value=1000.0, step=100.0)
        cap_note = st.text_input("備註 (選填)")
        
        if st.button("確認寫入資金紀錄"):
            acc_val = 'Both'
            if "System" in cap_account: acc_val = 'System'
            elif "User" in cap_account: acc_val = 'User'
            
            new_cap_record = {
                'ID': f"C{get_hkt_now().strftime('%Y%m%d%H%M%S')}",
                'Date': get_hkt_now().strftime('%Y-%m-%d %H:%M'),
                'Type': 'Deposit' if 'Deposit' in cap_action else 'Withdraw',
                'Account': acc_val,
                'Amount': float(cap_amount),
                'Note': cap_note
            }
            st.session_state.df_cap = pd.concat([st.session_state.df_cap, pd.DataFrame([new_cap_record])], ignore_index=True)
            save_db(st.session_state.df_cap, capital_file, cap_table)
            st.toast("✅ 資金紀錄雲端同步成功！系統與用家本金已自動重構。", icon="💰")
            st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("🔍 數據庫即時線上預覽與管理", use_container_width=True):
        preview_db_dialog(st.session_state.df_db, st.session_state.df_cap, db_file, capital_file, db_table, cap_table)

    t_pre, t_inplay, t_settle, t_ai = st.tabs(["📝 賽前建檔與投注", "⏱️ 即場賽事與預測", "⚖ 賽果結算與管理", "🤖 全局模型"])

    with t_pre:
        st.subheader("📝 賽事建檔與智能盤口走勢分析")
        
        if st.session_state.editing_bet_id:
            st.info(f"🛠️ **【修改/覆蓋模式】** 目前正在編輯未結算注單：`{st.session_state.editing_bet_id}`。修改後提交將**直接覆蓋**資料庫中的原有數據。")
            if st.button("❌ 取消修改 (恢復為新建注單)", key="cancel_edit_pre"):
                clear_edit_mode()
                st.rerun()
        elif st.session_state.last_bet_id:
            last_match = st.session_state.df_db[st.session_state.df_db['ID'] == st.session_state.last_bet_id]
            if not last_match.empty and last_match.iloc[0]['Status'] == 'Open':
                c_msg, c_btn = st.columns([3, 1])
                c_msg.info(f"💡 剛提交注單 ID: **{st.session_state.last_bet_id}** ({last_match.iloc[0]['Match']})。如發現資料有錯漏，可隨時載入修改。")
                if c_btn.button("✏ 載入該注單修改", key="load_last_pre"):
                    load_bet_to_edit(st.session_state.last_bet_id)
                    st.rerun()

        with st.expander("✏️ 載入 / 修改既有未結算注單 (Edit Open Bet)"):
            open_bets_list = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
            if open_bets_list.empty:
                st.caption("目前沒有未結算的注單。")
            else:
                opts_open = [f"{r['ID']} | {r['Date']} | {r['Match']} | {r['Bet_Type']} ({r['Selection']})" for _, r in open_bets_list.iterrows()]
                sel_open_bet = st.selectbox("選擇要修改的注單", opts_open, key="sel_open_edit_pre")
                if st.button("📥 載入所選注單資料至表單", key="btn_load_edit_pre"):
                    target_id = sel_open_bet.split(" | ")[0]
                    load_bet_to_edit(target_id)
                    st.rerun()
        st.divider()

        if sys_bankroll <= 0: st.warning("⚠ 目前系統可用資金不足！無法精確計算建議注碼。請先至側邊欄存入本金。")
        
        is_editing = bool(st.session_state.editing_bet_id)
        opts_tournaments = ["➕ 新增手動輸入..."] + sorted(list(set(st.session_state.df_db['Tournament_Name'].dropna().unique())))
        opts_teams = ["➕ 新增手動輸入..."] + sorted(list(set(st.session_state.df_db['Home_Team'].dropna().tolist() + st.session_state.df_db['Away_Team'].dropna().tolist())))
        
        st.markdown("##### 1. 賽事與球隊資料")
        col_t, col_c = st.columns(2)
        
        default_tourn_name = st.session_state.get('edit_t_name', '') if is_editing else ''
        default_tourn_idx = (opts_tournaments.index(default_tourn_name) if default_tourn_name in opts_tournaments else 0) if is_editing else 0
        
        sel_tournament = col_t.selectbox("賽事名稱 (Tournament Name)", opts_tournaments, index=default_tourn_idx, key="sel_tourn")
        tournament_name = col_t.text_input("輸入新賽事名稱", value=default_tourn_name, key="txt_tourn") if sel_tournament == "➕ 新增手動輸入..." else sel_tournament
        
        default_cat_name = st.session_state.get('edit_t_cat', CATEGORY_OPTIONS[0]) if is_editing else CATEGORY_OPTIONS[0]
        default_cat_idx = CATEGORY_OPTIONS.index(default_cat_name) if default_cat_name in CATEGORY_OPTIONS else 0
        
        if not is_editing and sel_tournament != "➕ 新增手動輸入...":
            match_rows = st.session_state.df_db[st.session_state.df_db['Tournament_Name'] == tournament_name]
            if not match_rows.empty:
                last_cat = match_rows.iloc[-1]['Tournament_Category']
                if last_cat in CATEGORY_OPTIONS:
                    default_cat_idx = CATEGORY_OPTIONS.index(last_cat)
                    
        tournament_category = col_c.selectbox("賽事分類 (Tournament Category)", CATEGORY_OPTIONS, index=default_cat_idx, key="sel_cat")

        col_h, col_a = st.columns(2)
        default_h_team = st.session_state.get('edit_h_team', '') if is_editing else ''
        default_h_idx = (opts_teams.index(default_h_team) if default_h_team in opts_teams else 0) if is_editing else 0
        sel_home = col_h.selectbox("主隊名稱", opts_teams, index=default_h_idx, key="sh")
        home_team = col_h.text_input("輸入新主隊", value=default_h_team, key="txt_h_team") if sel_home == "➕ 新增手動輸入..." else sel_home

        default_a_team = st.session_state.get('edit_a_team', '') if is_editing else ''
        default_a_idx = (opts_teams.index(default_a_team) if default_a_team in opts_teams else 0) if is_editing else 0
        sel_away = col_a.selectbox("客隊名稱", opts_teams, index=default_a_idx, key="sa")
        away_team = col_a.text_input("輸入新客隊", value=default_a_team, key="txt_a_team") if sel_away == "➕ 新增手動輸入..." else sel_away

        c_hr, c_ar = st.columns(2)
        ratings_list = ["S", "A", "B", "C", "D"]
        default_hr = st.session_state.get('edit_h_rating', 'C') if is_editing else 'C'
        default_ar = st.session_state.get('edit_a_rating', 'C') if is_editing else 'C'
        idx_hr = ratings_list.index(default_hr) if default_hr in ratings_list else 3
        idx_ar = ratings_list.index(default_ar) if default_ar in ratings_list else 3

        home_rating = c_hr.selectbox("主隊實力", ratings_list, index=idx_hr, key="sel_hr")
        away_rating = c_ar.selectbox("客隊實力", ratings_list, index=idx_ar, key="sel_ar")

        st.markdown("##### 2. 近 5 場狀態 (勝/和/敗)")
        f1, f2, f3, f4, f5, f6 = st.columns(6)
        hw_val = st.session_state.get('edit_hw', 3) if is_editing else 3
        hd_val = st.session_state.get('edit_hd', 1) if is_editing else 1
        hl_val = st.session_state.get('edit_hl', 1) if is_editing else 1
        aw_val = st.session_state.get('edit_aw', 2) if is_editing else 2
        ad_val = st.session_state.get('edit_ad', 2) if is_editing else 2
        al_val = st.session_state.get('edit_al', 1) if is_editing else 1

        home_form = f"{f1.number_input('主勝',0,10,hw_val,key='f1')}W{f2.number_input('主和',0,10,hd_val,key='f2')}D{f3.number_input('主敗',0,10,hl_val,key='f3')}L"
        away_form = f"{f4.number_input('客勝',0,10,aw_val,key='f4')}W{f5.number_input('客和',0,10,ad_val,key='f5')}D{f6.number_input('客敗',0,10,al_val,key='f6')}L"

        st.markdown("##### 3. 賽前盤口與賠率走勢紀錄 (JSON結構儲存)")
        if 'odds_history' not in st.session_state:
            st.session_state.odds_history = [{"id": 0, "type": "讓球", "record_time": "", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}]
        render_odds_section(st.session_state.odds_history, "pre")
        
        st.markdown("---")
        
        if st.button("🚀 賽前數據分析執行", type="primary", use_container_width=True):
            st.session_state.show_analysis = True
            df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
            
            rating_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
            hr_val = rating_map.get(home_rating, 3)
            ar_val = rating_map.get(away_rating, 3)
            
            candidates_base = []
            for idx, r in enumerate(st.session_state.odds_history):
                b_type, line_val = r['type'], float(r['line'])
                rec_time = r.get('record_time', '')
                if b_type == "讓球":
                    p_up = max(0.1, min(0.9, 0.5 + ((hr_val - ar_val) * 0.03)))
                    line_str = f"{line_val:g}"
                    label_h, label_a = (f"{line_str}主隊(上盤)", f"{line_str}客隊(下盤)") if line_val <= 0 else (f"{line_str}主隊(下盤)", f"{line_str}客隊(上盤)")
                    candidates_base.extend([
                        {'bet_type': b_type, 'selection': 'Home', 'base_prob': p_up, 'odds': float(r['upper']), 'line': line_val, 'label': label_h, 'history_idx': idx, 'record_time': rec_time},
                        {'bet_type': b_type, 'selection': 'Away', 'base_prob': 1-p_up, 'odds': float(r['lower']), 'line': line_val, 'label': label_a, 'history_idx': idx, 'record_time': rec_time}
                    ])
                else:
                    p_up = max(0.1, min(0.9, 0.5 + ((hr_val + ar_val - (6 if b_type=="入球大小" else 5)) * (0.02 if b_type=="入球大小" else 0.01))))
                    line_str = f"{line_val:g}"
                    label_over = f"{line_str}大盤(Over)"
                    label_under = f"{line_str}小盤(Under)"
                    candidates_base.extend([
                        {'bet_type': b_type, 'selection': 'Over', 'base_prob': p_up, 'odds': float(r['upper']), 'line': line_val, 'label': label_over, 'history_idx': idx, 'record_time': rec_time},
                        {'bet_type': b_type, 'selection': 'Under', 'base_prob': 1-p_up, 'odds': float(r['lower']), 'line': line_val, 'label': label_under, 'history_idx': idx, 'record_time': rec_time}
                    ])

            h_data = {'hr': hr_val, 'ar': ar_val, 'hf': extract_form_points(home_form), 'af': extract_form_points(away_form)}
            
            df_micro = df_settled[df_settled['Tournament_Name'] == tournament_name]
            df_macro = df_settled[df_settled['Tournament_Category'] == tournament_category]
            df_global = df_settled
            
            res_micro = evaluate_dimension(df_micro, "微觀維度 (同賽事)", candidates_base, rating_map, h_data)
            res_macro = evaluate_dimension(df_macro, "中觀維度 (同賽事分類)", candidates_base, rating_map, h_data)
            res_global = evaluate_dimension(df_global, "宏觀維度 (全域數據庫)", candidates_base, rating_map, h_data)
            
            valid_dims = [d for d in [res_micro, res_macro, res_global] if d['valid']]
            if valid_dims:
                best_dim = max(valid_dims, key=lambda x: x['score'])
            else:
                best_dim = res_global

            st.session_state.analysis_results = {
                'res_micro': res_micro, 'res_macro': res_macro, 'res_global': res_global,
                'best_dim': best_dim, 'h_data': h_data, 'candidates_base': candidates_base
            }

        if st.session_state.get('show_analysis', False) and 'analysis_results' in st.session_state:
            res = st.session_state.analysis_results
            st.markdown("---")
            st.subheader("📊 三維度數據分析與 EV 運算結果")
            
            d_tabs = st.tabs([res['res_micro']['dim'], res['res_macro']['dim'], res['res_global']['dim'], "🏆 最佳維度綜合推薦"])
            dims_list = [res['res_micro'], res['res_macro'], res['res_global']]
            
            for tab_idx, d_obj in enumerate(dims_list):
                with d_tabs[tab_idx]:
                    st.write(f"**狀態**: {d_obj['msg']} (樣本數: {d_obj['n']} 場)")
                    if d_obj['valid']:
                        st.metric("歷史勝率 / ROI", f"{d_obj['acc']*100:.1f}%", f"{d_obj['roi']*100:+.2f}% ROI")
                    
                    df_cand_view = pd.DataFrame([{
                        '盤口與選項': c['label'], '盤口線': c['line'], '賠率': c['odds'], 
                        '模型預測勝率': f"{c['prob']*100:.1f}%", '期望值 (EV)': f"{c['ev']:+.4f}"
                    } for c in d_obj['candidates']])
                    st.dataframe(df_cand_view, use_container_width=True)

            with d_tabs[3]:
                best_d = res['best_dim']
                st.success(f"⭐ 系統採用最佳維度：**{best_d['dim']}** (綜合評分最高)")
                best_cand = best_d['best']
                
                s_stake, r_stake = calc_suggested_stake(best_cand, sys_bankroll, sys_max_stake)
                u_stake = min(float(user_bankroll * 0.10), s_stake) if user_bankroll > 0 else 0.0

                st.markdown(f"### 🎯 AI 預測模型推薦決策")
                st.info(f"**推薦投注選項**: `{best_cand['label']}` | **賠率**: `{best_cand['odds']}` | **模型勝率**: `{best_cand['prob']*100:.1f}%` | **期望值 (EV)**: `{best_cand['ev']:+.4f}`")
                
                col_st1, col_st2 = st.columns(2)
                col_st1.metric("🤖 系統建議注碼 (Kelly)", f"${s_stake:,.2f}", f"原始凱利: ${r_stake:,.2f}")
                col_st2.metric("👤 用家建議注碼參考", f"${u_stake:,.2f}", f"可用資金: ${user_bankroll:,.2f}")

                st.markdown("##### 📥 確認並寫入投注記錄 (Commit Bet)")
                with st.form("commit_bet_form"):
                    f_sys_stake = st.number_input("系統投注金額 (System Stake)", min_value=0.0, value=float(s_stake), step=10.0)
                    f_user_stake = st.number_input("用家真實投注金額 (User Stake)", min_value=0.0, value=float(u_stake), step=10.0)
                    
                    sub_btn = st.form_submit_button("💾 確認提交注單並寫入資料庫", type="primary")
                    if sub_btn:
                        match_title = f"{home_team} vs {away_team}"
                        bet_id = st.session_state.editing_bet_id if st.session_state.editing_bet_id else f"B{get_hkt_now().strftime('%Y%m%d%H%M%S')}"
                        
                        odds_hist_json = json.dumps(st.session_state.odds_history, ensure_ascii=False)
                        
                        new_record = {
                            'ID': bet_id,
                            'Date': get_hkt_now().strftime('%Y-%m-%d %H:%M'),
                            'Status': 'Open',
                            'Tournament_Name': tournament_name,
                            'Tournament_Category': tournament_category,
                            'Match': match_title,
                            'Home_Team': home_team,
                            'Away_Team': away_team,
                            'Home_Rating': home_rating,
                            'Away_Rating': away_rating,
                            'Home_Form': home_form,
                            'Away_Form': away_form,
                            'Bet_Type': best_cand['bet_type'],
                            'Selection': best_cand['selection'],
                            'Initial_Line': float(best_cand['line']),
                            'Initial_Odds': float(best_cand['odds']),
                            'System_Stake': float(f_sys_stake),
                            'User_Stake': float(f_user_stake),
                            'Odds_History': odds_hist_json,
                            'InPlay_Minute': 0, 'Home_DA': 0, 'Away_DA': 0, 'Home_SoT': 0, 'Away_SoT': 0,
                            'Home_SoFF': 0, 'Away_SoFF': 0, 'Home_Red': 0, 'Away_Red': 0, 'Home_Sub': 0, 'Away_Sub': 0,
                            'Home_Possession': 50, 'Away_Possession': 50, 'Home_Goal': 0, 'Away_Goal': 0,
                            'Home_Corner': 0, 'Away_Corner': 0, 'Home_Goal_Conversion': 0.0, 'Away_Goal_Conversion': 0.0,
                            'Home_Firepower': 0.0, 'Away_Firepower': 0.0,
                            'Result_Label': '', 'System_Profit': 0.0, 'User_Profit': 0.0, 'Unit_Profit': 0.0, 'System_Payout': 0.0, 'User_Payout': 0.0
                        }

                        if st.session_state.editing_bet_id:
                            st.session_state.df_db = st.session_state.df_db[st.session_state.df_db['ID'] != bet_id]
                        
                        st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([new_record])], ignore_index=True)
                        save_db(st.session_state.df_db, db_file, db_table)
                        
                        st.session_state.last_bet_id = bet_id
                        clear_edit_mode()
                        st.toast(f"✅ 注單 {bet_id} 成功寫入雲端數據庫！", icon="🚀")
                        st.rerun()

    with t_inplay:
        st.subheader("⏱️️ 即場賽事實時更新與智慧火力分析")
        
        open_bets_inplay = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
        if open_bets_inplay.empty:
            st.info("目前沒有進行中的未結算賽事可供即場更新。請先於「賽前建檔與投注」建立賽事。")
        else:
            inplay_opts = [f"{r['ID']} | {r['Match']} (當前盤: {r['Bet_Type']} - {r['Selection']})" for _, r in open_bets_inplay.iterrows()]
            sel_inplay_match_str = st.selectbox("選擇要進行即場更新與分析的賽事", inplay_opts, key="sel_inplay_match")
            target_inplay_id = sel_inplay_match_str.split(" | ")[0]
            
            match_row = open_bets_inplay[open_bets_inplay['ID'] == target_inplay_id].iloc[0]
            
            st.markdown(f"#### 🏟️ 賽事: **{match_row['Match']}** | 原始賽前投注: `{match_row['Bet_Type']} ({match_row['Selection']})` @ `{match_row['Initial_Odds']}`")
            
            st.markdown("##### 1. 比賽即時動態數據輸入")
            col_min, col_hg, col_ag = st.columns(3)
            curr_minute = col_min.number_input("比賽進行分鐘數 (Minute)", min_value=1, max_value=120, value=int(match_row.get('InPlay_Minute', 45)), key="ip_min")
            curr_hg = col_hg.number_input("主隊即時入球 (Home Goals)", min_value=0, max_value=20, value=int(match_row.get('Home_Goal', 0)), key="ip_hg")
            curr_ag = col_ag.number_input("客隊即時入球 (Away Goals)", min_value=0, max_value=20, value=int(match_row.get('Away_Goal', 0)), key="ip_ag")

            col_hc, col_ac = st.columns(2)
            curr_hc = col_hc.number_input("主隊角球 (Home Corners)", min_value=0, max_value=30, value=int(match_row.get('Home_Corner', 0)), key="ip_hc")
            curr_ac = col_ac.number_input("客隊角球 (Away Corners)", min_value=0, max_value=30, value=int(match_row.get('Away_Corner', 0)), key="ip_ac")

            st.markdown("##### 統計與進攻指標")
            c1, c2, c3, c4 = st.columns(4)
            curr_hda = c1.number_input("主隊危險進攻 (DA)", 0, 200, int(match_row.get('Home_DA', 0)), key="ip_hda")
            curr_ada = c2.number_input("客隊危險進攻 (DA)", 0, 200, int(match_row.get('Away_DA', 0)), key="ip_ada")
            curr_hsot = c3.number_input("主隊射正 (SoT)", 0, 50, int(match_row.get('Home_SoT', 0)), key="ip_hsot")
            curr_asot = c4.number_input("客隊射正 (SoT)", 0, 50, int(match_row.get('Away_SoT', 0)), key="ip_asot")

            c5, c6, c7, c8 = st.columns(4)
            curr_hsoff = c5.number_input("主隊射斜 (SoFF)", 0, 50, int(match_row.get('Home_SoFF', 0)), key="ip_hsoff")
            curr_asoff = c6.number_input("客隊射斜 (SoFF)", 0, 50, int(match_row.get('Away_SoFF', 0)), key="ip_asoff")
            curr_hred = c7.number_input("主隊紅牌", 0, 3, int(match_row.get('Home_Red', 0)), key="ip_hred")
            curr_ared = c8.number_input("客隊紅牌", 0, 3, int(match_row.get('Away_Red', 0)), key="ip_ared")

            c9, c10 = st.columns(2)
            curr_hsub = c9.number_input("主隊換人", 0, 5, int(match_row.get('Home_Sub', 0)), key="ip_hsub")
            curr_asub = c10.number_input("客隊換人", 0, 5, int(match_row.get('Away_Sub', 0)), key="ip_asub")
            curr_hposs = st.slider("主隊控球率 (%)", 0, 100, int(match_row.get('Home_Possession', 50)), key="ip_hposs")

            # --- 自動計算進攻效率指標 ---
            h_conv = (curr_hsot / curr_hda * 100) if curr_hda > 0 else 0.0
            a_conv = (curr_asot / curr_ada * 100) if curr_ada > 0 else 0.0
            h_fire = (curr_hsot * 1.5 + curr_hsoff * 0.8 + curr_hc * 0.5) / (curr_minute / 90.0)
            a_fire = (curr_asot * 1.5 + curr_asoff * 0.8 + curr_ac * 0.5) / (curr_minute / 90.0)
            
            # 主客隊進攻產生角球效率 = 角球數量 / 危險進攻數量 * 100%
            h_corner_eff = (curr_hc / curr_hda * 100) if curr_hda > 0 else 0.0
            a_corner_eff = (curr_ac / curr_ada * 100) if curr_ada > 0 else 0.0

            st.markdown("---")
            st.markdown("##### 📈 實時進攻效率與指標看板")
            m_inf1, m_inf2, m_inf3, m_inf4 = st.columns(4)
            m_inf1.metric("主隊射正轉換率", f"{h_conv:.1f}%", f"火力指數: {h_fire:.2f}")
            m_inf2.metric("客隊射正轉換率", f"{a_conv:.1f}%", f"火力指數: {a_fire:.2f}")
            m_inf3.metric("主隊進攻產生角球效率", f"{h_corner_eff:.2f}%", f"角球:{curr_hc} / 危險進攻:{curr_hda}")
            m_inf4.metric("客隊進攻產生角球效率", f"{a_corner_eff:.2f}%", f"角球:{curr_ac} / 危險進攻:{curr_ada}")

            st.divider()

            # ==========================================
            # 第三項：即場投注 (提供即時盤口與馬會抽水自動計算)
            # ==========================================
            st.markdown("##### 3. 🎯 即場投注 (In-Play Betting & Margin Calculator)")
            st.caption("提供即時「入球、角球、讓球」盤口，具備手動/自動抽水計算功能。讓球盤口線預設為0，入球大小預設2.5，角球預設9.5。")

            if 'inplay_odds_history' not in st.session_state:
                try:
                    loaded_oh = json.loads(str(match_row.get('Odds_History', '[]')))
                    st.session_state.inplay_odds_history = loaded_oh if loaded_oh else [
                        {"id": 0, "type": "讓球", "record_time": get_hkt_now().strftime('%H:%M'), "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": True, "margin": 1.085},
                        {"id": 1, "type": "入球大小", "record_time": get_hkt_now().strftime('%H:%M'), "line": 2.5, "upper": 1.90, "lower": 1.90, "unlock": True, "margin": 1.085},
                        {"id": 2, "type": "角球大小", "record_time": get_hkt_now().strftime('%H:%M'), "line": 9.5, "upper": 1.90, "lower": 1.90, "unlock": True, "margin": 1.085}
                    ]
                except:
                    st.session_state.inplay_odds_history = [
                        {"id": 0, "type": "讓球", "record_time": "", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": True, "margin": 1.085},
                        {"id": 1, "type": "入球大小", "record_time": "", "line": 2.5, "upper": 1.90, "lower": 1.90, "unlock": True, "margin": 1.085},
                        {"id": 2, "type": "角球大小", "record_time": "", "line": 9.5, "upper": 1.90, "lower": 1.90, "unlock": True, "margin": 1.085}
                    ]

            # 互動式即場賠率與抽水計算區塊
            st.markdown("###### ⚙️ 即場賠率自動計算與手動切換互動設定")
            st.info("💡 **操作說明**：當您在「大/上賠率」輸入數值後，系統會自動根據馬會抽水公式（Margin = 1.085）計算出「小/下賠率」。若您直接修改「小/下賠率」，系統將自動覆蓋為您的手動數值並重新計算當前抽水百分比。")

            updated_inplay_odds = []
            for idx, item in enumerate(st.session_state.inplay_odds_history):
                with st.expander(f"盤口 #{idx+1} | 類型: {item.get('type', '讓球')} | 盤口線: {item.get('line', 0.0)}", expanded=(idx==0)):
                    c_t, c_l, c_u, c_d = st.columns(4)
                    ip_type = c_t.selectbox("盤口類型", ["讓球", "入球大小", "角球大小"], index=["讓球", "入球大小", "角球大小"].index(item.get('type', '讓球')) if item.get('type', '讓球') in ["讓球", "入球大小", "角球大小"] else 0, key=f"ip_type_{idx}")
                    default_line_val = 0.0 if ip_type == "讓球" else (2.5 if ip_type == "入球大小" else 9.5)
                    ip_line = c_l.number_input("讓球/大小盤口線", value=float(item.get('line', default_line_val)), step=0.25, key=f"ip_line_{idx}")
                    
                    # 大/上賠率輸入
                    ip_upper = c_u.number_input("大 / 上盤賠率", min_value=1.01, value=float(item.get('upper', 1.90)), step=0.01, format="%.2f", key=f"ip_upper_{idx}")
                    
                    # 馬會抽水公式自動計算對應的小/下賠率: 1 / (1.085 - 1/大賠率)
                    auto_lower = 1.90
                    try:
                        if ip_upper > 1 / 1.085:
                            auto_lower = round(1.0 / (1.085 - (1.0 / ip_upper)), 2)
                    except:
                        auto_lower = 1.90

                    # 取得當前 session_state 中使用者可能手動修改過的值
                    current_lower_input = st.session_state.get(f"ip_lower_{idx}", float(item.get('lower', auto_lower)))
                    
                    # 若使用者剛修改了大賠率，自動同步更新小賠率
                    if f"ip_upper_prev_{idx}" not in st.session_state:
                        st.session_state[f"ip_upper_prev_{idx}"] = ip_upper
                    
                    if st.session_state[f"ip_upper_prev_{idx}"] != ip_upper:
                        st.session_state[f"ip_upper_prev_{idx}"] = ip_upper
                        current_lower_input = auto_lower
                        st.session_state[f"ip_lower_{idx}"] = auto_lower

                    ip_lower = c_d.number_input("小 / 下盤賠率 (可手動覆蓋)", min_value=1.01, value=float(current_lower_input), step=0.01, format="%.2f", key=f"ip_lower_{idx}")
                    
                    # 計算抽水與百分比
                    margin_val = (1.0 / ip_upper) + (1.0 / ip_lower) if (ip_upper > 0 and ip_lower > 0) else 1.085
                    margin_pct = (margin_val - 1.0) * 100.0
                    st.caption(f"📊 當前盤口抽水 (Margin): **{margin_pct:.2f}%** (總和倍率: `{margin_val:.3f}`)")

                    updated_inplay_odds.append({
                        "id": idx,
                        "type": ip_type,
                        "record_time": get_hkt_now().strftime('%H:%M'),
                        "line": ip_line,
                        "upper": ip_upper,
                        "lower": ip_lower,
                        "unlock": True,
                        "margin": round(margin_val, 3)
                    })

            st.session_state.inplay_odds_history = updated_inplay_odds

            st.markdown("---")
            
            # --- 即場數據分析執行功能 (套用 tab-1 功能與 EV 運算) ---
            if st.button("🚀 即場數據分析執行 (In-Play EV & AI Analysis)", type="primary", use_container_width=True):
                st.session_state.show_inplay_analysis = True
                
                df_settled_ip = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
                rating_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
                hr_val = rating_map.get(match_row.get('Home_Rating', 'C'), 3)
                ar_val = rating_map.get(match_row.get('Away_Rating', 'C'), 3)
                
                inplay_candidates_base = []
                for idx, r in enumerate(st.session_state.inplay_odds_history):
                    b_type, line_val = r['type'], float(r['line'])
                    rec_time = r.get('record_time', '')
                    
                    # 結合剩餘時間與攻勢危險度計算期望值機率
                    time_factor = max(0.1, (90 - curr_minute) / 90.0)
                    da_diff_ratio = (curr_hda - curr_ada) / max(1, (curr_hda + curr_ada))
                    
                    if b_type == "讓球":
                        p_up = max(0.1, min(0.9, 0.5 + (da_diff_ratio * 0.3) + ((hr_val - ar_val) * 0.03)))
                        line_str = f"{line_val:g}"
                        label_h, label_a = (f"{line_str}主隊(上盤)", f"{line_str}客隊(下盤)") if line_val <= 0 else (f"{line_str}主隊(下盤)", f"{line_str}客隊(上盤)")
                        inplay_candidates_base.extend([
                            {'bet_type': f"{b_type} (即場)", 'selection': 'Home', 'base_prob': p_up, 'odds': float(r['upper']), 'line': line_val, 'label': label_h, 'history_idx': idx, 'record_time': rec_time},
                            {'bet_type': f"{b_type} (即場)", 'selection': 'Away', 'base_prob': 1-p_up, 'odds': float(r['lower']), 'line': line_val, 'label': label_a, 'history_idx': idx, 'record_time': rec_time}
                        ])
                    else:
                        goal_pressure = (curr_hg + curr_ag) / max(1, curr_minute / 15.0)
                        p_up = max(0.1, min(0.9, 0.5 + (goal_pressure * 0.15) + ((h_fire - a_fire) * 0.05)))
                        line_str = f"{line_val:g}"
                        label_over = f"{line_str}大盤(Over)"
                        label_under = f"{line_str}小盤(Under)"
                        inplay_candidates_base.extend([
                            {'bet_type': f"{b_type} (即場)", 'selection': 'Over', 'base_prob': p_up, 'odds': float(r['upper']), 'line': line_val, 'label': label_over, 'history_idx': idx, 'record_time': rec_time},
                            {'bet_type': f"{b_type} (即場)", 'selection': 'Under', 'base_prob': 1-p_up, 'odds': float(r['lower']), 'line': line_val, 'label': label_under, 'history_idx': idx, 'record_time': rec_time}
                        ])

                h_data = {'hr': hr_val, 'ar': ar_val, 'hf': extract_form_points(match_row.get('Home_Form', '3W1D1L')), 'af': extract_form_points(match_row.get('Away_Form', '2W2D1L'))}
                
                df_micro_ip = df_settled_ip[df_settled_ip['Tournament_Name'] == match_row['Tournament_Name']]
                df_macro_ip = df_settled_ip[df_settled_ip['Tournament_Category'] == match_row['Tournament_Category']]
                df_global_ip = df_settled_ip
                
                res_micro_ip = evaluate_dimension(df_micro_ip, "微觀維度 (同賽事)", inplay_candidates_base, rating_map, h_data)
                res_macro_ip = evaluate_dimension(df_macro_ip, "中觀維度 (同賽事分類)", inplay_candidates_base, rating_map, h_data)
                res_global_ip = evaluate_dimension(df_global_ip, "宏觀維度 (全域數據庫)", inplay_candidates_base, rating_map, h_data)
                
                valid_dims_ip = [d for d in [res_micro_ip, res_macro_ip, res_global_ip] if d['valid']]
                best_dim_ip = max(valid_dims_ip, key=lambda x: x['score']) if valid_dims_ip else res_global_ip

                st.session_state.inplay_analysis_results = {
                    'res_micro': res_micro_ip, 'res_macro': res_macro_ip, 'res_global': res_global_ip,
                    'best_dim': best_dim_ip, 'h_data': h_data, 'candidates_base': inplay_candidates_base
                }

            if st.session_state.get('show_inplay_analysis', False) and 'inplay_analysis_results' in st.session_state:
                res_ip = st.session_state.inplay_analysis_results
                st.markdown("---")
                st.subheader("📊 即場三維度數據分析與 EV 運算明細")
                
                ip_tabs = st.tabs([res_ip['res_micro']['dim'], res_ip['res_macro']['dim'], res_ip['res_global']['dim'], "🏆 即場最佳綜合推薦"])
                ip_dims_list = [res_ip['res_micro'], res_ip['res_macro'], res_ip['res_global']]
                
                for tab_idx, d_obj in enumerate(ip_dims_list):
                    with ip_tabs[tab_idx]:
                        st.write(f"**狀態**: {d_obj['msg']} (樣本數: {d_obj['n']} 場)")
                        if d_obj['valid']:
                            st.metric("歷史勝率 / ROI", f"{d_obj['acc']*100:.1f}%", f"{d_obj['roi']*100:+.2f}% ROI")
                        
                        df_cand_view_ip = pd.DataFrame([{
                            '即場盤口與選項': c['label'], '盤口線': c['line'], '賠率': c['odds'], 
                            '即場預測勝率': f"{c['prob']*100:.1f}%", '期望值 (EV)': f"{c['ev']:+.4f}"
                        } for c in d_obj['candidates']])
                        st.dataframe(df_cand_view_ip, use_container_width=True)

                with ip_tabs[3]:
                    best_d_ip = res_ip['best_dim']
                    st.success(f"⭐ 即場最佳分析維度：**{best_d_ip['dim']}**")
                    best_cand_ip = best_d_ip['best']
                    
                    s_stake_ip, r_stake_ip = calc_suggested_stake(best_cand_ip, sys_bankroll, sys_max_stake)
                    u_stake_ip = min(float(user_bankroll * 0.10), s_stake_ip) if user_bankroll > 0 else 0.0

                    st.markdown(f"### 🎯 即場 AI 預測模型推薦決策")
                    st.info(f"**推薦即場選項**: `{best_cand_ip['label']}` | **賠率**: `{best_cand_ip['odds']}` | **預測勝率**: `{best_cand_ip['prob']*100:.1f}%` | **EV**: `{best_cand_ip['ev']:+.4f}`")
                    
                    col_ist1, col_ist2 = st.columns(2)
                    col_ist1.metric("🤖 系統建議注碼", f"${s_stake_ip:,.2f}", f"凱利原始值: ${r_stake_ip:,.2f}")
                    col_ist2.metric("👤 用家建議注碼參考", f"${u_stake_ip:,.2f}", f"可用資金: ${user_bankroll:,.2f}")

                    st.markdown("##### 📥 確認即場投注決策並更新實時數據")
                    with st.form("commit_inplay_bet_form"):
                        f_sys_stake_ip = st.number_input("系統即場投注金額", min_value=0.0, value=float(s_stake_ip), step=10.0, key="fis_ip")
                        f_user_stake_ip = st.number_input("用家真實即場投注金額", min_value=0.0, value=float(u_stake_ip), step=10.0, key="fui_ip")
                        
                        sub_ip_btn = st.form_submit_button("💾 確認更新實時數據並寫入即場注單", type="primary")
                        if sub_ip_btn:
                            # 更新該筆資料庫中的即場動態數據
                            idx_to_update = st.session_state.df_db.index[st.session_state.df_db['ID'] == match_row['ID']].tolist()
                            if idx_to_update:
                                i_idx = idx_to_update[0]
                                st.session_state.df_db.at[i_idx, 'InPlay_Minute'] = curr_minute
                                st.session_state.df_db.at[i_idx, 'Home_Goal'] = curr_hg
                                st.session_state.df_db.at[i_idx, 'Away_Goal'] = curr_ag
                                st.session_state.df_db.at[i_idx, 'Home_Corner'] = curr_hc
                                st.session_state.df_db.at[i_idx, 'Away_Corner'] = curr_ac
                                st.session_state.df_db.at[i_idx, 'Home_DA'] = curr_hda
                                st.session_state.df_db.at[i_idx, 'Away_DA'] = curr_ada
                                st.session_state.df_db.at[i_idx, 'Home_SoT'] = curr_hsot
                                st.session_state.df_db.at[i_idx, 'Away_SoT'] = curr_asot
                                st.session_state.df_db.at[i_idx, 'Home_SoFF'] = curr_hsoff
                                st.session_state.df_db.at[i_idx, 'Away_SoFF'] = curr_asoff
                                st.session_state.df_db.at[i_idx, 'Home_Red'] = curr_hred
                                st.session_state.df_db.at[i_idx, 'Away_Red'] = curr_ared
                                st.session_state.df_db.at[i_idx, 'Home_Sub'] = curr_hsub
                                st.session_state.df_db.at[i_idx, 'Away_Sub'] = curr_asub
                                st.session_state.df_db.at[i_idx, 'Home_Possession'] = curr_hposs
                                st.session_state.df_db.at[i_idx, 'Home_Goal_Conversion'] = round(h_conv, 2)
                                st.session_state.df_db.at[i_idx, 'Away_Goal_Conversion'] = round(a_conv, 2)
                                st.session_state.df_db.at[i_idx, 'Home_Firepower'] = round(h_fire, 2)
                                st.session_state.df_db.at[i_idx, 'Away_Firepower'] = round(a_fire, 2)
                                
                                # 若用家選擇額外下注即場盤口，可新增一筆新注單
                                if f_sys_stake_ip > 0 or f_user_stake_ip > 0:
                                    new_ip_bet = match_row.copy()
                                    new_ip_bet['ID'] = f"IP{get_hkt_now().strftime('%Y%m%d%H%M%S')}"
                                    new_ip_bet['Bet_Type'] = best_cand_ip['bet_type']
                                    new_ip_bet['Selection'] = best_cand_ip['selection']
                                    new_ip_bet['Initial_Line'] = float(best_cand_ip['line'])
                                    new_ip_bet['Initial_Odds'] = float(best_cand_ip['odds'])
                                    new_ip_bet['System_Stake'] = float(f_sys_stake_ip)
                                    new_ip_bet['User_Stake'] = float(f_user_stake_ip)
                                    st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([new_ip_bet])], ignore_index=True)

                                save_db(st.session_state.df_db, db_file, db_table)
                                st.toast("✅ 即場實時數據與注單已成功同步更新至雲端數據庫！", icon="⏱️")
                                st.rerun()

    with t_settle:
        st.subheader("⚖ 賽果結算與注單管理")
        display_cumulative_metrics(st.session_state.df_db)
        
        open_bets_settle = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
        if open_bets_settle.empty:
            st.info("目前沒有需要結算的未結算注單 (Open Bets)。")
        else:
            st.markdown("##### 📝 待結算注單列表 (Open Bets Settlement)")
            for _, row in open_bets_settle.iterrows():
                with st.container():
                    st.markdown(f"**ID**: `{row['ID']}` | **賽事**: `{row['Match']}` | **盤口**: `{row['Bet_Type']} ({row['Selection']})` @ `{row['Initial_Odds']}` (線: {row['Initial_Line']})")
                    st.caption(f"系統注碼: ${row['System_Stake']:,.2f} \vert{} 用家注碼: ${row['User_Stake']:,.2f}")
                    
                    c_hg, c_ag, c_hc, c_ac = st.columns(4)
                    res_hg = c_hg.number_input("主隊全場入球", 0, 20, int(row.get('Home_Goal', 0)), key=f"s_hg_{row['ID']}")
                    res_ag = c_ag.number_input("客隊全場入球", 0, 20, int(row.get('Away_Goal', 0)), key=f"s_ag_{row['ID']}")
                    res_hc = c_hc.number_input("主隊全場角球", 0, 50, int(row.get('Home_Corner', 0)), key=f"s_hc_{row['ID']}")
                    res_ac = c_ac.number_input("客隊全場角球", 0, 50, int(row.get('Away_Corner', 0)), key=f"s_ac_{row['ID']}")
                    
                    if st.button(f"⚖ 確認結算此注單 ({row['ID']})", key=f"btn_settle_{row['ID']}", type="primary"):
                        s_prof, u_prof, s_pay, u_pay, u_pfit, label, diff = calculate_settlement(
                            row['Bet_Type'], row['Selection'], float(row['Initial_Line']), 
                            float(row['Initial_Odds']), float(row['System_Stake']), float(row['User_Stake']), 
                            res_hg, res_ag, res_hc, res_ac
                        )
                        
                        idx_match = st.session_state.df_db.index[st.session_state.df_db['ID'] == row['ID']].tolist()
                        if idx_match:
                            i = idx_match[0]
                            st.session_state.df_db.at[i, 'Status'] = 'Settled'
                            st.session_state.df_db.at[i, 'Home_Goal'] = res_hg
                            st.session_state.df_db.at[i, 'Away_Goal'] = res_ag
                            st.session_state.df_db.at[i, 'Home_Corner'] = res_hc
                            st.session_state.df_db.at[i, 'Away_Corner'] = res_ac
                            st.session_state.df_db.at[i, 'Result_Label'] = label
                            st.session_state.df_db.at[i, 'System_Profit'] = s_prof
                            st.session_state.df_db.at[i, 'User_Profit'] = u_prof
                            st.session_state.df_db.at[i, 'Unit_Profit'] = u_pfit
                            st.session_state.df_db.at[i, 'System_Payout'] = s_pay
                            st.session_state.df_db.at[i, 'User_Payout'] = u_pay
                            
                            save_db(st.session_state.df_db, db_file, db_table)
                            st.toast(f"✅ 注單 {row['ID']} 結算完成！結果: {label}", icon="⚖")
                            st.rerun()
                    st.divider()

    with t_ai:
        st.subheader("🤖 全局機器學習與策略效能檢視")
        df_all = st.session_state.df_db
        df_settled_all = df_all[df_all['Status'] == 'Settled']
        
        if len(df_settled_all) < 10:
            st.warning("⚠️ 已結算歷史樣本數小於 10 場，機器學習模型與策略效能分析可能不夠精確。建議多累積賽事紀錄。")
        else:
            st.success(f"✅ 當前模型有效訓練樣本數：**{len(df_settled_all)}** 場")
            total_sys_pnl_all = pd.to_numeric(df_settled_all['System_Profit'], errors='coerce').sum()
            total_usr_pnl_all = pd.to_numeric(df_settled_all['User_Profit'], errors='coerce').sum()
            win_count = len(df_settled_all[pd.to_numeric(df_settled_all['Unit_Profit'], errors='coerce') > 0])
            overall_acc = win_count / len(df_settled_all) if len(df_settled_all) > 0 else 0
            
            am1, am2, am3 = st.columns(3)
            am1.metric("整體預測勝率", f"{overall_acc*100:.1f}%")
            am2.metric("系統總淨利", f"${total_sys_pnl_all:,.2f}")
            am3.metric("用家真實總淨利", f"${total_usr_pnl_all:,.2f}")

if __name__ == '__main__':
    main()
