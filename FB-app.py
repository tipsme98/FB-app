import streamlit as st
import pandas as pd
import os
import json
import base64
import re
import io
import requests
from datetime import datetime, timedelta
import time

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
st.set_page_config(page_title="Actuarial and fund management system by Dr. EdwinPro", page_icon="⚽", layout="wide")

DB_COLUMNS = [
    'ID', 'Tipsme_ID', 'Date', 'Status', 
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

def load_db_github(repo, path, token):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
    res = requests.get(url, headers=headers)
    if res.status_code == 200: return pd.read_csv(io.StringIO(res.text))
    return None

def save_db_github(df, repo, path, token):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}"}
    res_get = requests.get(url, headers=headers)
    sha = res_get.json().get("sha") if res_get.status_code == 200 else None
    csv_content = df.to_csv(index=False)
    content_b64 = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
    payload = {"message": f"Auto-update {path} via Streamlit App", "content": content_b64}
    if sha: payload["sha"] = sha
    requests.put(url, json=payload, headers=headers)

def process_legacy_columns(df):
    if 'Stake' in df.columns and 'System_Stake' not in df.columns:
        df['System_Stake'] = df['Stake']; df['User_Stake'] = df['Stake']
    if 'Profit' in df.columns and 'System_Profit' not in df.columns:
        df['System_Profit'] = df['Profit']; df['User_Profit'] = df['Profit']
    if 'Payout' in df.columns and 'System_Payout' not in df.columns:
        df['System_Payout'] = df['Payout']; df['User_Payout'] = df['Payout']
    return df

def enforce_columns(df, columns):
    if 'Account' in columns and 'Account' not in df.columns: df['Account'] = 'Both'
    for col in columns:
        if col not in df.columns: df[col] = pd.Series(dtype='object')
    return df[columns]

def load_db(filename, columns, table_name):
    string_cols = ['ID', 'Tipsme_ID', 'Date', 'Status', 'Tournament_Name', 'Tournament_Category', 'Match', 'Home_Team', 'Away_Team', 'Home_Rating', 'Away_Rating', 'Home_Form', 'Away_Form', 'Bet_Type', 'Selection', 'Odds_History', 'Result_Label', 'Type', 'Account', 'Note']
    df = pd.DataFrame()
    if HAS_SQLALCHEMY and "DB_URL" in st.secrets and st.secrets["DB_URL"]:
        try:
            engine = create_engine(st.secrets["DB_URL"])
            df = pd.read_sql_table(table_name, engine)
            df = process_legacy_columns(df)
        except: pass

    if df.empty and "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets:
        try:
            gh_df = load_db_github(st.secrets["GITHUB_REPO"], filename, st.secrets["GITHUB_TOKEN"])
            if gh_df is not None: df = process_legacy_columns(gh_df)
        except: pass

    if df.empty and os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            if 'League' in df.columns and 'Tournament_Name' not in df.columns:
                df['Tournament_Name'] = df['League']; df['Tournament_Category'] = CATEGORY_OPTIONS[0]
            df = process_legacy_columns(df)
        except: pass
            
    if df.empty: df = pd.DataFrame(columns=columns)
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
                if df_to_db[col].dtype == 'object': df_to_db[col] = df_to_db[col].apply(lambda x: str(x) if pd.notna(x) else None)
            df_to_db.to_sql(table_name, engine, if_exists='replace', index=False)
        except: pass

    if "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets:
        try: save_db_github(df, st.secrets["GITHUB_REPO"], filename, st.secrets["GITHUB_TOKEN"])
        except: pass

    try: df.to_csv(filename, index=False)
    except: pass

# ==========================================
# 1.5 自動化抓取 API 模組 (深度擴展路由池)
# ==========================================
def fetch_api_data(url):
    # 強化防禦繞過
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "Origin": "https://www.tipsme.hk",
        "Referer": "https://www.tipsme.hk/",
        "Connection": "keep-alive"
    }
    try:
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200: 
            try:
                return response.json(), 200
            except:
                return None, "JSON Parse Error"
        return None, response.status_code
    except Exception as e:
        return None, str(e)

def get_match_odds(match_id):
    # 擴展多種可能的賠率端點
    urls = [
        f"https://api.tipsme.hk/api/v1/match/{match_id}/odds",
        f"https://api.tipsme.hk/api/Score/matchOdds/{match_id}",
        f"https://api.tipsme.hk/api/Score/odds/hkjc/{match_id}",
        f"https://api.tipsme.hk/api/Score/hkjc/odds/{match_id}",
        f"https://api.tipsme.hk/api/Score/match/{match_id}/odds",
        f"https://api.tipsme.hk/api/Score/odds/{match_id}",
        f"https://api.tipsme.hk/api/Score/odds/macau/{match_id}",
        f"https://tipsme-web.azurewebsites.net/api/Score/odds/hkjc/{match_id}",
        f"https://tipsme-web.azurewebsites.net/api/v1/match/{match_id}/odds"
    ]
    for url in urls:
        data, status = fetch_api_data(url)
        if status == 200 and data: return data
    return None

def get_match_fixtures(match_id):
    # 擴展多種可能的賽事資訊端點
    urls = [
        f"https://api.tipsme.hk/api/v1/match/{match_id}",
        f"https://api.tipsme.hk/api/Score/matchInfo/{match_id}",
        f"https://api.tipsme.hk/api/Score/match/{match_id}",
        f"https://api.tipsme.hk/api/Score/match/detail/{match_id}",
        f"https://api.tipsme.hk/api/Score/fixtures/{match_id}",
        f"https://tipsme-web.azurewebsites.net/api/Score/matchInfo/{match_id}"
    ]
    for url in urls:
        data, status = fetch_api_data(url)
        if status == 200 and data: return data
    return None

def get_matches_schedule(date_str):
    date_nodash = date_str.replace("-", "")
    # 擴展多種可能的賽程表端點，解決 404 問題
    urls = [
        f"https://api.tipsme.hk/api/v1/match/schedule/hkjc/{date_str}",
        f"https://api.tipsme.hk/api/v1/schedule/hkjc/{date_str}",
        f"https://api.tipsme.hk/api/Score/schedule/hkjc/{date_str}",
        f"https://api.tipsme.hk/api/Score/schedule/hkjc/{date_nodash}",
        f"https://api.tipsme.hk/api/Score/schedule/{date_str}",
        f"https://api.tipsme.hk/api/Score/schedule/hkjc?date={date_str}",
        f"https://tipsme-web.azurewebsites.net/api/Score/schedule/hkjc/{date_str}",
        f"https://tipsme-web.azurewebsites.net/api/v1/match/schedule/hkjc/{date_str}"
    ]
    for url in urls:
        data, status = fetch_api_data(url)
        if status == 200 and data: 
            if isinstance(data, dict):
                if 'data' in data: data = data['data']
                elif 'list' in data: data = data['list']
                elif 'matches' in data: data = data['matches']
            if isinstance(data, list) and len(data) > 0:
                return data, 200
    return None, 404

def extract_odds_history(odds_data):
    """深度遞迴搜尋 JSON，強制找出所有的賠率與盤口數據，並抓取變動時間"""
    new_history = []
    
    def find_odds_arrays(node):
        found = {}
        if isinstance(node, dict):
            for k, v in node.items():
                k_lower = k.lower()
                # 尋找典型的盤口陣列
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    if any(pk in k_lower for pk in ["letting", "hdc", "ah", "handicap", "asian", "讓球"]):
                        found["讓球"] = v
                    elif any(pk in k_lower for pk in ["ou", "hil", "overunder", "total", "goals", "入球"]):
                        found["入球大小"] = v
                    elif any(pk in k_lower for pk in ["corner", "chl", "corners", "角球"]):
                        found["角球大小"] = v
                elif isinstance(v, dict):
                    # 遞迴往下層尋找
                    sub_found = find_odds_arrays(v)
                    for sk, sv in sub_found.items():
                        if sk not in found: found[sk] = sv
        elif isinstance(node, list):
            for item in node:
                sub_found = find_odds_arrays(item)
                for sk, sv in sub_found.items():
                    if sk not in found: found[sk] = sv
        return found

    arrays = find_odds_arrays(odds_data)
    row_id = 0
    
    for bet_type_cn, target_array in arrays.items():
        if not isinstance(target_array, list): continue
        
        for item in target_array:
            if not isinstance(item, dict): continue
            
            def get_val(d, keys, default):
                for k, v in d.items():
                    if k.lower() in keys and v is not None:
                        try:
                            if isinstance(default, float):
                                s = str(v).replace('[', '').replace(']', '').replace('球', '').strip()
                                if '/' in s:
                                    parts = s.split('/')
                                    return (float(parts[0]) + float(parts[1])) / 2
                                return float(re.sub(r'[^\d\.\-]', '', s))
                            elif isinstance(default, str):
                                return str(v)
                        except: pass
                return default

            line = get_val(item, ['line', 'goal', 'p', 'handicap', 'matchgoal', 'ratio', 'goals', 'point'], 0.0)
            upper = get_val(item, ['h', 'home', 'homeodds', 'up', 'upper', 'over', 'overodds', 'high', 'h_odds', '大', '主'], 1.90)
            lower = get_val(item, ['a', 'away', 'awayodds', 'low', 'lower', 'under', 'underodds', 'a_odds', '小', '客'], 1.90)
            
            # 抓取並格式化時間
            time_str = get_val(item, ['time', 'updatedat', 'modifytime', 'date', 'updatetime'], "")
            if time_str:
                try:
                    time_str = pd.to_datetime(str(time_str).replace('T', ' ').replace('Z', '')).strftime("%d-%m %H:%M")
                except:
                    time_str = str(time_str)[:16]
            
            # 過濾無效資料
            if upper != 1.90 or lower != 1.90 or line != 0.0:
                new_history.append({
                    "id": row_id, 
                    "type": bet_type_cn, 
                    "time": time_str,
                    "line": line, 
                    "upper": upper, 
                    "lower": lower, 
                    "unlock": True
                })
                row_id += 1
                
    return new_history

def parse_and_fill_pre_match(match_id, default_h='', default_a='', default_l='', fallback_data=None):
    fixtures_data = get_match_fixtures(match_id)
    if not fixtures_data and fallback_data:
        fixtures_data = fallback_data
        
    odds_data = get_match_odds(match_id)
    success = False
    details = {}
    
    # 解析球隊資料
    h_name, a_name, l_name = default_h, default_a, default_l
    if fixtures_data:
        if isinstance(fixtures_data, dict) and 'data' in fixtures_data:
            fixtures_data = fixtures_data['data']
        if isinstance(fixtures_data, list) and len(fixtures_data) > 0:
            fixtures_data = fixtures_data[0]
            
        if isinstance(fixtures_data, dict):
            h_name = fixtures_data.get('homeName') or fixtures_data.get('home') or fixtures_data.get('homeTeamName') or h_name
            a_name = fixtures_data.get('awayName') or fixtures_data.get('away') or fixtures_data.get('awayTeamName') or a_name
            l_name = fixtures_data.get('leagueName') or fixtures_data.get('league') or fixtures_data.get('tournamentName') or l_name
            
    details = {'h': h_name, 'a': a_name, 'l': l_name}
    if h_name or a_name: success = True

    # 獲取賠率 (不再強塞假資料，防止批量同步污染 DB)
    odds_history_res = []
    if odds_data:
        new_history = extract_odds_history(odds_data)
        if new_history:
            odds_history_res = new_history
            success = True 
            
    # 如果抓不到，檢查 fallback_data 內是否隱含簡易賠率
    if not odds_history_res and fallback_data:
        new_history = extract_odds_history({"fallback": fallback_data})
        if new_history: odds_history_res = new_history

    return success, details, odds_history_res

# ==========================================
# 2. 資金、風控與累計算式
# ==========================================
def recalculate_bankroll_from_scratch(df_cap, df_db):
    if df_cap.empty: sys_dep = sys_wit = usr_dep = usr_wit = 0.0
    else:
        sys_cap = df_cap[df_cap['Account'].isin(['System', 'Both'])]
        usr_cap = df_cap[df_cap['Account'].isin(['User', 'Both'])]
        sys_dep = pd.to_numeric(sys_cap[sys_cap['Type'] == 'Deposit']['Amount'], errors='coerce').sum()
        sys_wit = pd.to_numeric(sys_cap[sys_cap['Type'] == 'Withdraw']['Amount'], errors='coerce').sum()
        usr_dep = pd.to_numeric(usr_cap[usr_cap['Type'] == 'Deposit']['Amount'], errors='coerce').sum()
        usr_wit = pd.to_numeric(usr_cap[usr_cap['Type'] == 'Withdraw']['Amount'], errors='coerce').sum()
    
    sys_net = max(0.0, sys_dep - sys_wit)
    usr_net = max(0.0, usr_dep - usr_wit)
    
    if df_db.empty: sys_profit = user_profit = sys_open = user_open = 0.0
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
    
    return (round(sys_dep, 2), round(sys_wit, 2), round(sys_net, 2), round(sys_profit, 2), round(sys_bankroll, 2), round(sys_max_stake, 2),
            round(usr_dep, 2), round(usr_wit, 2), round(usr_net, 2), round(user_profit, 2), round(user_bankroll, 2), round(user_max_stake, 2))

def calculate_settlement(bet_type, selection, line, odds, sys_stake, user_stake, h_g, a_g, h_c=0, a_c=0):
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
        res_label = "✅ 全贏"; sys_profit = sys_stake * (odds - 1); user_profit = user_stake * (odds - 1); unit_profit = round(odds - 1.0, 2)
        sys_payout = sys_stake + sys_profit; user_payout = user_stake + user_profit
    elif diff == 0.25:
        res_label = "🟢 贏半"; sys_profit = sys_stake * (odds - 1) / 2; user_profit = user_stake * (odds - 1) / 2; unit_profit = round((odds - 1.0) / 2.0, 2)
        sys_payout = sys_stake + sys_profit; user_payout = user_stake + user_profit
    elif diff == 0.0:
        res_label = "⚪ 走盤退本"; sys_profit = user_profit = 0.0; sys_payout = sys_stake; user_payout = user_stake; unit_profit = 0.0
    elif diff == -0.25:
        res_label = "🔴 輸半"; sys_profit = -sys_stake / 2; user_profit = -user_stake / 2; sys_payout = sys_stake / 2; user_payout = user_stake / 2; unit_profit = -0.50
    else:
        res_label = "❌ 全輸"; sys_profit = -sys_stake; user_profit = -user_stake; sys_payout = user_payout = 0.0; unit_profit = -1.00

    return round(sys_profit, 2), round(user_profit, 2), round(sys_payout, 2), round(user_payout, 2), unit_profit, res_label, diff

def display_cumulative_metrics(df):
    sys_profit = pd.to_numeric(df['System_Profit'], errors='coerce').sum()
    user_profit = pd.to_numeric(df['User_Profit'], errors='coerce').sum()
    user_payout = pd.to_numeric(df['User_Payout'], errors='coerce').sum()
    st.markdown("### 📊 數據庫累計總額看板 (Cumulative Summary)")
    m1, m2, m3 = st.columns(3)
    m1.metric("系統累積淨盈虧 (System Profit)", f"${sys_profit:,.2f}", delta=f"{sys_profit:,.2f}")
    m2.metric("用家真實淨盈虧 (User Profit)", f"${user_profit:,.2f}", delta=f"{user_profit:,.2f}")
    m3.metric("用家派彩總額 (User Payout)", f"${user_payout:,.2f}")
    st.divider()

def extract_form_points(form_str):
    try:
        w = int(re.search(r'(\d+)W', str(form_str)).group(1))
        d = int(re.search(r'(\d+)D', str(form_str)).group(1))
        return w * 3 + d * 1
    except: return 0

def parse_form_str(f_str):
    try: w = int(re.search(r'(\d+)W', str(f_str)).group(1))
    except: w = 0
    try: d = int(re.search(r'(\d+)D', str(f_str)).group(1))
    except: d = 0
    try: l = int(re.search(r'(\d+)L', str(f_str)).group(1))
    except: l = 0
    return w, d, l

def prepare_ml_dataset(df, rating_map):
    X, y = [], []
    for _, r in df.iterrows():
        try:
            hr = rating_map.get(r.get('Home_Rating', 'C'), 3); ar = rating_map.get(r.get('Away_Rating', 'C'), 3)
            hf = extract_form_points(r.get('Home_Form', '0W0D0L')); af = extract_form_points(r.get('Away_Form', '0W0D0L'))
            line = float(r.get('Initial_Line', 0)); odds = float(r.get('Initial_Odds', 1.90))
            X.append([hr, ar, hf, af, line, odds]); y.append(1 if float(r.get('Unit_Profit', 0)) > 0 else 0)
        except: continue
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
    else: roi = acc = 0

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
            except: pass

    if not model_success:
        for c in candidates:
            shift = ((acc - 0.5) * 0.2 + (roi * 0.1)) if valid else 0
            c['prob'] = max(0.05, min(0.95, c['base_prob'] + shift))

    for c in candidates: c['ev'] = c['prob'] * (c['odds'] - 1) - (1 - c['prob'])
    candidates = sorted(candidates, key=lambda x: x['ev'], reverse=True)
    score = (roi * 0.7) + (acc * 0.3) if valid else -1
    return {'dim': dim_name, 'valid': valid, 'msg': msg, 'roi': roi, 'acc': acc, 'n': n_samples, 'candidates': candidates, 'best': candidates[0], 'score': score}

def clear_edit_mode():
    st.session_state.editing_bet_id = None
    for k in list(st.session_state.keys()):
        if k.startswith('edit_'): del st.session_state[k]

def render_odds_section(odds_list, prefix="pre"):
    """渲染最終投注盤口的編輯區 (單一橫列)"""
    for i, row in enumerate(odds_list):
        r_id = row.get('id', i)
        up_key, type_key, unlock_key = f"{prefix}_up_{r_id}", f"{prefix}_t_{r_id}", f"{prefix}_u_{r_id}"
        low_key, line_key = f"{prefix}_low_{r_id}", f"{prefix}_line_{r_id}"
        
        c1, c2, c3, c4, c5, c6 = st.columns([2, 1.5, 1.5, 1.5, 1.5, 1])
        type_idx = ["讓球", "入球大小", "角球大小"].index(row['type']) if row['type'] in ["讓球", "入球大小", "角球大小"] else 0
        row['type'] = c1.selectbox(f"盤口類型", ["讓球", "入球大小", "角球大小"], key=type_key, index=type_idx)
        
        row['line'] = c2.number_input("盤口線", step=0.25, value=float(row['line']), key=line_key)
        up_lbl, low_lbl = ("主隊", "客隊") if row['type'] == "讓球" else ("大盤", "小盤")
        row['upper'] = c3.number_input(f"{up_lbl} 賠率", value=float(row['upper']), step=0.01, key=up_key)
        row['unlock'] = c4.checkbox("解鎖", value=row.get('unlock', False), key=unlock_key)
        
        if not row['unlock']: row['lower'] = c5.number_input(f"{low_lbl} 賠率", value=float(row['lower']), disabled=True, key=low_key)
        else: row['lower'] = c5.number_input(f"{low_lbl} 賠率", value=float(row['lower']), step=0.01, key=low_key)
            
        if c6.button("❌", key=f"{prefix}_del_{r_id}"):
            odds_list.pop(i)
            st.rerun()

def load_bet_to_edit(bet_id):
    match_df = st.session_state.df_db[st.session_state.df_db['ID'] == bet_id]
    if match_df.empty: return
    row = match_df.iloc[0]
    st.session_state.editing_bet_id = str(bet_id)
    st.session_state.edit_t_name = str(row.get('Tournament_Name', ''))
    st.session_state.edit_t_cat = str(row.get('Tournament_Category', CATEGORY_OPTIONS[0]))
    st.session_state.edit_h_team = str(row.get('Home_Team', ''))
    st.session_state.edit_a_team = str(row.get('Away_Team', ''))
    st.session_state.edit_h_rating = str(row.get('Home_Rating', 'C'))
    st.session_state.edit_a_rating = str(row.get('Away_Rating', 'C'))
    
    st.session_state.edit_hw, st.session_state.edit_hd, st.session_state.edit_hl = parse_form_str(row.get('Home_Form', '3W1D1L'))
    st.session_state.edit_aw, st.session_state.edit_ad, st.session_state.edit_al = parse_form_str(row.get('Away_Form', '2W2D1L'))
    
    try:
        oh = json.loads(str(row.get('Odds_History', '[]')))
        if isinstance(oh, list) and len(oh) > 0: 
            st.session_state.odds_history = oh
            # 取出最新的作為當前投注依據
            latest = {}
            for r in oh: latest[r['type']] = r.copy()
            st.session_state.current_odds = list(latest.values())
    except: pass
    
    st.session_state.edit_user_stake = float(row.get('User_Stake', 0.0))
    st.session_state.edit_bet_type = str(row.get('Bet_Type', '')).replace(" (即場)", "")
    st.session_state.edit_selection = str(row.get('Selection', 'Home'))

# ==========================================
# 6. 主程式 UI 
# ==========================================
def main():
    st.title("⚽ Actuarial and fund management system by Dr. EdwinPro")
    
    if 'undo_stack' not in st.session_state: st.session_state.undo_stack = []
    if 'editing_bet_id' not in st.session_state: st.session_state.editing_bet_id = None
    if 'odds_history' not in st.session_state: st.session_state.odds_history = []
    if 'current_odds' not in st.session_state: st.session_state.current_odds = [{"id": 0, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False}]
        
    db_file, capital_file = "football_betting_db.csv", "football_capital_db.csv"
    db_table, cap_table = "football_bets", "football_cap"
    
    st.session_state.df_db = load_db(db_file, DB_COLUMNS, db_table)
    st.session_state.df_cap = load_db(capital_file, CAPITAL_COLUMNS, cap_table)

    (sys_dep, sys_wit, sys_net, sys_pnl, sys_bankroll, sys_max_stake, 
     usr_dep, usr_wit, usr_net, usr_pnl, usr_bankroll, usr_max_stake) = recalculate_bankroll_from_scratch(st.session_state.df_cap, st.session_state.df_db)

    # 側邊欄與資料庫管理按鈕
    st.sidebar.header("⚙️ 系統設定與資金管理")
    st.sidebar.metric("系統可用資金 (Bankroll)", f"${sys_bankroll:,.2f}")
    st.sidebar.caption(f"🛑 系統單注上限: `${sys_max_stake:,.2f}`")
    
    st.sidebar.divider()
    if st.sidebar.button("📂 開啟/隱藏資料庫 (View DB)", use_container_width=True):
        st.session_state.show_db_viewer = not st.session_state.get('show_db_viewer', False)

    # 資料庫檢視區塊
    if st.session_state.get('show_db_viewer', False):
        st.markdown("### 🗄️ 系統資料庫即時檢視")
        st.dataframe(st.session_state.df_db, use_container_width=True)
        csv_data = st.session_state.df_db.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下載完整資料庫 (CSV)", data=csv_data, file_name="football_betting_db.csv", mime="text/csv")
        st.divider()

    t_pre, t_inplay, t_settle, t_ai = st.tabs(["📝 賽前建檔與投注", "⏱️ 即場賽事與預測", "⚖️ 賽果結算", "🤖 全局模型"])

    with t_pre:
        pre_t1, pre_t2, pre_t3 = st.tabs(["⚡ 單場一鍵抓取", "📅 按日期批量同步", "🎯 智能高價值推薦"])

        with pre_t1:
            st.markdown("##### ⚡ 一鍵智能抓取賽前數據與盤口走勢")
            col_id, col_btn = st.columns([2, 1])
            target_match_id = col_id.text_input("請輸入 Tipsme 賽事 ID (例如: 112684)", key="api_match_id")
            if col_btn.button("📥 獲取球隊與全盤口", use_container_width=True):
                if target_match_id:
                    target_match_id = target_match_id.strip()
                    with st.spinner('正在從 Tipsme 抓取數據與深度解析盤口...'):
                        success, details, odds_res = parse_and_fill_pre_match(target_match_id)
                        if success:
                            st.session_state.editing_bet_id = f"API_{target_match_id}"
                            st.session_state.current_tipsme_id = target_match_id
                            st.session_state.edit_h_team = details.get('h', '')
                            st.session_state.edit_a_team = details.get('a', '')
                            st.session_state.edit_t_name = details.get('l', '')
                            st.session_state.odds_history = odds_res
                            
                            # 從歷史走勢中提取最新盤口作為使用者目前編輯的依據
                            latest = {}
                            for r in odds_res: latest[r['type']] = r.copy()
                            if latest:
                                st.session_state.current_odds = list(latest.values())
                            else:
                                st.session_state.current_odds = [{"id": 0, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False}]
                            
                            st.success(f"✅ 成功載入賽事 {target_match_id}！歷史盤口與球隊資料已自動填入下方。")
                            time.sleep(1) # 讓使用者看見成功訊息
                            st.rerun()
                        else: 
                            st.error(f"❌ 抓取失敗。可能原因：(1) 賽事 ID 錯誤 (2) 該賽事尚無開盤資料 (3) 官方 API 已阻擋。")
                else: st.warning("請先輸入賽事 ID。")

        with pre_t2:
            st.markdown("##### 📅 自動掃描與批量建檔")
            st.caption("系統將自動抓取該日所有賽程，並深度解析所有盤口寫入資料庫。(已包含防阻擋降速機制)")
            c_date, c_sync = st.columns([2, 1])
            target_date = c_date.date_input("選擇賽事日期", value=datetime.today())
            date_str = target_date.strftime("%Y-%m-%d")
            
            if c_sync.button("🔄 同步該日所有賽事", type="primary", use_container_width=True):
                with st.spinner(f"正在與伺服器連線並掃描 {date_str} 賽事列表..."):
                    schedule_data, status_code = get_matches_schedule(date_str)
                    
                    if not schedule_data or not isinstance(schedule_data, list):
                        st.error(f"❌ 無法取得該日賽程表 (狀態碼: {status_code})。可能是 API 路由已變更或被阻擋。")
                    else:
                        seen_ids = set()
                        unique_valid_items = []
                        for item in schedule_data:
                            mid = str(item.get('matchId', item.get('id', '')))
                            if mid and mid not in seen_ids:
                                unique_valid_items.append(item)
                                seen_ids.add(mid)

                        if not unique_valid_items:
                            st.warning(f"⚠️ 找到了 {len(schedule_data)} 場賽事，但未能解析出有效的賽事 ID。")
                        else:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            success_count = 0
                            
                            for i, item in enumerate(unique_valid_items):
                                m_id = str(item.get('matchId', item.get('id', '')))
                                dh = item.get('homeName', item.get('home', item.get('homeTeamName', '')))
                                da = item.get('awayName', item.get('away', item.get('awayTeamName', '')))
                                dl = item.get('leagueName', item.get('league', item.get('tournamentName', '')))
                                
                                status_text.text(f"正在同步: {m_id} ({i+1}/{len(unique_valid_items)})...")
                                
                                success, details, odds_res = parse_and_fill_pre_match(m_id, default_h=dh, default_a=da, default_l=dl, fallback_data=item)
                                
                                if success and odds_res: # 確保有抓到盤口才算成功
                                    success_count += 1
                                    current_odds_json = json.dumps(odds_res, ensure_ascii=False)
                                    match_title = f"{details.get('h', 'Unknown')} vs {details.get('a', 'Unknown')}"
                                    
                                    # 抓取最新一筆盤口當作初始值存入 DB
                                    latest_odds = {}
                                    for r in odds_res: latest_odds[r['type']] = r
                                    init_l = float(latest_odds.get('讓球', {}).get('line', 0.0))
                                    init_u = float(latest_odds.get('讓球', {}).get('upper', 1.90))
                                    
                                    if not (st.session_state.df_db['Tipsme_ID'] == m_id).any():
                                        new_rec = {
                                            'ID': f"B{datetime.now().strftime('%Y%m%d%H%M%S%f')}", 
                                            'Tipsme_ID': m_id, 'Date': date_str, 'Status': 'Open', 
                                            'Match': match_title, 'Tournament_Name': details.get('l', ''), 
                                            'Odds_History': current_odds_json,
                                            'Initial_Line': init_l, 'Initial_Odds': init_u
                                        }
                                        st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([new_rec])], ignore_index=True)
                                    else:
                                        mask = st.session_state.df_db['Tipsme_ID'] == m_id
                                        st.session_state.df_db.loc[mask, 'Odds_History'] = current_odds_json
                                        st.session_state.df_db.loc[mask, 'Initial_Line'] = init_l
                                        st.session_state.df_db.loc[mask, 'Initial_Odds'] = init_u
                                        
                                progress_bar.progress((i + 1) / len(unique_valid_items))
                                time.sleep(0.8) # 延長延遲以避開 Rate Limit
                                
                            save_db(st.session_state.df_db, db_file, db_table)
                            st.success(f"✅ 批量同步完成！針對 {date_str} 掃描了 {len(unique_valid_items)} 場，成功寫入 {success_count} 場賽事的歷史走勢。")

        with pre_t3:
            st.markdown("##### 🎯 AI 系統自動推薦 (勝率達標且 EV > 0)")
            if st.button("🚀 掃描未開賽賽事尋找價值盤口"):
                open_matches = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
                if open_matches.empty: st.info("資料庫目前沒有 Open 狀態的未開賽賽事。請先執行批量同步。")
                else:
                    recommendations = []
                    rating_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
                    with st.spinner("模型高速運算中..."):
                        for _, row in open_matches.iterrows():
                            try:
                                history = json.loads(str(row.get('Odds_History', '[]')))
                                if not history: continue
                            except: continue
                            
                            # 取出各盤口的最新賠率
                            latest = {}
                            for r in history: latest[r['type']] = r
                            
                            hr_val = rating_map.get(row.get('Home_Rating', 'C'), 3)
                            ar_val = rating_map.get(row.get('Away_Rating', 'C'), 3)
                            
                            for b_type, r in latest.items():
                                line_val = float(r['line'])
                                u_odds, l_odds = float(r['upper']), float(r['lower'])
                                if b_type == "讓球":
                                    p_up = max(0.1, min(0.9, 0.5 + ((hr_val - ar_val) * 0.03)))
                                    ev_up = p_up * (u_odds - 1) - (1 - p_up)
                                    ev_down = (1 - p_up) * (l_odds - 1) - p_up
                                    if ev_up > 0.02 and p_up > 0.5: recommendations.append({"賽事": row['Match'], "盤口": f"讓球 [{line_val:+g}]", "推薦": "主隊(上盤)", "勝率": f"{p_up*100:.1f}%", "EV": f"{ev_up:.3f}", "賠率": u_odds})
                                    elif ev_down > 0.02 and (1-p_up) > 0.5: recommendations.append({"賽事": row['Match'], "盤口": f"讓球 [{line_val:+g}]", "推薦": "客隊(下盤)", "勝率": f"{(1-p_up)*100:.1f}%", "EV": f"{ev_down:.3f}", "賠率": l_odds})
                    if recommendations:
                        st.success(f"🔥 系統發現 {len(recommendations)} 個具備投資價值的黃金盤口！")
                        df_rec = pd.DataFrame(recommendations).sort_values(by="EV", ascending=False)
                        st.dataframe(df_rec, use_container_width=True, hide_index=True)
                    else: st.warning("⚠️ 目前各大盤口水位正常，未發現明顯高 EV 價值的賽事。")

        st.divider()
        
        # --- 賽事手動建檔與 AI 分析表單 ---
        is_editing = bool(st.session_state.editing_bet_id)
        st.markdown("##### 1. 賽事與球隊資料")
        col_t, col_c = st.columns(2)
        tournament_name = col_t.text_input("賽事名稱", value=st.session_state.get('edit_t_name', ''))
        tournament_category = col_c.selectbox("賽事分類", CATEGORY_OPTIONS, index=0)

        col_h, col_a = st.columns(2)
        home_team = col_h.text_input("主隊", value=st.session_state.get('edit_h_team', ''))
        away_team = col_a.text_input("客隊", value=st.session_state.get('edit_a_team', ''))

        c_hr, c_ar = st.columns(2)
        home_rating = c_hr.selectbox("主隊實力", ["S", "A", "B", "C", "D"], index=3)
        away_rating = c_ar.selectbox("客隊實力", ["S", "A", "B", "C", "D"], index=3)

        st.markdown("##### 2. 近 5 場狀態 (勝/和/敗)")
        f1, f2, f3, f4, f5, f6 = st.columns(6)
        hw_val, hd_val, hl_val = st.session_state.get('edit_hw', 3), st.session_state.get('edit_hd', 1), st.session_state.get('edit_hl', 1)
        aw_val, ad_val, al_val = st.session_state.get('edit_aw', 2), st.session_state.get('edit_ad', 2), st.session_state.get('edit_al', 1)
        home_form = f"{f1.number_input('主勝',0,10,hw_val)}W{f2.number_input('主和',0,10,hd_val)}D{f3.number_input('主敗',0,10,hl_val)}L"
        away_form = f"{f4.number_input('客勝',0,10,aw_val)}W{f5.number_input('客和',0,10,ad_val)}D{f6.number_input('客敗',0,10,al_val)}L"

        # --- 歷史走勢純展示區塊 (如 Tipsme 般表列) ---
        st.markdown("##### 3. 賽前盤口與賠率走勢紀錄")
        if 'odds_history' in st.session_state and st.session_state.odds_history:
            df_hist = pd.DataFrame(st.session_state.odds_history)
            if not df_hist.empty:
                for b_type in df_hist['type'].unique():
                    sub_df = df_hist[df_hist['type'] == b_type].copy()
                    sub_df['時間'] = sub_df.get('time', '').fillna('-')
                    sub_df['盤口線'] = sub_df['line'].apply(lambda x: f"[{x:+g}]")
                    sub_df['主/大'] = sub_df['upper'].apply(lambda x: f"{float(x):.2f}")
                    sub_df['客/小'] = sub_df['lower'].apply(lambda x: f"{float(x):.2f}")
                    
                    st.markdown(f"**🟢 {b_type} 走勢**")
                    st.dataframe(sub_df[['時間', '主/大', '盤口線', '客/小']], use_container_width=True, hide_index=True)
        else:
            st.caption("尚無歷史走勢資料。")

        # --- 最終投注編輯區 ---
        st.markdown("##### ✏️ 確認最終投注盤口 (AI 計算基準)")
        if 'current_odds' not in st.session_state or not st.session_state.current_odds: 
            st.session_state.current_odds = [{"id": 0, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False}]
        render_odds_section(st.session_state.current_odds, "pre")
        
        st.markdown("---")
        
        # --- AI 模型精算與結算寫入表單 ---
        if st.button("🚀 賽前數據分析執行", type="primary", use_container_width=True):
            st.session_state.show_analysis = True
            df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
            rating_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
            hr_val = rating_map.get(home_rating, 3); ar_val = rating_map.get(away_rating, 3)
            
            # 使用 current_odds (最後確認的盤口) 進行運算
            candidates_base = []
            for r in st.session_state.current_odds:
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
            res_macro = evaluate_dimension(df_settled, "宏觀 - 總數據", candidates_base, rating_map, h_data)
            
            best_bet = res_macro['best'] if 'best' in res_macro else candidates_base[0]
            suggested_stake = 0
            if 'ev' in best_bet and best_bet['ev'] > 0 and sys_bankroll > 0:
                b = best_bet['odds'] - 1
                kelly = max(0.0, min((best_bet['prob'] * b - (1 - best_bet['prob'])) / b, 0.10))
                suggested_stake = min(float(sys_max_stake), float(round((sys_bankroll * (kelly * 0.5)) / 10) * 10))
                
            st.session_state.analysis_result = {'best_bet': best_bet, 'stake': max(10.0, suggested_stake), 't_name': tournament_name, 't_cat': tournament_category}
            
        if st.session_state.get('show_analysis', False):
            res = st.session_state.analysis_result
            bb = res['best_bet']
            st.success("✅ EV 運算完成！")
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("💡 首選推薦", f"{bb['bet_type']} - {bb['label']}")
            mc2.metric(f"🎯 預期勝率", f"{bb.get('prob', bb.get('base_prob',0))*100:.1f}%")
            mc3.metric("📊 修正 EV", f"{bb.get('ev', 0):.3f}")

            with st.form("bet_form"):
                bc1, bc2 = st.columns(2)
                final_btype = bc1.selectbox("最終投注項目", [c['type'] for c in st.session_state.current_odds])
                final_sel = bc2.selectbox("最終投注方向", ["Home", "Away", "Over", "Under"])
                
                bc3, bc4 = st.columns(2)
                bc3.text_input("🤖 系統建議下注金額", f"${res['stake']:,.2f}", disabled=True)
                final_user_stake = bc4.number_input("👤 真實下注金額 ($)", min_value=0.0, step=10.0, value=float(res['stake']))
                
                submit_btn_label = f"🔄 確定修改覆蓋 (ID: {st.session_state.editing_bet_id})" if is_editing else "✅ 確定投注並寫入雲端資料庫"
                
                if st.form_submit_button(submit_btn_label):
                    target_id = st.session_state.get('editing_bet_id')
                    new_id = target_id if target_id and target_id.startswith('B') else f"B{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    tipsme_id = st.session_state.get('current_tipsme_id', '')
                    final_row = next((r for r in st.session_state.current_odds if r['type'] == final_btype), st.session_state.current_odds[-1])
                    line, odds = float(final_row['line']), float(final_row['upper']) if final_sel in ["Home", "Over"] else float(final_row['lower'])
                    
                    new_record = {
                        'ID': new_id, 'Tipsme_ID': tipsme_id, 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Status': 'Open',
                        'Tournament_Name': tournament_name, 'Tournament_Category': tournament_category, 
                        'Match': f"{home_team} vs {away_team}", 'Home_Team': home_team, 'Away_Team': away_team,
                        'Home_Rating': home_rating, 'Away_Rating': away_rating, 'Home_Form': home_form, 'Away_Form': away_form,
                        'Bet_Type': final_btype, 'Selection': final_sel, 'Initial_Line': line, 'Initial_Odds': odds, 
                        'System_Stake': res['stake'], 'User_Stake': final_user_stake,
                        'Odds_History': json.dumps(st.session_state.odds_history, ensure_ascii=False) # 保存完整歷史紀錄
                    }
                    
                    if target_id and (st.session_state.df_db['ID'] == target_id).any():
                        mask = st.session_state.df_db['ID'] == target_id
                        for col_name, val in new_record.items(): st.session_state.df_db.loc[mask, col_name] = val
                    else:
                        st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([new_record])], ignore_index=True)
                        
                    save_db(st.session_state.df_db, db_file, db_table)
                    st.success("✅ 注單同步成功！")
                    clear_edit_mode(); st.session_state.show_analysis = False; st.rerun()

    # --- 賽果結算分頁 ---
    with t_settle:
        st.subheader("⚖️ 賽果結算與資料庫維護")
        display_cumulative_metrics(st.session_state.df_db)
        open_bets = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
        if not open_bets.empty:
            for idx, row in open_bets.iterrows():
                with st.expander(f"📌 {row['Match']} - {row['Bet_Type']} ({row['Selection']}) | 盤口: {row['Initial_Line']}"):
                    col_b1, col_b2 = st.columns([3, 1])
                    col_b1.caption(f"注單 ID: `{row['ID']}` | 投注金額: 用家 ${row.get('User_Stake',0)} / 系統 ${row.get('System_Stake',0)}")
                    if col_b2.button("✏️ 載入此注單修改資料", key=f"btn_edit_settle_{row['ID']}"):
                        load_bet_to_edit(row['ID'])
                        st.success(f"已成功載入注單 `{row['ID']}`！請切換至「📝 賽前建檔」進行修改。")
                        
                    with st.form(f"settle_form_{row['ID']}"):
                        st.markdown("##### ⚽ 全場賽果輸入")
                        col1, col2 = st.columns(2)
                        h_g = col1.number_input("全場主隊入球數", min_value=0, value=int(row.get('Home_Goal', 0)) if pd.notna(row.get('Home_Goal')) else 0, key=f"hg_{row['ID']}")
                        a_g = col2.number_input("全場客隊入球數", min_value=0, value=int(row.get('Away_Goal', 0)) if pd.notna(row.get('Away_Goal')) else 0, key=f"ag_{row['ID']}")
                        col3, col4 = st.columns(2)
                        h_c = col3.number_input("全場主隊角球數", min_value=0, value=int(row.get('Home_Corner', 0)) if pd.notna(row.get('Home_Corner')) else 0, key=f"hc_{row['ID']}")
                        a_c = col4.number_input("全場客隊角球數", min_value=0, value=int(row.get('Away_Corner', 0)) if pd.notna(row.get('Away_Corner')) else 0, key=f"ac_{row['ID']}")
                        
                        if st.form_submit_button("確認賽果並雲端結算"):
                            sys_p, usr_p, sys_pay, usr_pay, u_prof, lbl, diff = calculate_calculate(
                                row['Bet_Type'], row['Selection'], float(row['Initial_Line']), float(row['Initial_Odds']), 
                                float(row.get('System_Stake', 0)), float(row.get('User_Stake', 0)), h_g, a_g, h_c, a_c
                            )
                            st.session_state.df_db['Result_Label'] = st.session_state.df_db['Result_Label'].astype('object')
                            st.session_state.df_db['Status'] = st.session_state.df_db['Status'].astype('object')
                            st.session_state.df_db.loc[idx, 'Home_Goal'] = h_g
                            st.session_state.df_db.loc[idx, 'Away_Goal'] = a_g
                            st.session_state.df_db.loc[idx, 'Home_Corner'] = h_c
                            st.session_state.df_db.loc[idx, 'Away_Corner'] = a_c
                            st.session_state.df_db.loc[idx, 'Result_Label'] = str(lbl)
                            st.session_state.df_db.loc[idx, 'System_Profit'] = float(sys_p)
                            st.session_state.df_db.loc[idx, 'User_Profit'] = float(usr_p)
                            st.session_state.df_db.loc[idx, 'System_Payout'] = float(sys_pay)
                            st.session_state.df_db.loc[idx, 'User_Payout'] = float(usr_pay)
                            st.session_state.df_db.loc[idx, 'Unit_Profit'] = float(u_prof)
                            st.session_state.df_db.loc[idx, 'Status'] = 'Settled'
                            
                            save_db(st.session_state.df_db, db_file, db_table)
                            st.success(f"結算完成！結果：{lbl}")
                            st.rerun()

    # --- 策略分析看板 ---
    with t_ai:
        st.subheader("🤖 全局機器學習模型與策略分析")
        df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled']
        
        if not df_settled.empty:
            df_chart = df_settled.copy()
            df_chart['Cum_Sys_Profit'] = pd.to_numeric(df_chart['System_Profit'], errors='coerce').cumsum()
            df_chart['Cum_User_Profit'] = pd.to_numeric(df_chart['User_Profit'], errors='coerce').cumsum()
            st.line_chart(df_chart[['Cum_Sys_Profit', 'Cum_User_Profit']])
        else:
            st.caption("尚無已結算數據可供展示走勢圖。")

if __name__ == "__main__":
    main()
