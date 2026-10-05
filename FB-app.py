import streamlit as st
import pandas as pd
import os
import json
import base64
import re
import io
import requests
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
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return pd.read_csv(io.StringIO(res.text))
    return None

def save_db_github(df, repo, path, token):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}"}
    res_get = requests.get(url, headers=headers)
    sha = res_get.json().get("sha") if res_get.status_code == 200 else None
    
    csv_content = df.to_csv(index=False)
    content_b64 = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
    
    payload = {
        "message": f"Auto-update {path} via Streamlit App",
        "content": content_b64
    }
    if sha:
        payload["sha"] = sha
        
    requests.put(url, json=payload, headers=headers)

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

def load_db(filename, columns, table_name):
    string_cols = [
        'ID', 'Date', 'Status', 'Tournament_Name', 'Tournament_Category', 
        'Match', 'Home_Team', 'Away_Team', 'Home_Rating', 'Away_Rating', 
        'Home_Form', 'Away_Form', 'Bet_Type', 'Selection', 'Odds_History', 'Result_Label',
        'Type', 'Account', 'Note'
    ]
    
    df = pd.DataFrame()
    # 策略 A: 嘗試 PostgreSQL 雲端資料庫
    if HAS_SQLALCHEMY and "DB_URL" in st.secrets and st.secrets["DB_URL"]:
        try:
            engine = create_engine(st.secrets["DB_URL"])
            df = pd.read_sql_table(table_name, engine)
            df = process_legacy_columns(df)
        except Exception:
            pass

    # 策略 B: 嘗試 GitHub API 自動同步
    if df.empty and "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets:
        try:
            gh_df = load_db_github(st.secrets["GITHUB_REPO"], filename, st.secrets["GITHUB_TOKEN"])
            if gh_df is not None:
                df = process_legacy_columns(gh_df)
        except Exception:
            pass

    # 策略 C: 本地 CSV 讀取防呆
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
        except Exception:
            pass

    if "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets:
        try:
            save_db_github(df, st.secrets["GITHUB_REPO"], filename, st.secrets["GITHUB_TOKEN"])
        except Exception:
            pass

    try:
        df.to_csv(filename, index=False)
    except Exception:
        pass

# ==========================================
# 1.5 自動化抓取 API 模組 (Auto-Scraper)
# ==========================================
def fetch_api_data(url):
    """通用的 API 請求函數"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def get_match_odds(match_id):
    """抓取馬會賠率變化"""
    url = f"https://tipsme-web.azurewebsites.net/api/Score/odds/hkjc/{match_id}"
    return fetch_api_data(url)

def get_match_fixtures(match_id):
    """抓取即場賽況及統計數據 (同時適用於賽果)"""
    url = f"https://tipsme-web.azurewebsites.net/api/Score/fixtures/{match_id}"
    return fetch_api_data(url)

def extract_odds_history(odds_data):
    """強大的盤口/賠率走勢萃取器，負責解構深層次 JSON 並轉換成系統需要的格式"""
    new_history = []
    row_id = 0
    
    def parse_line(line_str):
        try:
            line_str = str(line_str).replace('[', '').replace(']', '').replace('球', '')
            if '/' in line_str:
                parts = line_str.split('/')
                return (float(parts[0]) + float(parts[1])) / 2
            return float(line_str)
        except:
            return 0.0
            
    def parse_odds_val(val):
        try:
            clean_val = re.sub(r'[^\d\.]', '', str(val))
            return float(clean_val)
        except:
            return 1.90
            
    def get_odds(item, keys, default=1.90):
        for k in keys:
            if k in item and item[k] is not None:
                return parse_odds_val(item[k])
        return default

    mapping = {
        "讓球": ["letting", "hdc", "ah", "handicap", "讓球"],
        "入球大小": ["ou", "hil", "overunder", "入球大細", "入球大小"],
        "角球大小": ["corner", "chl", "corners", "角球大細", "角球大小"]
    }
    
    upper_keys = ['h', 'home', 'homeOdds', 'up', 'upper', 'over', 'overOdds', 'high', '大', '主']
    lower_keys = ['a', 'away', 'awayOdds', 'low', 'lower', 'under', 'underOdds', '小', '客', 'l']
    line_keys = ['line', 'goal', 'p', '盤', 'handicap']

    for bet_type_cn, possible_keys in mapping.items():
        target_data = []
        if isinstance(odds_data, dict):
            for k in possible_keys:
                if k in odds_data:
                    target_data = odds_data[k]
                    break
        elif isinstance(odds_data, list):
            for item in odds_data:
                if isinstance(item, dict):
                    t = item.get('type', '').lower()
                    if any(pk.lower() in t for pk in possible_keys):
                        target_data = item.get('history', [item])
                        break
                        
        if isinstance(target_data, dict):
            if 'history' in target_data:
                target_data = target_data['history']
            elif 'odds' in target_data:
                target_data = target_data['odds']
            else:
                target_data = [target_data]
                
        if isinstance(target_data, list):
            if len(target_data) > 12:
                step = len(target_data) // 10
                target_data = [target_data[0]] + target_data[1:-1:step] + [target_data[-1]]
                
            for item in target_data:
                if not isinstance(item, dict): continue
                
                line_val = 0.0
                for lk in line_keys:
                    if lk in item:
                        line_val = item[lk]
                        break
                line = parse_line(line_val)
                
                upper = get_odds(item, upper_keys, 1.90)
                lower = get_odds(item, lower_keys, 1.90)
                
                if upper != 1.90 or lower != 1.90 or line != 0.0: 
                    new_history.append({
                        "id": row_id, "type": bet_type_cn, "record_time": "", "line": line,
                        "upper": upper, "lower": lower, "unlock": True, "margin": 1.085
                    })
                    row_id += 1
                    
    return new_history

def parse_and_fill_inplay(match_id):
    """將抓取到的 API 數據填入 Session State (即場與賽果)"""
    data = get_match_fixtures(match_id)
    odds_data = get_match_odds(match_id)
    
    success = False
    
    if data and "teamStats" in data:
        team_stats_ft = data["teamStats"].get("ft", {})
        if team_stats_ft:
            goals = team_stats_ft.get("1", [0, 0])
            st.session_state.edit_h_g = int(goals[0])
            st.session_state.edit_a_g = int(goals[1])
            
            corners = team_stats_ft.get("2", [0, 0])
            st.session_state.edit_h_c = int(corners[0])
            st.session_state.edit_a_c = int(corners[1])
            
            red_cards = team_stats_ft.get("4", [0, 0])
            st.session_state.edit_h_red = int(red_cards[0])
            st.session_state.edit_a_red = int(red_cards[1])
            
            sot = team_stats_ft.get("21", [0, 0])
            st.session_state.edit_h_sot = int(sot[0])
            st.session_state.edit_a_sot = int(sot[1])
            
            poss = team_stats_ft.get("25", [50, 50])
            st.session_state.edit_h_poss = int(poss[0])
            success = True
            
    if odds_data:
        new_history = extract_odds_history(odds_data)
        if new_history:
            st.session_state.inplay_odds_history = new_history
            success = True
            
    return success

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
    st.session_state.edit_bet_type = str(row.get('Bet_Type', '')).replace(" (即場)", "") if pd.notna(row.get('Bet_Type')) else ''
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
                        'id': f"U{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                        'target': target_name,
                        'data': df_to_delete.copy(),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
                            'id': f"U{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                            'target': target_name,
                            'data': df_target.copy(),
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
        st.subheader("2. ↩️ 狀態重置 (Undo 復原中心)")
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

def get_last_odds_state(history, target_type, default_line):
    for row in reversed(history):
        if row['type'] == target_type:
            return {
                "type": target_type,
                "record_time": row.get('record_time', ''),
                "line": float(row.get('line', default_line)),
                "upper": float(row.get('upper', 1.90)),
                "lower": float(row.get('lower', 1.90)),
                "unlock": bool(row.get('unlock', False)),
                "margin": float(row.get('margin', 1.085))
            }
    return {
        "type": target_type,
        "record_time": "",
        "line": default_line,
        "upper": 1.90,
        "lower": 1.90,
        "unlock": False,
        "margin": 1.085
    }

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

    # 1. 以時間戳記作為區塊分界點
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
        
        # 尋找包含盤口線的行索引 (例如 [10.5] 或 10.5球)
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
        
        # 嚴格區分上下盤：盤口線之前歸上盤，盤口線所在行及之後歸下盤
        if line_idx == -1:
            upper_lines = block_lines
            lower_lines = []
        else:
            upper_lines = block_lines[:line_idx]
            lower_lines = block_lines[line_idx:]
            
        # 提取上盤賠率 (主隊/大盤) 並去重
        upper_odds_list = []
        for l_item in upper_lines:
            for n_str in re.findall(r'\b\d+(?:\.\d+)?\b', l_item):
                val = parse_odds_val(n_str)
                if val is not None and abs(val - active_line) > 1e-4:
                    upper_odds_list.append(val)
                    
        # 提取下盤賠率 (客隊/小盤) 並去重 (過濾掉中括號盤口數值)
        lower_odds_list = []
        for l_item in lower_lines:
            cleaned_l_item = re.sub(r'\[.*?\]', '', l_item)
            for n_str in re.findall(r'\b\d+(?:\.\d+)?\b', cleaned_l_item):
                val = parse_odds_val(n_str)
                if val is not None and abs(val - active_line) > 1e-4:
                    lower_odds_list.append(val)
                    
        # 去除數值重複的項目
        unique_upper = []
        for u in upper_odds_list:
            if not any(abs(u - existing) < 1e-4 for existing根據您提供的程式碼以及圖片中常見的 Streamlit `KeyError` 錯誤，這個問題通常發生在讀取字典鍵值（特別是在處理 `cand['bet_type']`）時，因為資料結構中偶爾會遺失該鍵值而導致程式崩潰。

我已經為您將相關的字典讀取方式全面升級為防呆的 `.get()` 方法，這能有效避免 `KeyError: 'bet_type'` 的發生，同時完全保留了您原有的所有邏輯與功能[cite: 1]。

以下是修正後的完整 `FB-app.py` 程式碼，您可以直接複製並覆蓋 GitHub 上的原始檔案：

```python
import streamlit as st
import pandas as pd
import os
import json
import base64
import re
import io
import requests
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
    url = f"[https://api.github.com/repos/](https://api.github.com/repos/){repo}/contents/{path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return pd.read_csv(io.StringIO(res.text))
    return None

def save_db_github(df, repo, path, token):
    url = f"[https://api.github.com/repos/](https://api.github.com/repos/){repo}/contents/{path}"
    headers = {"Authorization": f"token {token}"}
    res_get = requests.get(url, headers=headers)
    sha = res_get.json().get("sha") if res_get.status_code == 200 else None
    
    csv_content = df.to_csv(index=False)
    content_b64 = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
    
    payload = {
        "message": f"Auto-update {path} via Streamlit App",
        "content": content_b64
    }
    if sha:
        payload["sha"] = sha
        
    requests.put(url, json=payload, headers=headers)

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

def load_db(filename, columns, table_name):
    string_cols = [
        'ID', 'Date', 'Status', 'Tournament_Name', 'Tournament_Category', 
        'Match', 'Home_Team', 'Away_Team', 'Home_Rating', 'Away_Rating', 
        'Home_Form', 'Away_Form', 'Bet_Type', 'Selection', 'Odds_History', 'Result_Label',
        'Type', 'Account', 'Note'
    ]
    
    df = pd.DataFrame()
    # 策略 A: 嘗試 PostgreSQL 雲端資料庫
    if HAS_SQLALCHEMY and "DB_URL" in st.secrets and st.secrets["DB_URL"]:
        try:
            engine = create_engine(st.secrets["DB_URL"])
            df = pd.read_sql_table(table_name, engine)
            df = process_legacy_columns(df)
        except Exception:
            pass

    # 策略 B: 嘗試 GitHub API 自動同步
    if df.empty and "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets:
        try:
            gh_df = load_db_github(st.secrets["GITHUB_REPO"], filename, st.secrets["GITHUB_TOKEN"])
            if gh_df is not None:
                df = process_legacy_columns(gh_df)
        except Exception:
            pass

    # 策略 C: 本地 CSV 讀取防呆
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
        except Exception:
            pass

    if "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets:
        try:
            save_db_github(df, st.secrets["GITHUB_REPO"], filename, st.secrets["GITHUB_TOKEN"])
        except Exception:
            pass

    try:
        df.to_csv(filename, index=False)
    except Exception:
        pass

# ==========================================
# 1.5 自動化抓取 API 模組 (Auto-Scraper)
# ==========================================
def fetch_api_data(url):
    """通用的 API 請求函數"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def get_match_odds(match_id):
    """抓取馬會賠率變化"""
    url = f"[https://tipsme-web.azurewebsites.net/api/Score/odds/hkjc/](https://tipsme-web.azurewebsites.net/api/Score/odds/hkjc/){match_id}"
    return fetch_api_data(url)

def get_match_fixtures(match_id):
    """抓取即場賽況及統計數據 (同時適用於賽果)"""
    url = f"[https://tipsme-web.azurewebsites.net/api/Score/fixtures/](https://tipsme-web.azurewebsites.net/api/Score/fixtures/){match_id}"
    return fetch_api_data(url)

def extract_odds_history(odds_data):
    """強大的盤口/賠率走勢萃取器，負責解構深層次 JSON 並轉換成系統需要的格式"""
    new_history = []
    row_id = 0
    
    def parse_line(line_str):
        try:
            line_str = str(line_str).replace('[', '').replace(']', '').replace('球', '')
            if '/' in line_str:
                parts = line_str.split('/')
                return (float(parts[0]) + float(parts[1])) / 2
            return float(line_str)
        except:
            return 0.0
            
    def parse_odds_val(val):
        try:
            clean_val = re.sub(r'[^\d\.]', '', str(val))
            return float(clean_val)
        except:
            return 1.90
            
    def get_odds(item, keys, default=1.90):
        for k in keys:
            if k in item and item[k] is not None:
                return parse_odds_val(item[k])
        return default

    mapping = {
        "讓球": ["letting", "hdc", "ah", "handicap", "讓球"],
        "入球大小": ["ou", "hil", "overunder", "入球大細", "入球大小"],
        "角球大小": ["corner", "chl", "corners", "角球大細", "角球大小"]
    }
    
    upper_keys = ['h', 'home', 'homeOdds', 'up', 'upper', 'over', 'overOdds', 'high', '大', '主']
    lower_keys = ['a', 'away', 'awayOdds', 'low', 'lower', 'under', 'underOdds', '小', '客', 'l']
    line_keys = ['line', 'goal', 'p', '盤', 'handicap']

    for bet_type_cn, possible_keys in mapping.items():
        target_data = []
        if isinstance(odds_data, dict):
            for k in possible_keys:
                if k in odds_data:
                    target_data = odds_data[k]
                    break
        elif isinstance(odds_data, list):
            for item in odds_data:
                if isinstance(item, dict):
                    t = item.get('type', '').lower()
                    if any(pk.lower() in t for pk in possible_keys):
                        target_data = item.get('history', [item])
                        break
                        
        if isinstance(target_data, dict):
            if 'history' in target_data:
                target_data = target_data['history']
            elif 'odds' in target_data:
                target_data = target_data['odds']
            else:
                target_data = [target_data]
                
        if isinstance(target_data, list):
            if len(target_data) > 12:
                step = len(target_data) // 10
                target_data = [target_data[0]] + target_data[1:-1:step] + [target_data[-1]]
                
            for item in target_data:
                if not isinstance(item, dict): continue
                
                line_val = 0.0
                for lk in line_keys:
                    if lk in item:
                        line_val = item[lk]
                        break
                line = parse_line(line_val)
                
                upper = get_odds(item, upper_keys, 1.90)
                lower = get_odds(item, lower_keys, 1.90)
                
                if upper != 1.90 or lower != 1.90 or line != 0.0: 
                    new_history.append({
                        "id": row_id, "type": bet_type_cn, "record_time": "", "line": line,
                        "upper": upper, "lower": lower, "unlock": True, "margin": 1.085
                    })
                    row_id += 1
                    
    return new_history

def parse_and_fill_inplay(match_id):
    """將抓取到的 API 數據填入 Session State (即場與賽果)"""
    data = get_match_fixtures(match_id)
    odds_data = get_match_odds(match_id)
    
    success = False
    
    if data and "teamStats" in data:
        team_stats_ft = data["teamStats"].get("ft", {})
        if team_stats_ft:
            goals = team_stats_ft.get("1", [0, 0])
            st.session_state.edit_h_g = int(goals[0])
            st.session_state.edit_a_g = int(goals[1])
            
            corners = team_stats_ft.get("2", [0, 0])
            st.session_state.edit_h_c = int(corners[0])
            st.session_state.edit_a_c = int(corners[1])
            
            red_cards = team_stats_ft.get("4", [0, 0])
            st.session_state.edit_h_red = int(red_cards[0])
            st.session_state.edit_a_red = int(red_cards[1])
            
            sot = team_stats_ft.get("21", [0, 0])
            st.session_state.edit_h_sot = int(sot[0])
            st.session_state.edit_a_sot = int(sot[1])
            
            poss = team_stats_ft.get("25", [50, 50])
            st.session_state.edit_h_poss = int(poss[0])
            success = True
            
    if odds_data:
        new_history = extract_odds_history(odds_data)
        if new_history:
            st.session_state.inplay_odds_history = new_history
            success = True
            
    return success

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
    st.session_state.edit_bet_type = str(row.get('Bet_Type', '')).replace(" (即場)", "") if pd.notna(row.get('Bet_Type')) else ''
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
                        'id': f"U{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                        'target': target_name,
                        'data': df_to_delete.copy(),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
                            'id': f"U{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                            'target': target_name,
                            'data': df_target.copy(),
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
        st.subheader("2. ↩️ 狀態重置 (Undo 復原中心)")
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
                    
                st.success("✅ 資料已成功還原！")
                st.rerun()

# ==========================================
# 主程式
# ==========================================
def main():
    st.title("⚽ 智能足球精算與資金管理系統")
    st.markdown("Dr. EdwinPro 專屬精算分析與雲端資金風控系統，支援即時盤口、馬會 API 自動抓取與機器學習決策。")

    db_file = 'football_bets.csv'
    capital_file = 'capital_flow.csv'
    db_table = 'football_bets'
    cap_table = 'capital_flow'
    
    if 'df_db' not in st.session_state: st.session_state.df_db = load_db(db_file, DB_COLUMNS, db_table)
    if 'df_cap' not in st.session_state: st.session_state.df_cap = load_db(capital_file, CAPITAL_COLUMNS, cap_table)
    if 'undo_stack' not in st.session_state: st.session_state.undo_stack = []
    if 'odds_history' not in st.session_state: st.session_state.odds_history = []
    if 'inplay_odds_history' not in st.session_state: st.session_state.inplay_odds_history = []
    if 'editing_bet_id' not in st.session_state: st.session_state.editing_bet_id = None
    if 'api_tipsme_id' not in st.session_state: st.session_state.api_tipsme_id = ""
    
    df_db = st.session_state.df_db
    df_cap = st.session_state.df_cap

    is_editing = st.session_state.editing_bet_id is not None
    edit_id = st.session_state.editing_bet_id if is_editing else f"B{datetime.now().strftime('%Y%m%d%H%M%S')}"

    st.sidebar.header("🏦 資金配置管理中心")
    cap_type = st.sidebar.selectbox("流水類型 (Transaction)", ["Deposit (存入本金)", "Withdraw (提取本金)"])
    cap_target = st.sidebar.selectbox("資金帳戶對象", ["Both (系統與用家同時)", "System (僅限系統)", "User (僅限用家真實資金)"])
    cap_amt = st.sidebar.number_input("金額 (Amount)", min_value=0.0, step=100.0)
    cap_note = st.sidebar.text_input("資金備註 (Optional Note)")
    
    if st.sidebar.button("✍️ 記錄本金流水", type="primary"):
        c_type = "Deposit" if "Deposit" in cap_type else "Withdraw"
        t_target = cap_target.split(" ")[0]
        new_cap = pd.DataFrame([{
            'ID': f"C{datetime.now().strftime('%Y%m%d%H%M%S')}", 'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Type': c_type, 'Account': t_target, 'Amount': float(cap_amt), 'Note': cap_note
        }])
        st.session_state.df_cap = pd.concat([df_cap, new_cap], ignore_index=True)
        save_db(st.session_state.df_cap, capital_file, cap_table)
        st.sidebar.success(f"✅ 已成功記錄資金流水: {c_type} ${cap_amt} 於 {t_target} 帳戶")
        st.rerun()

    sys_res = recalculate_bankroll_from_scratch(st.session_state.df_cap, st.session_state.df_db)
    (sys_dep, sys_wit, sys_net, sys_profit, sys_bankroll, sys_max_stake,
     usr_dep, usr_wit, usr_net, usr_profit, usr_bankroll, usr_max_stake) = sys_res
    
    st.sidebar.markdown("### 📊 即時風控與資金水位")
    st.sidebar.markdown(f"**🤖 系統風控總水位 (System Bankroll)**: `${sys_bankroll:,.2f}`")
    st.sidebar.caption(f"淨存入: `${sys_net:,.2f}` | 淨盈虧: `${sys_profit:,.2f}`")
    st.sidebar.markdown(f"**💡 系統單場風險上限 (10%)**: `${sys_max_stake:,.2f}`")
    st.sidebar.divider()
    st.sidebar.markdown(f"**👤 您的真實總水位 (User Bankroll)**: `${usr_bankroll:,.2f}`")
    st.sidebar.caption(f"淨存入: `${usr_net:,.2f}` | 淨盈虧: `${usr_profit:,.2f}`")

    st.sidebar.markdown("---")
    if st.sidebar.button("🗄️ 開啟數據庫預覽與管理", use_container_width=True):
        preview_db_dialog(df_db, df_cap, db_file, capital_file, db_table, cap_table)
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 修改歷史注單")
    all_bets = []
    if not df_db.empty:
        all_bets = df_db.sort_values(by='Date', ascending=False)['ID'].tolist()
        
    if all_bets:
        sel_edit_id = st.sidebar.selectbox("選擇要修改的注單 ID", ["無"] + all_bets, index=0)
        
        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.button("載入注單", use_container_width=True):
                if sel_edit_id != "無":
                    load_bet_to_edit(sel_edit_id)
                    st.rerun()
                else:
                    st.sidebar.warning("請先選擇注單")
        with c2:
            if st.button("清除載入", use_container_width=True) and is_editing:
                clear_edit_mode()
                st.rerun()
                
        if is_editing:
            st.sidebar.success(f"✅ 正在修改注單: {st.session_state.editing_bet_id}")
    else:
        st.sidebar.info("資料庫目前沒有注單。")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 重整頁面 (Rerun)", use_container_width=True):
        st.rerun()

    if is_editing:
        st.info(f"🔄 **修改模式啟動**：您正在修改已存檔的注單 `{st.session_state.editing_bet_id}`。點擊左側「清除載入」即可返回新增模式。")
    
    t_pre, t_inplay, t_result, t_report = st.tabs(["📊 賽前分析與入單 (Pre-match)", "⚡ 即場走地追蹤 (In-Play)", "📝 賽果結算結帳 (Result)", "📈 戰績累計分析 (Report)"])

    rating_map = {'S': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1}
    r_options = list(rating_map.keys())

    # ==========================================
    # Tab 1: 賽前分析與入單 (Pre-match)
    # ==========================================
    with t_pre:
        st.header("1. 賽事基本資料建立")
        st.write("若您有馬會 API 的 Match ID (例如 tipsme)，輸入後可自動抓取即場資訊與盤口走勢。")
        
        api_col1, api_col2 = st.columns([3, 1])
        with api_col1:
            st.session_state.api_tipsme_id = st.text_input("⚡ 自動抓取 API Match ID (Tipsme格式)", value=st.session_state.api_tipsme_id, help="例如: a31b0b0a-3c0f-4886-be8d-8a58a74be656")
        with api_col2:
            st.write("")
            st.write("")
            if st.button("🤖 執行自動抓取", use_container_width=True):
                if st.session_state.api_tipsme_id:
                    with st.spinner("正在自動連接 API 抓取數據與盤口..."):
                        if parse_and_fill_inplay(st.session_state.api_tipsme_id):
                            st.success("✅ 成功從 API 匯入即場數據與賠率變化！請前往即場分頁查看。")
                        else:
                            st.warning("⚠️ API 抓取失敗或查無有效數據，請確認 ID 是否正確。")
                else:
                    st.warning("請先輸入 Match ID。")
                    
        st.divider()

        c1, c2 = st.columns(2)
        t_name = c1.text_input("賽事/聯賽名稱 (Tournament)", value=st.session_state.get('edit_t_name', ''))
        t_cat = c2.selectbox("賽事級別 (Category)", CATEGORY_OPTIONS, index=CATEGORY_OPTIONS.index(st.session_state.get('edit_t_cat', CATEGORY_OPTIONS[0])))

        st.subheader("2. 球隊基本面與近況分析 (Fundamental & Form)")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### 主隊 (Home)")
            h_team = st.text_input("主隊名稱", value=st.session_state.get('edit_h_team', ''), key="h_t")
            h_rating = st.selectbox("主隊評級", r_options, index=r_options.index(st.session_state.get('edit_h_rating', 'C')), key="h_r")
            st.markdown("**主隊近五場戰績 (Last 5)**")
            hc1, hc2, hc3 = st.columns(3)
            hw = hc1.number_input("勝 (W)", min_value=0, max_value=5, value=st.session_state.get('edit_hw', 3), key="hw")
            hd = hc2.number_input("平 (D)", min_value=0, max_value=5, value=st.session_state.get('edit_hd', 1), key="hd")
            hl = hc3.number_input("負 (L)", min_value=0, max_value=5, value=st.session_state.get('edit_hl', 1), key="hl")
        with c4:
            st.markdown("#### 客隊 (Away)")
            a_team = st.text_input("客隊名稱", value=st.session_state.get('edit_a_team', ''), key="a_t")
            a_rating = st.selectbox("客隊評級", r_options, index=r_options.index(st.session_state.get('edit_a_rating', 'C')), key="a_r")
            st.markdown("**客隊近五場戰績 (Last 5)**")
            ac1, ac2, ac3 = st.columns(3)
            aw = ac1.number_input("勝 (W)", min_value=0, max_value=5, value=st.session_state.get('edit_aw', 2), key="aw")
            ad = ac2.number_input("平 (D)", min_value=0, max_value=5, value=st.session_state.get('edit_ad', 2), key="ad")
            al = ac3.number_input("負 (L)", min_value=0, max_value=5, value=st.session_state.get('edit_al', 1), key="al")

        h_form = f"{hw}W{hd}D{hl}L"
        a_form = f"{aw}W{ad}D{al}L"

        st.divider()
        st.header("3. 系統自動盤口預測與 EV 分析器 (Machine Learning Engine)")
        
        st.markdown("""
        > 💡 **AI 三維度決策引擎運作原理**
        > 系統自動化讀取數據庫的歷史賽果，並對當前對戰組合進行多維度勝率演算。若該盤口具有優勢正期望值 (EV > 0)，系統將建議入注。
        """)

        df_settled = df_db[df_db['Status'] == 'Settled'].copy()
        
        h_data = {'hr': rating_map[h_rating], 'ar': rating_map[a_rating], 'hf': extract_form_points(h_form), 'af': extract_form_points(a_form)}
        
        candidates_base = [
            {'bet_type': '讓球', 'selection': 'Home', 'base_prob': 0.50, 'odds': 1.90, 'line': -0.5, 'label': f"{h_team} -0.5"},
            {'bet_type': '讓球', 'selection': 'Away', 'base_prob': 0.50, 'odds': 1.90, 'line': +0.5, 'label': f"{a_team} +0.5"},
            {'bet_type': '入球大小', 'selection': 'Over', 'base_prob': 0.45, 'odds': 1.85, 'line': 2.5, 'label': '大 2.5 球'},
            {'bet_type': '入球大小', 'selection': 'Under', 'base_prob': 0.55, 'odds': 1.95, 'line': 2.5, 'label': '小 2.5 球'},
            {'bet_type': '角球大小', 'selection': 'Over', 'base_prob': 0.50, 'odds': 1.90, 'line': 9.5, 'label': '大 9.5 角'},
        ]

        dims = []
        if not df_settled.empty:
            dims.append(evaluate_dimension(df_settled, "宏觀全局數據維度 (All Settled Data)", candidates_base, rating_map, h_data))
            
            df_cat = df_settled[df_settled['Tournament_Category'] == t_cat].copy()
            dims.append(evaluate_dimension(df_cat, f"同級別賽事維度 ({t_cat})", candidates_base, rating_map, h_data))
            
            mask_h = (df_settled['Home_Team'] == h_team) | (df_settled['Away_Team'] == h_team)
            mask_a = (df_settled['Home_Team'] == a_team) | (df_settled['Away_Team'] == a_team)
            df_teams = df_settled[mask_h | mask_a].copy()
            dims.append(evaluate_dimension(df_teams, "特定球隊歷史維度 (Team specific)", candidates_base, rating_map, h_data))
        else:
            st.info("系統目前尚無結算資料可供機器學習訓練，將採用預設基礎機率。")

        best_model = None
        if dims:
            valid_dims = [d for d in dims if d['valid']]
            if valid_dims:
                best_model = max(valid_dims, key=lambda x: x['score'])
            else:
                best_model = max(dims, key=lambda x: x['n']) 

        if best_model:
            st.success(f"🎯 **決策引擎已選定最佳運算維度：`{best_model.get('dim', '宏觀')}`** (樣本數: {best_model['n']}, 歷史命中率: {best_model['acc']:.2%})")
            
            disp_data = []
            for c in best_model.get('candidates', []):
                disp_data.append({
                    "盤口類型": c.get('bet_type', ''),
                    "選項": c.get('label', ''),
                    "預測勝率 (Prob)": f"{c.get('prob', 0):.1%}",
                    "賠率 (Odds)": c.get('odds', 1.90),
                    "期望值 (EV)": round(c.get('ev', 0), 3),
                    "系統建議": "✅ 值得投資" if c.get('ev', 0) > 0 else "❌ 不建議"
                })
            
            st.dataframe(pd.DataFrame(disp_data), use_container_width=True)

        st.divider()
        st.header("4. 決策入單 (Final Decision & Execution)")
        
        all_options = []
        option_map = {}
        
        if best_model:
            cand_list = best_model.get('candidates', [])
            for c in cand_list:
                opt_str = f"{c.get('bet_type', '')} | {c.get('label', '')} @ {c.get('odds', 1.90)}"
                if opt_str not in option_map:
                    all_options.append(opt_str)
                    option_map[opt_str] = c
                    
        custom_base = [
            {'bet_type': '讓球', 'selection': 'Home', 'line': 0.0, 'odds': 1.90, 'label': f"{h_team} (自訂)"},
            {'bet_type': '讓球', 'selection': 'Away', 'line': 0.0, 'odds': 1.90, 'label': f"{a_team} (自訂)"},
            {'bet_type': '入球大小', 'selection': 'Over', 'line': 2.5, 'odds': 1.90, 'label': '大球 (自訂)'},
            {'bet_type': '入球大小', 'selection': 'Under', 'line': 2.5, 'odds': 1.90, 'label': '小球 (自訂)'},
            {'bet_type': '角球大小', 'selection': 'Over', 'line': 9.5, 'odds': 1.90, 'label': '角大 (自訂)'},
            {'bet_type': '角球大小', 'selection': 'Under', 'line': 9.5, 'odds': 1.90, 'label': '角小 (自訂)'},
        ]
        
        for i, cb in enumerate(custom_base):
            c_str = f"🔧 自訂盤口: {cb.get('bet_type', '')} | {cb.get('label', '')}"
            if c_str not in option_map:
                all_options.append(c_str)
                cb_copy = cb.copy()
                cb_copy['is_custom'] = True
                option_map[c_str] = cb_copy

        st.write("請從下方挑選您決定實際入注的盤口 (可多選，或選擇自訂盤口)。系統將會根據 EV 替您計算最佳注碼。")
        
        default_sys = []
        if is_editing and st.session_state.get('edit_sys_stake', 0) > 0:
            eb = st.session_state.get('edit_bet_type')
            el = st.session_state.get('edit_line')
            es = st.session_state.get('edit_selection')
            for opt in all_options:
                c = option_map[opt]
                # [防呆修正]: c['bet_type'] 改為 c.get('bet_type')
                if c.get('bet_type') == eb and c.get('line') == el and c.get('selection') == es:
                    default_sys.append(opt)
                    break

        sys_selected = st.multiselect("🤖 系統投注 (System)", all_options, default=default_sys, help="被選中的盤口將自動計算凱利建議注碼")
        
        def calc_suggested_stake(cand, sys_bankroll, sys_max_stake):
            if cand.get('is_custom'): return 0.0, 0.0
            prob = cand.get('prob', 0)
            odds = cand.get('odds', 1.90)
            ev = cand.get('ev', -1)
            
            if ev > 0:
                fraction = ev / (odds - 1)
                fraction = max(0, min(fraction, 0.25))
                raw_stake = sys_bankroll * fraction
                
                if "讓球" in cand.get('bet_type', ''):
                    if prob >= 0.50 and ev >= 0.03:
                        if raw_stake < 200: raw_stake = 200.0
                        
                final_stake = min(raw_stake, sys_max_stake)
                return round(final_stake, -1), raw_stake
            return 0.0, 0.0

        sys_stakes = {}
        if sys_selected:
            st.markdown("##### ⚙️ 系統注碼微調")
            for sel in sys_selected:
                cand = option_map[sel]
                sug_stk, raw_stk = calc_suggested_stake(cand, sys_bankroll, sys_max_stake)
                
                sys_stakes[sel] = st.number_input(f"系統建議金額: {sel}", value=float(sug_stk), disabled=True, key=f"s_stk_{sel}")
                
                # [防呆修正]: cand['bet_type'] 改為 cand.get('bet_type', '')
                if "讓球" in cand.get('bet_type', '') and 0 < raw_stk < 200:
                    if cand.get('ev', 0) >= 0.03 and cand.get('prob', 0) >= 0.50:
                        st.caption(f"💡 `{cand.get('bet_type', '')}` EV/勝率達標，系統自動升級最低注碼 $200")
                    else:
                        st.caption(f"⚠️ `{cand.get('bet_type', '')}` EV未達標，系統建議放棄 (注碼 $0)")
                
                if cand.get('is_custom'):
                    cust_line = st.number_input(f"自訂盤口線 ({sel})", value=float(cand.get('line', 0.0)), step=0.25, key=f"c_line_{sel}")
                    cust_odds = st.number_input(f"自訂賠率 ({sel})", value=float(cand.get('odds', 1.90)), step=0.01, key=f"c_odds_{sel}")
                    cand['line'] = cust_line
                    cand['odds'] = cust_odds
                    cand['label'] = f"{cand.get('bet_type', '')} @ {cust_odds}"

        st.divider()
        
        default_usr = []
        if is_editing and st.session_state.get('edit_user_stake', 0) > 0:
            eb = st.session_state.get('edit_bet_type')
            el = st.session_state.get('edit_line')
            es = st.session_state.get('edit_selection')
            for opt in all_options:
                c = option_map[opt]
                # [防呆修正]: c['bet_type'] 改為 c.get('bet_type')
                if c.get('bet_type') == eb and c.get('line') == el and c.get('selection') == es:
                    default_usr.append(opt)
                    break

        user_selected = st.multiselect("👤 用家投注 (User)", all_options, default=default_usr, help="您可以選擇跟隨系統，或輸入自己判斷的金額 (包含自訂盤口)")
        user_stakes = {}
        
        if user_selected:
            st.markdown("##### 💰 用家自行決策注碼")
            for sel in user_selected:
                cand = option_map[sel]
                def_u_val = float(st.session_state.get('edit_user_stake', 0.0)) if is_editing and sel in default_usr else 0.0
                user_stakes[sel] = st.number_input(f"輸入真實入注金額: {sel}", min_value=0.0, step=50.0, value=def_u_val, key=f"u_stk_{sel}")
                
                if cand.get('is_custom') and sel not in sys_selected:
                    cust_line = st.number_input(f"自訂盤口線 ({sel})", value=float(cand.get('line', 0.0)), step=0.25, key=f"c_line_u_{sel}")
                    cust_odds = st.number_input(f"自訂賠率 ({sel})", value=float(cand.get('odds', 1.90)), step=0.01, key=f"c_odds_u_{sel}")
                    cand['line'] = cust_line
                    cand['odds'] = cust_odds
                    cand['label'] = f"{cand.get('bet_type', '')} @ {cust_odds}"

        submit_btn_label = f"💾 {'覆蓋儲存修改 (Save Edits)' if is_editing else '一鍵送出所有注單 (Submit)'}"
        
        if st.button(submit_btn_label, type="primary", use_container_width=True):
            if not sys_selected and not user_selected:
                st.warning("⚠️ 請至少選擇一個盤口入注。")
            elif not h_team or not a_team:
                st.warning("⚠️ 雙方球隊名稱不能為空。")
            else:
                final_selections = list(set(sys_selected + user_selected))
                new_rows = []
                
                for sel in final_selections:
                    cand = option_map[sel]
                    s_stake = sys_stakes.get(sel, 0.0)
                    u_stake = user_stakes.get(sel, 0.0)
                    
                    if s_stake == 0 and u_stake == 0: continue
                    
                    if is_editing and len(final_selections) == 1:
                        b_id = edit_id
                    else:
                        b_id = f"B{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    
                    new_rows.append({
                        'ID': b_id,
                        'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Status': 'Open',
                        'Tournament_Name': t_name, 'Tournament_Category': t_cat,
                        'Match': f"{h_team} vs {a_team}",
                        'Home_Team': h_team, 'Away_Team': a_team,
                        'Home_Rating': h_rating, 'Away_Rating': a_rating,
                        'Home_Form': h_form, 'Away_Form': a_form,
                        # [防呆修正]: cand['bet_type'] 改為 cand.get('bet_type', '')
                        'Bet_Type': cand.get('bet_type', ''), 'Selection': cand.get('selection', 'Home'), 'Initial_Line': cand.get('line', 0.0), 'Initial_Odds': cand.get('odds', 1.90),
                        'System_Stake': float(s_stake), 'User_Stake': float(u_stake),
                        'Odds_History': json.dumps([]),
                        'InPlay_Minute': 0,
                        'Home_DA': 0, 'Away_DA': 0, 'Home_SoT': 0, 'Away_SoT': 0, 'Home_SoFF': 0, 'Away_SoFF': 0,
                        'Home_Red': 0, 'Away_Red': 0, 'Home_Sub': 0, 'Away_Sub': 0, 'Home_Possession': 50, 'Away_Possession': 50,
                        'Home_Goal': 0, 'Away_Goal': 0, 'Home_Corner': 0, 'Away_Corner': 0,
                        'System_Profit': 0.0, 'User_Profit': 0.0, 'System_Payout': 0.0, 'User_Payout': 0.0, 'Unit_Profit': 0.0, 'Result_Label': ''
                    })
                
                if new_rows:
                    if is_editing:
                        st.session_state.df_db = df_db[df_db['ID'] != edit_id]
                        clear_edit_mode()
                        
                    st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame(new_rows)], ignore_index=True)
                    save_db(st.session_state.df_db, db_file, db_table)
                    st.success(f"✅ 成功寫入 {len(new_rows)} 筆未結算注單！請前往結算分頁或即場走地持續追蹤。")
                else:
                    st.warning("⚠️ 所選項目的注碼皆為 0，並未產生任何新注單。")

    # ==========================================
    # Tab 2: 即場走地追蹤 (In-Play)
    # ==========================================
    with t_inplay:
        st.header("⚡ 即場走地雷達與盤口變化 (In-Play Radar)")
        st.markdown("支援匯入 API `odds` 及手動紀錄。")
        
        ip_c1, ip_c2, ip_c3, ip_c4 = st.columns(4)
        ip_type = ip_c1.selectbox("盤口分類", ["讓球", "入球大小", "角球大小"])
        ip_time = ip_c2.text_input("發生時間 (e.g., 68')")
        ip_line = ip_c3.number_input("當前盤口線 (Line)", step=0.25, value=0.0)
        
        ip_c5, ip_c6, ip_c7 = st.columns(3)
        ip_upper = ip_c5.number_input("上盤/大球/主 賠率", step=0.01, value=1.90)
        ip_lower = ip_c6.number_input("下盤/小球/客 賠率", step=0.01, value=1.90)
        ip_unlock = ip_c7.checkbox("盤口開啟 (Unlock)", value=True)
        
        margin = round(1 / ip_upper + 1 / ip_lower - 1, 3) if ip_upper > 0 and ip_lower > 0 else 0
        st.caption(f"📊 馬會當前抽水率估算 (Margin): {margin:.1%}")

        if st.button("➕ 新增一筆盤口變化至雷達", use_container_width=True):
            new_record = {
                "id": len(st.session_state.inplay_odds_history),
                "type": ip_type, "record_time": ip_time, "line": ip_line,
                "upper": ip_upper, "lower": ip_lower, "unlock": ip_unlock, "margin": margin
            }
            st.session_state.inplay_odds_history.append(new_record)
            st.success("✅ 已記錄！您可以隨時在下方編輯，或者進入結算分頁寫入至注單中。")
            
        if st.session_state.inplay_odds_history:
            st.markdown("### 📈 即時盤口走勢表")
            
            edited_history = []
            for item in st.session_state.inplay_odds_history:
                with st.expander(f"ID:{item['id']} | {item['type']} | Line: {item['line']} | Upper: {item['upper']} / Lower: {item['lower']}", expanded=False):
                    hc1, hc2, hc3, hc4 = st.columns(4)
                    item['type'] = hc1.selectbox("類型", ["讓球", "入球大小", "角球大小"], index=["讓球", "入球大小", "角球大小"].index(item['type']), key=f"eh_type_{item['id']}")
                    item['record_time'] = hc2.text_input("時間", value=item['record_time'], key=f"eh_time_{item['id']}")
                    item['line'] = hc3.number_input("Line", value=float(item['line']), step=0.25, key=f"eh_line_{item['id']}")
                    item['upper'] = hc4.number_input("Upper", value=float(item['upper']), step=0.01, key=f"eh_up_{item['id']}")
                    
                    hc5, hc6, hc7 = st.columns([1,1,2])
                    item['lower'] = hc5.number_input("Lower", value=float(item['lower']), step=0.01, key=f"eh_lo_{item['id']}")
                    item['unlock'] = hc6.checkbox("Unlock", value=item['unlock'], key=f"eh_un_{item['id']}")
                    
                    if hc7.button("🗑️ 刪除此筆", key=f"eh_del_{item['id']}", use_container_width=True):
                        continue
                    edited_history.append(item)
                    
            st.session_state.inplay_odds_history = edited_history
            
            df_hist = pd.DataFrame(st.session_state.inplay_odds_history)
            if not df_hist.empty:
                st.dataframe(df_hist, use_container_width=True)

    # ==========================================
    # Tab 3: 賽果結算結帳 (Result)
    # ==========================================
    with t_result:
        st.header("📝 賽果輸入與一鍵結算 (Settlement)")
        open_bets = df_db[df_db['Status'] == 'Open'].copy()
        if open_bets.empty:
            st.info("🎉 恭喜！目前沒有尚未結算的注單。")
        else:
            bet_opts = [f"{r['ID']} | {r['Date']} | {r.get('Match', '')} | {r.get('Bet_Type', '')}: {r.get('Selection', '')}" for _, r in open_bets.iterrows()]
            selected_bet_str = st.selectbox("請選擇要結算的注單", bet_opts)
            selected_id = selected_bet_str.split(" | ")[0]
            
            target_row = open_bets[open_bets['ID'] == selected_id].iloc[0]
            
            st.markdown(f"**目前正在結算:** `{selected_id}` | `{target_row.get('Match', '')}` | `{target_row.get('Bet_Type', '')}` 選擇: `{target_row.get('Selection', '')}` @ `{target_row.get('Initial_Line', 0)}`")
            
            st.subheader("1. 最終比分與核心賽果")
            rg1, rg2, rc1, rc2 = st.columns(4)
            h_g = rg1.number_input("主隊進球 (Home Goals)", min_value=0, value=st.session_state.get('edit_h_g', 0))
            a_g = rg2.number_input("客隊進球 (Away Goals)", min_value=0, value=st.session_state.get('edit_a_g', 0))
            h_c = rc1.number_input("主隊角球 (Home Corners)", min_value=0, value=st.session_state.get('edit_h_c', 0))
            a_c = rc2.number_input("客隊角球 (Away Corners)", min_value=0, value=st.session_state.get('edit_a_c', 0))
            
            with st.expander("2. 📊 進階場上數據錄入 (Advanced Stats for Future ML)"):
                st.write("這些數據將大幅增加未來機器學習的準確度。若無資料可留空(0)。")
                adc1, adc2, adc3 = st.columns(3)
                h_da = adc1.number_input("主隊危險進攻 (Home DA)", min_value=0, value=st.session_state.get('edit_h_da', 0))
                a_da = adc1.number_input("客隊危險進攻 (Away DA)", min_value=0, value=st.session_state.get('edit_a_da', 0))
                
                h_sot = adc2.number_input("主隊射正 (Home SoT)", min_value=0, value=st.session_state.get('edit_h_sot', 0))
                a_sot = adc2.number_input("客隊射正 (Away SoT)", min_value=0, value=st.session_state.get('edit_a_sot', 0))
                
                h_poss = adc3.number_input("主隊控球率% (Home Poss.)", min_value=0, max_value=100, value=st.session_state.get('edit_h_poss', 50))
                a_poss = 100 - h_poss
                st.markdown(f"*自動推算客隊控球率:* **{a_poss}%**")

            if st.button("🔨 執行系統結算並歸檔 (Execute Settlement)", type="primary", use_container_width=True):
                sys_stake = float(target_row['System_Stake'])
                user_stake = float(target_row['User_Stake'])
                odds = float(target_row['Initial_Odds'])
                line = float(target_row['Initial_Line'])
                b_type = str(target_row['Bet_Type'])
                sel = str(target_row['Selection'])
                
                sys_prof, user_prof, sys_pay, user_pay, unit_prof, res_lbl, diff = calculate_settlement(
                    b_type, sel, line, odds, sys_stake, user_stake, h_g, a_g, h_c, a_c
                )
                
                idx = df_db.index[df_db['ID'] == selected_id].tolist()[0]
                
                df_db.at[idx, 'Status'] = 'Settled'
                df_db.at[idx, 'Home_Goal'] = h_g
                df_db.at[idx, 'Away_Goal'] = a_g
                df_db.at[idx, 'Home_Corner'] = h_c
                df_db.at[idx, 'Away_Corner'] = a_c
                df_db.at[idx, 'Home_DA'] = h_da
                df_db.at[idx, 'Away_DA'] = a_da
                df_db.at[idx, 'Home_SoT'] = h_sot
                df_db.at[idx, 'Away_SoT'] = a_sot
                df_db.at[idx, 'Home_Possession'] = h_poss
                df_db.at[idx, 'Away_Possession'] = a_poss
                
                df_db.at[idx, 'System_Profit'] = sys_prof
                df_db.at[idx, 'User_Profit'] = user_prof
                df_db.at[idx, 'System_Payout'] = sys_pay
                df_db.at[idx, 'User_Payout'] = user_pay
                df_db.at[idx, 'Unit_Profit'] = unit_prof
                df_db.at[idx, 'Result_Label'] = res_lbl
                
                if st.session_state.inplay_odds_history:
                    df_db.at[idx, 'Odds_History'] = json.dumps(st.session_state.inplay_odds_history)
                else:
                    df_db.at[idx, 'Odds_History'] = json.dumps([])

                save_db(df_db, db_file, db_table)
                st.session_state.inplay_odds_history = [] 
                st.success(f"✅ 結算完成！注單 {selected_id} 狀態已更新。系統盈虧: ${sys_prof}, 派彩: {res_lbl}。")
                st.rerun()

    # ==========================================
    # Tab 4: 戰績累計分析 (Report)
    # ==========================================
    with t_report:
        st.header("📈 戰績累計分析與圖表 (Performance Report)")
        
        df_settled = df_db[df_db['Status'] == 'Settled'].copy()
        if df_settled.empty:
            st.info("尚無結算的注單可供分析。")
        else:
            display_cumulative_metrics(df_settled)
            
            st.subheader("趨勢與分類表現")
            report_col1, report_col2 = st.columns(2)
            
            with report_col1:
                st.markdown("**按聯賽級別分類的淨盈虧**")
                cat_perf = df_settled.groupby('Tournament_Category')['System_Profit'].sum().reset_index()
                st.dataframe(cat_perf, use_container_width=True)
                
            with report_col2:
                st.markdown("**按盤口類型分類的勝率**")
                def calc_win_rate(group):
                    wins = len(group[group['Unit_Profit'] > 0])
                    total = len(group)
                    return round(wins / total * 100, 2) if total > 0 else 0
                
                type_perf = df_settled.groupby('Bet_Type').apply(calc_win_rate).reset_index()
                type_perf.columns = ['Bet_Type', 'Win Rate (%)']
                st.dataframe(type_perf, use_container_width=True)

if __name__ == '__main__':
    main()
