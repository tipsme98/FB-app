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

# 安全取得 st.secrets 金鑰防呆處理
def get_secret(key):
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return None

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

# GitHub API 讀取與寫入輔助函式 (已加上 timeout 避免卡死白屏)
def load_db_github(repo, path, token):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return pd.read_csv(io.StringIO(res.text))
    except Exception:
        pass
    return None

def save_db_github(df, repo, path, token):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}"}
    try:
        res_get = requests.get(url, headers=headers, timeout=10)
        sha = res_get.json().get("sha") if res_get.status_code == 200 else None
        
        csv_content = df.to_csv(index=False)
        content_b64 = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
        
        payload = {
            "message": f"Auto-update {path} via Streamlit App",
            "content": content_b64
        }
        if sha:
            payload["sha"] = sha
            
        requests.put(url, json=payload, headers=headers, timeout=10)
    except Exception:
        pass

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
    if HAS_SQLALCHEMY:
        db_url = get_secret("DB_URL")
        if db_url:
            try:
                engine = create_engine(db_url, connect_args={'connect_timeout': 5})
                df = pd.read_sql_table(table_name, engine)
                df = process_legacy_columns(df)
            except Exception:
                pass

    # 策略 B: 嘗試 GitHub API 自動同步
    if df.empty:
        gh_token = get_secret("GITHUB_TOKEN")
        gh_repo = get_secret("GITHUB_REPO")
        if gh_token and gh_repo:
            try:
                gh_df = load_db_github(gh_repo, filename, gh_token)
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
    if HAS_SQLALCHEMY:
        db_url = get_secret("DB_URL")
        if db_url:
            try:
                engine = create_engine(db_url, connect_args={'connect_timeout': 5})
                df_to_db = df.copy()
                for col in df_to_db.columns:
                    if df_to_db[col].dtype == 'object':
                        df_to_db[col] = df_to_db[col].apply(lambda x: str(x) if pd.notna(x) else None)
                df_to_db.to_sql(table_name, engine, if_exists='replace', index=False)
            except Exception:
                pass

    gh_token = get_secret("GITHUB_TOKEN")
    gh_repo = get_secret("GITHUB_REPO")
    if gh_token and gh_repo:
        try:
            save_db_github(df, gh_repo, filename, gh_token)
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
    """通用的 API 請求函數 (已補上 timeout 防止卡死)"""
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
                    margin = round((1/upper) + (1/lower), 3) if (upper > 0 and lower > 0) else 1.085
                    new_history.append({
                        "id": row_id, "type": bet_type_cn, "record_time": "", "line": line,
                        "upper": upper, "lower": lower, "unlock": True, "margin": margin
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

def parse_dt_for_comparison(dt_str, fallback_idx):
    """解析日期時間字串用於版本最新比對，若無有效時間則以 fallback_idx 為準"""
    if not dt_str or not str(dt_str).strip():
        return (datetime.min, fallback_idx)
    s = str(dt_str).strip()
    for fmt in ["%Y-%m-%d %H:%M", "%m-%d %H:%M", "%d-%m %H:%M", "%Y/%m/%d %H:%M", "%m/%d %H:%M"]:
        try:
            return (datetime.strptime(s, fmt), fallback_idx)
        except ValueError:
            pass
    return (datetime.min, fallback_idx)

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
            cleaned_l_item = re.sub(r'\[.*?\]', l_item)
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

# ==========================================
# 5.5 智能下注輔助計算
# ==========================================
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
    
    st.session_state.df_db = load_db(db_file, DB_COLUMNS, db_table)
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
                'ID': f"C{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
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

    t_pre, t_inplay, t_settle, t_ai = st.tabs(["📝 賽前建檔與投注", "⏱️ 即場賽事與預測", "⚖️ 賽果結算與管理", "🤖 全局模型"])

    with t_pre:
        st.subheader("📝 賽事建檔與智能盤口走勢分析")
        
        # --- 頂部修改與覆蓋控制區塊 ---
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

        if sys_bankroll <= 0: st.warning("⚠️ 目前系統可用資金不足！無法精確計算建議注碼。請先至側邊欄存入本金。")
        
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
            df_meso = df_settled[df_settled['Tournament_Category'] == tournament_category]
            df_macro = df_settled
            
            res_micro = evaluate_dimension(df_micro, "微觀 - 賽事名稱", candidates_base, rating_map, h_data)
            res_meso = evaluate_dimension(df_meso, "中觀 - 賽事分類", candidates_base, rating_map, h_data)
            res_macro = evaluate_dimension(df_macro, "宏觀 - 總數據", candidates_base, rating_map, h_data)
            
            valid_res = [r for r in [res_micro, res_meso, res_macro] if r['valid']]
            best_model = max(valid_res, key=lambda x: x['score']) if valid_res else res_macro
            if not valid_res: best_model['msg'] = "所有維度樣本數不足，降級為純基礎期望值運算。"

            best_bet = best_model['best'] if 'best' in best_model else candidates_base[0]

            st.session_state.analysis_result = {
                'micro': res_micro, 'meso': res_meso, 'macro': res_macro, 'best_model': best_model,
                'best_bet': best_bet, 't_name': tournament_name, 't_cat': tournament_category
            }
            
        if st.session_state.get('show_analysis', False):
            res = st.session_state.analysis_result
            st.success("✅ 三維度數據分析與 EV 運算完成！")
            
            # 建立每一個 (盤口類型, 盤口線) 最新時間點記錄的對照表
            latest_history_map = {}
            for idx, r in enumerate(st.session_state.odds_history):
                b_type = str(r.get('type', '讓球'))
                try:
                    line_val = float(r.get('line', 0.0))
                except:
                    line_val = 0.0
                rec_time = str(r.get('record_time', ''))
                
                key = (b_type, line_val)
                dt_tuple = parse_dt_for_comparison(rec_time, idx)
                if key not in latest_history_map or dt_tuple > latest_history_map[key][1]:
                    latest_history_map[key] = (idx, dt_tuple)

            c1, c2, c3 = st.columns(3)
            for col, r, title in zip([c1, c2, c3], [res['micro'], res['meso'], res['macro']], ["A. 微觀 (賽事名稱)", "B. 中觀 (賽事分類)", "C. 宏觀 (全局數據)"]):
                with col.container(border=True):
                    st.markdown(f"**{title}**")
                    if r['valid']:
                        st.write(f"樣本數: `{r['n']}` 場")
                        st.write(f"系統策略 ROI: `{r['roi']*100:.1f}%`")
                        st.write(f"歷史勝率: `{r['acc']*100:.1f}%`")
                    else:
                        st.warning(r['msg'])

            bm = res['best_model']
            bb = res['best_bet']
            dim_label_map = {"微觀 - 賽事名稱": "微觀", "中觀 - 賽事分類": "中觀", "宏觀 - 總數據": "宏觀"}
            dim_short = dim_label_map.get(bm['dim'], "宏觀")

            st.markdown(f"### 🧠 AI 預測模型推薦")
            
            st.markdown("#### 📊 所有盤口評估明細 (完整歷史時間點評估與 EV 運算)")
            cand_list = bm.get('candidates', [])
            
            category_order = ["讓球", "入球大小", "角球大小"]
            all_options = []
            option_map = {}
            
            if cand_list:
                df_show = pd.DataFrame(cand_list)
                df_show['推薦排序'] = range(1, len(df_show) + 1)
                df_show = df_show[['推薦排序', 'bet_type', 'line', 'label', 'odds', 'prob', 'ev']]
                df_show.columns = ['推薦排序', '盤口類型', '盤口線', '投注方向', '賠率', '預期勝率', '期望值 (EV)']
                df_show['預期勝率'] = df_show['預期勝率'].apply(lambda x: f"{x*100:.2f}%")
                df_show['期望值 (EV)'] = df_show['期望值 (EV)'].apply(lambda x: f"{x:.3f}")
                
                def highlight_first(row):
                    if row.name == 0:
                        return ['background-color: rgba(40, 167, 69, 0.2)'] * len(row)
                    return [''] * len(row)
                    
                st.dataframe(df_show.style.apply(highlight_first, axis=1), use_container_width=True)
                
                # 下拉選單去重與分組：按「讓球」、「入球大小」、「角球大小」分組，各盤口線僅保留最新版本
                for cat in category_order:
                    cat_cands = [c for c in cand_list if c.get('bet_type') == cat]
                    for c in cat_cands:
                        try:
                            line_val = float(c.get('line', 0.0))
                        except:
                            line_val = 0.0
                        c_idx = c.get('history_idx')
                        
                        latest_info = latest_history_map.get((cat, line_val))
                        latest_idx = latest_info[0] if latest_info else None
                        
                        if c_idx is None or c_idx == latest_idx:
                            opt_str = f"【{c['bet_type']}】{c['label']} @ {c['odds']}"
                            if opt_str not in option_map:
                                all_options.append(opt_str)
                                option_map[opt_str] = c

                # 編輯模式下防呆補全
                if is_editing:
                    eb = st.session_state.get('edit_bet_type')
                    el = st.session_state.get('edit_line')
                    es = st.session_state.get('edit_selection')
                    if eb and el is not None and es:
                        for c in cand_list:
                            if c['bet_type'] == eb and abs(float(c['line']) - float(el)) < 1e-4 and c['selection'] == es:
                                opt_str = f"【{c['bet_type']}】{c['label']} @ {c['odds']}"
                                if opt_str not in option_map:
                                    all_options.append(opt_str)
                                    option_map[opt_str] = c
                                break
            
            st.info(f"系統分析顯示，針對『{res['t_name']}』，採用『{bm['dim']}』級別的模型進行運算，其歷史準確率與 EV 獲利期望值最高，故本次投注策略依據此模型生成。")
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("💡 首選推薦", f"{bb['bet_type']} - {bb['label']}")
            mc2.metric(f"🎯 預期勝率 ({dim_short}修正)", f"{bb.get('prob', bb.get('base_prob',0))*100:.1f}%")
            mc3.metric("📊 修正 EV", f"{bb.get('ev', 0):.3f}")
            
            st.markdown("---")
            st.markdown("### 🎯 最終投注決策與注碼配置 (已簡化分組選單)")
            
            col_sys, col_usr = st.columns(2)
            
            with col_sys:
                st.markdown("#### 🤖 系統投注 (System)")
                st.caption("供模型學習及系統資金策略驗證使用")
                default_sys = []
                if is_editing and st.session_state.get('edit_sys_stake', 0) > 0:
                    eb = st.session_state.get('edit_bet_type')
                    el = st.session_state.get('edit_line')
                    es = st.session_state.get('edit_selection')
                    for opt, c in option_map.items():
                        if c['bet_type'] == eb and abs(float(c['line']) - float(el)) < 1e-4 and c['selection'] == es:
                            default_sys.append(opt)
                            break
                elif not is_editing and cand_list:
                    best_c = cand_list[0]
                    if best_c.get('ev', 0) > 0:
                        for opt, c in option_map.items():
                            if c['bet_type'] == best_c['bet_type'] and abs(float(c['line']) - float(best_c['line'])) < 1e-4 and c['selection'] == best_c['selection']:
                                default_sys.append(opt)
                                break
                
                sys_selected = st.multiselect("選擇系統投注項目", all_options, default=default_sys, key="sys_multi")
                sys_stakes = {}
                for sel in sys_selected:
                    cand = option_map[sel]
                    sug_stk, raw_stk = calc_suggested_stake(cand, sys_bankroll, sys_max_stake)
                    sys_stakes[sel] = st.number_input(f"系統建議金額: {sel}", value=float(sug_stk), disabled=True, key=f"s_stk_{sel}")
                    if "讓球" in cand['bet_type'] and 0 < raw_stk < 200:
                        if cand.get('ev', 0) >= 0.03 and cand.get('prob', 0) >= 0.50:
                            st.caption(f"💡 `{cand['bet_type']}` EV/勝率達標，系統自動升級最低注碼 $200")
                        else:
                            st.caption(f"⚠ `{cand['bet_type']}` EV未達標，系統建議放棄 (注碼 $0)")
            
            with col_usr:
                st.markdown("#### 👤 用家投注 (User)")
                st.caption("真實資金決策，不用於系統學習。可跟單或反買")
                default_usr = []
                if is_editing and st.session_state.get('edit_user_stake', 0) > 0:
                    eb = st.session_state.get('edit_bet_type')
                    el = st.session_state.get('edit_line')
                    es = st.session_state.get('edit_selection')
                    for opt, c in option_map.items():
                        if c['bet_type'] == eb and abs(float(c['line']) - float(el)) < 1e-4 and c['selection'] == es:
                            default_usr.append(opt)
                            break
                
                usr_selected = st.multiselect("選擇用家投注項目", all_options, default=default_usr, key="usr_multi")
                usr_stakes = {}
                for sel in usr_selected:
                    def_val = 100.0
                    if is_editing and sel in default_usr:
                        def_val = float(st.session_state.get('edit_user_stake', 100.0))
                    usr_stakes[sel] = st.number_input(f"用家自訂金額 ($): {sel}", min_value=0.0, step=10.0, value=def_val, key=f"u_stk_{sel}")
            
            st.write("")
            submit_btn_label = f"🔄 確定修改並覆蓋雲端資料庫 (ID: {st.session_state.editing_bet_id})" if is_editing else "✅ 確定投注並寫入雲端資料庫"
            
            if st.button(submit_btn_label, type="primary", use_container_width=True):
                all_keys = set(sys_selected + usr_selected)
                if not all_keys:
                    st.warning("⚠️ 請至少在系統或用家選擇一項投注！")
                else:
                    target_id = st.session_state.get('editing_bet_id')
                    if target_id:
                        st.session_state.df_db = st.session_state.df_db[st.session_state.df_db['ID'] != target_id]
                    
                    new_records = []
                    for i, sel in enumerate(all_keys):
                        cand = option_map[sel]
                        s_stk = sys_stakes.get(sel, 0.0)
                        u_stk = usr_stakes.get(sel, 0.0)
                        
                        if target_id and len(all_keys) == 1:
                            new_id = target_id
                        else:
                            new_id = f"B{datetime.now().strftime('%Y%m%d%H%M%S')}{i}"
                            
                        new_record = {
                            'ID': new_id, 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Status': 'Open',
                            'Tournament_Name': res['t_name'], 'Tournament_Category': res['t_cat'], 
                            'Match': f"{home_team} vs {away_team}", 'Home_Team': home_team, 'Away_Team': away_team,
                            'Home_Rating': home_rating, 'Away_Rating': away_rating, 'Home_Form': home_form, 'Away_Form': away_form,
                            'Bet_Type': cand['bet_type'], 'Selection': cand['selection'], 'Initial_Line': cand['line'], 'Initial_Odds': cand['odds'], 
                            'System_Stake': s_stk, 'User_Stake': u_stk,
                            'Odds_History': json.dumps(st.session_state.odds_history, ensure_ascii=False)
                        }
                        new_records.append(new_record)
                        
                    st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame(new_records)], ignore_index=True)
                    st.toast("✅ 投注紀錄雲端同步成功！", icon="📝")
                    st.session_state.last_bet_id = new_records[-1]['ID']
                    save_db(st.session_state.df_db, db_file, db_table)
                    
                    clear_edit_mode()
                    st.session_state.odds_history = [{"id": 0, "type": "讓球", "record_time": "", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}] 
                    st.session_state.show_analysis = False
                    st.rerun()

    with t_inplay:
        st.subheader("⏱ 即場賽事實時更新與智慧火力分析")
        
        # --- 頂部修改與覆蓋控制區塊 ---
        if st.session_state.editing_bet_id:
            st.info(f"🛠️ **【修改/覆蓋模式】** 目前正在編輯未結算注單：`{st.session_state.editing_bet_id}`。修改後提交將**直接覆蓋**資料庫中的原有數據。")
            if st.button("❌ 取消修改 (恢復為新增注單)", key="cancel_edit_inplay"):
                clear_edit_mode()
                st.rerun()
        elif st.session_state.last_bet_id:
            last_match = st.session_state.df_db[st.session_state.df_db['ID'] == st.session_state.last_bet_id]
            if not last_match.empty and last_match.iloc[0]['Status'] == 'Open':
                c_msg, c_btn = st.columns([3, 1])
                c_msg.info(f"💡 剛提交注單 ID: **{st.session_state.last_bet_id}** ({last_match.iloc[0]['Match']})。如發現資料有錯漏，可隨時載入修改。")
                if c_btn.button("✏️ 載入該注單修改", key="load_last_inplay"):
                    load_bet_to_edit(st.session_state.last_bet_id)
                    st.rerun()

        with st.expander("✏️ 載入 / 修改既有未結算即場注單 (Edit Open In-Play Bet)"):
            open_bets_list = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
            if open_bets_list.empty:
                st.caption("目前沒有未結算的注單。")
            else:
                opts_open = [f"{r['ID']} | {r['Date']} | {r['Match']} | {r['Bet_Type']} ({r['Selection']})" for _, r in open_bets_list.iterrows()]
                sel_open_bet = st.selectbox("選擇要修改的即場注單", opts_open, key="sel_open_edit_inplay")
                if st.button("📥 載入所選注單資料至表單", key="btn_load_edit_inplay"):
                    target_id = sel_open_bet.split(" | ")[0]
                    load_bet_to_edit(target_id)
                    st.rerun()
                    
        st.divider()

        st.markdown("##### ⚡ 一鍵同步即場賽況 (API)")
        col_api_id, col_api_btn = st.columns([2, 1])
        inplay_match_id = col_api_id.text_input("輸入賽事 ID", key="api_match_id")
        if col_api_btn.button("🔄 自動同步數據", key="btn_sync_inplay"):
            if inplay_match_id:
                if parse_and_fill_inplay(inplay_match_id):
                    st.success("✅ 即場數據與盤口同步成功！")
                    st.rerun()
                else:
                    st.warning("⚠️ 找不到賽事或 API 請求失敗。")

        if 'inplay_odds_history' not in st.session_state:
            st.session_state.inplay_odds_history = [{"id": 0, "type": "讓球", "record_time": "", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}]

        st.markdown("##### 1. 實時賽況與進攻數據")
        col_ip1, col_ip2, col_ip3 = st.columns(3)
        h_g = col_ip1.number_input("主隊入球", 0, 50, st.session_state.get('edit_h_g', 0), key="ip_hg")
        a_g = col_ip1.number_input("客隊入球", 0, 50, st.session_state.get('edit_a_g', 0), key="ip_ag")
        h_c = col_ip2.number_input("主隊角球", 0, 50, st.session_state.get('edit_h_c', 0), key="ip_hc")
        a_c = col_ip2.number_input("客隊角球", 0, 50, st.session_state.get('edit_a_c', 0), key="ip_ac")
        h_red = col_ip3.number_input("主隊紅牌", 0, 10, st.session_state.get('edit_h_red', 0), key="ip_hred")
        a_red = col_ip3.number_input("客隊紅牌", 0, 10, st.session_state.get('edit_a_red', 0), key="ip_ared")
        
        c1, c2, c3, c4 = st.columns(4)
        h_da = c1.number_input("主隊危險進攻", 0, 200, st.session_state.get('edit_h_da', 0), key="ip_hda")
        a_da = c1.number_input("客隊危險進攻", 0, 200, st.session_state.get('edit_a_da', 0), key="ip_ada")
        h_sot = c2.number_input("主隊射正", 0, 50, st.session_state.get('edit_h_sot', 0), key="ip_hsot")
        a_sot = c2.number_input("客隊射正", 0, 50, st.session_state.get('edit_a_sot', 0), key="ip_asot")
        h_soff = c3.number_input("主隊射偏", 0, 50, st.session_state.get('edit_h_soff', 0), key="ip_hsoff")
        a_soff = c3.number_input("客隊射偏", 0, 50, st.session_state.get('edit_a_soff', 0), key="ip_asoff")
        h_poss = c4.number_input("主隊控球率 (%)", 0, 100, st.session_state.get('edit_h_poss', 50), key="ip_hposs")
        a_poss = 100 - h_poss
        st.caption(f"客隊控球率自動計算為: {a_poss}%")
        
        if st.session_state.get('editing_bet_id'):
            if st.button("💾 儲存實時賽況至該注單", type="primary"):
                target_id = st.session_state.editing_bet_id
                mask = st.session_state.df_db['ID'] == target_id
                st.session_state.df_db.loc[mask, 'Home_Goal'] = h_g
                st.session_state.df_db.loc[mask, 'Away_Goal'] = a_g
                st.session_state.df_db.loc[mask, 'Home_Corner'] = h_c
                st.session_state.df_db.loc[mask, 'Away_Corner'] = a_c
                st.session_state.df_db.loc[mask, 'Home_Red'] = h_red
                st.session_state.df_db.loc[mask, 'Away_Red'] = a_red
                st.session_state.df_db.loc[mask, 'Home_DA'] = h_da
                st.session_state.df_db.loc[mask, 'Away_DA'] = a_da
                st.session_state.df_db.loc[mask, 'Home_SoT'] = h_sot
                st.session_state.df_db.loc[mask, 'Away_SoT'] = a_sot
                st.session_state.df_db.loc[mask, 'Home_SoFF'] = h_soff
                st.session_state.df_db.loc[mask, 'Away_SoFF'] = a_soff
                st.session_state.df_db.loc[mask, 'Home_Possession'] = h_poss
                st.session_state.df_db.loc[mask, 'Away_Possession'] = a_poss
                
                save_db(st.session_state.df_db, db_file, db_table)
                st.success("✅ 實時賽況已成功更新至資料庫！")
                clear_edit_mode()
                st.rerun()

    with t_settle:
        st.subheader("⚖️ 賽果結算與管理")
        open_bets = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
        
        if open_bets.empty:
            st.info("目前沒有待結算的注單。")
        else:
            settle_opts = [f"{r['ID']} | {r['Match']} | {r['Bet_Type']} ({r['Selection']}) | 盤口: {r['Initial_Line']}" for _, r in open_bets.iterrows()]
            sel_settle = st.selectbox("選擇要結算的注單", settle_opts)
            
            if sel_settle:
                target_id = sel_settle.split(" | ")[0]
                target_row = open_bets[open_bets['ID'] == target_id].iloc[0]
                
                # --- 修正程式碼末端截斷處並補齊下方所有結算與AI檢驗區塊 ---
                st.markdown("##### ⚡ 一鍵同步賽果 (API)")
                col_s_id, col_s_btn = st.columns([2, 1])
                settle_match_id = col_s_id.text_input("輸入賽事 ID", key="api_settle_match_id")
                if col_s_btn.button("🔄 自動同步賽果", key="btn_sync_settle"):
                    if settle_match_id:
                        if parse_and_fill_inplay(settle_match_id):
                            st.success("✅ 賽果數據同步成功！請確認下方欄位數據後點擊結算。")
                        else:
                            st.warning("⚠️ 找不到賽事或 API 請求失敗。")

                st.markdown("##### 手動結算賽果輸入")
                c_shg, c_sag = st.columns(2)
                s_hg = c_shg.number_input("全場主隊入球", 0, 50, st.session_state.get('edit_h_g', 0), key="s_hg")
                s_ag = c_sag.number_input("全場客隊入球", 0, 50, st.session_state.get('edit_a_g', 0), key="s_ag")
                
                c_shc, c_sac = st.columns(2)
                s_hc = c_shc.number_input("全場主隊角球", 0, 50, st.session_state.get('edit_h_c', 0), key="s_hc")
                s_ac = c_sac.number_input("全場客隊角球", 0, 50, st.session_state.get('edit_a_c', 0), key="s_ac")

                if st.button("⚖️ 確認結算此注單", type="primary"):
                    b_type = str(target_row['Bet_Type'])
                    sel = str(target_row['Selection'])
                    line = float(target_row['Initial_Line']) if pd.notna(target_row['Initial_Line']) else 0.0
                    odds = float(target_row['Initial_Odds']) if pd.notna(target_row['Initial_Odds']) else 1.90
                    sys_s = float(target_row['System_Stake']) if pd.notna(target_row['System_Stake']) else 0.0
                    usr_s = float(target_row['User_Stake']) if pd.notna(target_row['User_Stake']) else 0.0
                    
                    sys_p, usr_p, sys_pay, usr_pay, unit_p, res_label, diff = calculate_settlement(
                        b_type, sel, line, odds, sys_s, usr_s, s_hg, s_ag, s_hc, s_ac
                    )
                    
                    mask = st.session_state.df_db['ID'] == target_id
                    st.session_state.df_db.loc[mask, 'Status'] = 'Settled'
                    st.session_state.df_db.loc[mask, 'Home_Goal'] = s_hg
                    st.session_state.df_db.loc[mask, 'Away_Goal'] = s_ag
                    st.session_state.df_db.loc[mask, 'Home_Corner'] = s_hc
                    st.session_state.df_db.loc[mask, 'Away_Corner'] = s_ac
                    st.session_state.df_db.loc[mask, 'Result_Label'] = res_label
                    st.session_state.df_db.loc[mask, 'System_Profit'] = sys_p
                    st.session_state.df_db.loc[mask, 'User_Profit'] = usr_p
                    st.session_state.df_db.loc[mask, 'Unit_Profit'] = unit_p
                    st.session_state.df_db.loc[mask, 'System_Payout'] = sys_pay
                    st.session_state.df_db.loc[mask, 'User_Payout'] = usr_pay
                    
                    save_db(st.session_state.df_db, db_file, db_table)
                    st.success(f"✅ 結算完成！結果: {res_label} | 系統盈虧: ${sys_p} | 用家盈虧: ${usr_p}")
                    st.rerun()

    with t_ai:
        st.subheader("🤖 全局 AI 模型效能與數據統計")
        display_cumulative_metrics(st.session_state.df_db)
        
        st.markdown("##### 📌 各維度勝率與 ROI 統計")
        df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
        if not df_settled.empty:
            df_settled['System_Stake'] = pd.to_numeric(df_settled['System_Stake'], errors='coerce')
            df_settled['System_Profit'] = pd.to_numeric(df_settled['System_Profit'], errors='coerce')
            df_settled['Unit_Profit'] = pd.to_numeric(df_settled['Unit_Profit'], errors='coerce')
            
            grp_cat = df_settled.groupby('Tournament_Category').agg(
                場數=('ID', 'count'),
                系統總投注=('System_Stake', 'sum'),
                系統總盈虧=('System_Profit', 'sum'),
                勝場=('Unit_Profit', lambda x: (x > 0).sum())
            ).reset_index()
            grp_cat['勝率'] = (grp_cat['勝場'] / grp_cat['場數']).apply(lambda x: f"{x*100:.2f}%")
            grp_cat['ROI'] = (grp_cat['系統總盈虧'] / grp_cat['系統總投注']).fillna(0).apply(lambda x: f"{x*100:.2f}%")
            
            st.markdown("**1. 依賽事分類 (Tournament Category)**")
            st.dataframe(grp_cat, use_container_width=True)
            
            grp_type = df_settled.groupby('Bet_Type').agg(
                場數=('ID', 'count'),
                系統總投注=('System_Stake', 'sum'),
                系統總盈虧=('System_Profit', 'sum'),
                勝場=('Unit_Profit', lambda x: (x > 0).sum())
            ).reset_index()
            grp_type['勝率'] = (grp_type['勝場'] / grp_type['場數']).apply(lambda x: f"{x*100:.2f}%")
            grp_type['ROI'] = (grp_type['系統總盈虧'] / grp_type['系統總投注']).fillna(0).apply(lambda x: f"{x*100:.2f}%")
            
            st.markdown("**2. 依盤口類型 (Bet Type)**")
            st.dataframe(grp_type, use_container_width=True)
        else:
            st.info("尚無結算紀錄可供統計分析。")

if __name__ == "__main__":
    main()
