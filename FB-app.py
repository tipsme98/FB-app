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
    except Exception:
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
    
    sys_max_stake = max(0.0, (sys_net + sys_profit) * 0.10)
    user_max_stake = max(0.0, (usr_net + user_profit) * 0.10)
    
    return (
        round(sys_dep, 2), round(sys_wit, 2), round(sys_net, 2), round(sys_profit, 2), round(sys_bankroll, 2), round(sys_max_stake, 2),
        round(usr_dep, 2), round(usr_wit, 2), round(usr_net, 2), round(user_profit, 2), round(user_bankroll, 2), round(user_max_stake, 2)
    )

def calculate_settlement(bet_type, selection, line, odds, sys_stake, user_stake, h_g, a_g, h_c=0, a_c=0):
    diff = 0.0
    clean_btype = str(bet_type).replace(" (即場)", "")
    
    if clean_btype == '讓球':
        if selection in ['Home', '主勝', '主']: diff = h_g + line - a_g
        elif selection in ['Away', '客勝', '客']: diff = a_g - line - h_g
    elif clean_btype == '入球大小':
        total_goals = h_g + a_g
        if selection in ['Over', '大']: diff = total_goals - line
        elif selection in ['Under', '小']: diff = line - total_goals
    elif clean_btype == '角球大小':
        total_corners = h_c + a_c
        if selection in ['Over', '大']: diff = total_corners - line
        elif selection in ['Under', '小']: diff = line - total_corners

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
    
    st.session_state.edit_user_stake = float(row.get('User_Stake', 0.0)) if pd.notna(row.get('User_Stake')) else 0.0
    st.session_state.edit_bet_type = str(row.get('Bet_Type', '')).replace(" (即場)", "") if pd.notna(row.get('Bet_Type')) else ''
    st.session_state.edit_selection = str(row.get('Selection', 'Home')) if pd.notna(row.get('Selection')) else 'Home'

def clear_edit_mode():
    """清除修改模式狀態"""
    st.session_state.editing_bet_id = None
    for k in list(st.session_state.keys()):
        if k.startswith('edit_'):
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
        except Exception:
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
        st.subheader("2. ↩️️ 狀態重置 (Undo 復原中心)")
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

def render_odds_section(odds_history_state, prefix="pre"):
    st.markdown("💡 **智能解析與動態同步區：** 請分別貼上各盤口數據（包含跨行的日期及時間、盤口、整數或小數賠率）。系統會自動依時間區塊與上下盤位置智慧解析並自動去除重複賠率。您亦可直接於下方表格勾選刪除或修改資料。")
    
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
            
    df['刪除'] = False
            
    df_display = df[['刪除', 'record_time', 'type', 'line', 'upper', 'lower']].copy()
    df_display.columns = ['🗑️刪除', '📅日期及時間', '盤口類型', '盤口線', '主隊/大盤賠率', '客隊/小盤賠率']
    
    st.markdown("##### 📝 盤口與賠率走勢表 (可勾選刪除、修改日期時間、盤口或賠率)")
    edited_df = st.data_editor(
        df_display,
        num_rows="dynamic",
        column_config={
            "🗑️刪除": st.column_config.CheckboxColumn("刪除", default=False),
            "📅日期及時間": st.column_config.TextColumn("📅日期及時間", required=False),
            "盤口類型": st.column_config.SelectboxColumn("盤口類型", options=["讓球", "入球大小", "角球大小"], required=True),
            "盤口線": st.column_config.NumberColumn("盤口線", format="%.2f", required=True),
            "主隊/大盤賠率": st.column_config.NumberColumn("主隊/大盤賠率", min_value=1.01, format="%.2f", required=True),
            "客隊/小盤賠率": st.column_config.NumberColumn("客隊/小盤賠率", min_value=1.01, format="%.2f", required=True)
        },
        use_container_width=True,
        key=f"{prefix}_odds_editor"
    )
    
    new_history = []
    new_idx = 0
    for i, row in edited_df.iterrows():
        if row.get("🗑️刪除", False):
            continue
            
        try:
            up = float(row["主隊/大盤賠率"]) if pd.notna(row["主隊/大盤賠率"]) else 1.90
            lw = float(row["客隊/小盤賠率"]) if pd.notna(row["客隊/小盤賠率"]) else 1.90
            margin = (1/up) + (1/lw)
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
            "margin": margin
        })
        new_idx += 1
        
    odds_history_state.clear()
    odds_history_state.extend(new_history)

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
    if 'odds_history' not in st.session_state:
        st.session_state.odds_history = []
    if 'inplay_odds_history' not in st.session_state:
        st.session_state.inplay_odds_history = []
        
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
            
            act_type = 'Deposit' if 'Deposit' in cap_action else 'Withdraw'
            
            new_id = f"C{datetime.now().strftime('%Y%m%d%H%M%S')}"
            new_row = {
                'ID': new_id,
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Type': act_type,
                'Account': acc_val,
                'Amount': float(cap_amount),
                'Note': cap_note
            }
            
            st.session_state.df_cap = pd.concat([st.session_state.df_cap, pd.DataFrame([new_row])], ignore_index=True)
            save_db(st.session_state.df_cap, capital_file, cap_table)
            st.toast(f"✅ 成功寫入資金紀錄: {act_type} ${cap_amount}", icon="💰")
            st.rerun()

    if st.sidebar.button("📊 數據庫即時線上預覽與管理", use_container_width=True):
        preview_db_dialog(st.session_state.df_db, st.session_state.df_cap, db_file, capital_file, db_table, cap_table)

    # --- 主頁面分頁設計 ---
    t_pre, t_inplay, t_settle, t_ai = st.tabs([
        "⚽ 賽前精準建模與精確下注", 
        "⚡ 即場即時追單", 
        "🏁 賽果自動結算", 
        "🤖 全局模型與特徵對齊"
    ])

    # ==========================================
    # TAB 1: 賽前精準建模與精確下注
    # ==========================================
    with t_pre:
        display_cumulative_metrics(st.session_state.df_db)
        
        if st.session_state.editing_bet_id:
            st.info(f"✏️ 當前正在修改注單 ID: **{st.session_state.editing_bet_id}**")
            if st.button("❌ 取消修改並返回新建模式"):
                clear_edit_mode()
                st.rerun()

        st.subheader("1. 📋 賽事基本資訊與評級對陣")
        c1, c2, c3 = st.columns(3)
        with c1:
            t_name = st.text_input("賽事名稱 / 聯賽", value=st.session_state.get('edit_t_name', ''), placeholder="例如: 英超, 德甲")
            t_cat = st.selectbox("賽事類別", CATEGORY_OPTIONS, index=CATEGORY_OPTIONS.index(st.session_state.get('edit_t_cat', CATEGORY_OPTIONS[0])) if st.session_state.get('edit_t_cat') in CATEGORY_OPTIONS else 0)
        with c2:
            h_team = st.text_input("主隊名稱", value=st.session_state.get('edit_h_team', ''), placeholder="例如: 曼城")
            a_team = st.text_input("客隊名稱", value=st.session_state.get('edit_a_team', ''), placeholder="例如: 阿仙奴")
        with c3:
            h_rating = st.selectbox("主隊評級", ["A+", "A", "B", "C", "D"], index=["A+", "A", "B", "C", "D"].index(st.session_state.get('edit_h_rating', 'C')))
            a_rating = st.selectbox("客隊評級", ["A+", "A", "B", "C", "D"], index=["A+", "A", "B", "C", "D"].index(st.session_state.get('edit_a_rating', 'C')))

        st.markdown("##### 🏟 近況數據 (戰績紀錄)")
        fc1, fc2 = st.columns(2)
        with fc1:
            st.caption("主隊近 5 場戰績")
            hw = st.number_input("主隊 勝 (W)", min_value=0, max_value=5, value=st.session_state.get('edit_hw', 3))
            hd = st.number_input("主隊 和 (D)", min_value=0, max_value=5, value=st.session_state.get('edit_hd', 1))
            hl = st.number_input("主隊 負 (L)", min_value=0, max_value=5, value=st.session_state.get('edit_hl', 1))
            h_form_str = f"{hw}W{hd}D{hl}L"
        with fc2:
            st.caption("客隊近 5 場戰績")
            aw = st.number_input("客隊 勝 (W)", min_value=0, max_value=5, value=st.session_state.get('edit_aw', 2))
            ad = st.number_input("客隊 和 (D)", min_value=0, max_value=5, value=st.session_state.get('edit_ad', 2))
            al = st.number_input("客隊 負 (L)", min_value=0, max_value=5, value=st.session_state.get('edit_al', 1))
            a_form_str = f"{aw}W{ad}D{al}L"

        st.divider()
        st.subheader("2. 📈 賠率走勢紀錄")
        render_odds_section(st.session_state.odds_history, prefix="pre")

        st.divider()
        st.subheader("3. 🤖 三維度 ML 與 EV 精算推薦")
        
        rating_map = {"A+": 5, "A": 4, "B": 3, "C": 2, "D": 1}
        h_data = {
            'hr': rating_map.get(h_rating, 3),
            'ar': rating_map.get(a_rating, 3),
            'hf': hw * 3 + hd,
            'af': aw * 3 + ad
        }

        hd_state = get_last_odds_state(st.session_state.odds_history, "讓球", 0.0)
        ou_state = get_last_odds_state(st.session_state.odds_history, "入球大小", 2.5)
        cr_state = get_last_odds_state(st.session_state.odds_history, "角球大小", 9.5)

        candidates_base = [
            {"bet_type": "讓球", "selection": "Home", "line": hd_state['line'], "odds": hd_state['upper'], "base_prob": 0.50},
            {"bet_type": "讓球", "selection": "Away", "line": hd_state['line'], "odds": hd_state['lower'], "base_prob": 0.50},
            {"bet_type": "入球大小", "selection": "Over", "line": ou_state['line'], "odds": ou_state['upper'], "base_prob": 0.50},
            {"bet_type": "入球大小", "selection": "Under", "line": ou_state['line'], "odds": ou_state['lower'], "base_prob": 0.50},
            {"bet_type": "角球大小", "selection": "Over", "line": cr_state['line'], "odds": cr_state['upper'], "base_prob": 0.50},
            {"bet_type": "角球大小", "selection": "Under", "line": cr_state['line'], "odds": cr_state['lower'], "base_prob": 0.50},
        ]

        df_db = st.session_state.df_db
        settled_db = df_db[df_db['Status'] == 'Settled'] if not df_db.empty else pd.DataFrame()

        df_tourn = settled_db[settled_db['Tournament_Name'] == t_name] if not settled_db.empty and t_name else pd.DataFrame()
        df_rating = settled_db[(settled_db['Home_Rating'] == h_rating) & (settled_db['Away_Rating'] == a_rating)] if not settled_db.empty else pd.DataFrame()

        res_tourn = evaluate_dimension(df_tourn, "1. 聯賽維度", candidates_base, rating_map, h_data)
        res_rating = evaluate_dimension(df_rating, "2. 評級對陣維度", candidates_base, rating_map, h_data)
        res_global = evaluate_dimension(settled_db, "3. 全局歷史維度", candidates_base, rating_map, h_data)

        all_dims = [res_tourn, res_rating, res_global]
        best_dim = max(all_dims, key=lambda x: x['score'])
        best_c = best_dim['best']

        # 格式化顯示推薦選項（將盤口數值與選項整合顯示）
        sel_label_map = {
            'Home': f"主隊 ({best_c['line']:+.2f})" if best_c['bet_type'] == '讓球' else "主勝",
            'Away': f"客隊 ({-best_c['line']:+.2f})" if best_c['bet_type'] == '讓球' else "客勝",
            'Over': f"大 ({best_c['line']:.2f})",
            'Under': f"小 ({best_c['line']:.2f})"
        }
        disp_sel = sel_label_map.get(best_c['selection'], best_c['selection'])

        st.success(f"🌟 **首選精算推薦**: **【{best_c['bet_type']}】{disp_sel}** | 賠率: **{best_c['odds']:.2f}** | 盤口線: **{best_c['line']:.2f}** | 預期回報率 (EV): **{best_c['ev'] * 100:+.1f}%** | 勝率估算: **{best_c['prob'] * 100:.1f}%**")

        ev_cols = st.columns(3)
        for idx, res in enumerate(all_dims):
            with ev_cols[idx]:
                st.markdown(f"##### {res['dim']}")
                st.write(f"樣本數: {res['n']} 場 | ROI: {res['roi']*100:+.1f}% | 勝率: {res['acc']*100:.1f}%")
                if res['valid']:
                    bc = res['best']
                    b_sel_disp = sel_label_map.get(bc['selection'], bc['selection'])
                    st.info(f"最佳選擇: **{bc['bet_type']} - {b_sel_disp}** (盤口: {bc['line']:.2f}, 賠率: {bc['odds']:.2f}, EV: {bc['ev']*100:+.1f}%)")
                else:
                    st.caption(f"⚠️ {res['msg']}")

        st.divider()
        st.subheader("4. 💵 確認下注與寫入資料庫")
        
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            bet_type_sel = st.selectbox("玩法類型", ["讓球", "入球大小", "角球大小"], index=["讓球", "入球大小", "角球大小"].index(best_c['bet_type']) if best_c['bet_type'] in ["讓球", "入球大小", "角球大小"] else 0)
        with bc2:
            if bet_type_sel == "讓球":
                selection_sel = st.selectbox("下注目標", ["Home", "Away"], format_func=lambda x: f"主隊 ({h_team or 'Home'})" if x == 'Home' else f"客隊 ({a_team or 'Away'})")
                target_line = hd_state['line']
                target_odds = hd_state['upper'] if selection_sel == 'Home' else hd_state['lower']
            elif bet_type_sel == "入球大小":
                selection_sel = st.selectbox("下注目標", ["Over", "Under"], format_func=lambda x: "大 (Over)" if x == 'Over' else "小 (Under)")
                target_line = ou_state['line']
                target_odds = ou_state['upper'] if selection_sel == 'Over' else ou_state['lower']
            else:
                selection_sel = st.selectbox("下注目標", ["Over", "Under"], format_func=lambda x: "大 (Over)" if x == 'Over' else "小 (Under)")
                target_line = cr_state['line']
                target_odds = cr_state['upper'] if selection_sel == 'Over' else cr_state['lower']

        with bc3:
            final_line = st.number_input("盤口線 (Line)", value=float(target_line), step=0.25, format="%.2f")
            final_odds = st.number_input("賠率 (Odds)", value=float(target_odds), min_value=1.01, step=0.01, format="%.2f")

        with bc4:
            recommended_sys_stake = max(100.0, min(sys_max_stake, round((sys_bankroll * 0.05), 0))) if sys_bankroll > 0 else 100.0
            sys_stake_input = st.number_input("系統本金下注 ($)", min_value=0.0, value=float(recommended_sys_stake), step=100.0)
            user_stake_input = st.number_input("用家真實下注 ($)", min_value=0.0, value=float(st.session_state.get('edit_user_stake', recommended_sys_stake)), step=100.0)

        match_str = f"{h_team or '主隊'} vs {a_team or '客隊'}"
        
        btn_label = "💾 保存修改 (Update Bet)" if st.session_state.editing_bet_id else "🚀 確認下注並寫入資料庫 (Place Bet)"
        
        if st.button(btn_label, type="primary", use_container_width=True):
            if not h_team or not a_team:
                st.error("請填寫主隊與客隊名稱！")
            else:
                odds_hist_json = json.dumps(st.session_state.odds_history, ensure_ascii=False)
                bet_id = st.session_state.editing_bet_id if st.session_state.editing_bet_id else f"B{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                row_data = {
                    'ID': bet_id,
                    'Date': datetime.now().strftime('%Y-%m-%d'),
                    'Status': 'Open',
                    'Tournament_Name': t_name,
                    'Tournament_Category': t_cat,
                    'Match': match_str,
                    'Home_Team': h_team,
                    'Away_Team': a_team,
                    'Home_Rating': h_rating,
                    'Away_Rating': a_rating,
                    'Home_Form': h_form_str,
                    'Away_Form': a_form_str,
                    'Bet_Type': bet_type_sel,
                    'Selection': selection_sel,
                    'Initial_Line': final_line,
                    'Initial_Odds': final_odds,
                    'System_Stake': sys_stake_input,
                    'User_Stake': user_stake_input,
                    'Odds_History': odds_hist_json,
                    'InPlay_Minute': 0,
                    'Home_DA': 0, 'Away_DA': 0,
                    'Home_SoT': 0, 'Away_SoT': 0,
                    'Home_SoFF': 0, 'Away_SoFF': 0,
                    'Home_Red': 0, 'Away_Red': 0,
                    'Home_Sub': 0, 'Away_Sub': 0,
                    'Home_Possession': 50, 'Away_Possession': 50,
                    'Home_Goal': 0, 'Away_Goal': 0,
                    'Home_Corner': 0, 'Away_Corner': 0,
                    'Home_Goal_Conversion': 0.0, 'Away_Goal_Conversion': 0.0,
                    'Home_Firepower': 0.0, 'Away_Firepower': 0.0,
                    'Result_Label': '未結算',
                    'System_Profit': 0.0, 'User_Profit': 0.0, 'Unit_Profit': 0.0,
                    'System_Payout': 0.0, 'User_Payout': 0.0
                }

                if st.session_state.editing_bet_id:
                    st.session_state.df_db = st.session_state.df_db[st.session_state.df_db['ID'] != bet_id]
                
                st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([row_data])], ignore_index=True)
                save_db(st.session_state.df_db, db_file, db_table)
                
                st.toast(f"✅ 注單已成功保存！ID: {bet_id}", icon="⚽")
                clear_edit_mode()
                st.rerun()

    # ==========================================
    # TAB 2: 即場即時追單
    # ==========================================
    with t_inplay:
        st.subheader("⚡ 即場即時數據動態追單 (In-Play Live Tracking)")
        
        open_bets = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open'] if not st.session_state.df_db.empty else pd.DataFrame()
        
        col_sel, col_api = st.columns([2, 1])
        with col_sel:
            if not open_bets.empty:
                bet_options = ["新建/輸入賽事"] + [f"{r['ID']} | {r['Match']} | {r['Bet_Type']} ({r['Selection']})" for _, r in open_bets.iterrows()]
                selected_match_opt = st.selectbox("選擇既有進行中注單關聯賽事:", bet_options)
                if selected_match_opt != "新建/輸入賽事":
                    sel_id = selected_match_opt.split(" | ")[0]
                    if st.session_state.editing_bet_id != sel_id:
                        load_bet_to_edit(sel_id)
            else:
                st.info("目前無進行中注單，可手動輸入賽事進行即場分析。")

        with col_api:
            match_id_input = st.text_input("HKJC Match ID (自動抓取 API):", placeholder="例如: 12345")
            if st.button("🔄 同步即場數據與賠率", use_container_width=True):
                if match_id_input.strip():
                    with st.spinner("正在抓取 HKJC 即場數據與賠率..."):
                        if parse_and_fill_inplay(match_id_input.strip()):
                            st.toast("✅ API 數據與即場賠率同步成功！", icon="⚡")
                            st.rerun()
                        else:
                            st.error("無法取得即場數據，請確認 Match ID 是否正確。")
                else:
                    st.warning("請先輸入 Match ID。")

        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            inplay_min = st.number_input("比賽分鐘 (Minute)", min_value=1, max_value=120, value=st.session_state.get('edit_inplay_minute', 45))
            h_g = st.number_input("主隊進球 (Home Goal)", min_value=0, value=st.session_state.get('edit_h_g', 0))
            a_g = st.number_input("客隊進球 (Away Goal)", min_value=0, value=st.session_state.get('edit_a_g', 0))
        with ic2:
            h_c = st.number_input("主隊角球 (Home Corner)", min_value=0, value=st.session_state.get('edit_h_c', 0))
            a_c = st.number_input("客隊角球 (Away Corner)", min_value=0, value=st.session_state.get('edit_a_c', 0))
            h_da = st.number_input("主隊危險進攻 (Home DA)", min_value=0, value=st.session_state.get('edit_h_da', 0))
            a_da = st.number_input("客隊危險進攻 (Away DA)", min_value=0, value=st.session_state.get('edit_a_da', 0))
        with ic3:
            h_sot = st.number_input("主隊射正 (Home SoT)", min_value=0, value=st.session_state.get('edit_h_sot', 0))
            a_sot = st.number_input("客隊射正 (Away SoT)", min_value=0, value=st.session_state.get('edit_a_sot', 0))
            h_red = st.number_input("主隊紅牌 (Home Red)", min_value=0, value=st.session_state.get('edit_h_red', 0))
            a_red = st.number_input("客隊紅牌 (Away Red)", min_value=0, value=st.session_state.get('edit_a_red', 0))

        h_conv = (h_g / h_sot) if h_sot > 0 else (h_g / h_da if h_da > 0 else 0.0)
        a_conv = (a_g / a_sot) if a_sot > 0 else (a_g / a_da if a_da > 0 else 0.0)
        h_fire = (h_da * 0.3) + (h_sot * 0.5) + (h_c * 0.2)
        a_fire = (a_da * 0.3) + (a_sot * 0.5) + (a_c * 0.2)

        st.markdown("##### 📊 即場火力與轉化率指標")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("主隊火力指數", f"{h_fire:.1f}")
        m2.metric("客隊火力指數", f"{a_fire:.1f}")
        m3.metric("主隊進球轉化率", f"{h_conv * 100:.1f}%")
        m4.metric("客隊進球轉化率", f"{a_conv * 100:.1f}%")

        st.divider()
        st.subheader("📈 即場動態賠率走勢")
        render_odds_section(st.session_state.inplay_odds_history, prefix="inplay")

        st.divider()
        st.subheader("⚡ 建立即場追單 (In-Play Bet Placement)")
        
        ipc1, ipc2, ipc3, ipc4 = st.columns(4)
        with ipc1:
            ip_btype = st.selectbox("即場玩法", ["讓球 (即場)", "入球大小 (即場)", "角球大小 (即場)"])
        with ipc2:
            ip_sel = st.selectbox("下注選項", ["Home", "Away", "Over", "Under"], key="inplay_selection_sel")
        with ipc3:
            ip_line = st.number_input("即場盤口線", value=0.0 if "讓球" in ip_btype else 2.5, step=0.25, key="ip_line_input")
            ip_odds = st.number_input("即場賠率", value=1.90, min_value=1.01, step=0.01, key="ip_odds_input")
        with ipc4:
            ip_sys_stake = st.number_input("系統即場本金 ($)", min_value=0.0, value=100.0, step=50.0, key="ip_sys_stake")
            ip_usr_stake = st.number_input("用家即場本金 ($)", min_value=0.0, value=100.0, step=50.0, key="ip_usr_stake")

        if st.button("🚀 確認發送即場追單", type="primary", use_container_width=True):
            h_t = st.session_state.get('edit_h_team', '主隊')
            a_t = st.session_state.get('edit_a_team', '客隊')
            t_n = st.session_state.get('edit_t_name', '即場賽事')
            
            ip_id = f"IP{datetime.now().strftime('%Y%m%d%H%M%S')}"
            ip_row = {
                'ID': ip_id,
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Status': 'Open',
                'Tournament_Name': t_n,
                'Tournament_Category': CATEGORY_OPTIONS[0],
                'Match': f"{h_t} vs {a_t}",
                'Home_Team': h_t,
                'Away_Team': a_t,
                'Home_Rating': 'C', 'Away_Rating': 'C',
                'Home_Form': '0W0D0L', 'Away_Form': '0W0D0L',
                'Bet_Type': ip_btype,
                'Selection': ip_sel,
                'Initial_Line': ip_line,
                'Initial_Odds': ip_odds,
                'System_Stake': ip_sys_stake,
                'User_Stake': ip_usr_stake,
                'Odds_History': json.dumps(st.session_state.inplay_odds_history, ensure_ascii=False),
                'InPlay_Minute': inplay_min,
                'Home_DA': h_da, 'Away_DA': a_da,
                'Home_SoT': h_sot, 'Away_SoT': a_sot,
                'Home_SoFF': 0, 'Away_SoFF': 0,
                'Home_Red': h_red, 'Away_Red': a_red,
                'Home_Sub': 0, 'Away_Sub': 0,
                'Home_Possession': 50, 'Away_Possession': 50,
                'Home_Goal': h_g, 'Away_Goal': a_g,
                'Home_Corner': h_c, 'Away_Corner': a_c,
                'Home_Goal_Conversion': h_conv, 'Away_Goal_Conversion': a_conv,
                'Home_Firepower': h_fire, 'Away_Firepower': a_fire,
                'Result_Label': '未結算',
                'System_Profit': 0.0, 'User_Profit': 0.0, 'Unit_Profit': 0.0,
                'System_Payout': 0.0, 'User_Payout': 0.0
            }
            
            st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([ip_row])], ignore_index=True)
            save_db(st.session_state.df_db, db_file, db_table)
            st.toast(f"✅ 即場注單成功發送！ID: {ip_id}", icon="⚡")
            st.rerun()

    # ==========================================
    # TAB 3: 賽果自動結算
    # ==========================================
    with t_settle:
        st.subheader("🏁 賽果自動與手動結算 (Settlement Center)")
        
        open_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open'] if not st.session_state.df_db.empty else pd.DataFrame()
        
        if open_df.empty:
            st.success("🎉 目前無待結算的未完賽注單！")
        else:
            st.write(f"目前共有 **{len(open_df)}** 筆待結算注單。")
            
            for idx, r in open_df.iterrows():
                b_id = r['ID']
                match_name = r.get('Match', '')
                b_type = r.get('Bet_Type', '')
                selection = r.get('Selection', '')
                line = float(r.get('Initial_Line', 0))
                odds = float(r.get('Initial_Odds', 1.90))
                sys_stake = float(r.get('System_Stake', 0))
                usr_stake = float(r.get('User_Stake', 0))
                
                with st.expander(f"📌 注單 ID: {b_id} | {match_name} | {b_type} ({selection})", expanded=True):
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    with sc1:
                        st.write(f"**玩法**: {b_type}")
                        st.write(f"**選項**: {selection}")
                    with sc2:
                        st.write(f"**盤口線**: {line:.2f}")
                        st.write(f"**賠率**: {odds:.2f}")
                    with sc3:
                        st.write(f"**系統下注**: ${sys_stake:,.2f}")
                        st.write(f"**用家下注**: ${usr_stake:,.2f}")
                    with sc4:
                        st.write(f"**下注日期**: {r.get('Date', '')}")
                        
                    st.markdown("##### ⚽ 輸入終場賽果數據")
                    g_col1, g_col2, g_col3, g_col4 = st.columns(4)
                    with g_col1:
                        final_h_g = st.number_input("主隊進球", min_value=0, value=int(r.get('Home_Goal', 0)), key=f"s_hg_{b_id}")
                    with g_col2:
                        final_a_g = st.number_input("客隊進球", min_value=0, value=int(r.get('Away_Goal', 0)), key=f"s_ag_{b_id}")
                    with g_col3:
                        final_h_c = st.number_input("主隊角球", min_value=0, value=int(r.get('Home_Corner', 0)), key=f"s_hc_{b_id}")
                    with g_col4:
                        final_a_c = st.number_input("客隊角球", min_value=0, value=int(r.get('Away_Corner', 0)), key=f"s_ac_{b_id}")
                        
                    sys_p, usr_p, sys_pay, usr_pay, unit_p, res_lbl, diff = calculate_settlement(
                        b_type, selection, line, odds, sys_stake, usr_stake, 
                        final_h_g, final_a_g, final_h_c, final_a_c
                    )
                    
                    st.info(f"預計結算結果: **{res_lbl}** | 單位盈虧: **{unit_p:+.2f}** | 系統盈虧: **${sys_p:,.2f}** | 用家盈虧: **${usr_p:,.2f}**")
                    
                    if st.button(f"✅ 確認結算此注單 ({b_id})", key=f"btn_settle_{b_id}"):
                        st.session_state.df_db.loc[st.session_state.df_db['ID'] == b_id, 'Status'] = 'Settled'
                        st.session_state.df_db.loc[st.session_state.df_db['ID'] == b_id, 'Home_Goal'] = final_h_g
                        st.session_state.df_db.loc[st.session_state.df_db['ID'] == b_id, 'Away_Goal'] = final_a_g
                        st.session_state.df_db.loc[st.session_state.df_db['ID'] == b_id, 'Home_Corner'] = final_h_c
                        st.session_state.df_db.loc[st.session_state.df_db['ID'] == b_id, 'Away_Corner'] = final_a_c
                        st.session_state.df_db.loc[st.session_state.df_db['ID'] == b_id, 'Result_Label'] = res_lbl
                        st.session_state.df_db.loc[st.session_state.df_db['ID'] == b_id, 'System_Profit'] = sys_p
                        st.session_state.df_db.loc[st.session_state.df_db['ID'] == b_id, 'User_Profit'] = usr_p
                        st.session_state.df_db.loc[st.session_state.df_db['ID'] == b_id, 'Unit_Profit'] = unit_p
                        st.session_state.df_db.loc[st.session_state.df_db['ID'] == b_id, 'System_Payout'] = sys_pay
                        st.session_state.df_db.loc[st.session_state.df_db['ID'] == b_id, 'User_Payout'] = usr_pay
                        
                        save_db(st.session_state.df_db, db_file, db_table)
                        st.toast(f"🎉 注單 {b_id} 結算成功！[{res_lbl}]", icon="🏁")
                        st.rerun()

        st.divider()
        st.subheader("📜 已結算注單歷史總覽")
        settled_all = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'] if not st.session_state.df_db.empty else pd.DataFrame()
        if not settled_all.empty:
            st.dataframe(settled_all, use_container_width=True)
        else:
            st.caption("尚無已結算的歷史紀錄。")

    # ==========================================
    # TAB 4: 全局模型與特徵對齊
    # ==========================================
    with t_ai:
        st.subheader("🤖 全局機器學習模型與特徵對齊看板")
        
        settled_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'] if not st.session_state.df_db.empty else pd.DataFrame()
        
        n_settled = len(settled_df)
        if n_settled == 0:
            st.warning("⚠️ 數據庫中尚無已結算的賽事，無法進行機器學習與特徵對齊。")
        else:
            wins = len(settled_df[pd.to_numeric(settled_df['Unit_Profit'], errors='coerce') > 0])
            total_sys_stake = pd.to_numeric(settled_df['System_Stake'], errors='coerce').sum()
            total_sys_profit = pd.to_numeric(settled_df['System_Profit'], errors='coerce').sum()
            overall_winrate = (wins / n_settled) * 100 if n_settled > 0 else 0
            overall_roi = (total_sys_profit / total_sys_stake) * 100 if total_sys_stake > 0 else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("總已結算場數", f"{n_settled} 場")
            m2.metric("全局勝率 (Win Rate)", f"{overall_winrate:.1f}%")
            m3.metric("系統總淨盈虧", f"${total_sys_profit:,.2f}")
            m4.metric("全局 ROI", f"{overall_roi:+.1f}%")

            st.divider()
            st.markdown("##### 🌲 隨機森林 (RandomForest) 特徵重要性分析")
            
            rating_map = {"A+": 5, "A": 4, "B": 3, "C": 2, "D": 1}
            X, y = prepare_ml_dataset(settled_df, rating_map)
            
            if X is not None and len(np.unique(y)) > 1 and HAS_AI_MODULES and len(X) >= 5:
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X, y)
                
                feat_names = ['主隊評級', '客隊評級', '主隊近況分數', '客隊近況分數', '初盤盤口線', '初盤賠率']
                importances = clf.feature_importances_
                
                feat_df = pd.DataFrame({
                    '特徵名稱': feat_names,
                    '重要性 (Weight)': importances
                }).sort_values('重要性 (Weight)', ascending=False)
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.dataframe(feat_df, use_container_width=True)
                with c2:
                    st.bar_chart(feat_df.set_index('特徵名稱'))
            else:
                st.info("💡 樣本數量累積達 15 場以上後，系統將自動繪製隨機森林模型之特徵重要性權重圖表。")

            st.divider()
            st.markdown("##### 📊 玩法與類別分項表現")
            
            col_btype, col_cat = st.columns(2)
            with col_btype:
                st.caption("依玩法類型統計 (Bet Type Breakdown)")
                if 'Bet_Type' in settled_df.columns:
                    grp_btype = settled_df.groupby('Bet_Type').agg(
                        場數=('ID', 'count'),
                        系統淨盈虧=('System_Profit', lambda x: pd.to_numeric(x, errors='coerce').sum()),
                        用家淨盈虧=('User_Profit', lambda x: pd.to_numeric(x, errors='coerce').sum())
                    ).reset_index()
                    st.dataframe(grp_btype, use_container_width=True)
            with col_cat:
                st.caption("依賽事類別統計 (Tournament Category Breakdown)")
                if 'Tournament_Category' in settled_df.columns:
                    grp_cat = settled_df.groupby('Tournament_Category').agg(
                        場數=('ID', 'count'),
                        系統淨盈虧=('System_Profit', lambda x: pd.to_numeric(x, errors='coerce').sum()),
                        用家淨盈虧=('User_Profit', lambda x: pd.to_numeric(x, errors='coerce').sum())
                    ).reset_index()
                    st.dataframe(grp_cat, use_container_width=True)

if __name__ == "__main__":
    main()
