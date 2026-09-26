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
LOG_COLUMNS = ['ID', 'Date', 'Match', 'Analysis_Content', 'Confidence_Level']
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
        st.error(f"API 請求失敗: {e}")
        return None

def get_match_odds(match_id):
    """抓取馬會賠率變化"""
    url = f"https://tipsme-web.azurewebsites.net/api/Score/odds/hkjc/{match_id}"
    return fetch_api_data(url)

def get_match_fixtures(match_id):
    """抓取即場賽況及統計數據 (同時適用於賽果)"""
    url = f"https://tipsme-web.azurewebsites.net/api/Score/fixtures/{match_id}"
    return fetch_api_data(url)

def parse_and_fill_pre_match(match_id):
    """將抓取到的 API 數據填入 Session State (賽前)"""
    fixtures_data = get_match_fixtures(match_id)
    odds_data = get_match_odds(match_id)
    
    if fixtures_data:
        # 注意：這裡的 'homeName', 'awayName' 需要根據實際 JSON 欄位名稱調整
        st.session_state.edit_h_team = fixtures_data.get('homeName', fixtures_data.get('home', ''))
        st.session_state.edit_a_team = fixtures_data.get('awayName', fixtures_data.get('away', ''))
        st.session_state.edit_t_name = fixtures_data.get('leagueName', '')
        
    if odds_data:
        # 假設 JSON 裡面有 initial_line, home_odds 等
        # 我們自動幫使用者填入第一筆讓球盤口
        try:
            # 這裡需要根據您的 JSON 結構微調提取邏輯
            # 以下為示範邏輯
            st.session_state.odds_history = [{
                "id": 0, "type": "讓球", 
                "line": 0.0, # 替換為 odds_data 裡的初盤
                "upper": 1.90, # 替換為主隊賠率
                "lower": 1.90, # 替換為客隊賠率
                "unlock": False, "margin": 1.085
            }]
        except:
            pass
    return True

def parse_and_fill_inplay(match_id):
    """將抓取到的 API 數據填入 Session State (即場與賽果)"""
    data = get_match_fixtures(match_id)
    
    if data and "teamStats" in data:
        # 從 teamStats.ft (全場統計) 中提取數據
        # 如果是即場，這裡的 ft 會隨著比賽進行而更新
        team_stats_ft = data["teamStats"].get("ft", {})
        
        if team_stats_ft:
            # "1" 代表入球: [主隊入球, 客隊入球]
            goals = team_stats_ft.get("1", [0, 0])
            st.session_state.edit_h_g = int(goals[0])
            st.session_state.edit_a_g = int(goals[1])
            
            # "2" 代表角球: [主隊角球, 客隊角球]
            corners = team_stats_ft.get("2", [0, 0])
            st.session_state.edit_h_c = int(corners[0])
            st.session_state.edit_a_c = int(corners[1])
            
            # "4" 代表紅牌: [主隊紅牌, 客隊紅牌]
            red_cards = team_stats_ft.get("4", [0, 0])
            st.session_state.edit_h_red = int(red_cards[0])
            st.session_state.edit_a_red = int(red_cards[1])
            
            # "21" 代表射正 (Shot on Target): [主隊射正, 客隊射正]
            sot = team_stats_ft.get("21", [0, 0])
            st.session_state.edit_h_sot = int(sot[0])
            st.session_state.edit_a_sot = int(sot[1])
            
            # "25" 代表控球率 (Possession): [主隊控球率, 客隊控球率]
            poss = team_stats_ft.get("25", [50, 50])
            st.session_state.edit_h_poss = int(poss[0])
            
            return True
            
    return False

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
    m1.metric("系統累積淨盈虧 (System Profit)", f"${sys_profit:,.2f}", delta=f"{sys_profit:,.2f}")
    m2.metric("用家真實淨盈虧 (User Profit)", f"${user_profit:,.2f}", delta=f"{user_profit:,.2f}")
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
            if st.button("🗑️ 刪除選中紀錄", use_container_width=True):
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
                "line": float(row.get('line', default_line)),
                "upper": float(row.get('upper', 1.90)),
                "lower": float(row.get('lower', 1.90)),
                "unlock": bool(row.get('unlock', False)),
                "margin": float(row.get('margin', 1.085))
            }
    return {
        "type": target_type,
        "line": default_line,
        "upper": 1.90,
        "lower": 1.90,
        "unlock": False,
        "margin": 1.085
    }

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
    
    if ac1.button("➕ 新增讓球盤", key=f"{prefix}_add_hand", use_container_width=True):
        new_row = get_last_odds_state(odds_history_state, "讓球", 0.0)
        new_row["id"] = max_id + 1
        odds_history_state.append(new_row)
        st.rerun()
        
    if ac2.button("➕ 新增入球大小", key=f"{prefix}_add_goal", use_container_width=True):
        new_row = get_last_odds_state(odds_history_state, "入球大小", 2.5)
        new_row["id"] = max_id + 1
        odds_history_state.append(new_row)
        st.rerun()
        
    if ac3.button("➕ 新增角球大小", key=f"{prefix}_add_corn", use_container_width=True):
        new_row = get_last_odds_state(odds_history_state, "角球大小", 9.5)
        new_row["id"] = max_id + 1
        odds_history_state.append(new_row)
        st.rerun()

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
        
    st.sidebar.header("⚙️ 系統設定與資金管理")
    
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

        # --- 自動抓取賽前數據按鈕 ---
        with st.container(border=True):
            st.markdown("##### ⚡ 一鍵智能抓取賽前數據")
            col_id, col_btn = st.columns([2, 1])
            target_match_id = col_id.text_input("請輸入 Tipsme 賽事 ID (例如: 112684)", key="api_match_id")
            if col_btn.button("📥 自動獲取球隊與賠率", use_container_width=True):
                if target_match_id:
                    with st.spinner('正在從 Tipsme 抓取數據...'):
                        success = parse_and_fill_pre_match(target_match_id)
                        if success:
                            st.session_state.editing_bet_id = f"API_{target_match_id}" # 標記為API導入
                            st.success(f"✅ 成功載入賽事 {target_match_id} 的數據！已自動填寫下方表格。")
                            st.rerun()
                        else:
                            st.error("❌ 抓取失敗，請確認賽事 ID 是否正確。")
                else:
                    st.warning("請先輸入賽事 ID。")
        
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
                if c_btn.button("✏️ 載入該注單修改", key="load_last_pre"):
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
            upgrade_msg = ""
            if 'ev' in best_bet and best_bet['ev'] > 0 and sys_bankroll > 0:
                b = best_bet['odds'] - 1
                kelly = max(0.0, min((best_bet['prob'] * b - (1 - best_bet['prob'])) / b, 0.10))
                raw_stake = (sys_bankroll * (kelly * 0.5))
                suggested_stake = min(float(sys_max_stake), float(round(raw_stake / 10) * 10))
                
                if "讓球" in best_bet['bet_type']:
                    if 0 < suggested_stake < 200:
                        if best_bet.get('ev', 0) >= 0.03 and best_bet.get('prob', 0) >= 0.50:
                            suggested_stake = 200.0
                            upgrade_msg = "💡 **智能風控提示**：依據凱利公式，原計算注碼不足 $200。但因該讓球盤 EV (≥0.03) 與勝率 (≥50%) 均達標，系統判定具備高投資價值，建議升級至最低投注額 **$200**。"
                        else:
                            suggested_stake = 0.0
                            upgrade_msg = "⚠️ **智能風控提示**：依據凱利公式，原計算注碼不足 $200，且該讓球盤的期望值/勝率未達強制升級標準。系統建議 **放棄** 此次投注 (注碼歸 0)。"
                else:
                    suggested_stake = max(10.0, suggested_stake)

            st.session_state.analysis_result = {
                'micro': res_micro, 'meso': res_meso, 'macro': res_macro, 'best_model': best_model,
                'best_bet': best_bet, 'stake': suggested_stake, 't_name': tournament_name, 't_cat': tournament_category,
                'upgrade_msg': upgrade_msg
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
                        st.write(f"系統策略 ROI: `{r['roi']*100:.1f}%`")
                        st.write(f"歷史勝率: `{r['acc']*100:.1f}%`")
                    else:
                        st.warning(r['msg'])

            bm = res['best_model']
            bb = res['best_bet']
            dim_label_map = {"微觀 - 賽事名稱": "微觀", "中觀 - 賽事分類": "中觀", "宏觀 - 總數據": "宏觀"}
            dim_short = dim_label_map.get(bm['dim'], "宏觀")

            st.markdown(f"### 🧠 AI 預測模型推薦")
            
            st.markdown("#### 📊 所有盤口評估明細 (系統決策依據)")
            cand_list = bm.get('candidates', [])
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
            
            st.info(f"系統分析顯示，針對『{res['t_name']}』，採用『{bm['dim']}』級別的模型進行運算，其歷史準確率與 EV 獲利期望值最高，故本次投注策略依據此模型生成。")
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("💡 首選推薦", f"{bb['bet_type']} - {bb['label']}")
            mc2.metric(f"🎯 預期勝率 ({dim_short}修正)", f"{bb.get('prob', bb.get('base_prob',0))*100:.1f}%")
            mc3.metric("📊 修正 EV", f"{bb.get('ev', 0):.3f}")
            
            if res.get('upgrade_msg'):
                if "放棄" in res['upgrade_msg']:
                    st.warning(res['upgrade_msg'])
                else:
                    st.info(res['upgrade_msg'])

            with st.form("bet_form"):
                bc1, bc2 = st.columns(2)
                cand_btypes = [c['bet_type'] for c in bm.get('candidates', [bb])]
                default_btype_idx = 0
                if is_editing and st.session_state.get('edit_bet_type') in cand_btypes:
                    default_btype_idx = cand_btypes.index(st.session_state.get('edit_bet_type'))
                    
                final_btype = bc1.selectbox("最終投注項目", cand_btypes, index=default_btype_idx)
                
                sel_options = ["Home", "Away", "Over", "Under"]
                default_sel_idx = sel_options.index(bb['selection']) if bb['selection'] in sel_options else 0
                if is_editing and st.session_state.get('edit_selection') in sel_options:
                    default_sel_idx = sel_options.index(st.session_state.get('edit_selection'))
                    
                final_sel = bc2.selectbox("最終投注方向", sel_options, index=default_sel_idx)
                
                bc3, bc4 = st.columns(2)
                final_sys_stake = float(res['stake'] if res['stake'] > 0 else 0.0)
                bc3.text_input("🤖 系統建議下注金額 (System Stake) - 供分析學習用", f"${final_sys_stake:,.2f}", disabled=True)
                
                default_u_stake = float(st.session_state.get('edit_user_stake', final_sys_stake)) if is_editing else float(final_sys_stake)
                final_user_stake = bc4.number_input("👤 用家真實下注金額 (User Actual Stake) ($)", min_value=0.0, step=10.0, value=default_u_stake)
                
                final_row = next((r for r in st.session_state.odds_history if r['type'] == final_btype), st.session_state.odds_history[-1])
                line, odds = float(final_row['line']), float(final_row['upper']) if final_sel in ["Home", "Over"] else float(final_row['lower'])
                
                submit_btn_label = f"🔄 確定修改並覆蓋雲端資料庫 (ID: {st.session_state.editing_bet_id})" if is_editing else "✅ 確定投注並寫入雲端資料庫"
                
                if st.form_submit_button(submit_btn_label):
                    target_id = st.session_state.get('editing_bet_id')
                    new_id = target_id if target_id else f"B{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    
                    new_record = {
                        'ID': new_id, 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Status': 'Open',
                        'Tournament_Name': res['t_name'], 'Tournament_Category': res['t_cat'], 
                        'Match': f"{home_team} vs {away_team}", 'Home_Team': home_team, 'Away_Team': away_team,
                        'Home_Rating': home_rating, 'Away_Rating': away_rating, 'Home_Form': home_form, 'Away_Form': away_form,
                        'Bet_Type': final_btype, 'Selection': final_sel, 'Initial_Line': line, 'Initial_Odds': odds, 
                        'System_Stake': final_sys_stake, 'User_Stake': final_user_stake,
                        'Odds_History': json.dumps(st.session_state.odds_history, ensure_ascii=False)
                    }
                    
                    if target_id and (st.session_state.df_db['ID'] == target_id).any():
                        mask = st.session_state.df_db['ID'] == target_id
                        for col_name, val in new_record.items():
                            st.session_state.df_db.loc[mask, col_name] = val
                        st.toast(f"🔄 注單 {target_id} 修改成功並已覆蓋雲端資料庫！", icon="✅")
                    else:
                        st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([new_record])], ignore_index=True)
                        st.toast("✅ 投注紀錄雲端同步成功！", icon="📝")
                        
                    st.session_state.last_bet_id = new_id
                    save_db(st.session_state.df_db, db_file, db_table)
                    clear_edit_mode()
                    st.session_state.odds_history = [{"id": 0, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}] 
                    st.session_state.show_analysis = False
                    st.rerun()

    with t_inplay:
        st.subheader("⏱️ 即場賽事實時更新與智慧火力分析")
        
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

            def get_val(r, col, default=0, edit_key=None):
                if is_editing and edit_key and edit_key in st.session_state:
                    return int(st.session_state[edit_key])
                val = r.get(col)
                if pd.isna(val) or val == "": return default
                return int(float(val))

        # --- 自動抓取即場數據按鈕 ---
            with st.container(border=True):
                st.markdown("##### ⚡ 一鍵同步即場賽況")
                col_in_id, col_in_btn = st.columns([2, 1])
                inplay_match_id = col_in_id.text_input("輸入賽事 ID 進行同步 (例如: 112684)", key="api_inplay_id")
                if col_in_btn.button("🔄 自動同步即時比分與統計", use_container_width=True):
                    if inplay_match_id:
                        with st.spinner('正在同步最新即場數據...'):
                            if parse_and_fill_inplay(inplay_match_id):
                                st.success("✅ 即場數據同步成功！已更新下方火力分析。")
                                st.rerun()
                            else:
                                st.error("❌ 同步失敗。")
            
            st.markdown("##### 1. 實時數據輸入與自動效率計算")
            minute = st.number_input("比賽進行時間 (分鐘)", min_value=0, max_value=120, value=get_val(row, 'InPlay_Minute', 45, 'edit_inplay_minute'))
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 🏠 主隊實時數據")
                h_g = st.number_input("主隊入球", min_value=0, value=get_val(row, 'Home_Goal', 0, 'edit_h_g'))
                h_c = st.number_input("主隊角球", min_value=0, value=get_val(row, 'Home_Corner', 0, 'edit_h_c'))
                h_da = st.number_input("主隊危險進攻 (DA)", min_value=0, value=get_val(row, 'Home_DA', 0, 'edit_h_da'))
                h_sot = st.number_input("主隊射正 (SoT)", min_value=0, value=get_val(row, 'Home_SoT', 0, 'edit_h_sot'))
                h_soff = st.number_input("主隊射偏 (SoFF)", min_value=0, value=get_val(row, 'Home_SoFF', 0, 'edit_h_soff'))
                h_red = st.number_input("主隊紅牌", min_value=0, value=get_val(row, 'Home_Red', 0, 'edit_h_red'))
                h_sub = st.number_input("主隊換人", min_value=0, value=get_val(row, 'Home_Sub', 0, 'edit_h_sub'))
                
                h_poss = st.number_input("主隊控球率 (%)", min_value=0, max_value=100, value=get_val(row, 'Home_Possession', 50, 'edit_h_poss'))
                a_poss = max(0, 100 - h_poss)
                
                h_conv = (h_g / h_sot * 100) if h_sot > 0 else 0.0
                h_fire = ((h_sot + h_soff) / h_da * 100) if h_da > 0 else 0.0
                
                st.markdown(f"> 🎯 **主隊得分率**: `{h_conv:.1f}%` ({h_g}進球 / {h_sot}射正)")
                st.markdown(f"> ⚡ **主隊進攻火力**: `{h_fire:.1f}%` ({h_sot+h_soff}射門 / {h_da}危險進攻)")
                
            with c2:
                st.markdown("##### ✈️ 客隊實時數據")
                a_g = st.number_input("客隊入球", min_value=0, value=get_val(row, 'Away_Goal', 0, 'edit_a_g'))
                a_c = st.number_input("客隊角球", min_value=0, value=get_val(row, 'Away_Corner', 0, 'edit_a_c'))
                a_da = st.number_input("客隊危險進攻 (DA)", min_value=0, value=get_val(row, 'Away_DA', 0, 'edit_a_da'))
                a_sot = st.number_input("客隊射正 (SoT)", min_value=0, value=get_val(row, 'Away_SoT', 0, 'edit_a_sot'))
                a_soff = st.number_input("客隊射偏 (SoFF)", min_value=0, value=get_val(row, 'Away_SoFF', 0, 'edit_a_soff'))
                a_red = st.number_input("客隊紅牌", min_value=0, value=get_val(row, 'Away_Red', 0, 'edit_a_red'))
                a_sub = st.number_input("客隊換人", min_value=0, value=get_val(row, 'Away_Sub', 0, 'edit_a_sub'))
                
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
                upgrade_msg = ""
                if best_bet and best_bet['ev'] > 0 and sys_bankroll > 0:
                    b = best_bet['odds'] - 1
                    kelly = max(0.0, min((best_bet['prob'] * b - (1 - best_bet['prob'])) / b, 0.10))
                    raw_stake = (sys_bankroll * (kelly * 0.5))
                    suggested_stake = min(float(sys_max_stake), float(round(raw_stake / 10) * 10))
                    
                    if "讓球" in best_bet['bet_type']:
                        if 0 < suggested_stake < 200:
                            if best_bet.get('ev', 0) >= 0.03 and best_bet.get('prob', 0) >= 0.50:
                                suggested_stake = 200.0
                                upgrade_msg = "💡 **智能風控提示**：依據即場火力與凱利公式，原注碼不足 $200。但因該讓球盤即場 EV (≥0.03) 與勝率 (≥50%) 達標，系統判定具備高價值，建議升級至最低投注額 **$200**。"
                            else:
                                suggested_stake = 0.0
                                upgrade_msg = "⚠️ **智能風控提示**：即場計算注碼不足 $200，且讓球盤期望值/勝率未達強制升級標準。系統建議 **放棄** 此次即場投注 (注碼歸 0)。"
                    else:
                        suggested_stake = max(10.0, suggested_stake)
                
                st.session_state.inplay_analysis_result = {
                    'candidates': candidates, 'best_bet': best_bet,
                    'stake': suggested_stake, 'match_row': row.to_dict(),
                    'upgrade_msg': upgrade_msg
                }

            if st.session_state.get('show_inplay_analysis', False):
                res = st.session_state.inplay_analysis_result
                best_bet = res['best_bet']
                match_info = res['match_row']
                
                st.success("✅ 結合火力與剩餘時間的即場 EV 精算完成！")
                
                if res.get('candidates'):
                    st.markdown("#### 📊 所有即場盤口評估明細 (系統決策依據)")
                    df_inplay_show = pd.DataFrame(res['candidates'])
                    df_inplay_show['推薦排序'] = range(1, len(df_inplay_show) + 1)
                    df_inplay_show = df_inplay_show[['推薦排序', 'bet_type', 'line', 'label', 'odds', 'prob', 'ev']]
                    df_inplay_show.columns = ['推薦排序', '盤口類型', '盤口線', '投注方向', '賠率', '動態勝率', '期望值 (EV)']
                    df_inplay_show['動態勝率'] = df_inplay_show['動態勝率'].apply(lambda x: f"{x*100:.2f}%")
                    df_inplay_show['期望值 (EV)'] = df_inplay_show['期望值 (EV)'].apply(lambda x: f"{x:.3f}")
                    
                    def highlight_first_inplay(row):
                        if row.name == 0:
                            return ['background-color: rgba(40, 167, 69, 0.2)'] * len(row)
                        return [''] * len(row)
                        
                    st.dataframe(df_inplay_show.style.apply(highlight_first_inplay, axis=1), use_container_width=True)
                
                if best_bet:
                    st.info("系統已成功納入進攻火力效率與得分率，為您挑選出最佳價值的即場盤口：")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("💡 首選推薦", f"{best_bet['bet_type']} - {best_bet['label']}")
                    mc2.metric(f"🎯 動態勝率預測", f"{best_bet['prob']*100:.1f}%")
                    mc3.metric("📊 即場 EV", f"{best_bet['ev']:.3f}")
                    
                    if res.get('upgrade_msg'):
                        if "放棄" in res['upgrade_msg']:
                            st.warning(res['upgrade_msg'])
                        else:
                            st.info(res['upgrade_msg'])
                    
                    with st.form("inplay_bet_form"):
                        bc1, bc2 = st.columns(2)
                        cand_btypes = [c['bet_type'] for c in res['candidates']]
                        default_btype_idx = 0
                        if is_editing and st.session_state.get('edit_bet_type') in cand_btypes:
                            default_btype_idx = cand_btypes.index(st.session_state.get('edit_bet_type'))
                        final_btype = bc1.selectbox("最終投注項目", cand_btypes, index=default_btype_idx)
                        
                        sel_options = ["Home", "Away", "Over", "Under"]
                        default_sel_idx = sel_options.index(best_bet['selection']) if best_bet['selection'] in sel_options else 0
                        if is_editing and st.session_state.get('edit_selection') in sel_options:
                            default_sel_idx = sel_options.index(st.session_state.get('edit_selection'))
                        final_sel = bc2.selectbox("最終投注方向", sel_options, index=default_sel_idx)
                        
                        bc3, bc4 = st.columns(2)
                        final_sys_stake = float(res['stake'] if res['stake'] > 0 else 0.0)
                        bc3.text_input("🤖 系統建議下注金額 (System Stake) - 供分析學習用", f"${final_sys_stake:,.2f}", disabled=True)
                        
                        default_u_stake = float(st.session_state.get('edit_user_stake', final_sys_stake)) if is_editing else float(final_sys_stake)
                        final_user_stake = bc4.number_input("👤 用家真實下注金額 (User Actual Stake) ($)", min_value=0.0, step=10.0, value=default_u_stake)
                        
                        submit_btn_label = f"🔄 確定修改並覆蓋即場注單 (ID: {st.session_state.editing_bet_id})" if is_editing else "✅ 確認即場投注並扣除本金"
                        
                        if st.form_submit_button(submit_btn_label):
                            final_row = next((r for r in st.session_state.inplay_odds_history if r['type'] == final_btype), st.session_state.inplay_odds_history[-1])
                            line = float(final_row['line'])
                            odds = float(final_row['upper']) if final_sel in ["Home", "Over"] else float(final_row['lower'])
                            
                            target_id = st.session_state.get('editing_bet_id')
                            new_id = target_id if target_id else f"B{datetime.now().strftime('%Y%m%d%H%M%S')}_INPLAY"
                            
                            b_type_str = f"{final_btype} (即場)" if "即場" not in final_btype else final_btype
                            new_record = match_info.copy()
                            new_record.update({
                                'ID': new_id, 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                'Bet_Type': b_type_str, 'Selection': final_sel,
                                'Initial_Line': line, 'Initial_Odds': odds, 
                                'System_Stake': final_sys_stake, 'User_Stake': final_user_stake,
                                'Status': 'Open', 'Odds_History': json.dumps(st.session_state.inplay_odds_history, ensure_ascii=False),
                                'InPlay_Minute': minute, 'Home_Goal': h_g, 'Away_Goal': a_g, 'Home_Corner': h_c, 'Away_Corner': a_c,
                                'Home_DA': h_da, 'Away_DA': a_da, 'Home_SoT': h_sot, 'Away_SoT': a_sot,
                                'Home_SoFF': h_soff, 'Away_SoFF': a_soff, 'Home_Red': h_red, 'Away_Red': a_red,
                                'Home_Sub': h_sub, 'Away_Sub': a_sub, 'Home_Possession': h_poss, 'Away_Possession': a_poss,
                                'Home_Goal_Conversion': h_conv, 'Away_Goal_Conversion': a_conv, 
                                'Home_Firepower': h_fire, 'Away_Firepower': a_fire,
                                'Result_Label': '', 'System_Profit': 0, 'User_Profit': 0, 'Unit_Profit': 0, 'System_Payout': 0, 'User_Payout': 0
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

                            if target_id and (st.session_state.df_db['ID'] == target_id).any():
                                mask = st.session_state.df_db['ID'] == target_id
                                for col_name, val in new_record.items():
                                    st.session_state.df_db.loc[mask, col_name] = val
                                st.toast(f"🔄 即場注單 {target_id} 修改成功並已覆蓋雲端資料庫！", icon="✅")
                            else:
                                st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([new_record])], ignore_index=True)
                                st.toast("✅ 即場注單雲端同步成功！", icon="📝")

                            st.session_state.last_bet_id = new_id
                            save_db(st.session_state.df_db, db_file, db_table)
                            clear_edit_mode()
                            st.session_state.show_inplay_analysis = False
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
                    col_b1, col_b2 = st.columns([3, 1])
                    col_b1.caption(f"注單 ID: `{row['ID']}` | 投注金額: 用家 ${row.get('User_Stake',0)} / 系統 ${row.get('System_Stake',0)}")
                    if col_b2.button("✏️ 載入此注單修改資料", key=f"btn_edit_settle_{row['ID']}"):
                        load_bet_to_edit(row['ID'])
                        st.success(f"已成功載入注單 `{row['ID']}`！請切換至「📝 賽前建檔與投注」或「⏱️ 即場賽事」進行修改與覆蓋。")
                        
                    with st.form(f"settle_form_{row['ID']}"):
                        st.markdown("##### ⚽ 全場賽果輸入 (入球與角球)")
                        col_auto = st.columns(1)[0]
                        if col_auto.form_submit_button("⚡ 一鍵自動獲取完場比分與角球 (需輸入賽事ID)"):
                            # 這裡假設您的 ID 命名規則中包含真實 match_id，若無則可彈出輸入框
                            # 為簡化操作，這裡示範邏輯
                            st.info("請切換至即場賽事分頁使用自動同步，再回來點擊確認結算。")
                            
                        col1, col2 = st.columns(2)
                        h_g = col1.number_input("全場主隊入球數", min_value=0, value=int(row.get('Home_Goal', 0)) if pd.notna(row.get('Home_Goal')) else 0, key=f"hg_{row['ID']}")
                        a_g = col2.number_input("全場客隊入球數", min_value=0, value=int(row.get('Away_Goal', 0)) if pd.notna(row.get('Away_Goal')) else 0, key=f"ag_{row['ID']}")
                        
                        col3, col4 = st.columns(2)
                        h_c = col3.number_input("全場主隊角球數", min_value=0, value=int(row.get('Home_Corner', 0)) if pd.notna(row.get('Home_Corner')) else 0, key=f"hc_{row['ID']}")
                        a_c = col4.number_input("全場客隊角球數", min_value=0, value=int(row.get('Away_Corner', 0)) if pd.notna(row.get('Away_Corner')) else 0, key=f"ac_{row['ID']}")
                        
                        if st.form_submit_button("確認賽果並雲端結算"):
                            sys_p, usr_p, sys_pay, usr_pay, u_prof, lbl, diff = calculate_settlement(
                                row['Bet_Type'], row['Selection'], float(row['Initial_Line']), 
                                float(row['Initial_Odds']), float(row.get('System_Stake', 0)), float(row.get('User_Stake', 0)), 
                                h_g, a_g, h_c, a_c
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
                            st.success(f"結算完成並同步雲端！結果：{lbl} | 單位盈虧：{u_prof:+.2f} U")
                            st.rerun()
                            
        st.markdown("---")
        st.subheader("⚠️ 撤銷與回滾中心 (Settlement Rollback)")
        st.info("若發生結算錯誤，您可在此撤銷該筆結算。撤銷後，該賽事將恢復為「Open」未結算狀態，並重新顯示於上方的賽果輸入區，供您修改後重新結算。資金池會自動重構。")
        
        settled_bets = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].tail(5)
        
        if not settled_bets.empty:
            rollback_options = []
            for _, r in settled_bets.iterrows():
                rollback_options.append(f"{r['ID']} | [{r['Date']}] {r['Match']} | 結算狀態: {r['Result_Label']} | 用家盈虧: ${r.get('User_Profit', 0)}")
                
            sel_rollback = st.selectbox("請選擇要撤銷結算的最近紀錄：", rollback_options)
            
            if st.button("↩️ 撤銷結算並恢復為未結算狀態", type="primary"):
                target_id = sel_rollback.split(" | ")[0]
                mask = st.session_state.df_db['ID'] == target_id
                st.session_state.df_db.loc[mask, 'Status'] = 'Open'
                st.session_state.df_db.loc[mask, 'Result_Label'] = ''
                st.session_state.df_db.loc[mask, 'System_Profit'] = 0.0
                st.session_state.df_db.loc[mask, 'User_Profit'] = 0.0
                st.session_state.df_db.loc[mask, 'System_Payout'] = 0.0
                st.session_state.df_db.loc[mask, 'User_Payout'] = 0.0
                st.session_state.df_db.loc[mask, 'Unit_Profit'] = 0.0
                save_db(st.session_state.df_db, db_file, db_table)
                st.toast(f"✅ 已成功撤銷注單 {target_id} 的結算，恢復為 Open 狀態！", icon="↩️")
                st.rerun()

    with t_ai:
        st.subheader("🤖 全局機器學習模型與策略分析")
        df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled']
        
        if len(df_settled) < 15:
            st.info(f"📊 目前已結算樣本數為 {len(df_settled)} 場。當前已結算數據未達 15 場門檻，模型將以基礎機率與期望值運算為主。")
        else:
            st.success(f"🎉 目前共有 {len(df_settled)} 筆已結算賽事紀錄，全局機器學習模型運算中。")
            
        st.markdown("##### 📈 系統勝率與盈虧走勢看板")
        if not df_settled.empty:
            df_chart = df_settled.copy()
            df_chart['Cum_Sys_Profit'] = pd.to_numeric(df_chart['System_Profit'], errors='coerce').cumsum()
            df_chart['Cum_User_Profit'] = pd.to_numeric(df_chart['User_Profit'], errors='coerce').cumsum()
            st.line_chart(df_chart[['Cum_Sys_Profit', 'Cum_User_Profit']])
        else:
            st.caption("尚無已結算數據可供展示走勢圖。")

if __name__ == "__main__":
    main()
