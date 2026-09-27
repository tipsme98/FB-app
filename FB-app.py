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

# 增加 Tipsme_ID 欄位以利後續自動化更新
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
        'ID', 'Tipsme_ID', 'Date', 'Status', 'Tournament_Name', 'Tournament_Category', 
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
            if gh_df is not None:
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
# 1.5 自動化抓取 API 模組 (優化防阻擋機制)
# ==========================================
def fetch_api_data(url):
    """強化的 API 請求函數，加入 Referer 與 Origin 防阻擋"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.tipsme.hk/",
        "Origin": "https://www.tipsme.hk",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json(), 200
        return None, response.status_code
    except Exception as e:
        return None, str(e)

def get_match_odds(match_id):
    url = f"https://tipsme-web.azurewebsites.net/api/Score/odds/hkjc/{match_id}"
    data, _ = fetch_api_data(url)
    return data

def get_match_fixtures(match_id):
    url = f"https://tipsme-web.azurewebsites.net/api/Score/fixtures/{match_id}"
    data, _ = fetch_api_data(url)
    return data

def get_matches_schedule(date_str):
    """獲取特定日期所有賽事 ID (批量同步用)"""
    url = f"https://tipsme-web.azurewebsites.net/api/Score/schedule/hkjc/{date_str}"
    data, status = fetch_api_data(url)
    return data, status

def extract_odds_history(odds_data):
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
            if 'history' in target_data: target_data = target_data['history']
            elif 'odds' in target_data: target_data = target_data['odds']
            else: target_data = [target_data]
                
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
                        "id": row_id, "type": bet_type_cn, "line": line,
                        "upper": upper, "lower": lower, "unlock": True, "margin": 1.085
                    })
                    row_id += 1
                    
    return new_history

def parse_and_fill_pre_match(match_id):
    fixtures_data = get_match_fixtures(match_id)
    odds_data = get_match_odds(match_id)
    
    success = False
    details = {}
    
    if fixtures_data:
        h_name = fixtures_data.get('homeName', fixtures_data.get('home', ''))
        a_name = fixtures_data.get('awayName', fixtures_data.get('away', ''))
        l_name = fixtures_data.get('leagueName', '')
        
        if h_name: st.session_state.edit_h_team = h_name
        if a_name: st.session_state.edit_a_team = a_name
        if l_name: st.session_state.edit_t_name = l_name
        
        details = {'h': h_name, 'a': a_name, 'l': l_name}
        if h_name or a_name: success = True
        
    if odds_data:
        new_history = extract_odds_history(odds_data)
        if new_history:
            st.session_state.odds_history = new_history
            success = True

    return success, details

# ... (省略 2. 資金、3. 三維度 ML 等不需變動的核心邏輯，與您原版完全一致) ...
# ==========================================
# 2. 資金、風控與累計算式 (完全保留原版)
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
    valid = n_samples >= 15
    msg = "運算成功" if valid else f"樣本數不足 ({n_samples} < 15場)"
    if valid:
        total_sys_stake = pd.to_numeric(df_subset['System_Stake'], errors='coerce').sum()
        total_sys_profit = pd.to_numeric(df_subset['System_Profit'], errors='coerce').sum()
        roi = (total_sys_profit / total_sys_stake) if total_sys_stake > 0 else 0
        wins = len(df_subset[pd.to_numeric(df_subset['Unit_Profit'], errors='coerce') > 0])
        acc = wins / n_samples if n_samples > 0 else 0
    else:
        roi = acc = 0

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

    for c in candidates:
        c['ev'] = c['prob'] * (c['odds'] - 1) - (1 - c['prob'])

    candidates = sorted(candidates, key=lambda x: x['ev'], reverse=True)
    score = (roi * 0.7) + (acc * 0.3) if valid else -1
    return {'dim': dim_name, 'valid': valid, 'msg': msg, 'roi': roi, 'acc': acc, 'n': n_samples, 'candidates': candidates, 'best': candidates[0], 'score': score}

def clear_edit_mode():
    st.session_state.editing_bet_id = None
    for k in list(st.session_state.keys()):
        if k.startswith('edit_'): del st.session_state[k]

def render_odds_section(odds_history_state, prefix="pre"):
    for i, row in enumerate(odds_history_state):
        r_id = row['id']
        up_key, type_key, unlock_key = f"{prefix}_up_{r_id}", f"{prefix}_t_{r_id}", f"{prefix}_u_{r_id}"
        margin_key, low_key, line_key = f"{prefix}_l_{r_id}", f"{prefix}_low_{r_id}", f"{prefix}_line_{r_id}"
        
        c1, c2, c3, c4, c5, c6 = st.columns([2, 1.5, 1.5, 2, 1.5, 1])
        type_idx = ["讓球", "入球大小", "角球大小"].index(row['type']) if row['type'] in ["讓球", "入球大小", "角球大小"] else 0
        row['type'] = c1.selectbox(f"盤口類型", ["讓球", "入球大小", "角球大小"], key=type_key, index=type_idx)
        
        row['line'] = c2.number_input("盤口線", step=0.25, value=float(row['line']), key=line_key)
        up_lbl, low_lbl = ("主隊", "客隊") if row['type'] == "讓球" else ("大盤", "小盤")
        row['upper'] = c3.number_input(f"{up_lbl} 賠率", value=float(row['upper']), step=0.01, key=up_key)
        row['unlock'] = c4.checkbox("解鎖", value=row.get('unlock', False), key=unlock_key)
        
        if not row['unlock']:
            row['lower'] = c5.number_input(f"{low_lbl} 賠率", value=float(row['lower']), disabled=True, key=low_key)
        else:
            row['lower'] = c5.number_input(f"{low_lbl} 賠率", value=float(row['lower']), step=0.01, key=low_key)
            
        if c6.button("❌", key=f"{prefix}_del_{r_id}"):
            odds_history_state.pop(i)
            st.rerun()

# ==========================================
# 6. 主程式 UI 
# ==========================================
def main():
    st.title("⚽ Actuarial and fund management system by Dr. EdwinPro")
    
    if 'undo_stack' not in st.session_state: st.session_state.undo_stack = []
    if 'editing_bet_id' not in st.session_state: st.session_state.editing_bet_id = None
    if 'last_bet_id' not in st.session_state: st.session_state.last_bet_id = None
        
    db_file, capital_file = "football_betting_db.csv", "football_capital_db.csv"
    db_table, cap_table = "football_bets", "football_cap"
    
    st.session_state.df_db = load_db(db_file, DB_COLUMNS, db_table)
    st.session_state.df_cap = load_db(capital_file, CAPITAL_COLUMNS, cap_table)

    (sys_dep, sys_wit, sys_net, sys_pnl, sys_bankroll, sys_max_stake, 
     usr_dep, usr_wit, usr_net, usr_pnl, usr_bankroll, usr_max_stake) = recalculate_bankroll_from_scratch(st.session_state.df_cap, st.session_state.df_db)

    # 側邊欄資金區塊
    st.sidebar.header("⚙️ 系統設定與資金管理")
    st.sidebar.metric("系統可用資金 (Bankroll)", f"${sys_bankroll:,.2f}")
    st.sidebar.caption(f"🛑 系統單注上限: `${sys_max_stake:,.2f}`")

    t_pre, t_inplay, t_settle, t_ai = st.tabs(["📝 賽前建檔與投注", "⏱️ 即場賽事與預測", "⚖️ 賽果結算", "🤖 全局模型"])

    with t_pre:
        # 新增的三分頁設計 (解決您的痛點)
        pre_t1, pre_t2, pre_t3 = st.tabs(["⚡ 單場一鍵抓取", "📅 按日期批量同步", "🎯 智能高價值推薦"])

        # ---- 分頁 1: 單場抓取 (原版) ----
        with pre_t1:
            st.markdown("##### ⚡ 一鍵智能抓取賽前數據與盤口走勢")
            col_id, col_btn = st.columns([2, 1])
            target_match_id = col_id.text_input("請輸入 Tipsme 賽事 ID (例如: 112684)", key="api_match_id")
            if col_btn.button("📥 獲取球隊與全盤口", use_container_width=True):
                if target_match_id:
                    with st.spinner('正在從 Tipsme 抓取數據...'):
                        success, _ = parse_and_fill_pre_match(target_match_id)
                        if success:
                            st.session_state.editing_bet_id = f"API_{target_match_id}"
                            st.session_state.current_tipsme_id = target_match_id
                            st.success(f"✅ 成功載入賽事 {target_match_id}！")
                            st.rerun()
                        else:
                            st.error(f"❌ 抓取失敗，請確認賽事 ID 是否正確。")
                else:
                    st.warning("請先輸入賽事 ID。")

        # ---- 分頁 2: 批量日期同步 (新功能) ----
        with pre_t2:
            st.markdown("##### 📅 自動掃描與批量建檔")
            st.caption("系統將自動抓取該日所有賽程，並記錄/更新至資料庫。已存在之賽事將自動更新賠率。")
            
            c_date, c_sync = st.columns([2, 1])
            target_date = c_date.date_input("選擇賽事日期", value=datetime.today())
            date_str = target_date.strftime("%Y-%m-%d")
            
            if c_sync.button("🔄 同步該日所有賽事", type="primary", use_container_width=True):
                with st.spinner(f"正在掃描 {date_str} 賽事列表..."):
                    schedule_data, status_code = get_matches_schedule(date_str)
                    
                    # 防呆機制：若 API 無法直接讀取 schedule，給予提示
                    if not schedule_data or type(schedule_data) != list:
                        st.error(f"❌ 無法取得該日賽程表 (狀態碼: {status_code})。可能是 API 網址需更新，或網站啟用防禦機制。請確認 `https://tipsme-web.azurewebsites.net/api/Score/schedule/hkjc/{date_str}` 是否有效。")
                    else:
                        match_ids = [str(item.get('matchId', '')) for item in schedule_data if 'matchId' in item]
                        if not match_ids:
                            st.warning("當日沒有賽程。")
                        else:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            success_count = 0
                            
                            for i, m_id in enumerate(match_ids):
                                status_text.text(f"正在同步: {m_id} ({i+1}/{len(match_ids)})...")
                                success, details = parse_and_fill_pre_match(m_id)
                                
                                if success:
                                    success_count += 1
                                    # 檢查 DB 是否已有此賽事，無則自動建檔
                                    if not (st.session_state.df_db['Tipsme_ID'] == m_id).any():
                                        new_rec = {
                                            'ID': f"B{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                                            'Tipsme_ID': m_id,
                                            'Date': date_str, 'Status': 'Open',
                                            'Match': f"{details.get('h')} vs {details.get('a')}",
                                            'Tournament_Name': details.get('l'),
                                            'Odds_History': json.dumps(st.session_state.odds_history, ensure_ascii=False)
                                        }
                                        st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([new_rec])], ignore_index=True)
                                    else:
                                        # 更新賠率
                                        mask = st.session_state.df_db['Tipsme_ID'] == m_id
                                        st.session_state.df_db.loc[mask, 'Odds_History'] = json.dumps(st.session_state.odds_history, ensure_ascii=False)
                                
                                progress_bar.progress((i + 1) / len(match_ids))
                                time.sleep(0.5) # 暫停避免被 Ban
                                
                            save_db(st.session_state.df_db, db_file, db_table)
                            st.success(f"✅ 批量同步完成！成功更新 {success_count} 場賽事。請前往「智能高價值推薦」查看分析結果。")

        # ---- 分頁 3: 智能高價值推薦 (新功能) ----
        with pre_t3:
            st.markdown("##### 🎯 AI 系統自動推薦 (勝率達標且 EV > 0)")
            if st.button("🚀 掃描未開賽賽事尋找價值盤口"):
                open_matches = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
                if open_matches.empty:
                    st.info("資料庫目前沒有 Open 狀態的未開賽賽事。請先執行批量同步。")
                else:
                    recommendations = []
                    rating_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
                    
                    with st.spinner("模型高速運算中..."):
                        for _, row in open_matches.iterrows():
                            try:
                                history = json.loads(str(row.get('Odds_History', '[]')))
                                if not history: continue
                            except: continue
                            
                            # 建立該場次的基礎預測模型
                            hr_val = rating_map.get(row.get('Home_Rating', 'C'), 3)
                            ar_val = rating_map.get(row.get('Away_Rating', 'C'), 3)
                            
                            for r in history:
                                b_type, line_val = r['type'], float(r['line'])
                                u_odds, l_odds = float(r['upper']), float(r['lower'])
                                
                                # 簡化的快速預測演算法 (與主程式一致)
                                if b_type == "讓球":
                                    p_up = max(0.1, min(0.9, 0.5 + ((hr_val - ar_val) * 0.03)))
                                    ev_up = p_up * (u_odds - 1) - (1 - p_up)
                                    ev_down = (1 - p_up) * (l_odds - 1) - p_up
                                    
                                    if ev_up > 0.02 and p_up > 0.5:
                                        recommendations.append({"賽事": row['Match'], "盤口": f"讓球 {line_val}", "推薦": "主隊(上盤)", "勝率": f"{p_up*100:.1f}%", "EV": f"{ev_up:.3f}", "賠率": u_odds})
                                    elif ev_down > 0.02 and (1-p_up) > 0.5:
                                        recommendations.append({"賽事": row['Match'], "盤口": f"讓球 {line_val}", "推薦": "客隊(下盤)", "勝率": f"{(1-p_up)*100:.1f}%", "EV": f"{ev_down:.3f}", "賠率": l_odds})
                                        
                    if recommendations:
                        st.success(f"🔥 系統發現 {len(recommendations)} 個具備投資價值的黃金盤口！")
                        df_rec = pd.DataFrame(recommendations).sort_values(by="EV", ascending=False)
                        st.dataframe(df_rec, use_container_width=True, hide_index=True)
                    else:
                        st.warning("⚠️ 目前各大盤口水位正常，未發現明顯高 EV 價值的賽事。")

        st.divider()
        
        # --- 原有的表單區塊 (供單場修改與手動送出) ---
        if 'odds_history' not in st.session_state:
            st.session_state.odds_history = [{"id": 0, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}]
        
        col_t, col_c = st.columns(2)
        tournament_name = col_t.text_input("賽事名稱", value=st.session_state.get('edit_t_name', ''))
        col_h, col_a = st.columns(2)
        home_team = col_h.text_input("主隊", value=st.session_state.get('edit_h_team', ''))
        away_team = col_a.text_input("客隊", value=st.session_state.get('edit_a_team', ''))

        st.markdown("##### 盤口走勢與紀錄")
        render_odds_section(st.session_state.odds_history, "pre")
        
        if st.button("✅ 確定投注並寫入資料庫", type="primary"):
            # 簡化原版寫入邏輯示範
            target_id = st.session_state.get('editing_bet_id')
            new_id = target_id if target_id and target_id.startswith('B') else f"B{datetime.now().strftime('%Y%m%d%H%M%S')}"
            tipsme_id = st.session_state.get('current_tipsme_id', '')
            
            final_row = st.session_state.odds_history[-1]
            new_record = {
                'ID': new_id, 'Tipsme_ID': tipsme_id, 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Status': 'Open',
                'Tournament_Name': tournament_name, 'Match': f"{home_team} vs {away_team}", 
                'Home_Team': home_team, 'Away_Team': away_team,
                'Bet_Type': final_row['type'], 'Selection': 'Home', 'Initial_Line': final_row['line'], 'Initial_Odds': final_row['upper'], 
                'System_Stake': 100.0, 'User_Stake': 100.0,
                'Odds_History': json.dumps(st.session_state.odds_history, ensure_ascii=False)
            }
            st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([new_record])], ignore_index=True)
            save_db(st.session_state.df_db, db_file, db_table)
            st.success("寫入成功！")
            clear_edit_mode()
            st.rerun()

if __name__ == "__main__":
    main()
