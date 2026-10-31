import streamlit as st
import pandas as pd
import os
import json
import base64
import re
import io
import requests
import itertools
from datetime import datetime, timedelta, timezone

# ==========================================
# 0. 嘗試載入依賴套件 (AI與雲端資料庫) & 全局時區設定
# ==========================================
HKT = timezone(timedelta(hours=8))

def get_hkt_now():
    return datetime.now(HKT)

def _has_secret(key):
    """安全檢查 Streamlit Secrets (未設定 secrets.toml 時不會令程式崩潰)"""
    try:
        return key in st.secrets
    except Exception:
        return False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    import numpy as np
    HAS_AI_MODULES = True
except ImportError:
    HAS_AI_MODULES = False
    try:
        import numpy as np
    except ImportError:
        np = None

try:
    from sqlalchemy import create_engine
    import sqlalchemy
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# 選配套件：XGBoost / LightGBM (延遲載入：僅在真正訓練時才 import，加快啟動速度並降低記憶體)
def get_xgb_classifier():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        return None

def get_lgbm_classifier():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except ImportError:
        return None

# 限制數值函式庫執行緒 (避免在 CPU 受限的雲端容器上執行緒超訂，降低 CPU 峰值)
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
    HAS_THREADPOOLCTL = True
except ImportError:
    HAS_THREADPOOLCTL = False

# 深度學習/ML 分析最低樣本門檻 (讓球等小樣本盤口亦可訓練；EV 引擎仍維持 15 場)
ML_MIN_SAMPLES = 8

# ==========================================
# 1. 初始化設定與資料庫 Schema
# ==========================================
st.set_page_config(page_title="Actuarial and fund management system by Dr. EdwinPro (V11)", page_icon="⚽", layout="wide")

DB_COLUMNS = [
    'ID', 'Date', 'Status', 
    'Tournament_Name', 'Tournament_Category', 'Match', 'Match_Date', 'Home_Team', 'Away_Team', 
    'Home_Rating', 'Away_Rating', 'League_Size', 'Home_Form', 'Away_Form',
    'Bet_Type', 'Selection', 'Initial_Line', 'Initial_Odds', 
    'System_Stake', 'User_Stake', 'Odds_History',
    'InPlay_Minute', 'Home_DA', 'Away_DA', 'Home_SoT', 'Away_SoT', 'Home_SoFF', 'Away_SoFF',
    'Home_Red', 'Away_Red', 'Home_Sub', 'Away_Sub', 'Home_Possession', 'Away_Possession',
    'Home_Goal', 'Away_Goal', 'Home_Corner', 'Away_Corner', 
    'Home_Goal_Conversion', 'Away_Goal_Conversion', 'Home_Firepower', 'Away_Firepower',
    'Home_Corner_Eff', 'Away_Corner_Eff',
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
        'Match', 'Match_Date', 'Home_Team', 'Away_Team', 'Home_Rating', 'Away_Rating', 
        'Home_Form', 'Away_Form', 'Bet_Type', 'Selection', 'Odds_History', 'Result_Label',
        'Type', 'Account', 'Note'
    ]
    
    df = pd.DataFrame()
    # 優先從雲端 SQL 資料庫讀取
    if HAS_SQLALCHEMY and _has_secret("DB_URL") and st.secrets["DB_URL"]:
        try:
            engine = create_engine(st.secrets["DB_URL"])
            df = pd.read_sql_table(table_name, engine)
            df = process_legacy_columns(df)
        except Exception:
            pass

    # 其次從 GitHub 雲端倉庫讀取
    if df.empty and _has_secret("GITHUB_TOKEN") and _has_secret("GITHUB_REPO"):
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
    if HAS_SQLALCHEMY and _has_secret("DB_URL") and st.secrets["DB_URL"]:
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
    if _has_secret("GITHUB_TOKEN") and _has_secret("GITHUB_REPO"):
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
    
    # 修正：計算系統/用家單位利潤（僅計算有實際下注的注單）
    sys_staked = df[pd.to_numeric(df['System_Stake'], errors='coerce') > 0]
    usr_staked = df[pd.to_numeric(df['User_Stake'], errors='coerce') > 0]
    sys_unit = pd.to_numeric(sys_staked['Unit_Profit'], errors='coerce').sum()
    usr_unit = pd.to_numeric(usr_staked['Unit_Profit'], errors='coerce').sum()
    
    st.markdown("### 📊 數據庫累計總額看板 (Cumulative Summary)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("系統累積淨盈虧 (System Profit)", f"${sys_profit:,.2f}", delta=f"${sys_profit:,.2f}")
    m2.metric("用家真實淨盈虧 (User Profit)", f"${user_profit:,.2f}", delta=f"${user_profit:,.2f}")
    m3.metric("系統單位利潤 (Stake>0)", f"{sys_unit:.2f} U")
    m4.metric("用家單位利潤 (Stake>0)", f"{usr_unit:.2f} U")
    m5.metric("用家派彩總額 (User Payout)", f"${user_payout:,.2f}")
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

# ==========================================
# 2b. 聯賽排名與實力換算 (League Rank Helpers)
# ==========================================
# 舊字母評級對應聯賽中的約略百分位 (S=前10%, A=前30%, B=中游50%, C=後70%, D=後90%)
RATING_LETTER_PCTILE = {"S": 0.10, "A": 0.30, "B": 0.50, "C": 0.70, "D": 0.90}

def get_league_size(r, default=20):
    """安全讀取聯賽隊伍數目 (預設20隊)"""
    try:
        v = int(float(r.get('League_Size', default)))
        return v if v >= 2 else default
    except (TypeError, ValueError):
        return default

def _parse_rank_int(value):
    """將值解析為整數排名；無法解析回傳 None。
    支援 int/float (如 5.0) 及數字字串 (如 '5', '5.0')，經 pandas/資料庫往返仍可正確解析。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return int(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, 'is_integer'):
        try:
            if float(value).is_integer():
                return int(float(value))
            return None
        except (TypeError, ValueError, OverflowError):
            return None
    s = str(value).strip()
    if not s:
        return None
    try:
        f = float(s)
        if f.is_integer():
            return int(f)
    except (TypeError, ValueError, OverflowError):
        return None
    return None

def rating_to_rank(rating, league_size=20):
    """將 DB 儲存值換算為聯賽排名 (1=最強)。
    新數據為數字排名 (支援 5 / 5.0 / '5' 等形式)；舊數據為字母評級 (S/A/B/C/D)，依百分位換算為約略排名。"""
    n = max(2, int(league_size))
    r = _parse_rank_int(rating)
    if r is not None:
        return max(1, min(n, r))
    if rating is None:
        return max(1, round(n * 0.5))
    try:
        if pd.isna(rating):
            return max(1, round(n * 0.5))
    except (TypeError, ValueError):
        pass
    s = str(rating).strip().upper()
    pct = RATING_LETTER_PCTILE.get(s, 0.5)
    return max(1, min(n, round(n * pct)))

def rank_to_strength(rank, league_size=20):
    """聯賽排名轉換為 1~5 實力分 (1名=5分最強，最尾=1分)，與舊字母尺度一致。"""
    n = max(2, int(league_size))
    r = max(1, min(n, int(rank)))
    return 1.0 + 4.0 * (n - r) / (n - 1)

def rating_to_strength(rating, league_size=20):
    """DB 儲存值 (數字排名或舊字母) 轉換為 1~5 實力分，供 ML 特徵使用。
    訓練與預測一律統一先換算為排名再轉實力分，確保兩端表示一致。"""
    n = max(2, int(league_size))
    r = _parse_rank_int(rating)
    if r is None:
        try:
            if rating is None or pd.isna(rating):
                return rank_to_strength(max(1, round(n * 0.5)), n)
        except (TypeError, ValueError):
            pass
        s = str(rating).strip().upper()
        if s in RATING_LETTER_PCTILE:
            r = rating_to_rank(s, n)
        else:
            return rank_to_strength(max(1, round(n * 0.5)), n)
    return rank_to_strength(r, n)

def extract_odds_drift(r):
    """從 Odds_History 計算該注單方向的賠率走勢變化 (最後/最初 - 1)。
    依時間戳記排序 (無時間戳記的按儲存順序)，並截止於 Initial_Odds 對應的時間點，
    不會用到下注時間之後的賠率資料。"""
    try:
        oh = json.loads(str(r.get('Odds_History', '[]')))
        if not isinstance(oh, list) or len(oh) == 0:
            return 0.0
        sel = str(r.get('Selection', ''))
        bt = str(r.get('Bet_Type', '')).replace(' (即場)', '').strip()
        try:
            ln = round(float(r.get('Initial_Line', 0.0)), 3)
        except (TypeError, ValueError):
            ln = 0.0
        try:
            odds = float(r.get('Initial_Odds', 1.90))
        except (TypeError, ValueError):
            odds = None
        key = 'upper' if sel in ('Home', 'Over') else 'lower'
        matched = []
        for pos, h in enumerate(oh):
            try:
                if str(h.get('type', '')).replace(' (即場)', '').strip() == bt and round(float(h.get('line', 0.0)), 3) == ln:
                    matched.append((pos, h))
            except (TypeError, ValueError):
                continue
        if len(matched) < 2:
            return 0.0
        # 依時間戳記排序 (無時間戳記的維持儲存順序)
        matched.sort(key=lambda item: (_oh_sort_ts(item[1], item[0]), item[0]))
        # 截止於 Initial_Odds 對應的時間點 (最後一個賠率相同的快照)
        cutoff = None
        if odds is not None:
            for pos, h in matched:
                v = h.get(key)
                try:
                    if v is not None and abs(float(v) - odds) < 1e-6:
                        cutoff = (pos, h)
                except (TypeError, ValueError):
                    continue
        if cutoff is not None:
            use = [item for item in matched if _ts_cutoff_ok(item[1], item[0], cutoff[1], cutoff[0])]
            if cutoff not in use:
                use.append(cutoff)
        else:
            use = matched
        vals = []
        for pos, h in use:
            v = h.get(key)
            try:
                if v is not None and float(v) > 0:
                    vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if len(vals) >= 2 and vals[0] > 0:
            return vals[-1] / vals[0] - 1.0
    except Exception:
        pass
    return 0.0

def _oh_sort_ts(e, pos):
    """取得賠率快照的排序時間戳記；無時間戳記回傳 datetime.min (按儲存順序)"""
    try:
        return parse_dt_for_comparison(str(e.get('record_time', '')), pos)[0]
    except Exception:
        return datetime.min

def _ts_cutoff_ok(entry, pos, cand_entry, cand_pos):
    """判斷賠率快照 entry 是否在候選/截止快照 (cand) 之前或相同：
    兩者皆有時間戳記 → 比較時間；否則退回儲存順序 (位置) 比較"""
    e_ts = _oh_sort_ts(entry, pos)
    c_ts = _oh_sort_ts(cand_entry, cand_pos)
    if e_ts != datetime.min and c_ts != datetime.min:
        return e_ts <= c_ts
    return pos <= cand_pos

def _final_rating_value(raw_rating, current_rank, default_rank):
    """決定儲存的評級值：原值為數字排名、或用家實際修改過排名 → 儲存數字排名；
    原值為舊字母評級而排名未改動 → 保留原有字母，不自動轉換"""
    if _parse_rank_int(raw_rating) is not None:
        return int(current_rank)
    try:
        if int(current_rank) != int(default_rank):
            return int(current_rank)
    except (TypeError, ValueError):
        return int(current_rank)
    return raw_rating

def prepare_ml_dataset(df, rating_map):
    X, y = [], []
    for _, r in df.iterrows():
        try:
            ls = get_league_size(r)
            hr = rating_to_strength(r.get('Home_Rating', 'C'), ls)
            ar = rating_to_strength(r.get('Away_Rating', 'C'), ls)
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

    # 兩邊互補歸一化：同一盤口兩個方向加總 = 100% (相對勝率，僅作比較顯示)
    _pair_groups = {}
    for c in candidates:
        try:
            _gk = (str(c.get('bet_type')).replace(' (即場)', '').strip(), round(float(c.get('line', 0.0)), 3), c.get('history_idx'))
        except (TypeError, ValueError):
            continue
        _pair_groups.setdefault(_gk, []).append(c)
    for _g in _pair_groups.values():
        if len(_g) == 2:
            _sels = {str(c.get('selection')) for c in _g}
            if _sels == {'Home', 'Away'} or _sels == {'Over', 'Under'}:
                _p1 = float(_g[0].get('prob', 0.5))
                _p2 = float(_g[1].get('prob', 0.5))
                _tot = _p1 + _p2
                if _tot > 0 and np.isfinite(_tot):
                    _g[0]['relative_prob'] = _p1 / _tot
                    _g[1]['relative_prob'] = _p2 / _tot
    for c in candidates:
        if 'relative_prob' not in c:
            c['relative_prob'] = c.get('prob', c.get('base_prob', 0.5))

    for c in candidates:
        c['ev'] = c['prob'] * (c['odds'] - 1) - (1 - c['prob'])

    candidates = sorted(candidates, key=lambda x: x['ev'], reverse=True)
    score = (roi * 0.7) + (acc * 0.3) if valid else -1
    
    return {
        'dim': dim_name, 'valid': valid, 'msg': msg, 'roi': roi, 'acc': acc, 
        'n': n_samples, 'candidates': candidates, 'best': candidates[0], 'score': score
    }

# ==========================================
# 3b. 增強版多模型 ML 引擎 (深度學習/集成模型)
# ==========================================
def safe_num(series):
    """安全轉換為數值"""
    return pd.to_numeric(series, errors='coerce').fillna(0.0)

def add_account_unit_profit_columns(df):
    """為 DataFrame 加入 System_Unit_Profit 和 User_Unit_Profit 欄位"""
    df = df.copy()
    df['System_Unit_Profit'] = df.apply(
        lambda x: pd.to_numeric(x['Unit_Profit'], errors='coerce') if pd.to_numeric(x['System_Stake'], errors='coerce') > 0 else 0.0, axis=1
    )
    df['User_Unit_Profit'] = df.apply(
        lambda x: pd.to_numeric(x['Unit_Profit'], errors='coerce') if pd.to_numeric(x['User_Stake'], errors='coerce') > 0 else 0.0, axis=1
    )
    return df

def prepare_enhanced_ml_dataset(df, rating_map):
    """增強版特徵工程：加入更多特徵供深度學習模型使用。
    特徵包含：聯賽排名實力、近5場狀態、盤口線、賠率、賠率走勢變化 (odds_drift)、
    投注方向編碼 (主/客、大/小) 與實力方向性，讓模型學習『排名優勢 + 投注方向』的關聯。"""
    X, y = [], []
    for _, r in df.iterrows():
        try:
            ls = get_league_size(r)
            hr = rating_to_strength(r.get('Home_Rating', 'C'), ls)
            ar = rating_to_strength(r.get('Away_Rating', 'C'), ls)
            hf = extract_form_points(r.get('Home_Form', '0W0D0L'))
            af = extract_form_points(r.get('Away_Form', '0W0D0L'))
            line = float(r.get('Initial_Line', 0))
            odds = float(r.get('Initial_Odds', 1.90))
            stake = float(r.get('System_Stake', 0))
            # 基礎特徵
            rating_diff = hr - ar
            form_diff = hf - af
            stake_log = np.log1p(max(0, stake)) if np is not None else 0
            odds_margin = odds - 1.0
            # 賠率走勢變化特徵 (考慮盤口賠率變化)
            odds_drift = extract_odds_drift(r)
            # 投注方向編碼：讓模型知道這張注單是買哪一邊
            sel = str(r.get('Selection', ''))
            sel_home = 1.0 if sel == 'Home' else 0.0
            sel_over = 1.0 if sel == 'Over' else 0.0
            # 實力方向性：買主隊時實力差為正代表有利，買客隊時相反
            bt = str(r.get('Bet_Type', '')).replace(' (即場)', '').strip()
            if sel == 'Home':
                side_str_diff = hr - ar
            elif sel == 'Away':
                side_str_diff = ar - hr
            else:
                side_str_diff = 0.0
            is_hd = 1.0 if bt == '讓球' else 0.0
            is_ou = 1.0 if bt == '入球大小' else 0.0
            X.append([hr, ar, hf, af, line, odds, rating_diff, form_diff, stake_log, odds_margin,
                      odds_drift, sel_home, sel_over, side_str_diff, is_hd, is_ou])
            y.append(1 if float(r.get('Unit_Profit', 0)) > 0 else 0)
        except:
            continue
    return np.array(X) if len(X) > 0 else None, np.array(y) if len(y) > 0 else None

def train_ensemble_models(X, y):
    """訓練多個 ML/DL 模型，回傳 (best_model, model_results_dict)
    以單執行緒執行，避免在 CPU 受限的雲端容器上執行緒超訂。"""
    if HAS_THREADPOOLCTL:
        with _threadpool_limits(limits=1):
            return _train_ensemble_models_impl(X, y)
    return _train_ensemble_models_impl(X, y)

def _train_ensemble_models_impl(X, y):
    if X is None or y is None or len(np.unique(y)) < 2:
        return None, {}
    
    n_samples = len(y)
    # 安全防護：交叉驗證 fold 數必須 >= 2
    smallest_class_count = int(np.bincount(y).min())
    if smallest_class_count < 2:
        return None, {}
    cv_folds = min(5, smallest_class_count)
    results = {}
    models = {}
    
    # 1. RandomForest (基線)
    try:
        rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        scores = cross_val_score(rf, X, y, cv=cv_folds, scoring='accuracy', error_score=0)
        results['RandomForest'] = {'cv_mean': float(scores.mean()), 'cv_std': float(scores.std())}
        models['RandomForest'] = rf
    except Exception:
        pass
    
    # 2. GradientBoosting
    try:
        gb = GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=3)
        scores = cross_val_score(gb, X, y, cv=cv_folds, scoring='accuracy', error_score=0)
        results['GradientBoosting'] = {'cv_mean': float(scores.mean()), 'cv_std': float(scores.std())}
        models['GradientBoosting'] = gb
    except Exception:
        pass
    
    # 2b. XGBoost (選配：已安裝時自動加入集成，延遲載入)
    XGB_CLS = get_xgb_classifier()
    if XGB_CLS is not None:
        try:
            xgb = XGB_CLS(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='logloss', verbosity=0, n_jobs=1)
            scores = cross_val_score(xgb, X, y, cv=cv_folds, scoring='accuracy', error_score=0)
            results['XGBoost'] = {'cv_mean': float(scores.mean()), 'cv_std': float(scores.std())}
            models['XGBoost'] = xgb
        except Exception:
            pass
    
    # 2c. LightGBM (選配：已安裝時自動加入集成，延遲載入)
    LGBM_CLS = get_lgbm_classifier()
    if LGBM_CLS is not None:
        try:
            lgbm = LGBM_CLS(n_estimators=50, max_depth=3, random_state=42, verbose=-1, n_jobs=1)
            scores = cross_val_score(lgbm, X, y, cv=cv_folds, scoring='accuracy', error_score=0)
            results['LightGBM'] = {'cv_mean': float(scores.mean()), 'cv_std': float(scores.std())}
            models['LightGBM'] = lgbm
        except Exception:
            pass
    
    # 3. LogisticRegression (需標準化)
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(random_state=42, max_iter=200)
        scores = cross_val_score(lr, X_scaled, y, cv=cv_folds, scoring='accuracy', error_score=0)
        results['LogisticRegression'] = {'cv_mean': float(scores.mean()), 'cv_std': float(scores.std())}
        models['LogisticRegression'] = (lr, scaler)
    except Exception:
        pass
    
    # 4. MLPClassifier (輕量神經網絡，需較大樣本)
    if n_samples >= 100:
        try:
            scaler_mlp = StandardScaler()
            X_scaled_mlp = scaler_mlp.fit_transform(X)
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=300, alpha=0.01)
            scores = cross_val_score(mlp, X_scaled_mlp, y, cv=cv_folds, scoring='accuracy', error_score=0)
            results['MLP_NeuralNet'] = {'cv_mean': float(scores.mean()), 'cv_std': float(scores.std())}
            models['MLP_NeuralNet'] = (mlp, scaler_mlp)
        except Exception:
            pass
    
    # 5. VotingClassifier (集成投票)
    if len(models) >= 2:
        try:
            estimators = []
            for name, model in models.items():
                if isinstance(model, tuple):
                    estimators.append((name, model[0]))
                else:
                    estimators.append((name, model))
            voting = VotingClassifier(estimators=estimators, voting='soft')
            scores = cross_val_score(voting, X, y, cv=cv_folds, scoring='accuracy', error_score=0)
            results['Ensemble_Voting'] = {'cv_mean': float(scores.mean()), 'cv_std': float(scores.std())}
            models['Ensemble_Voting'] = voting
        except Exception:
            pass
    
    # 選擇最佳模型
    if not results:
        return None, {}
    
    best_name = max(results, key=lambda k: results[k]['cv_mean'])
    best_model = models.get(best_name)
    
    # 擬合最佳模型
    if best_model is not None:
        try:
            if isinstance(best_model, tuple):
                best_model[0].fit(best_model[1].fit_transform(X), y)
            else:
                best_model.fit(X, y)
        except Exception:
            pass
    
    return (best_name, best_model), results

def predict_with_model(model_info, x_input):
    """使用模型預測機率"""
    if model_info is None:
        return 0.5
    name, model = model_info
    try:
        if isinstance(model, tuple):
            clf, scaler = model
            x_scaled = scaler.transform(x_input)
            prob = clf.predict_proba(x_scaled)[0][1]
        else:
            prob = model.predict_proba(x_input)[0][1]
        return float(prob)
    except Exception:
        return 0.5

def evaluate_dimension_enhanced(df_subset, dim_name, candidates_base, rating_map, h_data):
    """增強版維度評估：使用多模型集成預測"""
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
    best_model_info = None
    model_results = {}
    
    if valid and HAS_AI_MODULES:
        X, y = prepare_enhanced_ml_dataset(df_subset, rating_map)
        if X is not None and len(np.unique(y)) > 1:
            best_model_info, model_results = train_ensemble_models(X, y)
            
            if best_model_info is not None:
                for c in candidates:
                    _c_sel = str(c.get('selection', ''))
                    _c_bt = str(c.get('bet_type', '')).replace(' (即場)', '').strip()
                    if _c_sel == 'Home':
                        _side_diff = h_data['hr'] - h_data['ar']
                    elif _c_sel == 'Away':
                        _side_diff = h_data['ar'] - h_data['hr']
                    else:
                        _side_diff = 0.0
                    x_input = np.array([[h_data['hr'], h_data['ar'], h_data['hf'], h_data['af'], c['line'], c['odds'], 
                                          h_data['hr'] - h_data['ar'], h_data['hf'] - h_data['af'], 
                                          0.0, c['odds'] - 1.0,
                                          float(c.get('odds_drift', 0.0) or 0.0),
                                          1.0 if _c_sel == 'Home' else 0.0,
                                          1.0 if _c_sel == 'Over' else 0.0,
                                          _side_diff,
                                          1.0 if _c_bt == '讓球' else 0.0,
                                          1.0 if _c_bt == '入球大小' else 0.0]])
                    prob = predict_with_model(best_model_info, x_input)
                    c['prob'] = (c['base_prob'] * 0.3) + (prob * 0.7)
                    c['model_prob'] = prob
                    c['model_name'] = best_model_info[0]
                    c['sample_count'] = n_samples
                model_success = True
    
    if not model_success:
        for c in candidates:
            shift = ((acc - 0.5) * 0.2 + (roi * 0.1)) if valid else 0
            c['prob'] = max(0.05, min(0.95, c['base_prob'] + shift))
            c['model_prob'] = c['prob']
            c['model_name'] = "基礎期望值"
            c['sample_count'] = n_samples
    
    # 兩邊互補歸一化：同一盤口 (相同類型+盤口線+時間點) 的兩個方向互相約束，
    # 歸一化後兩邊相對勝率加總 = 100%，方便直接比較應買哪一邊。
    # 相對化勝率只作兩邊比較顯示，不會取代模型勝率 (prob)，EV 及建議注碼仍以模型勝率計算。
    # 註：讓球整數盤存在走盤退款情況，相對化勝率為兩邊相對強度比較基準。
    _pair_groups = {}
    for c in candidates:
        try:
            _gk = (str(c.get('bet_type')).replace(' (即場)', '').strip(), round(float(c.get('line', 0.0)), 3), c.get('history_idx'))
        except (TypeError, ValueError):
            continue
        _pair_groups.setdefault(_gk, []).append(c)
    for _g in _pair_groups.values():
        if len(_g) == 2:
            _sels = {str(c.get('selection')) for c in _g}
            if _sels == {'Home', 'Away'} or _sels == {'Over', 'Under'}:
                _p1 = float(_g[0].get('prob', 0.5))
                _p2 = float(_g[1].get('prob', 0.5))
                _tot = _p1 + _p2
                if _tot > 0 and np.isfinite(_tot):
                    _g[0]['relative_prob'] = _p1 / _tot
                    _g[1]['relative_prob'] = _p2 / _tot
    for c in candidates:
        if 'relative_prob' not in c:
            c['relative_prob'] = c.get('prob', c.get('base_prob', 0.5))
    
    for c in candidates:
        c['ev'] = c['prob'] * (c['odds'] - 1) - (1 - c['prob'])
    
    candidates = sorted(candidates, key=lambda x: x['ev'], reverse=True)
    score = (roi * 0.7) + (acc * 0.3) if valid else -1
    
    return {
        'dim': dim_name, 'valid': valid, 'msg': msg, 'roi': roi, 'acc': acc, 
        'n': n_samples, 'candidates': candidates, 'best': candidates[0], 'score': score,
        'model_results': model_results, 'best_model_name': best_model_info[0] if best_model_info else "基礎期望值"
    }

# ==========================================
# 3c. V9 市場錨定 ML 引擎 (Market-Anchored ML Engine)
# ==========================================
# 設計重點：
#  1) 以「去水後的市場公平機率」為錨點，模型只學習『市場以外』的額外優勢，
#     並以時間序列前進式驗證 (Walk-Forward) 決定模型可信權重 w；模型未能勝過市場時 w=0 → 不建議投注。
#  2) 訓練資料由每場已結算比賽的完整 Odds_History 重建：所有盤口、所有盤口線、主客/大小兩邊皆成為樣本，
#     (不再只學習曾經下注的注單，消除選擇偏差，樣本量約增加 2.5 倍)。
#  3) 目標值為「有效勝率」e = (單位回報 + 1) / 賠率，完整處理亞洲盤贏半/輸半/走盤，EV = p × 賠率 − 1 為精確期望值。
#  4) 球隊特徵只使用『比賽日期之前』的數據 (近況、攻防數據、對賽、賽事環境)，杜絕未來數據洩漏。
#  5) 自動測試各特徵群組的組合 (例如 排名+賠率變化、排名+近5場+賠率變化+攻防數據...)，以樣本外 Log-Loss 選出最佳組合。
#  6) 以前進式回測計算各盤口類型的樣本外 ROI，長期虧損的盤口自動熔斷 (可於側邊欄手動解除)。

V9_MIN_MATCHES = 30           # 訓練所需最少已結算比賽數
V9_TEAM_LOOKBACK = 8          # 球隊近況/攻防數據回看場數
V9_SHRINK_K = 3.0             # 球隊數據收縮強度 (樣本少時向聯賽平均收縮)
V9_LEAGUE_SHRINK_K = 5.0      # 賽事環境數據收縮強度
V9_DEFAULTS = {'gf': 1.40, 'cf': 5.0, 'sot': 4.3, 'shots': 11.5, 'da': 45.0}
V9_BT_THRESHOLDS = [0.0, 0.02, 0.03, 0.05, 0.08]

V9_FEATURE_GROUPS = {
    '市場賠率': ['f_mkt_logit', 'f_margin'],
    '賠率變化': ['f_drift_side', 'f_drift_opp', 'f_nsnap'],
    '排名': ['f_rank_side'],
    '近5場戰績': ['f_form_side'],
    '球隊攻防數據': ['f_team_edge', 'f_team_shot_edge', 'f_team_n'],
    '對賽往績': ['f_h2h_edge', 'f_h2h_n'],
    '賽事環境': ['f_league_edge'],
    '盤口結構': ['f_is_hd', 'f_is_ou', 'f_side_hi', 'f_hd_home', 'f_ou_over', 'f_cr_over', 'f_quarter', 'f_line_norm'],
    '因素交互': ['f_x_rank_drift', 'f_x_form_drift', 'f_x_team_drift', 'f_x_rank_form', 'f_x_team_rank', 'f_x_mkt_drift'],
}
V9_ALL_FEATURES = [f for g in V9_FEATURE_GROUPS.values() for f in g]

V9_FEATURE_DESC = {
    'f_mkt_logit': ('市場賠率', '去水後市場公平勝率 (logit)，模型的錨點'),
    'f_margin': ('市場賠率', '該盤口抽水 (1/上賠率 + 1/下賠率)'),
    'f_drift_side': ('賠率變化', '本方向賠率由首個時間點至當前的變化率 (負數=受注降水)'),
    'f_drift_opp': ('賠率變化', '對面方向賠率變化率'),
    'f_nsnap': ('賠率變化', '該盤口賠率紀錄次數 (log)'),
    'f_rank_side': ('排名', '聯賽排名實力差 (按投注方向定正負)'),
    'f_form_side': ('近5場戰績', '近5場積分差 (按投注方向定正負)'),
    'f_team_edge': ('球隊攻防數據', '數據庫過往賽事：預期入球差/總入球/總角球 減 盤口線'),
    'f_team_shot_edge': ('球隊攻防數據', '數據庫過往賽事：射正/危險進攻攻防差'),
    'f_team_n': ('球隊攻防數據', '兩隊在數據庫內的過往場數 (數據可信度)'),
    'f_h2h_edge': ('對賽往績', '同一對手過往對賽：入球差/總入球/總角球 減 盤口線'),
    'f_h2h_n': ('對賽往績', '過往對賽場數'),
    'f_league_edge': ('賽事環境', '同一賽事過往平均入球/角球/主場優勢 減 盤口線'),
    'f_is_hd': ('盤口結構', '是否讓球盤'),
    'f_is_ou': ('盤口結構', '是否入球大小盤'),
    'f_side_hi': ('盤口結構', '投注方向 (主隊/大 = 1)'),
    'f_hd_home': ('盤口結構', '讓球 × 買主隊 (學習讓球主客方向偏差)'),
    'f_ou_over': ('盤口結構', '入球大小 × 買大 (學習入球大小方向偏差)'),
    'f_cr_over': ('盤口結構', '角球大小 × 買大 (學習角球大小方向偏差)'),
    'f_quarter': ('盤口結構', '是否四分一盤 (會出現贏半/輸半)'),
    'f_line_norm': ('盤口結構', '盤口線 (標準化)'),
    'f_x_rank_drift': ('因素交互', '排名 × 賠率變化'),
    'f_x_form_drift': ('因素交互', '近5場 × 賠率變化'),
    'f_x_team_drift': ('因素交互', '攻防數據 × 賠率變化'),
    'f_x_rank_form': ('因素交互', '排名 × 近5場'),
    'f_x_team_rank': ('因素交互', '攻防數據 × 排名'),
    'f_x_mkt_drift': ('因素交互', '市場勝率 × 賠率變化'),
}

LEGACY_16_FEATURES = [
    ('hr', '主隊實力分 (排名/字母評級換算 1~5)'),
    ('ar', '客隊實力分'),
    ('hf', '主隊近5場積分 (勝3和1)'),
    ('af', '客隊近5場積分'),
    ('line', '盤口線'),
    ('odds', '賠率'),
    ('rating_diff', '主客實力差'),
    ('form_diff', '主客近況積分差'),
    ('stake_log', 'log(系統注碼)：訓練用實際注碼、預測時固定為 0 → 訓練/預測不一致 (缺陷)'),
    ('odds_margin', '賠率 − 1 (與 odds 完全重複)'),
    ('odds_drift', '本方向賠率走勢變化'),
    ('sel_home', '是否買主隊'),
    ('sel_over', '是否買大'),
    ('side_str_diff', '按方向的實力差'),
    ('is_hd', '是否讓球盤'),
    ('is_ou', '是否入球大小盤'),
]


def v9_clean_bt(bt):
    return str(bt).replace(' (即場)', '').strip()


def v9_unit_return(diff, odds):
    """亞洲盤單位回報 (與 calculate_settlement 完全相同的判定規則)"""
    diff = round(float(diff), 2)
    odds = float(odds)
    if diff >= 0.5:
        return odds - 1.0
    if abs(diff - 0.25) < 1e-9:
        return (odds - 1.0) / 2.0
    if abs(diff) < 1e-9:
        return 0.0
    if abs(diff + 0.25) < 1e-9:
        return -0.5
    return -1.0


def v9_diff(btype, side, line, hg, ag, hc, ac):
    if btype == '讓球':
        return (hg + line - ag) if side == 'Home' else (ag - line - hg)
    if btype == '入球大小':
        return (hg + ag - line) if side == 'Over' else (line - hg - ag)
    return (hc + ac - line) if side == 'Over' else (line - hc - ac)


def v9_devig(o_side, o_opp):
    """比例法去水：回傳本方向公平機率"""
    try:
        a, b = 1.0 / float(o_side), 1.0 / float(o_opp)
        if a > 0 and b > 0:
            return a / (a + b), a + b
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return 0.5, 1.0


def v9_logit(p):
    p = min(0.98, max(0.02, float(p)))
    return float(np.log(p / (1.0 - p)))


def v9_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def v9_parse_oh_time(s, ref_date=None, fmt=None):
    """解析賠率紀錄時間。V10：同一份紀錄先自動偵測日期格式 (日-月 如 18-09 / 月-日 如 10-22)，
    無法判斷時選擇最接近比賽日期的解讀，並依比賽日期推斷年份"""
    ref = ref_date
    if ref is not None and hasattr(ref, 'to_pydatetime'):
        ref = ref.to_pydatetime()
    if ref is not None:
        ref = ref.replace(tzinfo=None, hour=23, minute=59)
    return v10_parse_time(s, fmt, ref)


def v9_group_markets(odds_history, ref_date=None):
    """將 Odds_History 依 (盤口類型, 盤口線) 分組並依時間排序 (無時間按儲存順序)"""
    markets = {}
    if not isinstance(odds_history, list):
        return markets
    _fmt = v10_detect_fmt(odds_history)
    for pos, h in enumerate(odds_history):
        try:
            bt = v9_clean_bt(h.get('type', ''))
            if bt not in ('讓球', '入球大小', '角球大小'):
                continue
            ln = round(float(h.get('line', 0.0)), 3)
            up, lw = float(h.get('upper')), float(h.get('lower'))
            if up <= 1.0 or lw <= 1.0 or not v11_valid_pair(up, lw):
                continue
        except (TypeError, ValueError):
            continue
        ts = v9_parse_oh_time(h.get('record_time', ''), ref_date, _fmt)
        markets.setdefault((bt, ln), []).append((ts or datetime.min, pos, up, lw))
    for k in markets:
        markets[k].sort(key=lambda x: (x[0], x[1]))
    return markets


def v9_safe_float(v, default=np.nan):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def v9_strength(rating, league_size):
    try:
        return float(rating_to_strength(rating, league_size))
    except Exception:
        return 3.0


def v9_build_match_table(df_settled):
    """由已結算注單建立『一場比賽一行』的賽果與賽況數據表 (含最完整的賽前賠率紀錄)"""
    if df_settled is None or df_settled.empty:
        return pd.DataFrame()
    d = df_settled.copy()
    d['_md'] = pd.to_datetime(d['Match_Date'], errors='coerce')
    d['_dt'] = pd.to_datetime(d['Date'], errors='coerce')
    d['_md'] = d['_md'].fillna(d['_dt'].dt.normalize())
    d = d[d['_md'].notna() & d['Match'].notna()]
    rows = []
    for (match, md), g in d.groupby(['Match', '_md'], sort=False):
        def first_num(col):
            if col not in g.columns:
                return np.nan
            s = pd.to_numeric(g[col], errors='coerce').dropna()
            return float(s.iloc[0]) if len(s) else np.nan
        hg, ag, hc, ac = first_num('Home_Goal'), first_num('Away_Goal'), first_num('Home_Corner'), first_num('Away_Corner')
        if np.isnan(hg) or np.isnan(ag):
            continue
        pre = g[~g['Bet_Type'].astype(str).str.contains('即場', na=False)]
        src = pre if not pre.empty else g
        best_oh, best_len = [], -1
        for s in src['Odds_History'].dropna():
            try:
                oh = json.loads(str(s))
                if isinstance(oh, list) and len(oh) > best_len:
                    best_oh, best_len = oh, len(oh)
            except Exception:
                continue
        r0 = g.iloc[0]
        ls = get_league_size(r0)
        _unk = v11_strength_unknown(r0.get('Home_Rating'), r0.get('Away_Rating'), ls, r0.get('Tournament_Name', ''), r0.get('Tournament_Category'))
        rows.append({
            'key': f"{match}|{md.strftime('%Y-%m-%d')}", 'match': match, 'date': md,
            'tourn': str(r0.get('Tournament_Name', '')), 'home': str(r0.get('Home_Team', '')), 'away': str(r0.get('Away_Team', '')),
            'hs': 3.0 if _unk else v9_strength(r0.get('Home_Rating'), ls), 'as': 3.0 if _unk else v9_strength(r0.get('Away_Rating'), ls),
            'unk': bool(_unk),
            'hfp': float(extract_form_points(r0.get('Home_Form', '0W0D0L'))), 'afp': float(extract_form_points(r0.get('Away_Form', '0W0D0L'))),
            'hg': hg, 'ag': ag, 'hc': hc if not np.isnan(hc) else np.nan, 'ac': ac if not np.isnan(ac) else np.nan,
            'hsot': first_num('Home_SoT'), 'asot': first_num('Away_SoT'),
            'hsoff': first_num('Home_SoFF'), 'asoff': first_num('Away_SoFF'),
            'hda': first_num('Home_DA'), 'ada': first_num('Away_DA'),
            'oh': best_oh, 'nbets': len(g), 'oh_len': max(0, best_len),
        })
    mt = pd.DataFrame(rows)
    if mt.empty:
        return mt
    # V10：同名球隊在不同日期的比賽是獨立比賽 (按 Match + Match_Date 區分)，不再自動合併
    return mt.sort_values('date').reset_index(drop=True)


def v9_find_duplicate_matches(df_db):
    """找出同一對賽在不同日期、但賽果完全相同的比賽 (可能是舊版即場更新把兩場比賽一起覆蓋所致)，供資料檢查"""
    try:
        d = df_db[df_db['Status'] == 'Settled'].copy()
        if d.empty:
            return pd.DataFrame()
        d['_md'] = d['Match_Date'].astype(str).str[:10]
        g = d.groupby('Match').agg(日期=('_md', lambda x: sorted(set(x))),
                                   比分=('Home_Goal', lambda x: sorted(set(pd.to_numeric(x, errors='coerce').dropna()))),
                                   客隊入球=('Away_Goal', lambda x: sorted(set(pd.to_numeric(x, errors='coerce').dropna()))),
                                   注單數=('ID', 'size'))
        g = g[(g['日期'].apply(len) > 1) & (g['比分'].apply(len) == 1) & (g['客隊入球'].apply(len) == 1)]
        if g.empty:
            return pd.DataFrame()
        g = g.reset_index()
        g['日期'] = g['日期'].apply(lambda x: ' / '.join(x))
        g['比分'] = g.apply(lambda r: f"{int(r['比分'][0])}-{int(r['客隊入球'][0])}", axis=1)
        return g[['Match', '日期', '比分', '注單數']].rename(columns={'Match': '比賽'})
    except Exception:
        return pd.DataFrame()


def v9_team_long(mt):
    """將比賽表轉換為『每隊每場一行』的長表 (進攻=for, 防守=against)"""
    if mt is None or mt.empty:
        return pd.DataFrame()
    recs = []
    for _, r in mt.iterrows():
        hsh = r['hsot'] + r['hsoff'] if not (np.isnan(r['hsot']) or np.isnan(r['hsoff'])) else np.nan
        ash = r['asot'] + r['asoff'] if not (np.isnan(r['asot']) or np.isnan(r['asoff'])) else np.nan
        recs.append({'team': r['home'], 'date': r['date'], 'tourn': r['tourn'], 'gf': r['hg'], 'ga': r['ag'], 'cf': r['hc'], 'ca': r['ac'],
                     'sotf': r['hsot'], 'sota': r['asot'], 'shf': hsh, 'sha': ash, 'daf': r['hda'], 'daa': r['ada']})
        recs.append({'team': r['away'], 'date': r['date'], 'tourn': r['tourn'], 'gf': r['ag'], 'ga': r['hg'], 'cf': r['ac'], 'ca': r['hc'],
                     'sotf': r['asot'], 'sota': r['hsot'], 'shf': ash, 'sha': hsh, 'daf': r['ada'], 'daa': r['hda']})
    return pd.DataFrame(recs)


def v9_league_profile(mt, tourn, before):
    """賽事環境 (只用 before 之前的比賽)：平均總入球、總角球、主隊淨勝球，向全局平均收縮"""
    prior = mt[mt['date'] < before] if (mt is not None and not mt.empty) else pd.DataFrame()
    g_goals = float((prior['hg'] + prior['ag']).mean()) if len(prior) else 2 * V9_DEFAULTS['gf']
    g_corn_s = (prior['hc'] + prior['ac']).dropna() if len(prior) else pd.Series(dtype=float)
    g_corn = float(g_corn_s.mean()) if len(g_corn_s) else 2 * V9_DEFAULTS['cf']
    g_hgd = float((prior['hg'] - prior['ag']).mean()) if len(prior) else 0.25
    if not np.isfinite(g_goals):
        g_goals = 2 * V9_DEFAULTS['gf']
    if not np.isfinite(g_hgd):
        g_hgd = 0.25
    t = prior[prior['tourn'] == tourn] if len(prior) else prior
    k = V9_LEAGUE_SHRINK_K
    n = len(t)
    goals = (float((t['hg'] + t['ag']).sum()) + k * g_goals) / (n + k) if n else g_goals
    tc = (t['hc'] + t['ac']).dropna() if n else pd.Series(dtype=float)
    corners = (float(tc.sum()) + k * g_corn) / (len(tc) + k) if len(tc) else g_corn
    hgd = (float((t['hg'] - t['ag']).sum()) + k * g_hgd) / (n + k) if n else g_hgd
    return {'goals': goals, 'corners': corners, 'hgd': hgd, 'n': n}


def v9_team_profile(tl, team, before, league):
    """球隊攻防數據 (只用 before 之前最近 V9_TEAM_LOOKBACK 場)，各項向賽事平均收縮"""
    prior_g = league['goals'] / 2.0
    prior_c = league['corners'] / 2.0
    if tl is None or tl.empty:
        h = pd.DataFrame()
    else:
        h = tl[(tl['team'] == team) & (tl['date'] < before)].sort_values('date').tail(V9_TEAM_LOOKBACK)
    k = V9_SHRINK_K

    def shr(col, prior):
        if h.empty or col not in h.columns:
            return prior
        s = pd.to_numeric(h[col], errors='coerce').dropna()
        return (float(s.sum()) + k * prior) / (len(s) + k)
    return {
        'gf': shr('gf', prior_g), 'ga': shr('ga', prior_g), 'cf': shr('cf', prior_c), 'ca': shr('ca', prior_c),
        'sotf': shr('sotf', V9_DEFAULTS['sot']), 'sota': shr('sota', V9_DEFAULTS['sot']),
        'daf': shr('daf', V9_DEFAULTS['da']), 'daa': shr('daa', V9_DEFAULTS['da']),
        'n': 0 if h.empty else len(h),
    }


def v9_h2h(mt, home, away, before):
    if mt is None or mt.empty:
        return []
    p = mt[(mt['date'] < before) & (((mt['home'] == home) & (mt['away'] == away)) | ((mt['home'] == away) & (mt['away'] == home)))]
    out = []
    for _, r in p.iterrows():
        same = (r['home'] == home)
        gd = (r['hg'] - r['ag']) if same else (r['ag'] - r['hg'])
        out.append({'gd': gd, 'tg': r['hg'] + r['ag'], 'tc': (r['hc'] + r['ac']) if not np.isnan(r['hc'] + r['ac']) else np.nan})
    return out


def v9_match_context(mt, tl, home, away, tourn, date, hs, as_, hfp, afp):
    league = v9_league_profile(mt, tourn, date)
    return {
        'hs': float(hs), 'as': float(as_), 'hfp': float(hfp), 'afp': float(afp),
        'hp': v9_team_profile(tl, home, date, league), 'ap': v9_team_profile(tl, away, date, league),
        'league': league, 'h2h': v9_h2h(mt, home, away, date),
    }


def v9_side_features(btype, side, line, o_side, o_opp, drift_side, drift_opp, nsnap, ctx):
    """按投注方向計算特徵 (s=+1 主隊/大，s=-1 客隊/小)，全部只用賽前資料"""
    s = 1.0 if side in ('Home', 'Over') else -1.0
    p_mkt, margin = v9_devig(o_side, o_opp)
    hp, ap, lg = ctx['hp'], ctx['ap'], ctx['league']
    hs, as_, hfp, afp = ctx['hs'], ctx['as'], ctx['hfp'], ctx['afp']
    h2h = ctx['h2h']
    line = float(line)
    if btype == '讓球':
        rank = s * (hs - as_) / 4.0
        form = s * (hfp - afp) / 15.0
        exp_gd = (hp['gf'] + ap['ga']) / 2.0 - (ap['gf'] + hp['ga']) / 2.0 + lg['hgd'] * 0.5
        team = s * (exp_gd + line)
        shot = s * (((hp['sotf'] + ap['sota']) / 2.0) - ((ap['sotf'] + hp['sota']) / 2.0)) / 3.0
        h2v = s * (float(np.mean([x['gd'] for x in h2h])) + line) if h2h else 0.0
        leag = s * (lg['hgd'] + line)
        line_norm = line / 1.5
    elif btype == '入球大小':
        rank = s * ((hs + as_) / 2.0 - 3.0) / 2.0
        form = s * ((hfp + afp) / 2.0 - 7.0) / 7.0
        exp_t = (hp['gf'] + ap['ga']) / 2.0 + (ap['gf'] + hp['ga']) / 2.0
        team = s * (exp_t - line)
        shot = s * (((hp['sotf'] + ap['sota']) / 2.0 + (ap['sotf'] + hp['sota']) / 2.0) - 2 * V9_DEFAULTS['sot']) / 3.0
        h2v = s * (float(np.mean([x['tg'] for x in h2h])) - line) if h2h else 0.0
        leag = s * (lg['goals'] - line)
        line_norm = (line - 2.5) / 1.0
    else:
        rank = s * abs(hs - as_) / 4.0
        form = s * abs(hfp - afp) / 15.0
        exp_c = (hp['cf'] + ap['ca']) / 2.0 + (ap['cf'] + hp['ca']) / 2.0
        team = s * (exp_c - line) / 2.0
        shot = s * (((hp['daf'] + ap['daa']) / 2.0 + (ap['daf'] + hp['daa']) / 2.0) - 2 * V9_DEFAULTS['da']) / 20.0
        tcs = [x['tc'] for x in h2h if not np.isnan(x['tc'])]
        h2v = s * (float(np.mean(tcs)) - line) / 2.0 if tcs else 0.0
        leag = s * (lg['corners'] - line) / 2.0
        line_norm = (line - 10.0) / 2.0
    team = float(np.clip(team, -3, 3))
    h2v = float(np.clip(h2v, -3, 3))
    leag = float(np.clip(leag, -3, 3))
    ds = float(np.clip(v9_safe_float(drift_side, 0.0), -0.5, 0.5))
    do = float(np.clip(v9_safe_float(drift_opp, 0.0), -0.5, 0.5))
    mk = v9_logit(p_mkt)
    f = {
        'f_mkt_logit': mk, 'f_margin': float(margin - 1.0),
        'f_drift_side': ds, 'f_drift_opp': do, 'f_nsnap': float(np.log1p(max(1, int(nsnap)))),
        'f_rank_side': float(rank), 'f_form_side': float(form),
        'f_team_edge': team, 'f_team_shot_edge': float(np.clip(shot, -3, 3)),
        'f_team_n': float(min(hp['n'], ap['n'])) / V9_TEAM_LOOKBACK,
        'f_h2h_edge': h2v, 'f_h2h_n': float(min(len(h2h), 3)) / 3.0,
        'f_league_edge': leag,
        'f_is_hd': 1.0 if btype == '讓球' else 0.0, 'f_is_ou': 1.0 if btype == '入球大小' else 0.0,
        'f_side_hi': 1.0 if s > 0 else 0.0,
        'f_hd_home': 1.0 if (btype == '讓球' and s > 0) else 0.0,
        'f_ou_over': 1.0 if (btype == '入球大小' and s > 0) else 0.0,
        'f_cr_over': 1.0 if (btype == '角球大小' and s > 0) else 0.0,
        'f_quarter': 1.0 if abs((abs(line) * 4) % 2 - 1) < 1e-6 else 0.0,
        'f_line_norm': float(line_norm),
    }
    f['f_x_rank_drift'] = f['f_rank_side'] * ds * 10
    f['f_x_form_drift'] = f['f_form_side'] * ds * 10
    f['f_x_team_drift'] = team * ds * 10
    f['f_x_rank_form'] = f['f_rank_side'] * f['f_form_side']
    f['f_x_team_rank'] = team * f['f_rank_side']
    f['f_x_mkt_drift'] = mk * ds * 10
    return f, p_mkt


def v9_build_training_set(mt):
    """每場比賽、每個盤口 (類型+盤口線) 取最後一個賠率時間點，主客/大小兩邊皆成為訓練樣本"""
    if mt is None or mt.empty:
        return pd.DataFrame()
    tl = v9_team_long(mt)
    rows = []
    for _, m in mt.iterrows():
        ctx = v9_match_context(mt, tl, m['home'], m['away'], m['tourn'], m['date'], m['hs'], m['as'], m['hfp'], m['afp'])
        markets = v9_group_markets(m['oh'], m['date'].to_pydatetime() if hasattr(m['date'], 'to_pydatetime') else None)
        for (bt, ln), snaps in markets.items():
            if bt == '角球大小' and (np.isnan(m['hc']) or np.isnan(m['ac'])):
                continue
            first, last = snaps[0], snaps[-1]
            for side in (['Home', 'Away'] if bt == '讓球' else ['Over', 'Under']):
                hi = side in ('Home', 'Over')
                o_side, o_opp = (last[2], last[3]) if hi else (last[3], last[2])
                o0_side, o0_opp = (first[2], first[3]) if hi else (first[3], first[2])
                f, p_mkt = v9_side_features(bt, side, ln, o_side, o_opp, o_side / o0_side - 1.0, o_opp / o0_opp - 1.0, len(snaps), ctx)
                diff = v9_diff(bt, side, ln, m['hg'], m['ag'], 0 if np.isnan(m['hc']) else m['hc'], 0 if np.isnan(m['ac']) else m['ac'])
                ret = v9_unit_return(diff, o_side)
                f.update({'key': m['key'], 'date': m['date'], 'tourn': m['tourn'], 'bet_type': bt, 'side': side, 'line': ln,
                          'odds': o_side, 'p_mkt': p_mkt, 'ret': ret, 'e': (ret + 1.0) / o_side})
                rows.append(f)
    return pd.DataFrame(rows)


def v9_soft_fit(kind, X, e):
    """以軟標籤 (有效勝率 e) 訓練：每行複製為 y=1 (權重 e) 與 y=0 (權重 1-e)，等同精確的期望值對數損失"""
    X = np.asarray(X, dtype=float)
    e = np.clip(np.asarray(e, dtype=float), 0.0, 1.0)
    X2 = np.vstack([X, X])
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w2 = np.concatenate([e, 1.0 - e]) + 1e-6
    if kind == 'LR':
        mdl = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=0.3, max_iter=1000))])
        mdl.fit(X2, y2, lr__sample_weight=w2)
    else:
        mdl = GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.8, min_samples_leaf=20, random_state=42)
        mdl.fit(X2, y2, sample_weight=w2)
    return mdl


def v9_soft_logloss(p, e):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    e = np.asarray(e, dtype=float)
    return float(-np.mean(e * np.log(p) + (1 - e) * np.log(1 - p)))


def v9_folds(dates, n_folds=5, init_frac=0.4):
    ud = sorted(pd.Series(dates).dropna().unique())
    if len(ud) < 4:
        return []
    start = max(1, int(len(ud) * init_frac))
    test_dates = ud[start:]
    if not test_dates:
        return []
    chunks = np.array_split(np.array(test_dates, dtype=object), min(n_folds, len(test_dates)))
    return [(c[0], c[-1]) for c in chunks if len(c)]


def v9_walk_forward(ts, cols, kind, folds):
    """時間序列前進式驗證：每一折只用之前日期的數據訓練，預測之後日期 (完全樣本外)"""
    preds = pd.Series(np.nan, index=ts.index)
    for d0, d1 in folds:
        tr = ts[ts['date'] < d0]
        te = ts[(ts['date'] >= d0) & (ts['date'] <= d1)]
        if len(tr) < 80 or te.empty:
            continue
        try:
            if kind == 'AVG':
                m1 = v9_soft_fit('LR', tr[cols].values, tr['e'].values)
                m2 = v9_soft_fit('GB', tr[cols].values, tr['e'].values)
                p = 0.5 * m1.predict_proba(te[cols].values)[:, 1] + 0.5 * m2.predict_proba(te[cols].values)[:, 1]
            else:
                m = v9_soft_fit(kind, tr[cols].values, tr['e'].values)
                p = m.predict_proba(te[cols].values)[:, 1]
            preds.loc[te.index] = p
        except Exception:
            continue
    return preds


def v9_blend(p_mkt, p_model, w):
    lm = np.log(np.clip(p_mkt, 1e-4, 1 - 1e-4) / (1 - np.clip(p_mkt, 1e-4, 1 - 1e-4)))
    lp = np.log(np.clip(p_model, 1e-4, 1 - 1e-4) / (1 - np.clip(p_model, 1e-4, 1 - 1e-4)))
    return v9_sigmoid((1 - w) * lm + w * lp)


def v9_market_e(ts):
    """市場基準的有效勝率：在市場公平賠率下 E[e] = 公平機率；轉換至實際賠率時 EV_mkt = p_mkt × 賠率 − 1"""
    return ts['p_mkt'].values


def v9_backtest(oof, thr):
    """每場比賽、每種盤口類型最多選一注 (EV 最高且 ≥ 門檻)，以樣本外預測計算真實回報"""
    if oof is None or oof.empty:
        return pd.DataFrame()
    d = oof[oof['ev'] >= thr]
    if d.empty:
        return pd.DataFrame(columns=oof.columns)
    idx = d.groupby(['key', 'bet_type'])['ev'].idxmax()
    return d.loc[idx]


def v9_train_engine(df_settled):
    """完整訓練流程：建表 → 特徵組合搜尋 → 模型選擇 → 市場錨定權重 → 回測與熔斷"""
    out = {'trained': False, 'msg': '', 'w': 0.0, 'cols': [], 'groups': [], 'kind': 'MARKET',
           'n_matches': 0, 'n_rows': 0, 'combo_table': [], 'model_table': [], 'bt_table': [], 'bt_type': {},
           'models': None, 'll_market': None, 'll_model': None, 'feat_importance': []}
    if not HAS_AI_MODULES:
        out['msg'] = '未安裝 scikit-learn，只能使用市場公平機率 (不建議投注)。'
        return out
    mt = v9_build_match_table(df_settled)
    out['n_matches'] = 0 if mt.empty else len(mt)
    if mt.empty or len(mt) < V9_MIN_MATCHES:
        out['msg'] = f'已結算比賽不足 {V9_MIN_MATCHES} 場 (目前 {out["n_matches"]} 場)，暫以市場公平機率運算。'
        return out
    ts = v9_build_training_set(mt)
    out['n_rows'] = len(ts)
    if ts.empty or len(ts) < 150:
        out['msg'] = '可用盤口樣本不足，暫以市場公平機率運算。'
        return out
    ts = ts.reset_index(drop=True)
    folds = v9_folds(ts['date'])
    if not folds:
        out['msg'] = '比賽日期不足以進行前進式驗證。'
        return out
    run_ctx = _threadpool_limits(limits=1) if HAS_THREADPOOLCTL else None
    try:
        if run_ctx is not None:
            run_ctx.__enter__()
        # --- 1) 特徵組合搜尋 (市場賠率為必選錨點 + 其餘群組任意組合，LR 快速評估) ---
        other = [g for g in V9_FEATURE_GROUPS if g != '市場賠率']
        valid_mask = None
        combos = []
        for k in range(0, len(other) + 1):
            for comb in itertools.combinations(other, k):
                if '因素交互' in comb and len(comb) == 1:
                    continue
                groups = ['市場賠率'] + list(comb)
                cols = [c for g in groups for c in V9_FEATURE_GROUPS[g]]
                p = v9_walk_forward(ts, cols, 'LR', folds)
                if valid_mask is None:
                    valid_mask = p.notna()
                m = valid_mask & p.notna()
                if m.sum() < 50:
                    continue
                combos.append({'groups': groups, 'cols': cols, 'll': v9_soft_logloss(p[m], ts.loc[m, 'e']), 'n': int(m.sum())})
        if not combos or valid_mask is None or valid_mask.sum() < 50:
            out['msg'] = '前進式驗證樣本不足。'
            return out
        vm = valid_mask
        ll_market = v9_soft_logloss(ts.loc[vm, 'p_mkt'], ts.loc[vm, 'e'])
        out['ll_market'] = ll_market
        combos.sort(key=lambda c: c['ll'])
        best_ll = combos[0]['ll']
        # 簡約原則：在最佳 Log-Loss 0.0005 範圍內選擇群組最少的組合，降低過度擬合
        near = [c for c in combos if c['ll'] <= best_ll + 0.0005]
        chosen = sorted(near, key=lambda c: (len(c['groups']), c['ll']))[0]
        out['combo_table'] = [{'特徵組合': ' + '.join(c['groups']), '群組數': len(c['groups']), '樣本外 Log-Loss': round(c['ll'], 5),
                               '相對市場改善': round(ll_market - c['ll'], 5)} for c in combos]
        cols = chosen['cols']
        out['cols'], out['groups'] = cols, chosen['groups']
        # --- 2) 模型選擇 (LR / GB / 平均集成) ---
        model_preds = {'LR': v9_walk_forward(ts, cols, 'LR', folds)}
        if len(ts) >= 400:
            model_preds['GB'] = v9_walk_forward(ts, cols, 'GB', folds)
            model_preds['AVG'] = (model_preds['LR'] + model_preds['GB']) / 2.0
        mtab = []
        best_kind, best_kind_ll = None, 1e9
        for kind, p in model_preds.items():
            m = vm & p.notna()
            if m.sum() < 50:
                continue
            ll = v9_soft_logloss(p[m], ts.loc[m, 'e'])
            mtab.append({'模型': {'LR': '邏輯迴歸 (LR)', 'GB': '梯度提升樹 (GB)', 'AVG': 'LR+GB 集成'}[kind], '樣本外 Log-Loss': round(ll, 5),
                         '相對市場改善': round(ll_market - ll, 5)})
            if ll < best_kind_ll:
                best_kind, best_kind_ll = kind, ll
        mtab.append({'模型': '市場公平機率 (基準)', '樣本外 Log-Loss': round(ll_market, 5), '相對市場改善': 0.0})
        out['model_table'] = mtab
        if best_kind is None:
            out['msg'] = '模型訓練失敗，暫以市場公平機率運算。'
            return out
        p_oof = model_preds[best_kind]
        # --- 3) 市場錨定權重 w (樣本外 Log-Loss 最低者；改善不足 0.0005 則 w=0) ---
        m = vm & p_oof.notna()
        best_w, best_w_ll = 0.0, ll_market
        for w in np.linspace(0.0, 1.0, 11):
            ll = v9_soft_logloss(v9_blend(ts.loc[m, 'p_mkt'].values, p_oof[m].values, w), ts.loc[m, 'e'])
            if ll < best_w_ll - 1e-9:
                best_w, best_w_ll = float(w), ll
        if ll_market - best_w_ll < 0.0005:
            best_w, best_w_ll = 0.0, ll_market
        out['w'], out['kind'], out['ll_model'] = best_w, best_kind, best_w_ll
        # --- 4) 回測 (樣本外) ---
        oof = ts.loc[m, ['key', 'date', 'bet_type', 'side', 'line', 'odds', 'p_mkt', 'ret']].copy()
        oof['p'] = v9_blend(oof['p_mkt'].values, p_oof[m].values, best_w)
        oof['ev'] = oof['p'] * oof['odds'] - 1.0
        out['oof'] = oof.reset_index(drop=True)
        bt_rows = []
        bt_type = {}
        for thr in V9_BT_THRESHOLDS:
            b = v9_backtest(oof, thr)
            n = len(b)
            bt_rows.append({'EV 門檻': f"{thr*100:.0f}%", '回測注數': n, '命中率': f"{(b['ret'] > 0).mean()*100:.1f}%" if n else '-',
                            '單位盈虧 (U)': round(float(b['ret'].sum()), 2) if n else 0.0,
                            'ROI': f"{b['ret'].mean()*100:.1f}%" if n else '-'})
            for bt in ('讓球', '入球大小', '角球大小'):
                bb = b[b['bet_type'] == bt] if n else b
                bt_type.setdefault(bt, {})[thr] = {'n': int(len(bb)), 'roi': float(bb['ret'].mean()) if len(bb) else 0.0,
                                                   'units': float(bb['ret'].sum()) if len(bb) else 0.0}
        # 市場整體 (隨機買任一邊) 的 ROI 作對照
        for bt in ('讓球', '入球大小', '角球大小'):
            sub = oof[oof['bet_type'] == bt]
            bt_type.setdefault(bt, {})['all'] = {'n': int(len(sub)), 'roi': float(sub['ret'].mean()) if len(sub) else 0.0}
        out['bt_table'] = bt_rows
        out['bt_type'] = bt_type
        # --- 5) 以全部數據擬合最終模型 ---
        models = {}
        if best_w > 0:
            if best_kind in ('LR', 'AVG'):
                models['LR'] = v9_soft_fit('LR', ts[cols].values, ts['e'].values)
            if best_kind in ('GB', 'AVG'):
                models['GB'] = v9_soft_fit('GB', ts[cols].values, ts['e'].values)
            if 'LR' in models:
                try:
                    coef = models['LR'].named_steps['lr'].coef_[0]
                    out['feat_importance'] = sorted([{'特徵': c, '群組': V9_FEATURE_DESC.get(c, ('', ''))[0],
                                                      '說明': V9_FEATURE_DESC.get(c, ('', ''))[1], '標準化係數': round(float(v), 4)}
                                                     for c, v in zip(cols, coef)], key=lambda x: -abs(x['標準化係數']))
                except Exception:
                    pass
        out['models'] = models
        out['trained'] = True
        if best_w > 0:
            out['msg'] = f"模型 ({best_kind}) 樣本外表現優於市場，市場錨定權重 w = {best_w:.1f}。"
        else:
            out['msg'] = "目前所有特徵組合的樣本外表現均未能穩定勝過市場賠率 → 系統以市場公平機率為準 (w = 0)，EV 會偏負，系統將建議少注或不注，以保護本金。"
        return out
    finally:
        if run_ctx is not None:
            try:
                run_ctx.__exit__(None, None, None)
            except Exception:
                pass


def v9_predict_p(engine, feats, p_mkt):
    """單一候選盤口：市場公平機率與模型機率按 w 混合，回傳 (最終有效勝率, 模型原始機率)"""
    if not engine or not engine.get('trained') or engine.get('w', 0) <= 0 or not engine.get('models'):
        return float(p_mkt), float(p_mkt)
    cols = engine['cols']
    x = np.array([[float(feats.get(c, 0.0)) for c in cols]])
    ps = []
    for name in ('LR', 'GB'):
        mdl = engine['models'].get(name)
        if mdl is not None:
            try:
                ps.append(float(mdl.predict_proba(x)[0][1]))
            except Exception:
                pass
    if not ps:
        return float(p_mkt), float(p_mkt)
    pm = float(np.mean(ps))
    return float(v9_blend(np.array([p_mkt]), np.array([pm]), engine['w'])[0]), pm


def v9_type_gate(engine, bt, thr, allow_override=False):
    """盤口熔斷：樣本外回測在該門檻下注數 ≥ 20 且 ROI ≤ -3% → 暫停該盤口類型"""
    if allow_override or not engine or not engine.get('bt_type'):
        return True, ''
    info = engine['bt_type'].get(bt, {})
    keys = [k for k in info.keys() if k != 'all']
    if not keys:
        return True, ''
    k = min(keys, key=lambda t: abs(t - thr))
    st_ = info.get(k, {})
    if st_.get('n', 0) >= 20 and st_.get('roi', 0.0) <= -0.03:
        return False, f"回測 {st_['n']} 注 ROI {st_['roi']*100:.1f}%"
    return True, ''


def v9_engine_input(df_settled):
    cols = ['ID', 'Date', 'Match', 'Match_Date', 'Tournament_Name', 'Tournament_Category', 'Home_Team', 'Away_Team', 'Home_Rating', 'Away_Rating', 'League_Size',
            'Home_Form', 'Away_Form', 'Bet_Type', 'Odds_History', 'Home_Goal', 'Away_Goal', 'Home_Corner', 'Away_Corner',
            'Home_SoT', 'Away_SoT', 'Home_SoFF', 'Away_SoFF', 'Home_DA', 'Away_DA']
    d = df_settled.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    d = d[cols].copy()
    for c in ['ID', 'Date', 'Match', 'Match_Date', 'Tournament_Name', 'Tournament_Category', 'Home_Team', 'Away_Team', 'Home_Rating', 'Away_Rating',
              'Home_Form', 'Away_Form', 'Bet_Type', 'Odds_History']:
        d[c] = d[c].apply(lambda v: None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v))
    # V11：去除名稱多餘空格 (' 托盧卡' 與 '托盧卡' 視為同一球隊)
    for c in ['Match', 'Tournament_Name', 'Home_Team', 'Away_Team']:
        d[c] = d[c].apply(lambda v: v11_clean_name(v) if v is not None else None)
    return d.reset_index(drop=True)


@st.cache_data(max_entries=8, show_spinner="🧠 V9 引擎訓練中：重建全盤口樣本、特徵組合搜尋、前進式回測 (首次較慢，完成後自動快取)...")
def v9_train_engine_cached(df_input):
    return v9_train_engine(df_input)


@st.cache_data(max_entries=8)
def v9_history_tables_cached(df_input):
    mt = v9_build_match_table(df_input)
    return mt, v9_team_long(mt)


def v9_get_engine(df_db):
    df_settled = df_db[df_db['Status'] == 'Settled'].copy() if not df_db.empty else df_db.copy()
    if df_settled.empty:
        return {'trained': False, 'msg': '尚無已結算數據。', 'w': 0.0, 'bt_type': {}, 'models': None}, pd.DataFrame(), pd.DataFrame()
    inp = v9_engine_input(df_settled)
    engine = v9_train_engine_cached(inp)
    mt, tl = v9_history_tables_cached(inp)
    return engine, mt, tl


def v9_score_candidates(engine, candidates, ctx, settings, nsnap_map=None):
    """為每個候選盤口計算：市場公平勝率、模型勝率、EV、相對市場優勢、熔斷狀態"""
    out = []
    pair = {}
    for c in candidates:
        pair.setdefault((v9_clean_bt(c['bet_type']), round(float(c['line']), 3), c.get('history_idx')), {})[c['selection']] = c
    for c in candidates:
        c = dict(c)
        bt = v9_clean_bt(c['bet_type'])
        grp = pair.get((bt, round(float(c['line']), 3), c.get('history_idx')), {})
        opp_sel = {'Home': 'Away', 'Away': 'Home', 'Over': 'Under', 'Under': 'Over'}[c['selection']]
        opp = grp.get(opp_sel)
        o_opp = float(opp['odds']) if opp else float(c['odds'])
        d_opp = float(opp.get('odds_drift', 0.0) or 0.0) if opp else 0.0
        nsnap = (nsnap_map or {}).get((bt, round(float(c['line']), 3), c.get('history_idx')), 1)
        feats, p_mkt = v9_side_features(bt, c['selection'], float(c['line']), float(c['odds']), o_opp,
                                        float(c.get('odds_drift', 0.0) or 0.0), d_opp, nsnap, ctx)
        p_final, p_model = v9_predict_p(engine, feats, p_mkt)
        if c.get('base_prob_override') is not None:
            p_final = float(c['base_prob_override'])
        c['p_mkt'] = p_mkt
        c['model_prob'] = p_model
        c['prob'] = p_final
        c['ev'] = p_final * float(c['odds']) - 1.0
        c['ev_mkt'] = p_mkt * float(c['odds']) - 1.0
        c['edge'] = p_final - p_mkt
        ok, why = v9_type_gate(engine, bt, settings['min_ev'], settings['override_gate'])
        c['gate_ok'], c['gate_msg'] = ok, why
        c['sample_count'] = engine.get('n_matches', 0) if engine else 0
        c['model_name'] = (f"V9-{engine.get('kind')} (w={engine.get('w', 0):.1f})" if engine and engine.get('trained') else '市場公平機率')
        if not ok:
            c['status'] = f"⛔ 熔斷 ({why})"
        elif c['ev'] >= settings['min_ev']:
            c['status'] = '✅ 建議'
        elif c['ev'] > 0:
            c['status'] = '🟡 EV 未達門檻'
        else:
            c['status'] = '➖ 負 EV'
        out.append(c)
    # 兩邊相對化 (僅顯示用)
    groups = {}
    for c in out:
        groups.setdefault((v9_clean_bt(c['bet_type']), round(float(c['line']), 3), c.get('history_idx')), []).append(c)
    for g in groups.values():
        tot = sum(float(x['prob']) for x in g)
        for x in g:
            x['relative_prob'] = float(x['prob']) / tot if (len(g) == 2 and tot > 0) else float(x['prob'])
    out.sort(key=lambda x: (x['gate_ok'], x['ev']), reverse=True)
    return out


# ---------- 即場 Poisson 模型 ----------
def v9_pois(lam, kmax=20):
    lam = max(1e-6, float(lam))
    k = np.arange(kmax + 1)
    from math import lgamma
    logp = k * np.log(lam) - lam - np.array([lgamma(i + 1) for i in k])
    p = np.exp(logp)
    return p / p.sum()


def v9_expected_ret_total(side, line, cur_total, lam, odds):
    p = v9_pois(lam, 30)
    er = 0.0
    for k, pk in enumerate(p):
        tot = cur_total + k
        diff = (tot - line) if side == 'Over' else (line - tot)
        er += pk * v9_unit_return(diff, odds)
    return er


def v9_expected_ret_hd(side, line, cur_h, cur_a, lam_h, lam_a, odds):
    ph, pa = v9_pois(lam_h, 12), v9_pois(lam_a, 12)
    er = 0.0
    for i, pi in enumerate(ph):
        for j, pj in enumerate(pa):
            hg, ag = cur_h + i, cur_a + j
            diff = (hg + line - ag) if side == 'Home' else (ag - line - hg)
            er += pi * pj * v9_unit_return(diff, odds)
    return er


def v9_solve_total_lambda(line, p_fair_over, lo=0.2, hi=8.0, cur=0):
    """求出使大盤在公平賠率下期望回報為 0 的 Poisson λ (市場隱含總入球/角球)"""
    o_f = 1.0 / max(0.02, min(0.98, p_fair_over))
    best, best_v = None, 1e9
    for lam in np.arange(lo, hi, 0.05):
        v = abs(v9_expected_ret_total('Over', line, cur, lam, o_f))
        if v < best_v:
            best, best_v = float(lam), v
    return best


def v9_solve_supremacy(line, p_fair_home, lam_t):
    o_f = 1.0 / max(0.02, min(0.98, p_fair_home))
    best, best_v = 0.0, 1e9
    for sup in np.arange(-3.0, 3.01, 0.05):
        lh, la = max(0.05, (lam_t + sup) / 2.0), max(0.05, (lam_t - sup) / 2.0)
        v = abs(v9_expected_ret_hd('Home', line, 0, 0, lh, la, o_f))
        if v < best_v:
            best, best_v = float(sup), v
    return best


def v9_main_market(odds_history, btype, ref_date=None):
    """取得某盤口類型最平衡 (主盤) 的最新賠率：回傳 (line, 上賠率, 下賠率) 或 None"""
    mk = v9_group_markets(odds_history, ref_date)
    best, best_gap = None, 1e9
    for (bt, ln), snaps in mk.items():
        if bt != btype:
            continue
        last = snaps[-1]
        gap = abs(1.0 / last[2] - 1.0 / last[3])
        if gap < best_gap:
            best, best_gap = (ln, last[2], last[3]), gap
    return best


def v9_prematch_lambdas(odds_history, ctx):
    """由賽前主盤賠率反推市場隱含 λ (主入球、客入球、總角球)；缺盤口時以數據庫攻防數據估算"""
    hp, ap, lg = ctx['hp'], ctx['ap'], ctx['league']
    lam_t = (hp['gf'] + ap['ga']) / 2.0 + (ap['gf'] + hp['ga']) / 2.0
    sup = (hp['gf'] + ap['ga']) / 2.0 - (ap['gf'] + hp['ga']) / 2.0 + lg['hgd'] * 0.5
    lam_c = (hp['cf'] + ap['ca']) / 2.0 + (ap['cf'] + hp['ca']) / 2.0
    src = {'goals': '數據庫攻防數據', 'hd': '數據庫攻防數據', 'corners': '數據庫攻防數據'}
    ou = v9_main_market(odds_history, '入球大小')
    if ou:
        p_over, _ = v9_devig(ou[1], ou[2])
        lam_t = v9_solve_total_lambda(ou[0], p_over, 0.3, 7.0)
        src['goals'] = f"入球大小主盤 {ou[0]:g}"
    hd = v9_main_market(odds_history, '讓球')
    if hd:
        p_home, _ = v9_devig(hd[1], hd[2])
        sup = v9_solve_supremacy(hd[0], p_home, lam_t)
        src['hd'] = f"讓球主盤 {hd[0]:g}"
    cr = v9_main_market(odds_history, '角球大小')
    if cr:
        p_over_c, _ = v9_devig(cr[1], cr[2])
        lam_c = v9_solve_total_lambda(cr[0], p_over_c, 2.0, 20.0)
        src['corners'] = f"角球大小主盤 {cr[0]:g}"
    lam_h = max(0.05, (lam_t + sup) / 2.0)
    lam_a = max(0.05, (lam_t - sup) / 2.0)
    return lam_h, lam_a, lam_c, src


def v9_inplay_lambdas(lam_h, lam_a, lam_c, minute, h_red, a_red, h_shots, a_shots, h_da, a_da):
    """即場剩餘時間 λ：按剩餘時間比例 × 攻勢強度 (射門/危險進攻相對正常節奏) × 紅牌調整"""
    rem = max(0.0, 90.0 - float(minute)) / 90.0
    m = max(1.0, float(minute))
    exp_shots_pm = V9_DEFAULTS['shots'] / 90.0
    exp_da_pm = V9_DEFAULTS['da'] / 90.0
    if minute >= 10:
        ih = float(np.clip(0.7 + 0.3 * (h_shots / m) / exp_shots_pm, 0.6, 1.6))
        ia = float(np.clip(0.7 + 0.3 * (a_shots / m) / exp_shots_pm, 0.6, 1.6))
        ic = float(np.clip(0.7 + 0.3 * ((h_da + a_da) / m) / (2 * exp_da_pm), 0.6, 1.6)) if (h_da + a_da) > 0 else 1.0
    else:
        ih = ia = ic = 1.0
    rh = (0.70 ** h_red) * (1.20 ** a_red)
    ra = (0.70 ** a_red) * (1.20 ** h_red)
    return lam_h * rem * ih * rh, lam_a * rem * ia * ra, lam_c * rem * ic, (ih, ia, ic)


def v9_calc_stake(cand, bankroll, max_stake, settings):
    """分數 Kelly：注碼 = 本金 × Kelly 分數 × (EV / (賠率−1))，上限為本金 max_pct 及系統單注上限，四捨五入至 $10"""
    if not cand.get('gate_ok', True) or bankroll <= 0:
        return 0.0, 0.0
    ev = float(cand.get('ev', 0.0))
    o = float(cand.get('odds', 1.9))
    if ev < settings['min_ev'] or o <= 1.0:
        return 0.0, 0.0
    f = max(0.0, ev / (o - 1.0))
    raw = bankroll * settings['kelly'] * f
    cap = min(float(max_stake) if max_stake > 0 else raw, bankroll * settings['max_pct'])
    stake = float(round(min(raw, cap) / 10.0) * 10.0)
    if stake < 10.0:
        stake = 0.0
    if settings.get('hd_min200') and '讓球' in str(cand.get('bet_type', '')) and 0 < stake < 200:
        stake = 200.0 if (ev >= 0.05 and bankroll * settings['max_pct'] >= 200) else 0.0
    return stake, raw


def v9_dimension_stats(df_subset, dim_name):
    """三維度歷史統計 (只作參考顯示)：樣本數、系統實注 ROI、勝率"""
    n = len(df_subset)
    out = {'dim': dim_name, 'n': n, 'valid': n >= 15, 'msg': '運算成功' if n >= 15 else f"樣本數不足 ({n} < 15張)",
           'roi': 0.0, 'acc': 0.0, 'n_bet': 0}
    if n == 0:
        return out
    stake = pd.to_numeric(df_subset['System_Stake'], errors='coerce').fillna(0.0)
    prof = pd.to_numeric(df_subset['System_Profit'], errors='coerce').fillna(0.0)
    up = pd.to_numeric(df_subset['Unit_Profit'], errors='coerce')
    out['n_bet'] = int((stake > 0).sum())
    out['roi'] = float(prof[stake > 0].sum() / stake[stake > 0].sum()) if stake.sum() > 0 else 0.0
    out['acc'] = float((up > 0).mean()) if up.notna().any() else 0.0
    return out


# ==========================================
# 3d. V10 看盤引擎 (Kanpan Engine：讀懂莊家盤口與水位變化)
# ==========================================
# 把「睇盤」七大要素量化為可驗證的訊號，以數據庫已結算比賽的完整賠率紀錄學習：
#  1) 實力評估      ：聯賽排名 + 近5場戰績 (公眾認知實力)、數據庫兩隊過往攻防數據
#  2) 初盤 vs 即時盤 ：以 Poisson 模型把「盤口線 + 水位」換算為同一單位 (隱含讓球數 / 隱含總入球)，計算由初盤到臨場的真實移動
#  3) 水位高低      ：收盤低水 / 中水 / 高水、同盤口降水 / 升水
#  4) 冷熱分析      ：市場隱含讓球數 與 公眾認知實力 的差 (盤口低於實力 = 冷；高於實力 = 熱)
#  5) 信賠不信變    ：所有判斷以收盤 (最後停留) 的盤口與水位為準，並識別臨場水位反覆
#  6) 信深不信淺    ：上盤 / 下盤、深盤 (≥1球)、淺盤
#  7) 反向思維      ：自動測試兩兩組合 (例如「實力佔優 + 盤口不變 (無防範)」、「熱門 + 盤口移離」)
# 每條規則以「只用過去日期」的前進式回測評分，回報向抽水基準收縮 (避免小樣本假象)，只有樣本外仍為正 EV 的規則才會啟用。

V10_MIN_N = 40            # 規則啟用最少歷史樣本
V10_SHRINK_K = 60.0       # 規則回報收縮強度 (等同 60 注抽水基準的先驗)
V10_PRIOR_ROI = -0.04     # 先驗回報 (約等於隨機投注扣除抽水)
V10_BT_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
V10_TYPES = ('讓球', '入球大小', '角球大小')

V10_BASE_RULE_DESC = {
    '實力近況佔優': ('1 實力評估', '排名 + 近5場戰績明顯較好 (讓球) / 兩隊進攻力偏強 (大小盤按方向)'),
    '實力近況較弱': ('1 實力評估', '排名 + 近5場戰績明顯較差'),
    '攻防數據看好': ('1 實力評估', '數據庫兩隊過往攻防數據推算結果優於盤口線'),
    '攻防數據不看好': ('1 實力評估', '數據庫兩隊過往攻防數據推算結果差於盤口線'),
    '盤水移向此方': ('2 初盤vs即時盤', '初盤→收盤的隱含讓球數/總入球向此方向移動 (莊家調整令此方更受捧)'),
    '盤水移離此方': ('2 初盤vs即時盤', '初盤→收盤的隱含讓球數/總入球離開此方向'),
    '盤水不變': ('2 初盤vs即時盤', '初盤至收盤幾乎沒有變化'),
    '升降盤移向此方': ('2 初盤vs即時盤', '主盤口線本身改變 (升盤/降盤) 並向此方向'),
    '升降盤移離此方': ('2 初盤vs即時盤', '主盤口線本身改變並離開此方向'),
    '收盤低水': ('3 水位', '收盤賠率 < 1.85 (低水)'),
    '收盤中水': ('3 水位', '收盤賠率 1.85 ~ 1.98'),
    '收盤高水': ('3 水位', '收盤賠率 > 1.98 (高水)'),
    '同盤降水': ('3 水位', '主盤口線不變而此方賠率下降'),
    '同盤升水': ('3 水位', '主盤口線不變而此方賠率上升'),
    '盤口低於實力(冷)': ('4 冷熱分析', '市場給此方的隱含讓球數低於其排名/近況應得水平'),
    '盤口高於實力(熱)': ('4 冷熱分析', '市場給此方的隱含讓球數高於其排名/近況應得水平'),
    '臨場水位反覆': ('5 信賠不信變', '水位曾明顯波動但收盤回到接近初盤位置'),
    '上盤(讓球方)': ('6 信深不信淺', '讓球盤中讓球的一方'),
    '下盤(受讓方)': ('6 信深不信淺', '讓球盤中受讓的一方'),
    '深盤上盤(讓≥1球)': ('6 信深不信淺', '讓球 1 球或以上的上盤'),
    '深盤下盤(受讓≥1球)': ('6 信深不信淺', '受讓 1 球或以上的下盤'),
    '淺盤(≤0.25球)': ('6 信深不信淺', '平手或半球以下的淺盤'),
    '主隊/大': ('方向', '讓球買主隊 / 大小盤買大'),
    '客隊/小': ('方向', '讓球買客隊 / 大小盤買小'),
}


def v10_detect_fmt(odds_history):
    """偵測一份賠率紀錄的日期格式：'dm' (日-月，如 18-09) / 'md' (月-日，如 10-22) / None (無法判斷)"""
    a = b = 0
    for h in odds_history or []:
        m = re.match(r'\s*(\d{1,2})[-/.](\d{1,2})(?![-/.]\d)', str(h.get('record_time', '')) if isinstance(h, dict) else '')
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            if x > 12 >= y:
                a += 1
            elif y > 12 >= x:
                b += 1
    if a and not b:
        return 'dm'
    if b and not a:
        return 'md'
    return None


def v10_parse_time(s, fmt, ref):
    """解析單一紀錄時間；年份及日/月次序不明時，選擇最接近比賽日期 (且不晚於比賽日期 3 日) 的解讀"""
    s = str(s or '').strip()
    if not s:
        return None
    for f in ("%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M"):
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    m = re.match(r'(\d{1,2})[-/.](\d{1,2})\s*(\d{1,2}):(\d{2})', s)
    if not m:
        return None
    x, y, hh, mm = map(int, m.groups())
    if ref is None:
        ref = get_hkt_now().replace(tzinfo=None)
    cands = []
    for f in ([fmt] if fmt else ['md', 'dm']):
        mo, d = (x, y) if f == 'md' else (y, x)
        for yr in (ref.year - 1, ref.year, ref.year + 1):
            try:
                cands.append(datetime(yr, mo, d, hh, mm))
            except ValueError:
                pass
    if not cands:
        return None
    return min(cands, key=lambda t: abs((t - ref).total_seconds()) + (1e10 if (t - ref).days > 3 else 0))


def v10_snapshots(odds_history, ref_date=None):
    """把賠率紀錄轉為排序後的快照列表 (無時間的紀錄依儲存位置排序)"""
    out = []
    if not isinstance(odds_history, list):
        return out
    ref = ref_date
    if ref is not None and hasattr(ref, 'to_pydatetime'):
        ref = ref.to_pydatetime()
    if ref is not None:
        ref = ref.replace(tzinfo=None, hour=23, minute=59)
    fmt = v10_detect_fmt(odds_history)
    for pos, h in enumerate(odds_history):
        try:
            bt = v9_clean_bt(h.get('type', ''))
            if bt not in V10_TYPES:
                continue
            ln = round(float(h.get('line', 0.0)), 3)
            up, lw = float(h.get('upper')), float(h.get('lower'))
            if up <= 1.0 or lw <= 1.0 or not v11_valid_pair(up, lw):
                continue
        except (TypeError, ValueError, AttributeError):
            continue
        ts = v10_parse_time(h.get('record_time', ''), fmt, ref)
        out.append({'t': ts if ts is not None else datetime.min + timedelta(minutes=pos), 'has_t': ts is not None,
                    'pos': pos, 'type': bt, 'line': ln, 'up': up, 'lw': lw})
    out.sort(key=lambda x: (x['t'], x['pos']))
    return out


def v10_timeline(snaps, btype):
    """同一時間點可能有多條盤口線：每個時間點取最平衡 (主盤) 的一條，回傳主盤時間序列"""
    by_t = {}
    for s in snaps:
        if s['type'] == btype:
            by_t.setdefault(s['t'], []).append(s)
    tl = []
    for t in sorted(by_t):
        g = by_t[t]
        best = min(g, key=lambda s: (abs(1.0 / s['up'] - 1.0 / s['lw']), s['pos']))
        tl.append(best)
    return tl


# ---- 向量化 Poisson：把 (盤口線, 公平機率) 換算為隱含讓球數 / 總入球 ----
_V10_SUP_GRID = np.round(np.arange(-3.5, 3.5001, 0.05), 4) if np is not None else None
_V10_GD_CACHE = {}


def _v10_pois_vec(lam, kmax):
    from math import lgamma
    lam = max(1e-6, float(lam))
    k = np.arange(kmax + 1)
    p = np.exp(k * np.log(lam) - lam - np.array([lgamma(i + 1) for i in k]))
    return p / p.sum()


def _v10_gd_matrix(lam_t):
    key = round(float(lam_t), 1)
    if key in _V10_GD_CACHE:
        return _V10_GD_CACHE[key]
    kmax = 12
    rows = []
    for sup in _V10_SUP_GRID:
        lh, la = max(0.05, (key + sup) / 2.0), max(0.05, (key - sup) / 2.0)
        P = np.outer(_v10_pois_vec(lh, kmax), _v10_pois_vec(la, kmax))
        gd = np.zeros(2 * kmax + 1)
        for i in range(kmax + 1):
            for j in range(kmax + 1):
                gd[i - j + kmax] += P[i, j]
        rows.append(gd)
    M = np.array(rows)
    _V10_GD_CACHE[key] = M
    return M


def _v10_ret_vec(diff, o):
    d = np.round(diff, 2)
    r = np.full(d.shape, -1.0)
    r[d >= 0.5] = o - 1.0
    r[np.isclose(d, 0.25)] = (o - 1.0) / 2.0
    r[np.isclose(d, 0.0)] = 0.0
    r[np.isclose(d, -0.25)] = -0.5
    return r


def v10_implied_sup(line, p_home, lam_t=2.6):
    """主隊隱含淨勝球 (正數 = 主隊較強)"""
    M = _v10_gd_matrix(max(0.6, min(6.0, lam_t)))
    of = 1.0 / min(0.97, max(0.03, p_home))
    gd = np.arange(-12, 13)
    er = M @ _v10_ret_vec(gd + float(line), of)
    return float(_V10_SUP_GRID[int(np.argmin(np.abs(er)))])


_V10_TOT_CACHE = {}


def v10_implied_total(line, p_over, lo=0.3, hi=8.0, step=0.05):
    key = (lo, hi, step)
    if key not in _V10_TOT_CACHE:
        lams = np.arange(lo, hi, step)
        _V10_TOT_CACHE[key] = (lams, np.array([_v10_pois_vec(l, 40) for l in lams]))
    lams, M = _V10_TOT_CACHE[key]
    of = 1.0 / min(0.97, max(0.03, p_over))
    er = M @ _v10_ret_vec(np.arange(41) - float(line), of)
    return float(lams[int(np.argmin(np.abs(er)))])


def v10_implied(btype, snap, lam_t):
    p_hi, _ = v9_devig(snap['up'], snap['lw'])
    if btype == '讓球':
        return v10_implied_sup(snap['line'], p_hi, lam_t)
    if btype == '入球大小':
        return v10_implied_total(snap['line'], p_hi, 0.3, 8.0, 0.05)
    return v10_implied_total(snap['line'], p_hi, 2.0, 22.0, 0.1)


def v10_market_rows(snaps, ctx, hs, as_, hfp, afp, unk=False):
    """為一場比賽計算每種盤口 (收盤主盤) 兩個方向的看盤特徵；回傳列表 (不含賽果)"""
    rows = []
    tls = {bt: v10_timeline(snaps, bt) for bt in V10_TYPES}
    lam_t = 2.6
    if tls['入球大小']:
        try:
            lam_t = v10_implied('入球大小', tls['入球大小'][-1], 2.6)
        except Exception:
            lam_t = 2.6
    for bt, tl in tls.items():
        if not tl:
            continue
        o, c = tl[0], tl[-1]
        try:
            io, ic = v10_implied(bt, o, lam_t), v10_implied(bt, c, lam_t)
        except Exception:
            io = ic = 0.0
        p_c_hi, _ = v9_devig(c['up'], c['lw'])
        p_o_hi, _ = v9_devig(o['up'], o['lw'])
        same_line = abs(c['line'] - o['line']) < 1e-9
        # 同主盤口線時的水位序列，用於識別臨場水位反覆
        p_series = [v9_devig(x['up'], x['lw'])[0] for x in tl if abs(x['line'] - c['line']) < 1e-9]
        swing = (max(p_series) - min(p_series)) if len(p_series) >= 3 else 0.0
        for side in (['Home', 'Away'] if bt == '讓球' else ['Over', 'Under']):
            s = 1.0 if side in ('Home', 'Over') else -1.0
            hi = s > 0
            odds = c['up'] if hi else c['lw']
            opp = c['lw'] if hi else c['up']
            p_c = p_c_hi if hi else 1 - p_c_hi
            p_o = p_o_hi if hi else 1 - p_o_hi
            if bt == '讓球':
                mv_line = s * -(c['line'] - o['line'])
                pub = s * ((hs - as_) / 4.0 + (hfp - afp) / 15.0)
                giver = (side == 'Home' and c['line'] < 0) or (side == 'Away' and c['line'] > 0)
            else:
                mv_line = s * (c['line'] - o['line'])
                pub = s * (((hs + as_) / 2.0 - 3.0) / 2.0 + ((hfp + afp) / 2.0 - 7.0) / 7.0)
                giver = False
            try:
                f9, _ = v9_side_features(bt, side, c['line'], odds, opp, 0.0, 0.0, 1, ctx)
                team_edge = float(f9.get('f_team_edge', 0.0))
            except Exception:
                team_edge = 0.0
            rows.append({
                'bet_type': bt, 'side': side, 's': s, 'line': c['line'], 'line_o': o['line'], 'odds': odds, 'opp': opp,
                'odds_o': o['up'] if hi else o['lw'], 'p_c': p_c, 'p_o': p_o,
                'imp_o': io, 'imp_c': ic, 'mv_imp': s * (ic - io), 'mv_line': mv_line,
                'mv_p': (p_c - p_o) if same_line else 0.0, 'swing': swing, 'nsnap': len(tl),
                'pub': pub, 'team_edge': team_edge, 'giver': bool(giver), 'deep': bool(bt == '讓球' and abs(c['line']) >= 1.0),
                'shallow': bool(bt == '讓球' and abs(c['line']) <= 0.25),
                'hs': hs, 'as': as_, 'hfp': hfp, 'afp': afp, 'close_pos': c['pos'], 'unk': bool(unk),
            })
    return rows


def v10_apply_resid(D, coef):
    """冷熱分析：市場隱含值 與『排名/近況應得水平』(以數據庫擬合) 的差，按方向定正負"""
    D = D.copy()
    D['resid'] = 0.0
    for bt in V10_TYPES:
        idx = D['bet_type'] == bt
        if not idx.any() or bt not in coef:
            continue
        b = np.array(coef[bt])
        x = D.loc[idx]
        if bt == '讓球':
            fit = b[0] + b[1] * (x['hs'] - x['as']) + b[2] * (x['hfp'] - x['afp'])
        else:
            fit = b[0] + b[1] * (x['hs'] + x['as']) + b[2] * (x['hfp'] + x['afp'])
        scale = 1.0 if bt != '角球大小' else 0.5
        D.loc[idx, 'resid'] = x['s'] * (x['imp_c'] - fit) * scale
    if 'unk' in D.columns:
        # V11：盃賽 / 洲際賽 / 無排名 → 不知道兩隊實力，不能判斷盤口『冷熱』
        D.loc[D['unk'].astype(bool), 'resid'] = 0.0
    return D


def v10_fit_resid(D):
    coef = {}
    for bt in V10_TYPES:
        m = D[(D['bet_type'] == bt) & (D['s'] > 0)]
        if 'unk' in m.columns:
            m = m[~m['unk'].astype(bool)]
        if len(m) < 20:
            coef[bt] = [float(m['imp_c'].mean()) if len(m) else 0.0, 0.0, 0.0]
            continue
        if bt == '讓球':
            X = np.c_[np.ones(len(m)), m['hs'] - m['as'], m['hfp'] - m['afp']]
        else:
            X = np.c_[np.ones(len(m)), m['hs'] + m['as'], m['hfp'] + m['afp']]
        coef[bt] = [float(v) for v in np.linalg.lstsq(X, m['imp_c'].values, rcond=None)[0]]
    return coef


def v10_base_masks(D):
    hd = D['bet_type'] == '讓球'
    mv, o = D['mv_imp'], D['odds']
    return {
        '實力近況佔優': D['pub'] > 0.25,
        '實力近況較弱': D['pub'] < -0.25,
        '攻防數據看好': D['team_edge'] > 0.3,
        '攻防數據不看好': D['team_edge'] < -0.3,
        '盤水移向此方': mv > 0.04,
        '盤水移離此方': mv < -0.04,
        '盤水不變': mv.abs() <= 0.04,
        '升降盤移向此方': D['mv_line'] > 0,
        '升降盤移離此方': D['mv_line'] < 0,
        '收盤低水': o < 1.85,
        '收盤中水': (o >= 1.85) & (o <= 1.98),
        '收盤高水': o > 1.98,
        '同盤降水': D['mv_p'] > 0.01,
        '同盤升水': D['mv_p'] < -0.01,
        '盤口低於實力(冷)': D['resid'] < -0.15,
        '盤口高於實力(熱)': D['resid'] > 0.15,
        '臨場水位反覆': (D['swing'] > 0.03) & (D['mv_p'].abs() <= 0.01),
        '上盤(讓球方)': hd & D['giver'].astype(bool),
        '下盤(受讓方)': hd & ~D['giver'].astype(bool) & (D['line'].abs() > 1e-9),
        '深盤上盤(讓≥1球)': hd & D['giver'].astype(bool) & D['deep'].astype(bool),
        '深盤下盤(受讓≥1球)': hd & ~D['giver'].astype(bool) & D['deep'].astype(bool),
        '淺盤(≤0.25球)': hd & D['shallow'].astype(bool),
        '主隊/大': D['s'] > 0,
        '客隊/小': D['s'] < 0,
    }


def v10_rule_matrix(D, rule_names=None):
    """回傳 (規則名稱列表, 布林矩陣 rows×rules)：單一規則 + 兩兩組合"""
    base = v10_base_masks(D)
    keys = list(base.keys())
    names, cols = [], []
    for k in keys:
        names.append(k)
        cols.append(base[k].values.astype(bool))
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            names.append(f"{keys[i]} + {keys[j]}")
            cols.append((base[keys[i]] & base[keys[j]]).values.astype(bool))
    M = np.column_stack(cols) if cols else np.zeros((len(D), 0), dtype=bool)
    if rule_names is not None:
        pos = {n: i for i, n in enumerate(names)}
        sel = [pos[n] for n in rule_names if n in pos]
        return [names[i] for i in sel], M[:, sel]
    return names, M


def v10_rule_stats(D, M, names, mask_rows=None, minn=V10_MIN_N, k=V10_SHRINK_K, prior=V10_PRIOR_ROI):
    """每條規則 × 每種盤口類型：樣本數、ROI、收縮後 ROI (作為該規則的期望值估計)"""
    ret = D['ret'].values
    out = {}
    for bt in V10_TYPES:
        tm = (D['bet_type'] == bt).values
        if mask_rows is not None:
            tm = tm & mask_rows
        Mt = M & tm[:, None]
        n = Mt.sum(axis=0)
        s = (Mt * ret[:, None]).sum(axis=0)
        w = (Mt * (ret[:, None] > 0)).sum(axis=0)
        shr = (s + k * prior) / (n + k)
        out[bt] = {'n': n, 'sum': s, 'win': w, 'shr': shr}
    return out


def v10_pick(D, M, stats, rows_mask, thr, minn=V10_MIN_N):
    """在 rows_mask 範圍內，為每行找出『已啟用且收縮 ROI 最高』的規則；回傳 (best_shr, best_rule_idx)"""
    best = np.full(len(D), -9.0)
    best_i = np.full(len(D), -1)
    for bt in V10_TYPES:
        st_ = stats[bt]
        active = (st_['n'] >= minn) & (st_['shr'] >= thr)
        if not active.any():
            continue
        tm = ((D['bet_type'] == bt).values & rows_mask)
        if not tm.any():
            continue
        vals = np.where(M[tm][:, active], st_['shr'][active][None, :], -9.0)
        idx_active = np.where(active)[0]
        am = vals.argmax(axis=1)
        bv = vals[np.arange(len(am)), am]
        rows = np.where(tm)[0]
        best[rows] = bv
        best_i[rows] = np.where(bv > -9.0, idx_active[am], -1)
    return best, best_i


def v10_build_dataset(mt):
    """由比賽表 (含完整賽前賠率紀錄) 建立看盤訓練集：每場、每盤口類型、兩個方向，以收盤主盤計算真實回報"""
    if mt is None or mt.empty:
        return pd.DataFrame()
    tl = v9_team_long(mt)
    rows = []
    for _, m in mt.iterrows():
        date = m['date']
        snaps = v10_snapshots(m['oh'], date)
        if not snaps:
            continue
        ctx = v9_match_context(mt, tl, m['home'], m['away'], m['tourn'], date, m['hs'], m['as'], m['hfp'], m['afp'])
        for r in v10_market_rows(snaps, ctx, m['hs'], m['as'], m['hfp'], m['afp'], bool(m.get('unk', False))):
            bt = r['bet_type']
            if bt == '角球大小' and (np.isnan(m['hc']) or np.isnan(m['ac'])):
                continue
            diff = v9_diff(bt, r['side'], r['line'], m['hg'], m['ag'],
                           0 if np.isnan(m['hc']) else m['hc'], 0 if np.isnan(m['ac']) else m['ac'])
            r['ret'] = v9_unit_return(diff, r['odds'])
            r['key'] = m['key']
            r['date'] = date
            r['match'] = m['match']
            r['tourn'] = m['tourn']
            rows.append(r)
    return pd.DataFrame(rows)


def v10_ml_compare(D, folds, thr=0.02):
    """對照組：以相同看盤特徵訓練 LR / GB (前進式)，比較盈利能力"""
    out = []
    if not HAS_AI_MODULES or D.empty:
        return out
    X = D.copy()
    X['e'] = (X['ret'] + 1.0) / X['odds']
    X['ml'] = np.log(np.clip(X['p_c'], 0.02, 0.98) / (1 - np.clip(X['p_c'], 0.02, 0.98)))
    for bt, nm in (('讓球', 'hd'), ('入球大小', 'ou'), ('角球大小', 'cr')):
        X['is_' + nm] = (X['bet_type'] == bt).astype(float)
        for c in ('s', 'mv_imp', 'pub', 'resid', 'mv_p'):
            X[f'{c}_{nm}'] = X[c] * X['is_' + nm]
    cols = ['ml', 'odds', 'mv_imp', 'mv_line', 'mv_p', 'pub', 'resid', 'team_edge', 'swing', 'is_hd', 'is_ou'] + \
           [f'{c}_{nm}' for nm in ('hd', 'ou', 'cr') for c in ('s', 'mv_imp', 'pub', 'resid', 'mv_p')]
    X['giver_f'] = X['giver'].astype(float)
    cols.append('giver_f')
    for kind, label in (('LR', '機器學習 LR (看盤特徵)'), ('GB', '機器學習 GB (看盤特徵)')):
        P = pd.Series(np.nan, index=X.index)
        for d0, d1 in folds:
            tr = X['date'] < d0
            te = (X['date'] >= d0) & (X['date'] <= d1)
            if tr.sum() < 100 or te.sum() == 0:
                continue
            try:
                mdl = v9_soft_fit(kind, X.loc[tr, cols].values, X.loc[tr, 'e'].values)
                P[te] = mdl.predict_proba(X.loc[te, cols].values)[:, 1]
            except Exception:
                continue
        o = X[P.notna()].copy()
        if o.empty:
            continue
        o['ev'] = P[P.notna()] * o['odds'] - 1.0
        b = o[o['ev'] >= thr].sort_values('ev', ascending=False).drop_duplicates(['key', 'bet_type'])
        out.append({'策略 / 模型': label, '回測注數': len(b), 'ROI': f"{b['ret'].mean()*100:.1f}%" if len(b) else '-',
                    '單位盈虧 (U)': round(float(b['ret'].sum()), 2) if len(b) else 0.0})
    return out


def v10_train_engine(df_settled):
    out = {'trained': False, 'msg': '', 'n_matches': 0, 'n_rows': 0, 'coef': {}, 'rules': [], 'rule_names': [],
           'stats': {}, 'bt': {}, 'bt_type': {}, 'compare': [], 'params': {'minn': V10_MIN_N, 'k': V10_SHRINK_K, 'prior': V10_PRIOR_ROI}}
    mt = v9_build_match_table(df_settled)
    out['n_matches'] = 0 if mt.empty else len(mt)
    if mt.empty or len(mt) < V9_MIN_MATCHES:
        out['msg'] = f"已結算比賽不足 {V9_MIN_MATCHES} 場，看盤引擎暫未啟用。"
        return out
    D = v10_build_dataset(mt)
    if D.empty or len(D) < 200:
        out['msg'] = '可用的賠率紀錄不足，看盤引擎暫未啟用。'
        return out
    D = D.reset_index(drop=True)
    coef = v10_fit_resid(D)
    D = v10_apply_resid(D, coef)
    names, M = v10_rule_matrix(D)
    out['n_rows'] = len(D)
    out['coef'] = coef
    # --- 前進式回測：每一折只用之前日期計算規則表現並選擇規則，再預測之後日期 ---
    folds = v9_folds(D['date'], n_folds=6, init_frac=0.4)
    bt_res = {}
    bt_type = {bt: {} for bt in V10_TYPES}
    for thr in V10_BT_THRESHOLDS:
        picks = []
        for d0, d1 in folds:
            tr = (D['date'] < d0).values
            te = ((D['date'] >= d0) & (D['date'] <= d1)).values
            if tr.sum() < 150 or te.sum() == 0:
                continue
            stt = v10_rule_stats(D, M, names, mask_rows=tr)
            best, best_i = v10_pick(D, M, stt, te, thr)
            sel = np.where(te & (best_i >= 0))[0]
            if len(sel) == 0:
                continue
            P = D.loc[sel, ['key', 'bet_type', 'side', 'ret', 'odds']].copy()
            P['shr'] = best[sel]
            P = P.sort_values('shr', ascending=False).drop_duplicates(['key', 'bet_type'])
            picks.append(P)
        Pk = pd.concat(picks) if picks else pd.DataFrame(columns=['key', 'bet_type', 'side', 'ret', 'odds', 'shr'])
        n = len(Pk)
        bt_res[thr] = {'n': n, 'roi': float(Pk['ret'].mean()) if n else 0.0, 'units': float(Pk['ret'].sum()) if n else 0.0,
                       'hit': float((Pk['ret'] > 0).mean()) if n else 0.0}
        for bt in V10_TYPES:
            sub = Pk[Pk['bet_type'] == bt]
            bt_type[bt][thr] = {'n': int(len(sub)), 'roi': float(sub['ret'].mean()) if len(sub) else 0.0,
                                'units': float(sub['ret'].sum()) if len(sub) else 0.0}
    test_mask = (D['date'] >= folds[0][0]).values if folds else np.zeros(len(D), dtype=bool)
    for bt in V10_TYPES:
        sub = D[test_mask & (D['bet_type'] == bt).values]
        bt_type[bt]['all'] = {'n': int(len(sub)), 'roi': float(sub['ret'].mean()) if len(sub) else 0.0}
    out['bt'] = bt_res
    out['bt_type'] = bt_type
    # --- 以全部數據計算最終規則表現 (用於即將比賽的預測) ---
    stt = v10_rule_stats(D, M, names)
    rules = []
    for bt in V10_TYPES:
        s_ = stt[bt]
        for i, nm in enumerate(names):
            n = int(s_['n'][i])
            if n < 20:
                continue
            se = float(D.loc[M[:, i] & (D['bet_type'] == bt).values, 'ret'].std()) / np.sqrt(n) if n > 1 else 1.0
            roi = float(s_['sum'][i] / n)
            rules.append({'rule': nm, 'bet_type': bt, 'n': n, 'roi': roi, 'shr': float(s_['shr'][i]),
                          'hit': float(s_['win'][i] / n), 'units': float(s_['sum'][i]), 'z': roi / se if se > 0 else 0.0})
    rules.sort(key=lambda r: -r['shr'])
    out['rules'] = rules
    out['rule_names'] = names
    # --- 對照組 ---
    compare = []
    if test_mask.any():
        sub = D[test_mask]
        compare.append({'策略 / 模型': '隨機買任一邊 (市場基準)', '回測注數': int(len(sub)), 'ROI': f"{sub['ret'].mean()*100:.1f}%",
                        '單位盈虧 (U)': round(float(sub['ret'].sum()), 2)})
    for thr in (0.0, 0.02, 0.03):
        r = bt_res.get(thr, {})
        compare.append({'策略 / 模型': f'V10 看盤規則引擎 (EV 門檻 {thr*100:.0f}%)', '回測注數': r.get('n', 0),
                        'ROI': f"{r.get('roi', 0)*100:.1f}%" if r.get('n') else '-', '單位盈虧 (U)': round(r.get('units', 0.0), 2)})
    try:
        compare.extend(v10_ml_compare(D, folds, 0.02))
    except Exception:
        pass
    out['compare'] = compare
    out['trained'] = True
    r2 = bt_res.get(0.03, {})
    out['msg'] = (f"看盤引擎已由 {out['n_matches']} 場比賽、{len(D)} 個收盤主盤樣本學習 {len(names)} 條規則 (24 條單一規則 + 兩兩組合)。"
                  f"前進式回測 (EV 門檻 3%)：{r2.get('n', 0)} 注，ROI {r2.get('roi', 0)*100:.1f}%。")
    return out


@st.cache_data(max_entries=8, show_spinner="📈 V10 看盤引擎學習中：重建初盤→收盤盤口水位變化、測試看盤規則組合、前進式回測...")
def v10_train_engine_cached(df_input):
    return v10_train_engine(df_input)


def v10_get_engine(df_db):
    df_settled = df_db[df_db['Status'] == 'Settled'].copy() if not df_db.empty else df_db.copy()
    if df_settled.empty:
        return {'trained': False, 'msg': '尚無已結算數據。', 'rules': [], 'bt': {}, 'bt_type': {}, 'compare': [], 'coef': {}}
    return v10_train_engine_cached(v9_engine_input(df_settled))


def v10_type_gate(engine, bt, thr, override=False):
    """看盤盤口熔斷：該盤口類型前進式回測 (相近門檻) 注數 ≥ 20 且 ROI ≤ -3% → 暫停"""
    if override or not engine or not engine.get('bt_type'):
        return True, ''
    info = engine['bt_type'].get(bt, {})
    keys = [k for k in info if k != 'all']
    if not keys:
        return True, ''
    k = min(keys, key=lambda t: abs(t - thr))
    s_ = info[k]
    if s_.get('n', 0) >= 20 and s_.get('roi', 0.0) <= -0.03:
        return False, f"看盤回測 {s_['n']} 注 ROI {s_['roi']*100:.1f}%"
    return True, ''


def v10_predict(engine, odds_history, ctx, hs, as_, hfp, afp, ref_date, thr, unk=False):
    """即將比賽：以最新 (收盤) 主盤計算看盤特徵，找出觸發且已啟用的最佳規則
    回傳 {(bet_type, line, side): {...}}"""
    res = {}
    if not engine or not engine.get('trained'):
        return res
    snaps = v10_snapshots(odds_history, ref_date)
    if not snaps:
        return res
    rows = v10_market_rows(snaps, ctx, hs, as_, hfp, afp, unk)
    if not rows:
        return res
    D = v10_apply_resid(pd.DataFrame(rows), engine.get('coef', {}))
    names, M = v10_rule_matrix(D)
    lookup = {(r['rule'], r['bet_type']): r for r in engine.get('rules', [])}
    minn = engine.get('params', {}).get('minn', V10_MIN_N)
    for i in range(len(D)):
        row = D.iloc[i]
        fired = []
        for j in np.where(M[i])[0]:
            st_ = lookup.get((names[j], row['bet_type']))
            if st_ and st_['n'] >= minn:
                fired.append(st_)
        fired.sort(key=lambda r: -r['shr'])
        best = fired[0] if fired else None
        fired_z = [f for f in fired if f.get('z', 0.0) >= V11_Z_MIN]
        best_z = fired_z[0] if fired_z else None
        active = [f for f in fired if f['shr'] >= thr]
        res[(row['bet_type'], round(float(row['line']), 3), row['side'])] = {
            'ev_kp': float(best['shr']) if best else None,
            'kp_rule': best['rule'] if best else '',
            'kp_n': int(best['n']) if best else 0,
            'kp_roi': float(best['roi']) if best else 0.0,
            'kp_active': len(active),
            'kp_fired': [f"{f['rule']} (n={f['n']}, ROI {f['roi']*100:.1f}%, 估計EV {f['shr']*100:+.1f}%, z={f.get('z', 0):.1f})" for f in fired[:5]],
            'ev_kz': float(best_z['shr']) if best_z else None, 'kz_rule': best_z['rule'] if best_z else '',
            'mv_imp': float(row['mv_imp']), 'imp_o': float(row['imp_o']), 'imp_c': float(row['imp_c']),
            'line_o': float(row['line_o']), 'odds_o': float(row['odds_o']), 'pub': float(row['pub']), 'resid': float(row['resid']),
            'close_pos': int(row['close_pos']),
        }
    return res


def v10_apply_to_candidates(cands, kp_map, mode, settings, kp_engine, arena=None, league_ctx=None):
    """把看盤 / V9 結果套用到候選盤口，並依策略模式決定最終 EV 與狀態。
    V11 自動模式：只有在模型擂台通過樣本外驗證的『策略 × 盤口類型』才會實際落注，其餘只作模擬注 ($0)"""
    thr = settings['min_ev']
    league_ctx = league_ctx or {}
    arena = arena or {}
    act = arena.get('active', {}) or {}
    for c in cands:
        bt = v9_clean_bt(c['bet_type'])
        c['ev_v9'] = float(c.get('ev', 0.0))
        c['prob_v9'] = float(c.get('prob', 0.0))
        key = (bt, round(float(c['line']), 3), c['selection'])
        # 只有『收盤主盤』(最新時間點最平衡的盤口線) 的最新賠率才配對看盤訊號
        kp = kp_map.get(key) if c.get('is_latest', True) else None
        c['ev_kp'] = kp.get('ev_kp') if kp else None
        c['kp_rule'] = kp.get('kp_rule', '') if kp else ''
        c['kp_fired'] = kp.get('kp_fired', []) if kp else []
        c['kp_n'] = kp.get('kp_n', 0) if kp else 0
        c['ev_kz'] = kp.get('ev_kz') if kp else None
        c['kz_rule'] = kp.get('kz_rule', '') if kp else ''
        c['paper'] = False
        c['strategy'] = 'v9'
        if mode == V11_MODE_AUTO:
            chosen = None
            for sk in ('kz', 'kp', 'v9'):
                if not act.get(sk, {}).get(bt):
                    continue
                ev_s = {'kz': c['ev_kz'], 'kp': c['ev_kp'], 'v9': c['ev_v9']}[sk]
                if ev_s is None:
                    continue
                if ev_s >= max(thr, V11_STRAT_THR[sk]):
                    chosen = (sk, ev_s)
                    break
            if chosen:
                c['strategy'], c['ev'] = chosen
                c['engine_used'] = V11_STRAT_NAMES[chosen[0]]
                c['gate_ok'], c['gate_msg'] = True, ''
            else:
                opts = [(sk, e) for sk, e in (('kp', c['ev_kp']), ('v9', c['ev_v9'])) if e is not None]
                sk, e = max(opts, key=lambda t: t[1]) if opts else ('v9', c['ev_v9'])
                c['strategy'], c['ev'] = sk, float(e)
                c['engine_used'] = '模擬'
                c['paper'] = True
                c['gate_ok'], c['gate_msg'] = False, '策略未通過樣本外驗證'
            c['prob'] = (c['ev'] + 1.0) / float(c['odds'])
        elif mode == V11_MODE_STRICT:
            if c['ev_kz'] is not None:
                c['ev'] = float(c['ev_kz'])
                c['prob'] = (c['ev'] + 1.0) / float(c['odds'])
                c['gate_ok'], c['gate_msg'] = True, ''
                c['engine_used'], c['strategy'] = '看盤', 'kz'
                c['kp_rule'] = c['kz_rule'] or c['kp_rule']
            else:
                c['gate_ok'], c['gate_msg'] = False, '無嚴格看盤訊號'
                c['engine_used'], c['strategy'] = '看盤 (無訊號)', 'kz'
        elif mode in ('看盤規則', '看盤規則 + V9') and c['ev_kp'] is not None:
            ok, why = v10_type_gate(kp_engine, bt, thr, settings['override_gate'])
            c['ev'] = float(c['ev_kp'])
            c['prob'] = (c['ev'] + 1.0) / float(c['odds'])
            c['gate_ok'], c['gate_msg'] = ok, why
            c['engine_used'], c['strategy'] = '看盤', 'kp'
        elif mode == '看盤規則':
            # 只用看盤規則：沒有看盤訊號的盤口不給注碼 (V9 EV 只作參考)
            c['gate_ok'], c['gate_msg'] = False, '無看盤訊號'
            c['engine_used'], c['strategy'] = '看盤 (無訊號)', 'kp'
        else:
            c['engine_used'] = 'V9'
        if c.get('paper'):
            c['status'] = '📝 模擬注 (策略未通過驗證，$0)' if c['ev'] >= thr else ('🟡 EV 未達門檻' if c['ev'] > 0 else '➖ 負 EV')
        elif mode in ('看盤規則', V11_MODE_STRICT) and c.get('engine_used') == '看盤 (無訊號)':
            c['status'] = '➖ 非收盤主盤 / 無看盤訊號'
        elif not c.get('gate_ok', True):
            c['status'] = f"⛔ 熔斷 ({c.get('gate_msg', '')})"
        elif c['ev'] >= thr:
            c['status'] = '✅ 建議'
        elif c['ev'] > 0:
            c['status'] = '🟡 EV 未達門檻'
        else:
            c['status'] = '➖ 負 EV'
        # V11 聯賽層面保護：新聯賽觀察期 / 聯賽熔斷 (只影響原本會建議落注的盤口)
        if str(c.get('status', '')).startswith('✅') and not settings.get('override_gate'):
            n_lg = int(league_ctx.get('n_matches', 999))
            lg = (arena.get('league', {}) or {}).get(c.get('strategy'), {}).get(league_ctx.get('tourn'))
            if settings.get('new_league_paper', True) and n_lg < V11_NEW_LEAGUE_MIN:
                c['status'] = f"📝 模擬注 (新聯賽觀察期：只有 {n_lg} 場已結算 < {V11_NEW_LEAGUE_MIN})"
                c['gate_ok'], c['gate_msg'], c['paper'] = False, '新聯賽觀察期', True
            elif settings.get('league_breaker', True) and lg and lg['n'] >= V11_LG_MIN_N and lg['roi'] <= V11_LG_MAX_ROI:
                c['status'] = f"⛔ 聯賽熔斷 (此聯賽樣本外 {lg['n']} 注 ROI {lg['roi']*100:.0f}%)"
                c['gate_ok'], c['gate_msg'] = False, '聯賽熔斷'
    # 同一盤口類型只保留看盤估計 EV 較高的一方 (避免同時建議主客兩邊 / 大小兩邊)，與前進式回測邏輯一致
    best_by_type = {}
    for c in cands:
        if c.get('strategy') in ('kp', 'kz') and c.get('ev_kp') is not None and c.get('engine_used') != 'V9':
            bt = v9_clean_bt(c['bet_type'])
            if bt not in best_by_type or c['ev'] > best_by_type[bt]['ev']:
                best_by_type[bt] = c
    for c in cands:
        if c.get('strategy') in ('kp', 'kz') and c.get('ev_kp') is not None and c.get('engine_used') != 'V9' and best_by_type.get(v9_clean_bt(c['bet_type'])) is not c:
            if str(c.get('status', ''))[:1] in ('✅', '🟡', '📝'):
                c['status'] = '↔ 對邊看盤訊號較強'
            c['gate_ok'], c['gate_msg'] = False, '對邊看盤訊號較強'
    cands.sort(key=lambda x: (str(x.get('status', '')).startswith('✅'), str(x.get('status', '')).startswith('📝'),
                              x.get('ev_kp') is not None, x.get('ev', -9)), reverse=True)
    return cands


# ---------- 設定永久儲存 (與數據庫同步到 GitHub / SQL / 本地) ----------
SETTINGS_FILE = "football_settings_db.csv"
SETTINGS_TABLE = "football_settings"
SETTINGS_COLUMNS = ['Key', 'Value']
V11_MODE_AUTO = '🛡 自動 (只用已驗證策略)'
V11_MODE_STRICT = '看盤嚴格規則 (z≥1.5)'
V10_SETTING_DEFAULTS = {
    'v9_min_ev': 3.0, 'v9_kelly': 0.25, 'v9_max_pct': 3.0, 'v9_max_picks': 1,
    'v9_hd_min200': False, 'v9_override_gate': False, 'v10_mode': V11_MODE_AUTO,
    'v11_new_league_paper': True, 'v11_league_breaker': True, 'v11_migrated': True,
}
V10_MODE_OPTIONS = [V11_MODE_AUTO, '看盤規則', V11_MODE_STRICT, '看盤規則 + V9', '只用 V9 模型']
V11_LS_PREFIX = 'LS|'


def _v10_cast(key, val):
    d = V10_SETTING_DEFAULTS[key]
    try:
        if isinstance(d, bool):
            return str(val).strip().lower() in ('true', '1', 'yes')
        if isinstance(d, int):
            return int(float(val))
        if isinstance(d, float):
            return float(val)
        return str(val)
    except (TypeError, ValueError):
        return d


def v10_load_settings_into_state():
    """啟動時讀取已儲存的設定 (只在本次瀏覽工作階段第一次執行)"""
    if st.session_state.get('_v10_settings_loaded'):
        return
    saved = {}
    ls_map = {}
    ign = set()
    try:
        df_s = load_db(SETTINGS_FILE, SETTINGS_COLUMNS, SETTINGS_TABLE)
        for _, r in df_s.iterrows():
            k = str(r.get('Key', ''))
            if k in V10_SETTING_DEFAULTS and pd.notna(r.get('Value')):
                saved[k] = _v10_cast(k, r.get('Value'))
            elif k.startswith('IGN|'):
                ign.add(k[4:])
            elif k.startswith(V11_LS_PREFIX) and pd.notna(r.get('Value')):
                try:
                    ls_map[k[len(V11_LS_PREFIX):]] = int(float(r.get('Value')))
                except (TypeError, ValueError):
                    pass
    except Exception:
        saved = {}
    st.session_state['_v11_ls_map'] = ls_map
    st.session_state['_v11_ign'] = ign
    # V11 升級：舊版儲存的建議引擎改為新的自動保護模式 (只執行一次，之後用家可自行更改)
    if saved and 'v11_migrated' not in saved:
        saved['v10_mode'] = V11_MODE_AUTO
        saved['v11_migrated'] = True
        st.session_state['_v11_migrated_now'] = True
    for k, d in V10_SETTING_DEFAULTS.items():
        v = saved.get(k, d)
        if k == 'v9_kelly' and v not in [0.1, 0.15, 0.2, 0.25, 0.33, 0.5]:
            v = 0.25
        if k == 'v10_mode' and v not in V10_MODE_OPTIONS:
            v = d
        st.session_state[k] = v
    st.session_state['_v10_settings_loaded'] = True
    st.session_state['_v10_settings_saved'] = bool(saved)


def v10_save_settings(silent=False):
    rows = [{'Key': k, 'Value': str(st.session_state.get(k, d))} for k, d in V10_SETTING_DEFAULTS.items()]
    rows += [{'Key': f"{V11_LS_PREFIX}{t}", 'Value': str(int(n))} for t, n in (st.session_state.get('_v11_ls_map', {}) or {}).items()]
    rows += [{'Key': f"IGN|{k}", 'Value': '1'} for k in sorted(st.session_state.get('_v11_ign', set()) or set())]
    try:
        save_db(pd.DataFrame(rows, columns=SETTINGS_COLUMNS), SETTINGS_FILE, SETTINGS_TABLE)
        st.session_state['_v10_settings_saved'] = True
        if not silent:
            st.toast("✅ 投注策略設定已永久儲存，下次開啟程式會自動載入。", icon="💾")
    except Exception as e:
        st.toast(f"設定儲存失敗: {e}", icon="⚠")


def v10_md_series(df):
    """比賽日期 (YYYY-MM-DD)；缺 Match_Date 時以下注日期代替"""
    md = pd.to_datetime(df['Match_Date'], errors='coerce') if 'Match_Date' in df.columns else pd.Series(pd.NaT, index=df.index)
    dt = pd.to_datetime(df['Date'], errors='coerce') if 'Date' in df.columns else pd.Series(pd.NaT, index=df.index)
    md = md.fillna(dt)
    return md.dt.strftime('%Y-%m-%d').fillna('未知日期')


def v10_match_groups(df):
    """按『賽事 + 對賽 + 比賽日期』分組 (最新日期排最前)"""
    if df is None or df.empty:
        return pd.DataFrame(columns=['Tournament_Name', 'Match', '_md', 'Bet_Count'])
    d = df.copy()
    d['_md'] = v10_md_series(d)
    d['Tournament_Name'] = d['Tournament_Name'].fillna('').astype(str)
    d['Match'] = d['Match'].fillna('').astype(str)
    g = d.groupby(['Tournament_Name', 'Match', '_md']).size().reset_index(name='Bet_Count')
    return g.sort_values(['_md', 'Tournament_Name', 'Match'], ascending=[False, True, True]).reset_index(drop=True)


def v10_match_mask(df, tourn, match, md):
    return ((df['Tournament_Name'].fillna('').astype(str) == str(tourn)) &
            (df['Match'].fillna('').astype(str) == str(match)) &
            (v10_md_series(df) == str(md)))


# ==========================================
# 3c. V11 數據質素、聯賽隊伍數目、模型擂台 (樣本外驗證後才落注)
# ==========================================
V11_DB_FILE = "football_betting_db.csv"
V11_DB_TABLE = "football_bets"
# 常見聯賽隊伍數目 (2026-27 球季)；用家輸入過或手動修改的數目永遠優先
V11_DEFAULT_LEAGUE_SIZES = {
    '英格蘭超級聯賽': 20, '西班牙甲組聯賽': 20, '意大利甲組聯賽': 20, '德國甲組聯賽': 18, '法國甲組聯賽': 18,
    '荷蘭甲組聯賽': 18, '葡萄牙超級聯賽': 18, '日本職業聯賽': 20, '南韓職業聯賽': 12, '澳洲職業聯賽': 12,
    '墨西哥超級聯賽': 18, '美國職業大聯盟': 30, '英格蘭冠軍聯賽': 24, '蘇格蘭超級聯賽': 12, '比利時甲組聯賽': 16,
    '土耳其超級聯賽': 18, '沙地職業聯賽': 18, '中國超級聯賽': 16, '德國乙組聯賽': 18, '西班牙乙組聯賽': 22,
    '意大利乙組聯賽': 20, '法國乙組聯賽': 18, '日本乙組聯賽': 20,
}
V11_NA_TOKENS = {'NA', 'N/A', '無', '無排名', '-', ''}
V11_CUP_CATS = ("國際盃賽 (Cup)", "國內盃賽 (Domestic Cup)", "友誼賽 (Friendly)")
V11_INTL_HINTS = ('亞冠', '歐聯', '歐霸', '歐協', '世冠', '自由盃', '南美', '中北美', '世界盃', '國際')
V11_CUP_HINTS = ('盃', '杯', '亞冠', '歐聯', '歐霸', '歐協', '友誼')
V11_STRAT_NAMES = {'kp': 'V10 看盤規則 (EV≥3%)', 'kz': 'V11 嚴格看盤規則 (z≥1.5)', 'v9': 'V9 市場錨定機器學習 (EV≥2%)',
                   'lb': '聯賽專屬偏差模型 (對照組)', 'mkt': '隨機買任一邊 (市場基準)'}
V11_STRAT_THR = {'kp': 0.03, 'kz': 0.0, 'v9': 0.02}
V11_ACT_MIN_N = 50          # 策略啟用：該盤口類型樣本外注數下限
V11_ACT_MIN_ROI = 0.02      # 策略啟用：樣本外 ROI 下限
V11_ACT_HALF_MIN_N = 15     # 前後兩段測試期各自最少注數 (兩段都要賺錢，避免偶然)
V11_LG_MIN_N = 15           # 聯賽熔斷：該聯賽樣本外注數下限
V11_LG_MAX_ROI = -0.15      # 聯賽熔斷：樣本外 ROI ≤ -15%
V11_NEW_LEAGUE_MIN = 15     # 新聯賽觀察期：已結算比賽少於此數只作模擬注
V11_Z_MIN = 1.5


def v11_clean_name(s):
    """去除名稱前後及重複空格 (例如 ' 托盧卡' → '托盧卡')，避免同一球隊被當成兩隊"""
    if s is None:
        return s
    try:
        if isinstance(s, float) and np.isnan(s):
            return s
    except TypeError:
        pass
    t = re.sub(r'\s+', ' ', str(s)).strip()
    if ' vs ' in t:
        t = ' vs '.join(p.strip() for p in t.split(' vs '))
    return t


def v11_is_na_rating(v):
    if v is None:
        return True
    try:
        if isinstance(v, float) and np.isnan(v):
            return True  # CSV 讀入時 'NA' 會變成空值 (NaN)，同樣視為無排名
    except TypeError:
        pass
    return str(v).strip().upper() in V11_NA_TOKENS  # 空白 = 未填排名，同樣視為實力未知


def v11_is_cup_like(tourn, cat=None):
    t = str(tourn or '')
    if t in V11_DEFAULT_LEAGUE_SIZES:
        return False
    if cat is not None and str(cat) in V11_CUP_CATS:
        return True
    return any(h in t for h in V11_CUP_HINTS)


def v11_is_intl(tourn, cat=None):
    t = str(tourn or '')
    if t in V11_DEFAULT_LEAGUE_SIZES:
        return False
    return str(cat) == "國際盃賽 (Cup)" or any(h in t for h in V11_INTL_HINTS)


def v11_strength_unknown(h_rating, a_rating, ls, tourn='', cat=None):
    """實力未知：排名標記為『無排名』、國際盃賽 (不同聯賽的排名無法比較)、或盃賽中兩隊排名都是未改動的預設中位數"""
    if v11_is_na_rating(h_rating) or v11_is_na_rating(a_rating):
        return True
    if v11_is_intl(tourn, cat):
        return True
    if v11_is_cup_like(tourn, cat):
        rh, ra = _parse_rank_int(h_rating), _parse_rank_int(a_rating)
        if rh is not None and rh == ra:
            return True
    return False


def v11_valid_pair(up, lw):
    """賠率是否合理：兩邊均 > 1.0、≤ 15，且抽水 (1/上 + 1/下) 在 1.00 ~ 1.25 之間"""
    try:
        up, lw = float(up), float(lw)
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(up) and np.isfinite(lw)) or up <= 1.0 or lw <= 1.0 or up > 15 or lw > 15:
        return False
    m = 1.0 / up + 1.0 / lw
    return 1.0 <= m <= 1.25


def v11_ls_map():
    return dict(st.session_state.get('_v11_ls_map', {}) or {})


def v11_lookup_league_size(tourn, df_db=None):
    """回傳 (隊伍數目, 來源說明)。優先次序：用家儲存 → 系統內建 → 數據庫最近一次輸入 → 20"""
    t = v11_clean_name(tourn) or ''
    m = v11_ls_map()
    if t in m:
        try:
            return int(m[t]), '已記住的用家設定'
        except (TypeError, ValueError):
            pass
    if t in V11_DEFAULT_LEAGUE_SIZES:
        return V11_DEFAULT_LEAGUE_SIZES[t], '系統內建 (2026-27 球季)'
    if df_db is not None and not df_db.empty and t:
        d = df_db[df_db['Tournament_Name'].astype(str).map(v11_clean_name) == t]
        s = pd.to_numeric(d.get('League_Size', pd.Series(dtype=float)), errors='coerce').dropna()
        s = s[(s >= 2) & (s <= 40)]
        if len(s):
            return int(s.iloc[-1]), '數據庫最近一次輸入'
    return 20, '預設值'


def v11_save_league_size(tourn, n, silent=False):
    t = v11_clean_name(tourn) or ''
    if not t or t == "➕ 新增手動輸入...":
        return
    try:
        n = int(n)
    except (TypeError, ValueError):
        return
    m = v11_ls_map()
    if m.get(t) == n:
        return
    m[t] = n
    st.session_state['_v11_ls_map'] = m
    v10_save_settings(silent=True)
    if not silent:
        st.toast(f"💾 已記住「{t}」有 {n} 支隊伍，下次選擇此賽事會自動填寫。", icon="📝")


def _v11_on_ls_change(tourn):
    v11_save_league_size(tourn, st.session_state.get('league_size', 20))


def v11_default_no_rank(tourn, df_db):
    """新增注單時『此賽事沒有聯賽排名』的預設值：盃賽/洲際賽，或此賽事過往資料已標記為無排名"""
    t = v11_clean_name(tourn) or ''
    if not t or t == "➕ 新增手動輸入...":
        return False
    if df_db is not None and not df_db.empty:
        d = df_db[df_db['Tournament_Name'].astype(str).map(v11_clean_name) == t]
        if len(d) and d['Home_Rating'].map(v11_is_na_rating).tail(10).mean() >= 0.5:
            return True
    return v11_is_cup_like(t)


# ---------- 數據質素檢查 ----------
def _v11_issue(level, cat, row_or_none, msg, fix, ids=None, tourn='', match='', md=''):
    if row_or_none is not None:
        tourn = tourn or str(row_or_none.get('Tournament_Name', ''))
        match = match or str(row_or_none.get('Match', ''))
    return {'嚴重程度': level, '類別': cat, '賽事': tourn, '比賽': match, '比賽日期': md,
            '注單ID': ', '.join(ids) if ids else '', '問題說明': msg, '建議處理': fix}


def v11_data_audit(df_db):
    """全面檢查數據庫：名稱空格、重複注單、賽事標籤錯誤、同隊短時間內出賽兩次、賠率輸入錯誤、
    近況不等於 5 場、排名未填、隊伍數目錯誤、結算結果與比分不符等"""
    issues = []
    if df_db is None or df_db.empty:
        return pd.DataFrame(columns=['嚴重程度', '類別', '賽事', '比賽', '比賽日期', '注單ID', '問題說明', '建議處理'])
    d = df_db.copy()
    d['_md'] = v10_md_series(d)
    d['_ID'] = d['ID'].astype(str)
    for c in ('Home_Team', 'Away_Team', 'Tournament_Name', 'Match'):
        d[c] = d[c].where(d[c].notna(), '')
    d['_h'] = d['Home_Team'].astype(str).map(v11_clean_name)
    d['_a'] = d['Away_Team'].astype(str).map(v11_clean_name)
    d['_t'] = d['Tournament_Name'].astype(str).map(v11_clean_name)
    d['_m'] = d['Match'].astype(str).map(v11_clean_name)

    # 1) 名稱多餘空格
    for c, lab in (('Home_Team', '主隊'), ('Away_Team', '客隊'), ('Tournament_Name', '賽事'), ('Match', '對賽')):
        bad = d[d[c].astype(str) != d[c].astype(str).map(v11_clean_name)]
        for v, g in bad.groupby(c):
            issues.append(_v11_issue('🟠 中', '名稱多餘空格', g.iloc[0], f"{lab}名稱「{v}」含多餘空格，與「{v11_clean_name(v)}」被當成不同球隊/賽事，球隊歷史數據被分割。",
                                     '按下方「🧹 修正名稱空格」一鍵修正', list(g['_ID'])[:6], md=g['_md'].iloc[0]))

    # 2) 重複注單
    kcols = ['_m', '_md', 'Bet_Type', 'Selection', 'Initial_Line', 'Initial_Odds']
    dd = d.copy()
    for c in ('Initial_Line', 'Initial_Odds'):
        dd[c] = pd.to_numeric(dd[c], errors='coerce').round(3)
    for _, g in dd.groupby(kcols, dropna=False):
        if len(g) > 1:
            issues.append(_v11_issue('🔴 高', '重複注單', g.iloc[0], f"同一比賽、盤口、方向、盤口線及賠率出現 {len(g)} 張相同注單 ({g.iloc[0]['Bet_Type']} {g.iloc[0]['Selection']} {g.iloc[0]['Initial_Line']} @{g.iloc[0]['Initial_Odds']})，盈虧及學習樣本會被重複計算。",
                                     '如非真的下注兩次，請在下方「刪除重複注單」刪除多餘的一張', list(g['_ID']), md=g.iloc[0]['_md']))

    # 3) 同一比賽內資料不一致
    for (m_, md_), g in d.groupby(['_m', '_md']):
        if g['_t'].nunique() > 1:
            cnt = g['_t'].value_counts()
            minor = cnt.index[-1]
            issues.append(_v11_issue('🔴 高', '同一比賽賽事名稱不一致', g.iloc[0], f"同一場比賽同時被記錄為：{'、'.join(f'{k} ({v} 張)' for k, v in cnt.items())}。",
                                     f"請把「{minor}」的注單改為正確賽事", list(g[g['_t'] == minor]['_ID']), tourn='/'.join(cnt.index), md=md_))
        for col, lab in (('Home_Rating', '主隊排名'), ('Away_Rating', '客隊排名'), ('Home_Form', '主隊近況'), ('Away_Form', '客隊近況'), ('League_Size', '隊伍數目')):
            vals = g[col].dropna().astype(str).map(lambda x: x[:-2] if x.endswith('.0') else x)
            vals = vals[vals != '']
            if vals.nunique() > 1:
                cnt = vals.value_counts()
                minor = cnt.index[-1]
                ids_ = list(g[g[col].astype(str).map(lambda x: x[:-2] if x.endswith('.0') else x) == minor]['_ID'])
                issues.append(_v11_issue('🟡 低', '同一比賽資料不一致', g.iloc[0], f"{lab}在同一場比賽有不同數值：{'、'.join(f'{k} ({v} 張)' for k, v in cnt.items())}。系統學習時只取其中一個。",
                                         '請統一為正確數值', ids_[:6], md=md_))

    # 4) 同一球隊短時間內出賽兩次 / 球隊出現在不常見的賽事
    mu = d.drop_duplicates(['_m', '_md'])[['_m', '_md', '_h', '_a', '_t', '_ID']]
    long = pd.concat([mu.rename(columns={'_h': 'team'})[['team', '_m', '_md', '_t', '_ID']],
                      mu.rename(columns={'_a': 'team'})[['team', '_m', '_md', '_t', '_ID']]])
    long = long[long['team'].astype(str) != '']
    long['_dt'] = pd.to_datetime(long['_md'], errors='coerce')
    seen = set()
    for team, g in long.dropna(subset=['_dt']).sort_values('_dt').groupby('team'):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            gap = (g.loc[i, '_dt'] - g.loc[i - 1, '_dt']).days
            if gap <= 2 and g.loc[i, '_m'] != g.loc[i - 1, '_m']:
                key = (team, g.loc[i - 1, '_md'], g.loc[i, '_md'])
                if key in seen:
                    continue
                seen.add(key)
                lvl = '🔴 高' if gap == 0 else '🟡 低'
                issues.append(_v11_issue(lvl, '同一球隊短時間內出賽兩次', None,
                                         f"「{team}」在 {g.loc[i - 1, '_md']} ({g.loc[i - 1, '_m']}) 及 {g.loc[i, '_md']} ({g.loc[i, '_m']}) 相隔 {gap} 日出賽。" + ('同一天踢兩場，必定有日期或球隊輸入錯誤。' if gap == 0 else '請確認日期是否正確 (盃賽/洲際賽可能屬實)。'),
                                         '檢查比賽日期或球隊名稱', [g.loc[i - 1, '_ID'], g.loc[i, '_ID']], tourn=f"{g.loc[i - 1, '_t']} / {g.loc[i, '_t']}", match=team, md=g.loc[i, '_md']))
    home_t = long.groupby('team')['_t'].agg(lambda x: x.value_counts().to_dict())
    for _, r in mu.iterrows():
        if v11_is_cup_like(r['_t']) or v11_is_intl(r['_t']):
            continue  # 盃賽/洲際賽的球隊本來就屬於其他聯賽，不屬錯誤
        bad = True
        for tm in (r['_h'], r['_a']):
            tc = home_t.get(tm, {})
            others = sum(v for k, v in tc.items() if k != r['_t'])
            if tc.get(r['_t'], 0) > 1 or others < 2:
                bad = False
        if bad and r['_t']:
            issues.append(_v11_issue('🔴 高', '賽事名稱可能選錯', None, f"兩隊通常參加其他賽事 (主隊：{'、'.join(home_t.get(r['_h'], {}).keys())})，但此場被記錄為「{r['_t']}」。",
                                     '修改為正確賽事名稱 (錯誤標籤會污染聯賽統計及熔斷判斷)', [r['_ID']], tourn=r['_t'], match=r['_m'], md=r['_md']))

    # 5) 賠率輸入錯誤
    for (m_, md_), g in d.groupby(['_m', '_md']):
        bad_list = []
        for s in g['Odds_History'].dropna().unique():
            try:
                oh = json.loads(str(s))
            except Exception:
                continue
            if not isinstance(oh, list):
                continue
            for h in oh:
                try:
                    bt, ln, up, lw = v9_clean_bt(h.get('type', '')), float(h.get('line', 0)), float(h.get('upper')), float(h.get('lower'))
                except (TypeError, ValueError, AttributeError):
                    continue
                if up <= 1.0 or lw <= 1.0:
                    continue
                tag = f"{bt} {ln:g} @ {up:g}/{lw:g}"
                if not v11_valid_pair(up, lw):
                    bad_list.append(tag + f" (抽水 {1/up + 1/lw:.2f})")
                elif bt == '讓球' and abs(ln) >= 3.5:
                    bad_list.append(tag + " (讓球盤口線異常深，請確認)")
        if bad_list:
            bad_list = list(dict.fromkeys(bad_list))
            issues.append(_v11_issue('🟠 中', '賠率輸入錯誤', g.iloc[0], f"不合理的賠率紀錄：{'；'.join(bad_list[:6])}。正常抽水約 1.03~1.12；兩邊同一數值或任何一邊 >15 多數是輸入/貼上錯誤。V11 訓練時已自動忽略這些紀錄。",
                                     '修正該場的賠率紀錄', list(g['_ID'])[:3], md=md_))

    # 6) 近況不等於 5 場 / 未改動的預設近況
    mrow = d.drop_duplicates(['_m', '_md'])
    for _, r in mrow.iterrows():
        msgs = []
        for col, lab in (('Home_Form', '主隊'), ('Away_Form', '客隊')):
            w, dr, l_ = parse_form_str(r.get(col, ''))
            if w + dr + l_ != 5:
                msgs.append(f"{lab} {r.get(col)} 合共 {w + dr + l_} 場")
        if msgs:
            issues.append(_v11_issue('🟡 低', '近況場數不等於 5', r, '；'.join(msgs) + '。近 5 場勝和負合計應為 5。', '修正近況', [r['_ID']], md=r['_md']))
        elif str(r.get('Home_Form')) == '3W1D1L' and str(r.get('Away_Form')) == '2W2D1L':
            issues.append(_v11_issue('🟡 低', '近況可能未填 (預設值)', r, '兩隊近況與表單預設值完全相同 (3W1D1L / 2W2D1L)。', '確認是否已輸入實際近況', [r['_ID']], md=r['_md']))

    # 7) 排名未填 / 隊伍數目 / 賽事分類 (按賽事彙總)
    for t, g in d.drop_duplicates(['_m', '_md']).groupby('_t'):
        if not t:
            continue
        ls_vals = pd.to_numeric(g['League_Size'], errors='coerce')
        mid = (ls_vals.fillna(20) // 2).astype(int)
        hr = g['Home_Rating'].map(_parse_rank_int)
        ar = g['Away_Rating'].map(_parse_rank_int)
        default_rank = (hr == mid) & (ar == mid)
        na_rank = g['Home_Rating'].map(v11_is_na_rating)
        if default_rank.sum() >= max(3, 0.6 * len(g)) and not na_rank.all():
            issues.append(_v11_issue('🔴 高', '排名未填 (全部是預設值)', None, f"{default_rank.sum()}/{len(g)} 場兩隊排名都是預設值 {int(mid.iloc[0])}。系統會誤以為兩隊實力相同，產生錯誤的『冷熱』判斷。",
                                     '盃賽/洲際賽請在下方「賽事設定」標記為「沒有聯賽排名」；聯賽請輸入實際排名', None, tourn=t))
        elif 0 < default_rank.sum() < max(3, 0.6 * len(g)):
            for _, r in g[default_rank].iterrows():
                issues.append(_v11_issue('🟡 低', '排名可能未填 (預設值)', r, f"兩隊排名都是預設值 {int((pd.to_numeric(r.get('League_Size'), errors='coerce') or 20) // 2)}。", '確認排名', [r['_ID']], md=r['_md']))
        exp_ls = V11_DEFAULT_LEAGUE_SIZES.get(t)
        if exp_ls is not None:
            wrong = g[ls_vals.notna() & (ls_vals != exp_ls)]
            if len(wrong):
                issues.append(_v11_issue('🟠 中', '聯賽隊伍數目錯誤', None, f"{len(wrong)} 場比賽的隊伍數目為 {sorted(set(ls_vals[ls_vals.notna() & (ls_vals != exp_ls)].astype(int)))}，{t} 應為 {exp_ls} 隊。隊伍數目錯誤會令排名換算的實力分偏差。",
                                         '按下方「賽事設定」統一隊伍數目', list(wrong['_ID'])[:6], tourn=t))
        if ls_vals.isna().mean() > 0.3:
            issues.append(_v11_issue('🟡 低', '隊伍數目缺失', None, f"{int(ls_vals.isna().sum())}/{len(g)} 場沒有記錄隊伍數目 (系統以 20 隊計算)。", '按下方「賽事設定」統一隊伍數目', None, tourn=t))
        cats = g['Tournament_Category'].dropna().astype(str).unique()
        if v11_is_cup_like(t) and any(c.startswith('國內聯賽') for c in cats):
            issues.append(_v11_issue('🟡 低', '賽事分類可能錯誤', None, f"「{t}」屬盃賽/洲際賽，但分類為「國內聯賽」。", '按下方「賽事設定」修改分類', None, tourn=t))
        letters = g['Home_Rating'].dropna().astype(str).str.upper().isin(list(RATING_LETTER_PCTILE.keys())).mean()
        if 0 < letters < 1:
            issues.append(_v11_issue('🟡 低', '排名格式混合', None, f"{letters*100:.0f}% 比賽使用舊字母評級 (S~D)，其餘使用數字排名。字母評級只能換算為約略排名。", '如有時間可改為實際排名', None, tourn=t))

    # 8) 結算結果與比分不符
    sd = d[d['Status'] == 'Settled']
    for _, r in sd.iterrows():
        try:
            hg, ag = float(r.get('Home_Goal')), float(r.get('Away_Goal'))
            hc, ac = r.get('Home_Corner'), r.get('Away_Corner')
            if np.isnan(hg) or np.isnan(ag):
                continue
            if '角球' in str(r.get('Bet_Type', '')) and (pd.isna(hc) or pd.isna(ac)):
                continue
            hc = 0 if pd.isna(hc) else int(float(hc))
            ac = 0 if pd.isna(ac) else int(float(ac))
            sp, up_, _, _, _, lab, _ = calculate_settlement(r['Bet_Type'], r['Selection'], r['Initial_Line'], r['Initial_Odds'],
                                                             r['System_Stake'], r['User_Stake'], int(hg), int(ag), hc, ac)
        except Exception:
            continue
        s0 = pd.to_numeric(r.get('System_Profit'), errors='coerce')
        u0 = pd.to_numeric(r.get('User_Profit'), errors='coerce')
        if (pd.notna(s0) and abs(float(sp) - float(s0)) > 0.6) or (pd.notna(u0) and abs(float(up_) - float(u0)) > 0.6):
            issues.append(_v11_issue('🔴 高', '結算結果與比分不符', r, f"已儲存結果「{r.get('Result_Label')}」系統盈虧 {s0} / 用家 {u0}，但按已儲存比分 {int(hg)}-{int(ag)} (角球 {hc}-{ac}) 重新計算應為「{lab}」系統 {sp:.1f} / 用家 {up_:.1f}。比分或結算其中之一錯誤。",
                                     '先確認比分；比分正確 → 按下方「重新計算結算」；比分錯誤 → 到結算頁修改比分', [r['_ID']], md=r['_md']))
    out = pd.DataFrame(issues)
    if out.empty:
        return pd.DataFrame(columns=['嚴重程度', '類別', '賽事', '比賽', '比賽日期', '注單ID', '問題說明', '建議處理'])
    order = {'🔴 高': 0, '🟠 中': 1, '🟡 低': 2}
    out['_o'] = out['嚴重程度'].map(order)
    return out.sort_values(['_o', '類別', '比賽日期']).drop(columns='_o').reset_index(drop=True)


V11_IGN_PREFIX = 'IGN|'


def v11_issue_key(row):
    """每項問題的固定編號 (數據改動後編號會改變，問題會重新出現)"""
    import hashlib
    raw = '|'.join(str(row.get(c, '')) for c in ('類別', '賽事', '比賽', '比賽日期', '注單ID', '問題說明'))
    return hashlib.md5(raw.encode('utf-8')).hexdigest()[:12]


def v11_ignored_keys():
    return set(st.session_state.get('_v11_ign', set()) or set())


def v11_set_ignored(keys):
    st.session_state['_v11_ign'] = set(keys)
    v10_save_settings(silent=True)


def v11_issues_active(issues):
    """移除用家已確認無誤的項目"""
    if issues is None or issues.empty:
        return issues
    ign = v11_ignored_keys()
    if not ign:
        return issues
    return issues[~issues['編號'].isin(ign)]


@st.cache_data(max_entries=4, show_spinner="🧹 正在檢查數據質素...")
def v11_data_audit_cached(df_db):
    out = v11_data_audit(df_db)
    if out is not None and not out.empty:
        out = out.copy()
        out['編號'] = out.apply(v11_issue_key, axis=1)
    elif out is not None:
        out = out.copy()
        out['編號'] = pd.Series(dtype=str)
    return out


def v11_settlement_mismatch_ids(issues):
    if issues is None or issues.empty:
        return []
    x = issues[issues['類別'] == '結算結果與比分不符']
    return [i.strip() for s in x['注單ID'] for i in str(s).split(',') if i.strip()]


def v11_clear_model_caches():
    for f in (v9_train_engine_cached, v9_history_tables_cached, v10_train_engine_cached, v11_arena_cached, v11_data_audit_cached):
        try:
            f.clear()
        except Exception:
            pass


def _v11_commit_db(df_new, msg, undo_rows=None):
    if undo_rows is not None and len(undo_rows):
        st.session_state.setdefault('undo_stack', [])
        st.session_state.undo_stack.append({'id': f"U{get_hkt_now().strftime('%Y%m%d%H%M%S%f')}", 'target': 'bets',
                                            'data': undo_rows.copy(), 'timestamp': get_hkt_now().strftime('%Y-%m-%d %H:%M:%S')})
        st.session_state.undo_stack = st.session_state.undo_stack[-10:]
    st.session_state.df_db = enforce_columns(df_new.reset_index(drop=True), DB_COLUMNS)
    save_db(st.session_state.df_db, V11_DB_FILE, V11_DB_TABLE)
    v11_clear_model_caches()
    st.toast(msg, icon="🧹")
    st.rerun()


def render_v11_data_quality():
    """V11 數據質素檢查面板 + 安全修正工具"""
    df_db = st.session_state.df_db
    all_issues = v11_data_audit_cached(df_db)
    ign = v11_ignored_keys()
    issues = all_issues
    st.markdown("##### 🧹 數據質素檢查 (錯誤數據比虧損數據更傷害模型)")
    with st.expander("ℹ️ 如何處理每一項問題？", expanded=False):
        st.markdown(
            "1. **數據真的錯了** → 到「⚖ 賽果結算與管理 → 已完成注單資料修改」用注單 ID 搜尋並修改，或用下方一鍵修正。修改後該項會自動消失。\n"
            "2. **你已核實數據正確** (例如盤口真的是 -3.75、球隊真的兩日內出賽) → 在表格最右的「✅ 確認無誤」打勾，再按「💾 儲存確認」。"
            "該項之後不再計入問題數目，亦不會再出現提示 (設定會永久儲存)。\n"
            "3. 若之後該場數據再被修改，問題內容不同，系統會重新檢查並再次提示。")
    if all_issues.empty:
        st.success("✅ 沒有發現數據問題。")
    else:
        issues = all_issues[~all_issues['編號'].isin(ign)]
        n_conf = int(all_issues['編號'].isin(ign).sum())
        cnt = issues['嚴重程度'].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 高", int(cnt.get('🔴 高', 0)))
        c2.metric("🟠 中", int(cnt.get('🟠 中', 0)))
        c3.metric("🟡 低", int(cnt.get('🟡 低', 0)))
        c4.metric("✅ 已確認無誤", n_conf)
        if issues.empty:
            st.success("✅ 所有問題已修正或已確認無誤。")
        show_conf = st.checkbox("顯示已確認無誤的項目 (可取消確認)", value=False, key="v11_dq_show_conf")
        base = all_issues if show_conf else issues
        cats = ['全部'] + list(dict.fromkeys(base['類別']))
        pick = st.selectbox("篩選問題類別", cats, key="v11_dq_filter")
        show = base if pick == '全部' else base[base['類別'] == pick]
        if not show.empty:
            view = show.copy()
            view.insert(0, '✅ 確認無誤', view['編號'].isin(ign))
            ed = st.data_editor(view, use_container_width=True, hide_index=True, key=f"v11_dq_editor_{pick}_{show_conf}",
                                disabled=[c for c in view.columns if c != '✅ 確認無誤'],
                                column_config={'✅ 確認無誤': st.column_config.CheckboxColumn('✅ 確認無誤', help='已核實數據正確，不需要修改'),
                                               '編號': None})
            b1, b2 = st.columns(2)
            if b1.button("💾 儲存確認", key="v11_dq_save_ign", use_container_width=True, type="primary"):
                new_ign = set(ign)
                for _, r in ed.iterrows():
                    if bool(r['✅ 確認無誤']):
                        new_ign.add(r['編號'])
                    else:
                        new_ign.discard(r['編號'])
                new_ign &= set(all_issues['編號'])  # 清走已不存在的舊編號
                v11_set_ignored(new_ign)
                st.toast(f"✅ 已儲存：{len(new_ign)} 項標記為確認無誤。", icon="💾")
                st.rerun()
            if b2.button("↩ 取消全部確認", key="v11_dq_clear_ign", use_container_width=True, disabled=not ign):
                v11_set_ignored(set())
                st.rerun()
        st.caption("注單 ID 可在「⚖ 賽果結算與管理 → 已完成注單資料修改」搜尋並修改。修改後模型會自動重新訓練。")
        issues = all_issues

    st.markdown("##### 🛠 一鍵安全修正")
    f1, f2 = st.columns(2)
    with f1.container(border=True):
        st.markdown("**① 修正名稱多餘空格**")
        st.caption("去除球隊、賽事及對賽名稱前後/重複空格，合併被分割的球隊歷史。不會改動任何盈虧。")
        if st.button("🧹 修正名稱空格", key="v11_fix_space", use_container_width=True):
            d = df_db.copy()
            for c in ('Home_Team', 'Away_Team', 'Tournament_Name', 'Match'):
                d[c] = d[c].map(v11_clean_name)
            _v11_commit_db(d, "✅ 已修正名稱空格")
    with f2.container(border=True):
        st.markdown("**② 刪除重複注單**")
        dup = issues[issues['類別'] == '重複注單'] if not issues.empty else issues
        opts = []
        for s in dup.get('注單ID', pd.Series(dtype=str)):
            ids_ = [i.strip() for i in str(s).split(',') if i.strip()]
            opts.extend(ids_[1:])
        sel = st.multiselect("選擇要刪除的重複注單 (預設保留每組第一張)", opts, default=opts, key="v11_dup_sel")
        if st.button("🗑 刪除選中的重複注單", key="v11_fix_dup", use_container_width=True, disabled=not sel):
            rows = df_db[df_db['ID'].astype(str).isin(sel)]
            _v11_commit_db(df_db[~df_db['ID'].astype(str).isin(sel)], f"✅ 已刪除 {len(rows)} 張重複注單 (可在數據清理與還原中復原)", undo_rows=rows)
    f3, f4 = st.columns(2)
    with f3.container(border=True):
        st.markdown("**③ 按比分重新計算結算**")
        mm = v11_settlement_mismatch_ids(issues)
        st.caption("只在你已確認比分正確後使用。比分錯誤請到結算頁修改比分。")
        sel2 = st.multiselect("選擇要重新計算的注單", mm, default=[], key="v11_resettle_sel")
        if st.button("🔄 重新計算選中注單", key="v11_fix_settle", use_container_width=True, disabled=not sel2):
            d = df_db.copy()
            for bid in sel2:
                mk = d['ID'].astype(str) == bid
                if not mk.any():
                    continue
                r = d[mk].iloc[0]
                try:
                    hc = 0 if pd.isna(r.get('Home_Corner')) else int(float(r.get('Home_Corner')))
                    ac = 0 if pd.isna(r.get('Away_Corner')) else int(float(r.get('Away_Corner')))
                    sp, up_, spay, upay, unit, lab, _ = calculate_settlement(r['Bet_Type'], r['Selection'], r['Initial_Line'], r['Initial_Odds'],
                                                                             r['System_Stake'], r['User_Stake'], int(float(r['Home_Goal'])), int(float(r['Away_Goal'])), hc, ac)
                except Exception as e:
                    st.error(f"注單 {bid} 重算失敗：{e}")
                    continue
                d.loc[mk, ['System_Profit', 'User_Profit', 'System_Payout', 'User_Payout', 'Unit_Profit', 'Result_Label']] = [sp, up_, spay, upay, unit, lab]
            _v11_commit_db(d, f"✅ 已重新計算 {len(sel2)} 張注單")
    with f4.container(border=True):
        st.markdown("**④ 賽事設定 (分類 / 隊伍數目 / 無排名)**")
        tours = sorted(t for t in df_db['Tournament_Name'].dropna().astype(str).map(v11_clean_name).unique() if t)
        if tours:
            t = st.selectbox("賽事", tours, key="v11_ts_tourn")
            dt = df_db[df_db['Tournament_Name'].astype(str).map(v11_clean_name) == t]
            cur_cat = dt['Tournament_Category'].dropna().astype(str).mode()
            cur_cat = cur_cat.iloc[0] if len(cur_cat) else CATEGORY_OPTIONS[0]
            sug_cat = cur_cat
            if v11_is_cup_like(t) and cur_cat.startswith('國內聯賽'):
                sug_cat = CATEGORY_OPTIONS[2] if v11_is_intl(t) else CATEGORY_OPTIONS[3]
            new_cat = st.selectbox("賽事分類", CATEGORY_OPTIONS, index=CATEGORY_OPTIONS.index(sug_cat) if sug_cat in CATEGORY_OPTIONS else 0, key=f"v11_ts_cat_{t}")
            ls0, _src = v11_lookup_league_size(t, df_db)
            new_ls = st.number_input(f"隊伍數目 (目前自動填寫：{ls0}，來源：{_src})", 2, 40, int(ls0), key=f"v11_ts_ls_{t}")
            hr_mid = dt['Home_Rating'].map(_parse_rank_int)
            ar_mid = dt['Away_Rating'].map(_parse_rank_int)
            ls_v = pd.to_numeric(dt['League_Size'], errors='coerce').fillna(20) // 2
            n_def = int(((hr_mid == ls_v) & (ar_mid == ls_v)).sum())
            mark_na = st.checkbox(f"把排名為預設值的 {n_def} 張注單標記為「沒有聯賽排名」(盃賽/洲際賽適用)", value=v11_is_cup_like(t) and n_def > 0, key=f"v11_ts_na_{t}")
            if st.button("💾 套用到此賽事全部注單", key="v11_ts_apply", use_container_width=True):
                d = df_db.copy()
                mk = d['Tournament_Name'].astype(str).map(v11_clean_name) == t
                d.loc[mk, 'Tournament_Category'] = new_cat
                d.loc[mk, 'League_Size'] = int(new_ls)
                if mark_na:
                    hr2 = d.loc[mk, 'Home_Rating'].map(_parse_rank_int)
                    ar2 = d.loc[mk, 'Away_Rating'].map(_parse_rank_int)
                    lsv = pd.to_numeric(df_db.loc[mk, 'League_Size'], errors='coerce').fillna(20) // 2
                    sel_na = mk.copy()
                    sel_na.loc[mk] = ((hr2 == lsv) & (ar2 == lsv)).values
                    d.loc[sel_na, 'Home_Rating'] = 'NA'
                    d.loc[sel_na, 'Away_Rating'] = 'NA'
                v11_save_league_size(t, new_ls, silent=True)
                _v11_commit_db(d, f"✅ 已更新「{t}」的分類、隊伍數目及排名設定")


# ---------- 模型擂台：所有模型在同一測試期、完全樣本外比較 ----------
def v11_wf_rules(D, M, names, folds, thr, zmin=None):
    ret = D['ret'].values
    picks = []
    for d0, d1 in folds:
        tr = (D['date'] < d0).values
        te = ((D['date'] >= d0) & (D['date'] <= d1)).values
        if tr.sum() < 150 or te.sum() == 0:
            continue
        stt = v10_rule_stats(D, M, names, mask_rows=tr)
        if zmin is not None:
            for bt in V10_TYPES:
                tm = (D['bet_type'] == bt).values & tr
                n = stt[bt]['n'].astype(float)
                s = stt[bt]['sum']
                ss = ((M & tm[:, None]) * (ret[:, None] ** 2)).sum(axis=0)
                nn = np.maximum(n, 1.0)
                var = np.where(n > 1, (ss - s ** 2 / nn) / np.maximum(n - 1, 1.0), 1.0)
                se = np.sqrt(np.maximum(var, 1e-9) / nn)
                z = (s / nn) / se
                stt[bt]['shr'] = np.where(z >= zmin, stt[bt]['shr'], -9.0)
        best, bi = v10_pick(D, M, stt, te, thr)
        sel = np.where(te & (bi >= 0))[0]
        if len(sel) == 0:
            continue
        P = D.loc[sel].copy()
        P['shr'] = best[sel]
        picks.append(P.sort_values('shr', ascending=False).drop_duplicates(['key', 'bet_type']))
    return pd.concat(picks) if picks else pd.DataFrame(columns=list(D.columns) + ['shr'])


def v11_league_bias(D, folds, k=20.0, thr=0.0, minn=15):
    picks = []
    for d0, d1 in folds:
        tr = D[D['date'] < d0]
        te = D[(D['date'] >= d0) & (D['date'] <= d1)]
        if tr.empty or te.empty:
            continue
        g = tr.groupby(['tourn', 'bet_type', 'side'])['ret'].agg(['sum', 'size'])
        g['shr'] = (g['sum'] + k * V10_PRIOR_ROI) / (g['size'] + k)
        x = te.join(g, on=['tourn', 'bet_type', 'side'])
        x = x[(x['size'] >= minn) & (x['shr'] >= thr)]
        if len(x):
            picks.append(x.sort_values('shr', ascending=False).drop_duplicates(['key', 'bet_type']))
    return pd.concat(picks) if picks else pd.DataFrame(columns=list(D.columns) + ['shr'])


def _v11_stat(P):
    n = int(len(P))
    if n == 0:
        return {'n': 0, 'roi': 0.0, 'units': 0.0, 'hit': 0.0, 'ci': 0.0}
    r = P['ret'].astype(float)
    return {'n': n, 'roi': float(r.mean()), 'units': float(r.sum()), 'hit': float((r > 0).mean()),
            'ci': float(1.96 * r.std(ddof=1) / np.sqrt(n)) if n > 1 else 1.0}


def v11_arena(df_input):
    out = {'trained': False, 'msg': '', 'summary': [], 'by_type': [], 'by_league': [], 'active': {}, 'league': {},
           'type_stats': {}, 'n_matches': 0, 't0': None}
    mt = v9_build_match_table(df_input)
    if mt.empty or len(mt) < V9_MIN_MATCHES:
        out['msg'] = f"已結算比賽不足 {V9_MIN_MATCHES} 場，模型擂台暫未啟用 (所有建議只作模擬注)。"
        return out
    D = v10_build_dataset(mt)
    if D.empty or len(D) < 200:
        out['msg'] = '可用的賠率紀錄不足，模型擂台暫未啟用 (所有建議只作模擬注)。'
        return out
    D = D.reset_index(drop=True)
    D = v10_apply_resid(D, v10_fit_resid(D))
    names, M = v10_rule_matrix(D)
    folds = v9_folds(D['date'], n_folds=6, init_frac=0.4)
    if not folds:
        out['msg'] = '比賽日期不足以進行前進式回測。'
        return out
    t0 = folds[0][0]
    res = {'kp': v11_wf_rules(D, M, names, folds, V11_STRAT_THR['kp']),
           'kz': v11_wf_rules(D, M, names, folds, V11_STRAT_THR['kz'], zmin=V11_Z_MIN)}
    try:
        e9 = v9_train_engine_cached(df_input)
        o = e9.get('oof') if isinstance(e9, dict) else None
    except Exception:
        o = None
    tmap = dict(zip(mt['key'], mt['tourn']))
    if o is not None and len(o):
        o = o.copy()
        o['date'] = pd.to_datetime(o['date'])
        x = o[(o['ev'] >= V11_STRAT_THR['v9']) & (o['date'] >= t0)]
        if len(x):
            x = x.loc[x.groupby(['key', 'bet_type'])['ev'].idxmax()].copy()
            x['tourn'] = x['key'].map(tmap)
        res['v9'] = x
    else:
        res['v9'] = pd.DataFrame(columns=['key', 'date', 'bet_type', 'side', 'ret', 'tourn'])
    res['lb'] = v11_league_bias(D, folds)
    res['mkt'] = D[D['date'] >= t0]
    test_dates = D.loc[D['date'] >= t0, 'date']
    mid = test_dates.quantile(0.5) if len(test_dates) else t0
    leagues = sorted(t for t in D['tourn'].dropna().unique())
    for sk, P in res.items():
        s_all = _v11_stat(P)
        h1, h2 = _v11_stat(P[P['date'] < mid]), _v11_stat(P[P['date'] >= mid])
        out['summary'].append({'策略 / 模型': V11_STRAT_NAMES[sk], '樣本外注數': s_all['n'],
                               'ROI': f"{s_all['roi']*100:+.1f}%" if s_all['n'] else '-',
                               '95% 誤差範圍': f"±{s_all['ci']*100:.1f}%" if s_all['n'] > 1 else '-',
                               '單位盈虧 (U)': round(s_all['units'], 2), '命中率': f"{s_all['hit']*100:.1f}%" if s_all['n'] else '-',
                               '前半段 ROI': f"{h1['roi']*100:+.1f}% ({h1['n']})" if h1['n'] else '-',
                               '後半段 ROI': f"{h2['roi']*100:+.1f}% ({h2['n']})" if h2['n'] else '-'})
        out['type_stats'][sk] = {}
        row_t = {'策略 / 模型': V11_STRAT_NAMES[sk]}
        for bt in V10_TYPES:
            Pb = P[P['bet_type'] == bt]
            sb, b1, b2 = _v11_stat(Pb), _v11_stat(Pb[Pb['date'] < mid]), _v11_stat(Pb[Pb['date'] >= mid])
            ok = (sk in V11_STRAT_THR and sb['n'] >= V11_ACT_MIN_N and sb['roi'] >= V11_ACT_MIN_ROI and
                  b1['n'] >= V11_ACT_HALF_MIN_N and b2['n'] >= V11_ACT_HALF_MIN_N and b1['roi'] > 0 and b2['roi'] > 0)
            out['active'].setdefault(sk, {})[bt] = bool(ok)
            out['type_stats'][sk][bt] = {'n': sb['n'], 'roi': sb['roi'], 'h1': b1, 'h2': b2}
            row_t[bt] = (f"{sb['roi']*100:+.1f}% ({sb['n']})" + (' ✅' if ok else '')) if sb['n'] else '-'
        out['by_type'].append(row_t)
        row_l = {'策略 / 模型': V11_STRAT_NAMES[sk]}
        out['league'][sk] = {}
        for L in leagues:
            sl = _v11_stat(P[P['tourn'] == L]) if 'tourn' in P.columns else _v11_stat(P.iloc[0:0])
            out['league'][sk][L] = {'n': sl['n'], 'roi': sl['roi']}
            row_l[L] = f"{sl['roi']*100:+.0f}% ({sl['n']})" if sl['n'] else '-'
        out['by_league'].append(row_l)
    out['n_matches'] = len(mt)
    out['t0'] = str(pd.Timestamp(t0).date())
    out['trained'] = True
    n_act = sum(v for d_ in out['active'].values() for v in d_.values())
    out['msg'] = (f"模型擂台：{len(mt)} 場比賽，測試期由 {out['t0']} 開始 (之前的比賽只用作訓練)。"
                  + (f"目前有 {n_act} 個『策略 × 盤口類型』通過驗證，可以實際落注。" if n_act else
                     "目前沒有任何策略通過驗證 (樣本外 ≥ 50 注、ROI ≥ +2%、前後兩段都賺錢)，自動模式只會提供模擬注。"))
    return out


@st.cache_data(max_entries=8, show_spinner="🏟 V11 模型擂台：所有模型在同一測試期進行樣本外比較 (首次較慢)...")
def v11_arena_cached(df_input):
    return v11_arena(df_input)


def v11_get_arena(df_db):
    df_settled = df_db[df_db['Status'] == 'Settled'].copy() if not df_db.empty else df_db.copy()
    if df_settled.empty:
        return {'trained': False, 'msg': '尚無已結算數據。', 'active': {}, 'league': {}, 'summary': [], 'by_type': [], 'by_league': []}
    return v11_arena_cached(v9_engine_input(df_settled))


def v11_league_match_count(df_db, tourn):
    if df_db is None or df_db.empty:
        return 0
    t = v11_clean_name(tourn)
    d = df_db[(df_db['Status'] == 'Settled') & (df_db['Tournament_Name'].astype(str).map(v11_clean_name) == t)]
    if d.empty:
        return 0
    return int(pd.DataFrame({'m': d['Match'].astype(str).map(v11_clean_name), 'd': v10_md_series(d)}).drop_duplicates().shape[0])


def render_v11_arena_panel():
    ar = v11_get_arena(st.session_state.df_db)
    st.info(ar.get('msg', ''))
    if not ar.get('trained'):
        return
    st.markdown("##### 🏟 整體比較 (同一測試期、完全樣本外：每一折只用之前日期訓練，再預測之後日期)")
    st.dataframe(pd.DataFrame(ar['summary']), use_container_width=True, hide_index=True)
    st.caption("95% 誤差範圍：例如 ROI −2% ±8% 代表真實回報可能在 −10% 至 +6% 之間，數百注的結果仍有很大運氣成分。"
               "「前半段 / 後半段」把測試期分成兩半，真正有效的策略兩段都應該賺錢。")
    st.markdown("##### 🎯 各盤口類型 (✅ = 通過驗證，自動模式會實際落注)")
    st.dataframe(pd.DataFrame(ar['by_type']), use_container_width=True, hide_index=True)
    st.caption(f"通過驗證條件：樣本外 ≥ {V11_ACT_MIN_N} 注、ROI ≥ +{V11_ACT_MIN_ROI*100:.0f}%，而且前後兩段各 ≥ {V11_ACT_HALF_MIN_N} 注並且都賺錢。"
               "每次新增賽果後會自動重新驗證，策略可能因此啟用或停用。")
    st.markdown("##### 🌏 各聯賽 (ROI 及注數)")
    st.dataframe(pd.DataFrame(ar['by_league']).set_index('策略 / 模型').T, use_container_width=True)
    st.caption(f"個別聯賽通常只有 10~40 注，誤差範圍超過 ±30%，不足以判斷「哪個模型適合哪個聯賽」。系統因此不會為個別聯賽挑選模型 "
               f"(「聯賽專屬偏差模型」正是這樣做，結果最差)，只用作熔斷：某聯賽樣本外 ≥ {V11_LG_MIN_N} 注且 ROI ≤ {V11_LG_MAX_ROI*100:.0f}% 時暫停該聯賽的實注。")


def v9_get_settings():
    return {
        'min_ev': float(st.session_state.get('v9_min_ev', V10_SETTING_DEFAULTS['v9_min_ev'])) / 100.0,
        'kelly': float(st.session_state.get('v9_kelly', 0.25)),
        'max_pct': float(st.session_state.get('v9_max_pct', 3.0)) / 100.0,
        'hd_min200': bool(st.session_state.get('v9_hd_min200', False)),
        'override_gate': bool(st.session_state.get('v9_override_gate', False)),
        'max_picks': int(st.session_state.get('v9_max_picks', 1)),
        'mode': str(st.session_state.get('v10_mode', V11_MODE_AUTO)),
        'new_league_paper': bool(st.session_state.get('v11_new_league_paper', True)),
        'league_breaker': bool(st.session_state.get('v11_league_breaker', True)),
    }


# ==========================================
# 4. 注單載入與修改輔助邏輯 (新增/覆蓋)
# ==========================================
# 賽前建檔表單所有固定 key 的 widget：載入注單時必須清除，
# 否則 Streamlit widget 會沿用舊值，導致已載入的資料無法顯示在表單上
EDIT_FORM_WIDGET_KEYS = [
    'sel_tourn', 'txt_tourn', 'sel_cat', 'match_date_input', 'sh', 'txt_h_team', 'sa', 'txt_a_team',
    'league_size', 'h_rank', 'a_rank', 'sel_hr', 'sel_ar',
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6',
    'pre_paste_hd', 'pre_paste_ou', 'pre_paste_cr', 'pre_odds_editor',
    'sys_multi', 'usr_multi'
]

def reset_edit_form_widgets():
    """清除賽前建檔表單 widget 的 session_state key，
    令下一個 rerun 時所有表單欄位重新以 edit_* 預設值初始化，正確顯示已載入的注單資料"""
    for k in EDIT_FORM_WIDGET_KEYS:
        st.session_state.pop(k, None)
    for k in list(st.session_state.keys()):
        if k.startswith('s_stk_') or k.startswith('u_stk_'):
            st.session_state.pop(k, None)
    # 清除貼上區簽名，避免 render_odds_section 誤重新解析覆蓋已載入的賠率歷史
    st.session_state.pop('pre_last_paste_sig', None)
    st.session_state.pop('_v11_ls_marker', None)
    st.session_state.pop('v11_no_rank', None)

def load_bet_to_edit(bet_id):
    """將雲端數據庫中的注單載入至 session_state 供使用者重新修改"""
    match_df = st.session_state.df_db[st.session_state.df_db['ID'] == bet_id]
    if match_df.empty:
        return
    row = match_df.iloc[0]
    st.session_state.editing_bet_id = str(bet_id)
    
    st.session_state.edit_t_name = str(row.get('Tournament_Name', '')) if pd.notna(row.get('Tournament_Name')) else ''
    st.session_state.edit_t_cat = str(row.get('Tournament_Category', CATEGORY_OPTIONS[0])) if pd.notna(row.get('Tournament_Category')) else CATEGORY_OPTIONS[0]
    _match_date_str = get_row_match_date(row)
    st.session_state.edit_match_date = _match_date_str
    st.session_state.edit_h_team = str(row.get('Home_Team', '')) if pd.notna(row.get('Home_Team')) else ''
    st.session_state.edit_a_team = str(row.get('Away_Team', '')) if pd.notna(row.get('Away_Team')) else ''
    # 聯賽排名：新數據為數字排名，舊數據為字母評級 (自動換算為約略排名)
    _league_size = get_league_size(row)
    st.session_state.edit_league_size = _league_size
    st.session_state.edit_h_rank = rating_to_rank(row.get('Home_Rating'), _league_size)
    st.session_state.edit_a_rank = rating_to_rank(row.get('Away_Rating'), _league_size)
    st.session_state.edit_h_rating_raw = str(row.get('Home_Rating', '')) if pd.notna(row.get('Home_Rating')) else ''
    st.session_state.edit_a_rating_raw = str(row.get('Away_Rating', '')) if pd.notna(row.get('Away_Rating')) else ''
    
    hw, hd, hl = parse_form_str(row.get('Home_Form', '3W1D1L'))
    aw, ad, al = parse_form_str(row.get('Away_Form', '2W2D1L'))
    st.session_state.edit_hw, st.session_state.edit_hd, st.session_state.edit_hl = hw, hd, hl
    st.session_state.edit_aw, st.session_state.edit_ad, st.session_state.edit_al = aw, ad, al
    
    try:
        oh = json.loads(str(row.get('Odds_History', '[]')))
        if isinstance(oh, list) and len(oh) > 0:
            st.session_state.odds_history = oh
            st.session_state.inplay_odds_history = oh
        else:
            raise ValueError("empty")
    except:
        # 賠率歷史缺失或格式無效時重設為預設值，避免顯示上一注單的賠率歷史
        st.session_state.odds_history = [{"id": 0, "type": "讓球", "record_time": "", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}]
        st.session_state.inplay_odds_history = list(st.session_state.odds_history)
    
    # 清除上一注單的舊分析結果，避免表單顯示新注單但分析結果仍是舊的
    st.session_state.show_analysis = False
    st.session_state.pop('analysis_result', None)
        
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
    
    # 清除表單 widget 舊值，確保下一個 rerun 正確顯示已載入的資料
    # (包括賽事/球隊/分類/日期/近5場狀態/賠率走勢表/投注選單等所有欄位)
    reset_edit_form_widgets()

def clear_edit_mode():
    """清除修改模式狀態"""
    st.session_state.editing_bet_id = None
    keys_to_del = [k for k in st.session_state.keys() if k.startswith('edit_')]
    for k in keys_to_del:
        del st.session_state[k]

# ==========================================
# 4b. 已完成注單資料修補工具 (開啟已儲存賽事更新數據)
# ==========================================
def is_missing_match_date(v):
    """判斷 Match_Date 是否缺失 (None/NaN/空字串)"""
    if pd.isna(v):
        return True
    s = str(v).strip()
    return s == '' or s.lower() in ('none', 'nan', 'nat')

# 已完成注單表格編輯器中的數值欄位 (轉為數字型別方便直接編輯)
SETTLED_EDITOR_NUMCOLS = [
    'League_Size', 'InPlay_Minute', 'Home_DA', 'Away_DA', 'Home_SoT', 'Away_SoT', 'Home_SoFF', 'Away_SoFF',
    'Home_Red', 'Away_Red', 'Home_Sub', 'Away_Sub', 'Home_Possession', 'Away_Possession',
    'Home_Goal', 'Away_Goal', 'Home_Corner', 'Away_Corner',
    'Home_Goal_Conversion', 'Away_Goal_Conversion', 'Home_Firepower', 'Away_Firepower',
    'Home_Corner_Eff', 'Away_Corner_Eff',
    'Initial_Line', 'Initial_Odds', 'System_Stake', 'User_Stake',
    'System_Profit', 'User_Profit', 'Unit_Profit', 'System_Payout', 'User_Payout'
]

@st.dialog("🗂 已完成注單資料修改 (可直接編輯任何欄位)", width="large")
def edit_settled_bets_dialog(db_file, db_table):
    """開啟資料庫中已儲存的注單 (含已完成 Settled) 進行更新：可直接在表格修改任何欄位，不限於比賽日期"""
    st.caption("此工具可開啟已完成 (Settled) 的注單，直接在表格中修改任何欄位 (賽事、球隊、排名、盤口、賠率、注碼、賽果、盈虧等)。ID 欄位為唯讀，確保正確對應原注單。修改後按儲存即自動同步雲端數據庫。")
    
    settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
    if settled.empty:
        st.info("目前沒有已完成的注單。")
        return
    
    n_missing = int(settled['Match_Date'].apply(is_missing_match_date).sum())
    
    tab_match, tab_id = st.tabs(["📋 按比賽場次表格編輯", "🔍 按注單 ID 單筆編輯"])
    
    with tab_match:
        st.write(f"目前共有 **{len(settled)}** 張已完成注單，其中 **{n_missing}** 張缺少比賽日期。")
        
        # 依「賽事 + 對賽 + 記錄日期」分組，避免不同日期的同組合對賽被誤改
        settled_groups = settled.copy()
        settled_groups['_Record_Date'] = pd.to_datetime(settled_groups['Date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('無記錄日期')
        
        groups = settled_groups.fillna({'Tournament_Name': '', 'Match': ''}).groupby(['Tournament_Name', 'Match', '_Record_Date'], dropna=False)
        group_indices = []
        group_opts = []
        for (tname, match, record_date), g in groups:
            group_indices.append(list(g.index))
            existing_dates = sorted({str(d)[:10] for d in g['Match_Date'] if not is_missing_match_date(d)})
            miss_cnt = int(g['Match_Date'].apply(is_missing_match_date).sum())
            date_disp = "、".join(existing_dates) if existing_dates else "未設定"
            group_opts.append(f"{tname} | {match} | 記錄日期: {record_date} | 比賽日期: {date_disp} | 缺日期: {miss_cnt}/{len(g)} 張")
        
        sel_group = st.selectbox("選擇要修改的比賽場次", group_opts, key="sbd_group_sel")
        g_idx = group_opts.index(sel_group)
        selected_indices = group_indices[g_idx]
        
        _render_settled_editor(settled.loc[selected_indices], editor_key=f"sbd_match_editor_{g_idx}",
                               save_key="sbd_save_group", db_file=db_file, db_table=db_table,
                               button_label="💾 儲存此場次所有修改")
        
        st.divider()
        with st.expander("⚡ 一鍵智能補填：以記錄日期 (Date) 填補所有缺失的比賽日期"):
            st.caption("系統會將所有缺少 Match_Date 的注單，自動以該注單的記錄日期 (Date 欄位) 作為比賽日期填入。")
            if st.button("⚡ 執行一鍵補填", key="sbd_auto_fill"):
                miss_mask = (st.session_state.df_db['Status'] == 'Settled') & st.session_state.df_db['Match_Date'].apply(is_missing_match_date)
                miss_idx = st.session_state.df_db.index[miss_mask]
                if len(miss_idx) == 0:
                    st.toast("目前沒有缺少比賽日期的注單。", icon="✅")
                else:
                    for idx in miss_idx:
                        dv = st.session_state.df_db.at[idx, 'Date']
                        parsed = pd.to_datetime(dv, errors='coerce')
                        fill_val = parsed.strftime('%Y-%m-%d') if pd.notna(parsed) else get_hkt_now().strftime('%Y-%m-%d')
                        st.session_state.df_db.at[idx, 'Match_Date'] = fill_val
                    save_db(st.session_state.df_db, db_file, db_table)
                    st.toast(f"✅ 已智能補填 {len(miss_idx)} 張注單的比賽日期！", icon="⚡")
                    st.rerun()
    
    with tab_id:
        def _id_opt(r):
            cur = str(r.get('Match_Date', ''))[:10] if not is_missing_match_date(r.get('Match_Date')) else "❌未設定"
            return f"{r['ID']} | {r.get('Match', '')} | {r.get('Bet_Type', '')} | 日期: {cur}"
        
        id_opts = [_id_opt(r) for _, r in settled.iterrows()]
        sel_id_str = st.selectbox("選擇要修改的注單 (ID)", id_opts, key="sbd_id_sel")
        target_id = str(sel_id_str).split(" | ")[0]
        
        target_rows = settled[settled['ID'] == target_id]
        if target_rows.empty:
            st.warning("找不到所選注單。")
            return
        
        _render_settled_editor(target_rows, editor_key=f"sbd_id_editor_{target_id}",
                               save_key="sbd_save_single", db_file=db_file, db_table=db_table,
                               button_label="💾 儲存此注單的修改")

def _render_settled_editor(df_rows, editor_key, save_key, db_file, db_table, button_label):
    """渲染已完成注單的完整欄位表格編輯器 (可修改任何欄位)，並提供儲存功能"""
    # 使用資料庫實際欄位 (保留所有欄位，不遺漏)
    cols = [c for c in st.session_state.df_db.columns if c in df_rows.columns]
    edit_df = df_rows[cols].copy()
    
    # 數值欄位轉為數字型別方便直接編輯；無法轉換的儲存格保留原值 (不會被靜默變成空值)
    for col in SETTLED_EDITOR_NUMCOLS:
        if col in edit_df.columns:
            conv = pd.to_numeric(edit_df[col], errors='coerce')
            bad_mask = conv.isna() & edit_df[col].notna() & (edit_df[col].astype(str).str.strip() != '') & (edit_df[col].astype(str).str.lower() != 'nan')
            if bad_mask.any():
                edit_df.loc[~bad_mask, col] = conv[~bad_mask]
            else:
                edit_df[col] = conv
    
    # 編輯器 key 加入修訂版號：儲存後版號 +1，避免切換場次時沿用舊編輯狀態
    _rev = int(st.session_state.get('sbd_editor_rev', 0))
    
    st.markdown("##### 📝 直接編輯注單資料 (可修改任何欄位)")
    edited = st.data_editor(
        edit_df,
        num_rows="fixed",
        use_container_width=True,
        key=f"{editor_key}_r{_rev}",
        column_config={
            'ID': st.column_config.TextColumn('ID (唯讀)', disabled=True),
            'Status': st.column_config.SelectboxColumn('Status', options=['Open', 'Settled'], required=True),
            'Bet_Type': st.column_config.SelectboxColumn('Bet_Type', options=['讓球', '入球大小', '角球大小', '讓球 (即場)', '入球大小 (即場)', '角球大小 (即場)'], required=False),
            'Selection': st.column_config.SelectboxColumn('Selection', options=['Home', 'Away', 'Over', 'Under'], required=False),
        }
    )
    
    recompute = st.checkbox("💾 儲存時自動重算盈虧欄位 (僅重算有改動賠率/注碼/賽果的注單)", value=True, key=f"{save_key}_recompute_r{_rev}",
                           help="只會重算「有修改」結算相關欄位 (賠率/注碼/入球/角球/方向) 且已有賽果數據的注單，其他注單的盈虧欄位原封不動。")
    
    if st.button(button_label, type="primary", use_container_width=True, key=save_key):
        _save_edited_settled_bets(edited, df_rows[cols], db_file, db_table, recompute=recompute)

def _cell_equal(a, b, numeric=False):
    """比較兩個儲存格值是否相等 (數值欄位容許字串/數字型別差異)"""
    try:
        a_na = pd.isna(a)
        b_na = pd.isna(b)
        if a_na and b_na:
            return True
        if a_na or b_na:
            return False
        if numeric:
            return abs(float(a) - float(b)) < 1e-9
        sa, sb = str(a), str(b)
        if sa in ('', 'nan', 'None') and sb in ('', 'nan', 'None'):
            return True
        return sa == sb
    except (TypeError, ValueError):
        return str(a) == str(b)

# 會影響結算結果的欄位：只有這些欄位有改動時才需要重算盈虧
SETTLEMENT_DRIVING_COLS = ['Bet_Type', 'Selection', 'Initial_Line', 'Initial_Odds',
                           'System_Stake', 'User_Stake',
                           'Home_Goal', 'Away_Goal', 'Home_Corner', 'Away_Corner', 'Status']

def _save_edited_settled_bets(edited, original_df, db_file, db_table, recompute=True):
    """將表格編輯結果按 ID 寫回數據庫：只寫入有改動的儲存格。
    勾選自動重算時：只有「結算相關欄位有改動」且「該盤口所需的賽果數據齊全」的注單才會重算盈虧；
    若結算相關欄位有改動但賽果數據缺失，該等改動會被拒絕 (不會以 0 代替缺失數據胡亂重算)，
    並清楚列出被拒絕的注單。任何計算錯誤都會中止儲存並顯示原因。"""
    df_db = st.session_state.df_db
    df_work = df_db.copy()
    updated_ids = []
    rejected_ids = []
    
    def _market_required_scores(bet_type):
        bt = str(bet_type).replace(' (即場)', '').strip()
        if bt == '角球大小':
            return ['Home_Corner', 'Away_Corner']
        return ['Home_Goal', 'Away_Goal']
    
    for _, erow in edited.iterrows():
        bet_id = str(erow.get('ID', ''))
        if not bet_id or bet_id in ('', 'nan', 'None'):
            continue
        mask = df_work['ID'] == bet_id
        if mask.sum() == 0:
            continue
        orig_rows = original_df[original_df['ID'] == bet_id]
        if orig_rows.empty:
            continue
        orig_row = orig_rows.iloc[0]
        
        changed_cols = []
        for col in edited.columns:
            if col == 'ID':
                continue
            is_num = col in SETTLED_EDITOR_NUMCOLS
            if not _cell_equal(erow[col], orig_row.get(col), numeric=is_num):
                changed_cols.append(col)
        
        if not changed_cols:
            continue
        
        settle_changed = set(changed_cols) & set(SETTLEMENT_DRIVING_COLS)
        
        # 自動重算開啟時：結算相關欄位有改動但該盤口所需賽果數據缺失 → 拒絕該等改動
        if recompute and settle_changed:
            cur_row = df_work[mask].iloc[0]
            if str(cur_row.get('Status', '')) == 'Settled':
                required_scores = _market_required_scores(cur_row.get('Bet_Type', ''))
                if any(pd.isna(cur_row.get(c)) for c in required_scores):
                    rejected_ids.append(bet_id)
                    continue
        
        # 寫入有改動的儲存格 (僅改動部分)
        for col in changed_cols:
            df_work.loc[mask, col] = erow[col]
        updated_ids.append(bet_id)
        
        # 僅當結算相關欄位有改動時才重算盈虧
        if recompute and settle_changed:
            row = df_work[mask].iloc[0]
            if str(row.get('Status', '')) != 'Settled':
                continue
            score_cols = ['Home_Goal', 'Away_Goal', 'Home_Corner', 'Away_Corner']
            if any(pd.isna(row.get(c)) for c in score_cols):
                continue
            try:
                hg = int(float(row.get('Home_Goal')))
                ag = int(float(row.get('Away_Goal')))
                hc = int(float(row.get('Home_Corner')))
                ac = int(float(row.get('Away_Corner')))
                sys_profit, usr_profit, sys_payout, usr_payout, unit_profit, res_label, _diff = calculate_settlement(
                    row['Bet_Type'], row['Selection'], row['Initial_Line'], row['Initial_Odds'],
                    row['System_Stake'], row['User_Stake'], hg, ag, hc, ac
                )
            except Exception as e:
                st.error(f"❌ 注單 {bet_id} 盈虧重算失敗：{e}。已中止儲存，資料庫未有任何改動。")
                return
            df_work.loc[mask, 'System_Profit'] = sys_profit
            df_work.loc[mask, 'User_Profit'] = usr_profit
            df_work.loc[mask, 'System_Payout'] = sys_payout
            df_work.loc[mask, 'User_Payout'] = usr_payout
            df_work.loc[mask, 'Unit_Profit'] = unit_profit
            df_work.loc[mask, 'Result_Label'] = res_label
    
    if rejected_ids:
        st.error(f"❌ 以下注單的結算相關欄位修改被拒絕 (缺少該盤口所需的賽果數據，系統不會以 0 代替胡亂重算)：{', '.join(rejected_ids)}。其他改動已正常儲存；如確定要手動修改該等欄位，請取消勾選「自動重算盈虧」後重試。")
    
    if not updated_ids:
        if not rejected_ids:
            st.toast("沒有偵測到任何改動，資料庫保持不變。", icon="✅")
        return
    
    # 驗證通過才提交：一次性寫回，避免部分寫入
    st.session_state.df_db = df_work
    save_db(df_work, db_file, db_table)
    st.session_state.sbd_editor_rev = int(st.session_state.get('sbd_editor_rev', 0)) + 1
    msg = f"✅ 已更新 {len(updated_ids)} 張注單的改動欄位並同步雲端！"
    st.toast(msg, icon="🗂")
    st.rerun()

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
        
        # 修正：分別計算系統/用家單位利潤（僅計算有實際下注的注單）
        sys_staked = show_df[pd.to_numeric(show_df['System_Stake'], errors='coerce') > 0]
        usr_staked = show_df[pd.to_numeric(show_df['User_Stake'], errors='coerce') > 0]
        sys_unit_profit = pd.to_numeric(sys_staked['Unit_Profit'], errors='coerce').sum()
        usr_unit_profit = pd.to_numeric(usr_staked['Unit_Profit'], errors='coerce').sum()
        
        # 顯示盈虧修正指標
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("原始單位利潤 (全部)", f"{total_unit:.2f} U")
        sm2.metric("系統單位利潤 (Stake>0)", f"{sys_unit_profit:.2f} U")
        sm3.metric("用家單位利潤 (Stake>0)", f"{usr_unit_profit:.2f} U")
        sm4.metric("系統淨盈虧", f"${total_sys_profit:,.2f}")
        st.caption("💡 原始單位利潤包含未下注注單的賠率結果；系統/用家單位利潤僅計算有實際下注 (Stake>0) 的注單，與全局模型一致。")
        st.divider()

        summary_data = {col: None for col in show_df.columns}
        if 'ID' in summary_data: summary_data['ID'] = "TOTAL (總計)"
        if 'System_Profit' in summary_data: summary_data['System_Profit'] = round(total_sys_profit, 2)
        if 'User_Profit' in summary_data: summary_data['User_Profit'] = round(total_user_profit, 2)
        if 'Unit_Profit' in summary_data: summary_data['Unit_Profit'] = round(sys_unit_profit, 2)
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

def parse_dt_for_comparison(dt_str, fallback_idx, fmt=None, ref=None):
    """解析日期時間字串用於版本最新比對 (V10：支援 日-月 / 月-日 自動判斷)，若無有效時間則以 fallback_idx 為準"""
    if not dt_str or not str(dt_str).strip():
        return (datetime.min, fallback_idx)
    if ref is None:
        ref = get_hkt_now().replace(tzinfo=None)
    dt = v10_parse_time(str(dt_str).strip(), fmt, ref)
    return (dt if dt is not None else datetime.min, fallback_idx)

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
    
    # 僅在貼上內容有變化時才重新解析，避免不相關的 rerun (例如載入注單修改)
    # 覆蓋已載入或手動編輯的賠率歷史
    _paste_sig = f"{raw_hd.strip()}||{raw_ou.strip()}||{raw_cr.strip()}"
    _last_sig = st.session_state.get(f"{prefix}_last_paste_sig", None)
    if _paste_sig != _last_sig:
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
        st.session_state[f"{prefix}_last_paste_sig"] = _paste_sig
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
    """V9 建議注碼 (分數 Kelly + 熔斷 + EV 門檻)，參數設定見側邊欄「V9 投注策略與風控設定」"""
    return v9_calc_stake(cand, sys_bankroll, sys_max_stake, v9_get_settings())

# ==========================================
# 5.5 即場投注輔助函式 (賠率連動與機率計算)
# ==========================================
MARGIN_CONST = 1.085

def _calc_lower_from_upper(upper, margin=MARGIN_CONST):
    """根據馬會抽水公式由大賠率算小賠率: 1 / (margin - 1/大賠率)"""
    if upper and upper > 1.0:
        denom = margin - 1.0 / upper
        if denom > 0:
            return round(1.0 / denom, 2)
    return 1.90

def auto_calc_lower(prefix):
    """當大/上賠率變動時，自動計算小/下賠率並重設手動旗標"""
    upper = st.session_state.get(f"{prefix}_upper", 1.90)
    st.session_state[f"{prefix}_lower"] = _calc_lower_from_upper(upper)
    st.session_state[f"{prefix}_manual"] = False

def mark_manual_lower(prefix):
    """當小/下賠率被手動修改時，標記為手動模式"""
    st.session_state[f"{prefix}_manual"] = True

def calc_live_margin_str(upper, lower):
    """計算抽水百分比字串"""
    try:
        if upper > 0 and lower > 0:
            margin = (1.0 / upper) + (1.0 / lower)
            pct = (margin - 1.0) * 100.0
            return f"{pct:.2f}% (margin: {margin:.3f})"
        return "8.50% (margin: 1.085)"
    except:
        return "8.50% (margin: 1.085)"

def calc_inplay_base_prob(bet_type, selection, line, h_g, a_g, h_c, a_c, ip_minute, h_firepower, a_firepower, h_corner_eff, a_corner_eff):
    """結合剩餘時間與攻勢危險度計算即場調整後的基礎機率"""
    remaining_min = max(0, 90 - ip_minute)
    time_factor = remaining_min / 90.0

    if bet_type == "讓球":
        if selection == "Home":
            current_diff = h_g + line - a_g
            if current_diff > 0:
                base = 0.7 + min(0.2, current_diff * 0.05)
            elif current_diff == 0:
                base = 0.5
            else:
                base = 0.3 - min(0.2, abs(current_diff) * 0.05)
            base += (h_firepower - a_firepower) * 0.0003 * time_factor
        else:
            current_diff = a_g - line - h_g
            if current_diff > 0:
                base = 0.7 + min(0.2, current_diff * 0.05)
            elif current_diff == 0:
                base = 0.5
            else:
                base = 0.3 - min(0.2, abs(current_diff) * 0.05)
            base += (a_firepower - h_firepower) * 0.0003 * time_factor
    elif bet_type == "入球大小":
        total_goals = h_g + a_g
        total_firepower = h_firepower + a_firepower
        if selection == "Over":
            if total_goals > line:
                base = 0.95
            else:
                gap = line - total_goals
                scoring_rate = min(1.0, total_firepower / 100.0)
                base = min(0.85, 0.3 + scoring_rate * time_factor * 0.5 - gap * 0.1)
        else:
            if total_goals > line:
                base = 0.05
            else:
                gap = line - total_goals
                scoring_rate = min(1.0, total_firepower / 100.0)
                over_prob = min(0.85, 0.3 + scoring_rate * time_factor * 0.5 - gap * 0.1)
                base = 1 - over_prob
    elif bet_type == "角球大小":
        total_corners = h_c + a_c
        total_corner_eff = h_corner_eff + a_corner_eff
        if selection == "Over":
            if total_corners > line:
                base = 0.95
            else:
                gap = line - total_corners
                corner_rate = min(1.0, total_corner_eff / 100.0)
                base = min(0.85, 0.3 + corner_rate * time_factor * 0.5 - gap * 0.08)
        else:
            if total_corners > line:
                base = 0.05
            else:
                gap = line - total_corners
                corner_rate = min(1.0, total_corner_eff / 100.0)
                over_prob = min(0.85, 0.3 + corner_rate * time_factor * 0.5 - gap * 0.08)
                base = 1 - over_prob
    else:
        base = 0.5

    return max(0.05, min(0.95, base))

# ==========================================
# 5.6 時間維度盈虧分析輔助函式
# ==========================================
PERIOD_LABELS = {
    'D': '每日',
    'W': '每周',
    'Q': '每季',
    'Y': '每年',
}
PERIOD_FREQ = {
    'D': 'D',
    'W': 'W',
    'Q': 'Q',
    'Y': 'Y',
}

def get_row_match_date(row):
    """安全取得比賽日期：Match_Date -> Date -> 今天"""
    md = row.get('Match_Date', '')
    if pd.notna(md) and str(md).strip():
        return str(md)[:10]
    dv = row.get('Date', '')
    parsed = pd.to_datetime(dv, errors='coerce')
    if pd.notna(parsed):
        return parsed.strftime('%Y-%m-%d')
    return get_hkt_now().strftime('%Y-%m-%d')

@st.cache_data(max_entries=64)
def compute_period_pnl(df, profit_col, date_col='Match_Date', period='D'):
    """按指定時間維度分組計算各期淨盈虧，回傳 DataFrame (period_label, total)
    優先使用 Match_Date，舊資料無此欄位時退回 Date。"""
    if df.empty:
        return pd.DataFrame(columns=['period', profit_col])
    tmp = df.copy()
    # 優先使用 Match_Date，若不存在或為空則退回 Date
    if date_col in tmp.columns:
        tmp['_dt'] = pd.to_datetime(tmp[date_col], errors='coerce')
        # 對於 Match_Date 為空的記錄，退回 Date
        if 'Date' in tmp.columns:
            date_fallback = pd.to_datetime(tmp['Date'], errors='coerce')
            tmp['_dt'] = tmp['_dt'].fillna(date_fallback)
    elif 'Date' in tmp.columns:
        tmp['_dt'] = pd.to_datetime(tmp['Date'], errors='coerce')
    else:
        return pd.DataFrame(columns=['period', profit_col])
    tmp = tmp.dropna(subset=['_dt'])
    tmp[profit_col] = pd.to_numeric(tmp[profit_col], errors='coerce').fillna(0.0)
    if tmp.empty:
        return pd.DataFrame(columns=['period', profit_col])
    freq = PERIOD_FREQ.get(period, 'D')
    tmp['_period'] = tmp['_dt'].dt.to_period(freq).astype(str)
    grouped = tmp.groupby('_period')[profit_col].sum().reset_index()
    grouped.columns = ['period', profit_col]
    grouped = grouped.sort_values('period').reset_index(drop=True)
    return grouped

@st.cache_data(max_entries=32)
def compute_period_pnl_table(df, sys_col='System_Profit', usr_col='User_Profit', date_col='Match_Date'):
    """計算全部四個時間維度的系統與用家淨盈虧，回傳 dict"""
    result = {}
    for period in ['D', 'W', 'Q', 'Y']:
        sys_df = compute_period_pnl(df, sys_col, date_col, period)
        usr_df = compute_period_pnl(df, usr_col, date_col, period)
        if not sys_df.empty:
            sys_total = sys_df[sys_col].iloc[-1]  # 最新一期
            sys_all = sys_df[sys_col].sum()
        else:
            sys_total = 0.0
            sys_all = 0.0
        if not usr_df.empty:
            usr_total = usr_df[usr_col].iloc[-1]
            usr_all = usr_df[usr_col].sum()
        else:
            usr_total = 0.0
            usr_all = 0.0
        result[period] = {
            'sys_latest': round(float(sys_total), 2),
            'usr_latest': round(float(usr_total), 2),
            'sys_all': round(float(sys_all), 2),
            'usr_all': round(float(usr_all), 2),
            'sys_df': sys_df,
            'usr_df': usr_df,
        }
    return result

def render_pnl_chart_section(df_settled, key_prefix, title_prefix=""):
    """渲染可選擇系列的盈虧折線圖區塊"""
    if df_settled.empty:
        return
    
    # 預先計算所有時間序列
    series_data = {}
    for period in ['D', 'W', 'Q', 'Y']:
        sys_df = compute_period_pnl(df_settled, 'System_Profit', 'Match_Date', period)
        usr_df = compute_period_pnl(df_settled, 'User_Profit', 'Match_Date', period)
        if not sys_df.empty:
            series_data[f"系統{PERIOD_LABELS[period]}盈虧"] = sys_df.set_index('period')['System_Profit']
        if not usr_df.empty:
            series_data[f"用家{PERIOD_LABELS[period]}盈虧"] = usr_df.set_index('period')['User_Profit']
    
    if not series_data:
        st.caption("暫無可用於繪製折線圖的數據。")
        return
    
    st.markdown(f"#### 📈 {title_prefix}盈虧折線圖" if title_prefix else "#### 📈 盈虧折線圖")
    
    all_series = list(series_data.keys())
    
    # 預設選擇系統每日及用家每日
    default_series = [s for s in all_series if '每日' in s]
    if not default_series:
        default_series = all_series[:2] if len(all_series) >= 2 else all_series[:1]
    
    selected_series = st.multiselect(
        f"選擇要顯示的盈虧線條",
        all_series,
        default=default_series,
        key=f"{key_prefix}_series_select"
    )
    
    if not selected_series:
        st.caption("請至少選擇一條線條以顯示折線圖。")
        return
    
    # 依時間頻率分組渲染
    freq_groups = {}
    for s in selected_series:
        for period_key, period_label in PERIOD_LABELS.items():
            if period_label in s:
                freq = PERIOD_FREQ[period_key]
                if freq not in freq_groups:
                    freq_groups[freq] = []
                freq_groups[freq].append(s)
                break
    
    for freq, series_list in freq_groups.items():
        period_label = next((v for k, v in PERIOD_LABELS.items() if PERIOD_FREQ[k] == freq), freq)
        # 合併同頻率的系列
        merged = None
        for s in series_list:
            if s in series_data:
                if merged is None:
                    merged = series_data[s].to_frame(name=s)
                else:
                    merged = merged.join(series_data[s].to_frame(name=s), how='outer')
        
        if merged is not None and not merged.empty:
            merged = merged.fillna(0.0)
            st.caption(f"**{period_label}盈虧折線圖** ({', '.join(series_list)})")
            st.line_chart(merged, use_container_width=True)
            
            # 也提供累計盈虧圖
            cumulative = merged.cumsum()
            st.caption(f"**{period_label}累計盈虧折線圖** ({', '.join(series_list)})")
            st.line_chart(cumulative, use_container_width=True)

@st.cache_data(max_entries=16)
def build_time_dimension_pnl_records(df_settled):
    """構建時間維度盈虧記錄表：按盤口類型 x 時間維度記錄盈虧"""
    if df_settled.empty:
        return pd.DataFrame()
    
    df = add_account_unit_profit_columns(df_settled)
    records = []
    bet_types = ["讓球", "入球大小", "角球大小"]
    
    for bt in bet_types:
        sub_df = df[df['Bet_Type'].astype(str).str.contains(bt, na=False, regex=False)]
        if sub_df.empty:
            continue
        
        for period in ['D', 'W', 'Q', 'Y']:
            for profit_col, account_label in [('System_Profit', '系統'), ('User_Profit', '用家')]:
                pnl_df = compute_period_pnl(sub_df, profit_col, 'Match_Date', period)
                if pnl_df.empty:
                    continue
                
                # 使用與 compute_period_pnl 相同的分組邏輯計算注單數
                tmp = sub_df.copy()
                if 'Match_Date' in tmp.columns:
                    tmp['_dt'] = pd.to_datetime(tmp['Match_Date'], errors='coerce')
                    if 'Date' in tmp.columns:
                        date_fallback = pd.to_datetime(tmp['Date'], errors='coerce')
                        tmp['_dt'] = tmp['_dt'].fillna(date_fallback)
                elif 'Date' in tmp.columns:
                    tmp['_dt'] = pd.to_datetime(tmp['Date'], errors='coerce')
                tmp = tmp.dropna(subset=['_dt'])
                if tmp.empty:
                    continue
                freq = PERIOD_FREQ.get(period, 'D')
                tmp['_period'] = tmp['_dt'].dt.to_period(freq).astype(str)
                period_counts = tmp.groupby('_period').size().to_dict()
                
                for _, row in pnl_df.iterrows():
                    period_label = row['period']
                    total = float(row[profit_col])
                    bet_count = period_counts.get(period_label, 0)
                    records.append({
                        '盤口類型': bt,
                        '時間維度': PERIOD_LABELS[period],
                        '期間': period_label,
                        '帳戶': account_label,
                        '淨盈虧': round(total, 2),
                        '注單數': bet_count,
                    })
    
    if not records:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(records)
    return result_df

def render_time_dimension_pnl_tab(df_settled, sys_bankroll, sys_max_stake):
    """渲染時間維度盈虧記錄分析頁面"""
    if df_settled.empty:
        st.info("目前尚無已結算的賽事，無法進行時間維度盈虧分析。")
        return
    
    df = add_account_unit_profit_columns(df_settled)
    
    st.markdown("### 📅 時間維度盈虧記錄分析")
    st.caption("本板塊按盤口類型（讓球/入球大小/角球大小）與時間維度（每日/每周/每季/每年）記錄盈虧；深度學習與機器學習盈虧分析已移至『全局模型』板塊。")
    
    # --- 盈虧總覽指標 ---
    st.markdown("#### 📊 盈虧總覽")
    
    total_bets = len(df_settled)
    sys_pnl = pd.to_numeric(df_settled['System_Profit'], errors='coerce').sum()
    usr_pnl = pd.to_numeric(df_settled['User_Profit'], errors='coerce').sum()
    sys_unit = df['System_Unit_Profit'].sum()
    usr_unit = df['User_Unit_Profit'].sum()
    sys_staked_bets = len(df[pd.to_numeric(df['System_Stake'], errors='coerce') > 0])
    
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
    tc1.metric("已結算注單", f"{total_bets} 張")
    tc2.metric("系統下注注單", f"{sys_staked_bets} 張")
    tc3.metric("系統總盈虧", f"${sys_pnl:,.2f}")
    tc4.metric("系統單位利潤", f"{sys_unit:.2f} U")
    tc5.metric("用家單位利潤", f"{usr_unit:.2f} U")
    
    st.caption("💡 注意：單位利潤僅計算系統/用家有實際下注 (Stake > 0) 的注單，與全局模型一致。零下注注單的盈虧結果不計入。")
    
    # --- 全局時間維度盈虧 (由『全局模型』板塊合併至此，避免重複) ---
    st.divider()
    st.markdown("#### 📅 全局時間維度盈虧 (按比賽日期分組)")
    st.caption("按賽前輸入的比賽日期 (Match_Date) 分組；舊資料無比賽日期時會退回記錄日期 (Date)。")
    
    global_pnl = compute_period_pnl_table(df_settled)
    gp1, gp2, gp3, gp4 = st.columns(4)
    with gp1:
        st.markdown("**每日淨盈虧**")
        st.metric("系統 (最新一期)", f"${global_pnl['D']['sys_latest']:,.2f}")
        st.metric("用家 (最新一期)", f"${global_pnl['D']['usr_latest']:,.2f}")
    with gp2:
        st.markdown("**每周淨盈虧**")
        st.metric("系統 (最新一期)", f"${global_pnl['W']['sys_latest']:,.2f}")
        st.metric("用家 (最新一期)", f"${global_pnl['W']['usr_latest']:,.2f}")
    with gp3:
        st.markdown("**每季淨盈虧**")
        st.metric("系統 (最新一期)", f"${global_pnl['Q']['sys_latest']:,.2f}")
        st.metric("用家 (最新一期)", f"${global_pnl['Q']['usr_latest']:,.2f}")
    with gp4:
        st.markdown("**每年淨盈虧**")
        st.metric("系統 (最新一期)", f"${global_pnl['Y']['sys_latest']:,.2f}")
        st.metric("用家 (最新一期)", f"${global_pnl['Y']['usr_latest']:,.2f}")
    
    render_pnl_chart_section(df_settled, "global", "全局")
    
    # --- 各盤口類型時間維度盈虧記錄表 ---
    st.divider()
    st.markdown("#### 📋 時間維度盈虧記錄明細表")
    
    pnl_records = build_time_dimension_pnl_records(df_settled)
    if not pnl_records.empty:
        # 提供篩選
        filter_col1, filter_col2 = st.columns(2)
        sel_bt = filter_col1.selectbox("篩選盤口類型", ["全部"] + list(pnl_records['盤口類型'].unique()), key="tdp_bt_filter")
        sel_period = filter_col2.selectbox("篩選時間維度", ["全部"] + list(pnl_records['時間維度'].unique()), key="tdp_period_filter")
        
        filtered = pnl_records.copy()
        if sel_bt != "全部":
            filtered = filtered[filtered['盤口類型'] == sel_bt]
        if sel_period != "全部":
            filtered = filtered[filtered['時間維度'] == sel_period]
        
        st.dataframe(
            filtered,
            column_config={
                "淨盈虧": st.column_config.NumberColumn("淨盈虧", format="$%.2f"),
                "注單數": st.column_config.NumberColumn("注單數", format="%d"),
            },
            use_container_width=True
        )
        
        # 下載
        st.download_button(
            "📥 下載時間維度盈虧記錄 (CSV)",
            filtered.to_csv(index=False).encode('utf-8-sig'),
            "time_dimension_pnl_records.csv",
            "text/csv",
            key="btn_down_tdp_csv"
        )
    else:
        st.warning("暫無可用的時間維度盈虧記錄。")
    
    # --- 各盤口類型時間維度盈虧對比 ---
    st.divider()
    st.markdown("#### 📈 各盤口類型時間維度盈虧對比")
    
    bet_types = ["讓球", "入球大小", "角球大小"]
    bt_tabs = st.tabs(bet_types)
    
    for i, bt in enumerate(bet_types):
        with bt_tabs[i]:
            sub_df = df[df['Bet_Type'].astype(str).str.contains(bt, na=False, regex=False)]
            if sub_df.empty:
                st.write(f"暫無 {bt} 結算紀錄。")
                continue
            
            bt_pnl = compute_period_pnl_table(sub_df)
            
            btc1, btc2, btc3, btc4 = st.columns(4)
            with btc1:
                st.markdown("**每日盈虧**")
                st.metric("系統", f"${bt_pnl['D']['sys_latest']:,.2f}")
                st.metric("用家", f"${bt_pnl['D']['usr_latest']:,.2f}")
            with btc2:
                st.markdown("**每周盈虧**")
                st.metric("系統", f"${bt_pnl['W']['sys_latest']:,.2f}")
                st.metric("用家", f"${bt_pnl['W']['usr_latest']:,.2f}")
            with btc3:
                st.markdown("**每季盈虧**")
                st.metric("系統", f"${bt_pnl['Q']['sys_latest']:,.2f}")
                st.metric("用家", f"${bt_pnl['Q']['usr_latest']:,.2f}")
            with btc4:
                st.markdown("**每年盈虧**")
                st.metric("系統", f"${bt_pnl['Y']['sys_latest']:,.2f}")
                st.metric("用家", f"${bt_pnl['Y']['usr_latest']:,.2f}")
            
            render_pnl_chart_section(sub_df, f"tdp_bt_{i}", bt)
    
    # --- 盤口類型對比盈虧折線圖 ---
    st.divider()
    render_bet_type_comparison_chart(df_settled, bet_types, "tdp_compare")

def render_bet_type_comparison_chart(df_settled, bet_types, key_prefix):
    """渲染盤口類型對比盈虧折線圖：可選擇 盤口類型 x 系統/用家 x 每日/每周/每季/每年"""
    if df_settled.empty:
        return
    
    series_data = {}
    for bt in bet_types:
        sub_df = df_settled[df_settled['Bet_Type'].astype(str).str.contains(bt, na=False, regex=False)]
        if sub_df.empty:
            continue
        for period in ['D', 'W', 'Q', 'Y']:
            sys_df = compute_period_pnl(sub_df, 'System_Profit', 'Match_Date', period)
            usr_df = compute_period_pnl(sub_df, 'User_Profit', 'Match_Date', period)
            if not sys_df.empty:
                series_data[f"系統-{bt}-{PERIOD_LABELS[period]}"] = sys_df.set_index('period')['System_Profit']
            if not usr_df.empty:
                series_data[f"用家-{bt}-{PERIOD_LABELS[period]}"] = usr_df.set_index('period')['User_Profit']
    
    if not series_data:
        st.caption("暫無可用於繪製盤口對比折線圖的數據。")
        return
    
    st.markdown("#### 📈 盤口類型對比盈虧折線圖")
    
    all_series = list(series_data.keys())
    default_series = [s for s in all_series if '每日' in s]
    if not default_series:
        default_series = all_series[:2] if len(all_series) >= 2 else all_series[:1]
    
    selected_series = st.multiselect(
        "選擇要顯示的盈虧線條 (可跨盤口類型、帳戶、頻率任意組合)",
        all_series,
        default=default_series,
        key=f"{key_prefix}_bt_series_select"
    )
    
    if not selected_series:
        st.caption("請至少選擇一條線條以顯示折線圖。")
        return
    
    # 依時間頻率分組渲染
    freq_groups = {}
    for s in selected_series:
        for period_key, period_label in PERIOD_LABELS.items():
            if period_label in s:
                freq = PERIOD_FREQ[period_key]
                if freq not in freq_groups:
                    freq_groups[freq] = []
                freq_groups[freq].append(s)
                break
    
    for freq, series_list in freq_groups.items():
        period_label = next((v for k, v in PERIOD_LABELS.items() if PERIOD_FREQ[k] == freq), freq)
        merged = None
        for s in series_list:
            if s in series_data:
                if merged is None:
                    merged = series_data[s].to_frame(name=s)
                else:
                    merged = merged.join(series_data[s].to_frame(name=s), how='outer')
        
        if merged is not None and not merged.empty:
            merged = merged.fillna(0.0)
            st.caption(f"**{period_label}盈虧折線圖** ({', '.join(series_list)})")
            st.line_chart(merged, use_container_width=True)
            
            cumulative = merged.cumsum()
            st.caption(f"**{period_label}累計盈虧折線圖** ({', '.join(series_list)})")
            st.line_chart(cumulative, use_container_width=True)

# ==========================================
# 5.7 深度學習與機器學習盈虧分析 (全局模型板塊)
# ==========================================
@st.cache_data(max_entries=16, show_spinner="🧠 正在訓練 ML 模型 (首次訓練較慢，完成後會自動快取)...")
def run_ml_analysis_cached(df_settled):
    """執行 ML 分析並回傳可序列化的摘要結果。
    @st.cache_data 快取：資料庫內容變更時自動重新計算，
    避免每次畫面刷新 (rerun) 都重複訓練模型，大幅降低 CPU 用量。"""
    rating_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
    bet_types = ["讓球", "入球大小", "角球大小"]
    analysis = {}
    
    for bt in bet_types:
        sub_df = df_settled[df_settled['Bet_Type'].astype(str).str.contains(bt, na=False, regex=False)]
        n = len(sub_df)
        entry = {'n': n, 'trained': False, 'class_ok': True}
        
        if n < ML_MIN_SAMPLES:
            analysis[bt] = entry
            continue
        
        X, y = prepare_enhanced_ml_dataset(sub_df, rating_map)
        if X is None or len(np.unique(y)) < 2:
            entry['class_ok'] = False
            analysis[bt] = entry
            continue
        
        best_model_info, model_results = train_ensemble_models(X, y)
        
        if model_results:
            entry['trained'] = True
            entry['model_results'] = {
                name: {'cv_mean': float(res['cv_mean']), 'cv_std': float(res['cv_std'])}
                for name, res in model_results.items()
            }
            entry['best_model'] = best_model_info[0] if best_model_info else None
            
            # 時間趨勢統計 (最近10場)
            recent = sub_df.tail(min(10, n))
            recent_win_rate = len(recent[pd.to_numeric(recent['Unit_Profit'], errors='coerce') > 0]) / len(recent) if len(recent) > 0 else 0.0
            recent_odds = pd.to_numeric(recent['Initial_Odds'], errors='coerce').mean()
            entry['recent_win_rate'] = float(recent_win_rate)
            entry['recent_odds'] = float(recent_odds) if pd.notna(recent_odds) else 0.0
        
        analysis[bt] = entry
    
    return analysis

def render_v10_kanpan_panel(v9_set):
    """V10 看盤引擎板塊：模型 / 策略比較、前進式回測、熔斷、最強看盤規則、規則說明"""
    kp = v10_get_engine(st.session_state.df_db)
    st.info(kp.get('msg', ''))
    if not kp.get('trained'):
        return
    st.markdown("##### 🏁 哪個模型最能讀懂莊家？(全部為前進式樣本外回測，同一測試期間)")
    if kp.get('compare'):
        st.dataframe(pd.DataFrame(kp['compare']), use_container_width=True, hide_index=True)
    st.caption("隨機買任一邊 = 不看盤的基準 (約等於馬會抽水)。看盤規則引擎：每一折只用之前日期計算每條規則的表現並挑選規則，再對之後日期下注，"
               "完全沒有使用未來數據。機器學習 LR / GB 使用相同的看盤特徵作對照。")
    rows = []
    for thr in V10_BT_THRESHOLDS:
        r = kp['bt'].get(thr, {})
        rows.append({'EV 門檻': f"{thr*100:.0f}%", '回測注數': r.get('n', 0), '命中率 (贏半或以上)': f"{r.get('hit', 0)*100:.1f}%" if r.get('n') else '-',
                     '單位盈虧 (U)': round(r.get('units', 0.0), 2), 'ROI': f"{r.get('roi', 0)*100:.1f}%" if r.get('n') else '-'})
    st.markdown("##### 📊 看盤引擎前進式回測 (不同 EV 門檻)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    rows = []
    for bt in V10_TYPES:
        info = kp['bt_type'].get(bt, {})
        keys = [k for k in info if k != 'all']
        k = min(keys, key=lambda t: abs(t - v9_set['min_ev'])) if keys else None
        sel = info.get(k, {}) if k is not None else {}
        ok, why = v10_type_gate(kp, bt, v9_set['min_ev'], v9_set['override_gate'])
        rows.append({'盤口類型': bt, '隨機買任一邊 ROI': f"{info.get('all', {}).get('roi', 0)*100:.1f}%" if info.get('all') else '-',
                     f'看盤回測注數 (門檻 {v9_set["min_ev"]*100:.1f}%)': sel.get('n', 0),
                     '看盤回測 ROI': f"{sel.get('roi', 0)*100:.1f}%" if sel.get('n') else '-',
                     '單位盈虧 (U)': round(sel.get('units', 0.0), 2), '狀態': '✅ 允許' if ok else f'⛔ 熔斷 ({why})'})
    st.markdown("##### 🚦 各盤口類型看盤回測與熔斷")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("##### 🏆 目前啟用的看盤規則 (按收縮後估計 EV 排序)")
    minn = kp.get('params', {}).get('minn', V10_MIN_N)
    act = [r for r in kp.get('rules', []) if r['n'] >= minn and r['shr'] >= v9_set['min_ev']]
    if act:
        def _grp(name):
            return ' / '.join(sorted({V10_BASE_RULE_DESC.get(p.strip(), ('-', ''))[0] for p in name.split(' + ')}))
        st.dataframe(pd.DataFrame([{'看盤規則': r['rule'], '盤口類型': r['bet_type'], '看盤要素': _grp(r['rule']), '歷史樣本': r['n'],
                                    '歷史 ROI': f"{r['roi']*100:.1f}%", '收縮後估計 EV': f"{r['shr']*100:+.1f}%", '命中率': f"{r['hit']*100:.1f}%",
                                    'z 值': round(r['z'], 2)} for r in act[:60]]), use_container_width=True, hide_index=True)
        st.caption(f"共 {len(act)} 條規則達到啟用條件 (樣本 ≥ {minn} 且估計 EV ≥ {v9_set['min_ev']*100:.1f}%)。z 值 ≥ 2 代表統計上較可信；"
                   "z 值低於 1.5 的規則仍可能只是隨機波動，系統以收縮估計及前進式回測降低這個風險。")
    else:
        st.info("目前沒有看盤規則達到啟用條件。")
    with st.expander("🔴 最差的看盤規則 (與莊為敵的反面：這些情況長期虧損，應避免)"):
        worst = sorted([r for r in kp.get('rules', []) if r['n'] >= minn], key=lambda r: r['shr'])[:30]
        st.dataframe(pd.DataFrame([{'看盤規則': r['rule'], '盤口類型': r['bet_type'], '歷史樣本': r['n'], '歷史 ROI': f"{r['roi']*100:.1f}%",
                                    '收縮後估計 EV': f"{r['shr']*100:+.1f}%"} for r in worst]), use_container_width=True, hide_index=True)
    with st.expander("📚 24 條基本看盤規則說明 (對應睇盤七大要素)"):
        st.dataframe(pd.DataFrame([{'看盤要素': v[0], '規則': k, '定義': v[1]} for k, v in V10_BASE_RULE_DESC.items()]),
                     use_container_width=True, hide_index=True)
        st.caption("系統同時自動測試全部兩兩組合 (例如「實力近況佔優 + 盤水移離此方」、「盤口低於實力(冷) + 收盤高水」)，即「反向思維」與「與莊為敵」的組合判斷。")
    st.warning("⚠ 看盤規則來自數百場比賽的統計，數百注的 ROI 標準誤約 ±5%，任何結果都不保證未來盈利。請持續記錄、定期重新訓練，並以小注碼驗證。")


def render_dl_ml_analysis_section(df_settled):
    """V9 全局模型板塊：特徵說明、特徵組合搜尋結果、模型比較、前進式回測、盤口熔斷狀態"""
    st.markdown("### 🧠 V11 模型擂台 + V10 看盤引擎 + V9 市場錨定機器學習分析")
    if not HAS_AI_MODULES:
        st.warning("未安裝 sklearn 套件，無法執行 ML 模型訓練。")
        return
    engine, _mt, _tl = v9_get_engine(st.session_state.df_db)
    v9_set = v9_get_settings()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("訓練比賽數", f"{engine.get('n_matches', 0)} 場")
    k2.metric("全盤口訓練樣本", f"{engine.get('n_rows', 0)} 個")
    k3.metric("市場錨定權重 w", f"{engine.get('w', 0.0):.1f}")
    k4.metric("採用模型", str(engine.get('kind', 'MARKET')))
    st.info(engine.get('msg', ''))
    _dups = v9_find_duplicate_matches(st.session_state.df_db)
    if not _dups.empty:
        st.warning(f"🧹 資料品質提示：發現 {len(_dups)} 對同名比賽在不同日期的賽果完全相同。V10 已把不同日期視為獨立比賽；"
                   "如果它們是被舊版即場更新 (按球隊名稱合併) 錯誤覆蓋，請到「已完成注單資料修改」更正其中一場的賽果。")
        st.dataframe(_dups, use_container_width=True, hide_index=True)
    if engine.get('trained'):
        st.write(f"自動選出的特徵組合: **{' + '.join(engine.get('groups') or ['市場賠率'])}**")
        if engine.get('ll_market') is not None and engine.get('ll_model') is not None:
            st.caption(f"樣本外 Log-Loss：市場 {engine['ll_market']:.5f} → 模型 {engine['ll_model']:.5f} (越低越好；改善 {engine['ll_market'] - engine['ll_model']:.5f})")

    ta, tq, t0, t1, t2, t3, t4 = st.tabs(["🏟 V11 模型擂台", "🧹 V11 數據質素檢查", "📈 V10 看盤引擎", "📉 V9 前進式回測 / 熔斷", "🧩 特徵組合排名", "🤖 模型比較 / 特徵權重", "📚 特徵說明 (新舊對照)"])
    with ta:
        render_v11_arena_panel()
    with tq:
        render_v11_data_quality()
    with t0:
        render_v10_kanpan_panel(v9_set)
    with t1:
        st.caption("回測方法：每一折只用『之前日期』的比賽訓練，預測『之後日期』的比賽 (完全樣本外)；每場每種盤口類型最多選一注 (EV 最高且 ≥ 門檻)。")
        if engine.get('bt_table'):
            st.dataframe(pd.DataFrame(engine['bt_table']), use_container_width=True, hide_index=True)
        rows = []
        for bt in ('讓球', '入球大小', '角球大小'):
            info = (engine.get('bt_type') or {}).get(bt, {})
            ok, why = v9_type_gate(engine, bt, v9_set['min_ev'], v9_set['override_gate'])
            keys = [k for k in info.keys() if k != 'all']
            k = min(keys, key=lambda t: abs(t - v9_set['min_ev'])) if keys else None
            sel = info.get(k, {}) if k is not None else {}
            rows.append({'盤口類型': bt,
                         '隨機買任一邊 ROI (市場基準)': f"{info.get('all', {}).get('roi', 0.0)*100:.1f}%" if info.get('all') else '-',
                         f'模型回測注數 (門檻 {v9_set["min_ev"]*100:.0f}%)': sel.get('n', 0),
                         '模型回測 ROI': f"{sel.get('roi', 0.0)*100:.1f}%" if sel.get('n') else '-',
                         '模型回測單位盈虧 (U)': round(sel.get('units', 0.0), 2) if sel.get('n') else 0.0,
                         '狀態': '✅ 允許' if ok else f'⛔ 熔斷 ({why})'})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("熔斷規則：樣本外回測注數 ≥ 20 且 ROI ≤ −3% 的盤口類型暫停建議。可於側邊欄「V10 投注策略與風控設定」解除 (不建議)。")
        st.warning("⚠ 回測結果有統計波動：數百注的 ROI 標準誤約 ±5%。任何模型都無法保證盈利；請以小注碼、嚴格門檻、長期記錄驗證。")

    with t2:
        st.caption("系統自動測試『市場賠率』+ 其餘 8 個因素群組的所有組合 (共 255 種)，例如「排名+賠率變化」、「排名+近5場戰績+賠率變化」、「排名+近5場戰績+賠率變化+球隊攻防數據」等，"
                   "以樣本外 Log-Loss 排名 (越低越好)。在最佳值 0.0005 範圍內會優先選擇群組最少的組合，以降低過度擬合。")
        if engine.get('combo_table'):
            ct = pd.DataFrame(engine['combo_table'])
            st.dataframe(ct.head(40), use_container_width=True, hide_index=True)
            single = []
            for g in V9_FEATURE_GROUPS:
                if g == '市場賠率':
                    continue
                r = ct[ct['特徵組合'] == f"市場賠率 + {g}"]
                if not r.empty:
                    single.append({'因素群組': g, '單獨加入市場後的改善': float(r.iloc[0]['相對市場改善'])})
            if single:
                st.markdown("##### 各因素『獨立』加入市場賠率後的效果")
                st.dataframe(pd.DataFrame(single).sort_values('單獨加入市場後的改善', ascending=False), use_container_width=True, hide_index=True)
                st.caption("正數 = 該因素提供了市場以外的額外資訊；負數或接近 0 = 該因素已被賠率反映或數據量不足。")
        else:
            st.info("數據不足，尚未進行特徵組合搜尋。")

    with t3:
        if engine.get('model_table'):
            st.dataframe(pd.DataFrame(engine['model_table']), use_container_width=True, hide_index=True)
        if engine.get('feat_importance'):
            st.markdown("##### 邏輯迴歸標準化係數 (正數 = 提高該方向勝率)")
            st.dataframe(pd.DataFrame(engine['feat_importance']), use_container_width=True, hide_index=True)
        elif engine.get('trained'):
            st.caption("w = 0 時不擬合最終模型 (系統以市場公平機率為準)。")

    with t4:
        st.markdown("##### 舊版 V8 的 16 個學習特徵")
        st.dataframe(pd.DataFrame([{'#': i + 1, '特徵': n, '學習內容': d} for i, (n, d) in enumerate(LEGACY_16_FEATURES)]), use_container_width=True, hide_index=True)
        st.markdown("##### V9 特徵 (按因素群組)")
        st.dataframe(pd.DataFrame([{'群組': V9_FEATURE_DESC[f][0], '特徵': f, '學習內容': V9_FEATURE_DESC[f][1]} for f in V9_ALL_FEATURES]),
                     use_container_width=True, hide_index=True)

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
    v10_load_settings_into_state()
    
    db_file = "football_betting_db.csv"
    capital_file = "football_capital_db.csv"
    db_table = "football_bets"
    cap_table = "football_cap"

    # --- 數據雲端同步狀態提示與手動刷新 ---
    has_sql = HAS_SQLALCHEMY and _has_secret("DB_URL") and st.secrets["DB_URL"]
    has_gh = _has_secret("GITHUB_TOKEN") and _has_secret("GITHUB_REPO")
    
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
    else:
        st.session_state.df_db = enforce_columns(st.session_state.df_db, DB_COLUMNS)
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


    with st.sidebar.expander("🎯 V10 投注策略與風控設定", expanded=False):
        st.caption("✅ 已載入上次儲存的設定" if st.session_state.get('_v10_settings_saved') else "目前為預設設定 (更改後會自動永久儲存)")
        if st.session_state.pop('_v11_migrated_now', False):
            st.info("V11 已把建議引擎升級為「🛡 自動 (只用已驗證策略)」。如想沿用舊模式可在下方更改。")
        st.selectbox("建議引擎", V10_MODE_OPTIONS, key="v10_mode", on_change=v10_save_settings,
                     help="🛡 自動 (預設)：只有在『模型擂台』通過樣本外驗證 (≥50 注、ROI ≥ +2%、前後兩段都賺錢) 的策略 × 盤口類型才實際落注，其餘只顯示模擬注 ($0)。"
                          "看盤規則：V10 邏輯。看盤嚴格規則：只用 z ≥ 1.5 (統計上較可信) 的看盤規則。看盤規則 + V9：有看盤訊號用看盤估計 EV，否則用 V9。只用 V9 模型：舊版 V9 邏輯。")
        st.checkbox("新聯賽觀察期 (已結算 < 15 場只作模擬注)", key="v11_new_league_paper", on_change=v10_save_settings,
                    help="新加入的聯賽未有足夠數據，模型對其特性一無所知；先記錄賽果讓系統學習，再實際落注。")
        st.checkbox("聯賽熔斷 (某聯賽樣本外 ≥15 注且 ROI ≤ −15% 暫停實注)", key="v11_league_breaker", on_change=v10_save_settings)
        st.number_input("最低 EV 門檻 (%)", min_value=0.0, max_value=20.0, step=0.5, key="v9_min_ev", on_change=v10_save_settings,
                        help="只有估計期望值 ≥ 此門檻的盤口才會被建議。0% = 注數多但包含大量優勢極薄、估計誤差內的投注；3% = 注數少但每注質素較高。")
        st.select_slider("Kelly 分數", options=[0.1, 0.15, 0.2, 0.25, 0.33, 0.5], key="v9_kelly", on_change=v10_save_settings,
                         help="注碼 = 本金 × Kelly 分數 × EV ÷ (賠率 − 1)。高於 0.25 時注碼隨估計優勢急升，估計誤差會令回撤急劇放大，建議 0.25 或以下。")
        st.number_input("單注上限 (% 系統本金)", min_value=0.5, max_value=10.0, step=0.5, key="v9_max_pct", on_change=v10_save_settings)
        st.number_input("每場最多建議注數", min_value=1, max_value=3, step=1, key="v9_max_picks", on_change=v10_save_settings,
                        help="同一場比賽的多個盤口高度相關，建議每場只下 1 注。")
        st.checkbox("啟用舊版「讓球最低 $200」規則", key="v9_hd_min200", on_change=v10_save_settings,
                    help="舊版會把 EV 偏低的讓球注碼強制升至 $200，放大了邊緣投注的虧損，預設關閉。")
        st.checkbox("解除盤口熔斷 (不建議)", key="v9_override_gate", on_change=v10_save_settings,
                    help="樣本外回測 ≥ 20 注且 ROI ≤ −3% 的盤口類型會自動暫停建議。")
        _sb1, _sb2 = st.columns(2)
        if _sb1.button("💾 儲存設定", use_container_width=True, key="v10_save_btn"):
            v10_save_settings()
        if _sb2.button("↺ 回復預設", use_container_width=True, key="v10_reset_btn"):
            for _k, _d in V10_SETTING_DEFAULTS.items():
                st.session_state[_k] = _d
            v10_save_settings()
            st.rerun()
        if st.button("♻ 重新訓練 V9 模型 + V10 看盤引擎", use_container_width=True, key="v9_retrain"):
            v9_train_engine_cached.clear()
            v9_history_tables_cached.clear()
            v10_train_engine_cached.clear()
            v11_arena_cached.clear()
            v11_data_audit_cached.clear()
            st.toast("已清除模型快取，下次分析時會重新訓練。(投注設定不受影響)", icon="♻")

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
    if st.sidebar.button("🗂 已完成注單資料修改 (可直接編輯任何欄位)", use_container_width=True):
        edit_settled_bets_dialog(db_file, db_table)

    t_pre, t_inplay, t_settle, t_ai, t_time_pnl = st.tabs(["📝 賽前建檔與投注", "⏱️ 即場賽事與預測", "⚖ 賽果結算與管理", "🤖 全局模型", "📅 時間維度盈虧記錄分析"])

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
        try:
            _iss = v11_issues_active(v11_data_audit_cached(st.session_state.df_db))
            _nh = int((_iss['嚴重程度'] == '🔴 高').sum()) if not _iss.empty else 0
            if _nh:
                st.warning(f"🧹 數據質素檢查發現 {_nh} 項嚴重問題 (重複注單、賽事標籤錯誤、排名未填、結算不符等)，會令模型學錯。請到「🤖 全局模型 → 🧹 V11 數據質素檢查」修正，或核實數據正確後勾選「✅ 確認無誤」。")
        except Exception:
            pass
        
        is_editing = bool(st.session_state.editing_bet_id)
        opts_tournaments = ["➕ 新增手動輸入..."] + sorted(list(set(st.session_state.df_db['Tournament_Name'].dropna().unique())))
        opts_teams = ["➕ 新增手動輸入..."] + sorted(list(set(st.session_state.df_db['Home_Team'].dropna().tolist() + st.session_state.df_db['Away_Team'].dropna().tolist())))
        
        st.markdown("##### 1. 賽事與球隊資料")
        col_t, col_c = st.columns(2)
        
        default_tourn_name = st.session_state.get('edit_t_name', '') if is_editing else ''
        default_tourn_idx = (opts_tournaments.index(default_tourn_name) if default_tourn_name in opts_tournaments else 0) if is_editing else 0
        
        sel_tournament = col_t.selectbox("賽事名稱 (Tournament Name)", opts_tournaments, index=default_tourn_idx, key="sel_tourn")
        tournament_name = col_t.text_input("輸入新賽事名稱", value=default_tourn_name, key="txt_tourn") if sel_tournament == "➕ 新增手動輸入..." else sel_tournament
        tournament_name = v11_clean_name(tournament_name) or ''
        
        default_cat_name = st.session_state.get('edit_t_cat', CATEGORY_OPTIONS[0]) if is_editing else CATEGORY_OPTIONS[0]
        default_cat_idx = CATEGORY_OPTIONS.index(default_cat_name) if default_cat_name in CATEGORY_OPTIONS else 0
        
        if not is_editing and sel_tournament != "➕ 新增手動輸入...":
            match_rows = st.session_state.df_db[st.session_state.df_db['Tournament_Name'] == tournament_name]
            if not match_rows.empty:
                last_cat = match_rows.iloc[-1]['Tournament_Category']
                if last_cat in CATEGORY_OPTIONS:
                    default_cat_idx = CATEGORY_OPTIONS.index(last_cat)
        if not is_editing and tournament_name and v11_is_cup_like(tournament_name) and CATEGORY_OPTIONS[default_cat_idx].startswith('國內聯賽'):
            default_cat_idx = 2 if v11_is_intl(tournament_name) else 3
                    
        tournament_category = col_c.selectbox("賽事分類 (Tournament Category)", CATEGORY_OPTIONS, index=default_cat_idx, key="sel_cat")

        # 比賽日期輸入 (用於時間維度盈虧分組)
        _default_md = st.session_state.get('edit_match_date', get_hkt_now().strftime('%Y-%m-%d')) if is_editing else get_hkt_now().strftime('%Y-%m-%d')
        try:
            _default_md_dt = datetime.strptime(_default_md, '%Y-%m-%d').date()
        except:
            _default_md_dt = get_hkt_now().date()
        match_date_input = st.date_input("比賽日期 / 下注日期 (Match Date)", value=_default_md_dt, key="match_date_input")
        match_date = match_date_input.strftime('%Y-%m-%d')

        col_h, col_a = st.columns(2)
        default_h_team = st.session_state.get('edit_h_team', '') if is_editing else ''
        default_h_idx = (opts_teams.index(default_h_team) if default_h_team in opts_teams else 0) if is_editing else 0
        sel_home = col_h.selectbox("主隊名稱", opts_teams, index=default_h_idx, key="sh")
        home_team = col_h.text_input("輸入新主隊", value=default_h_team, key="txt_h_team") if sel_home == "➕ 新增手動輸入..." else sel_home
        home_team = v11_clean_name(home_team) or ''

        default_a_team = st.session_state.get('edit_a_team', '') if is_editing else ''
        default_a_idx = (opts_teams.index(default_a_team) if default_a_team in opts_teams else 0) if is_editing else 0
        sel_away = col_a.selectbox("客隊名稱", opts_teams, index=default_a_idx, key="sa")
        away_team = col_a.text_input("輸入新客隊", value=default_a_team, key="txt_a_team") if sel_away == "➕ 新增手動輸入..." else sel_away
        away_team = v11_clean_name(away_team) or ''

        # V11：按賽事自動填寫隊伍數目 (用家修改後會記住，下次自動沿用)
        _eid = str(st.session_state.editing_bet_id or '')
        _ls_marker = f"{_eid}|{tournament_name}"
        _prev_marker = st.session_state.get('_v11_ls_marker')
        _ls_src = st.session_state.get('_v11_ls_src', '')
        if _prev_marker != _ls_marker or 'league_size' not in st.session_state:
            if is_editing and (_prev_marker is None or not str(_prev_marker).startswith(f"{_eid}|")):
                _ls_val, _ls_src = int(st.session_state.get('edit_league_size', 20)), '已載入注單'
                _nr = v11_is_na_rating(st.session_state.get('edit_h_rating_raw')) or v11_is_na_rating(st.session_state.get('edit_a_rating_raw'))
            else:
                _ls_val, _ls_src = v11_lookup_league_size(tournament_name, st.session_state.df_db)
                _nr = v11_default_no_rank(tournament_name, st.session_state.df_db)
            st.session_state['league_size'] = int(max(2, min(40, _ls_val)))
            st.session_state['v11_no_rank'] = bool(_nr)
            st.session_state['_v11_ls_marker'] = _ls_marker
            st.session_state['_v11_ls_src'] = _ls_src
        no_rank = st.checkbox("此賽事沒有聯賽排名 (盃賽 / 洲際賽，例如亞冠二級、日本聯賽盃)", key="v11_no_rank",
                              help="勾選後系統把兩隊實力視為『未知』：不會用排名計算實力差及盤口冷熱，避免以預設排名 (例如 10 vs 10) 誤導模型。")
        c_ls, c_hr, c_ar = st.columns(3)
        league_size = c_ls.number_input("聯賽隊伍數目 (自動填寫，可手動修改)", min_value=2, max_value=40, key="league_size",
                                       on_change=_v11_on_ls_change, args=(tournament_name,),
                                       help="系統會按賽事自動填寫 (例如英超 20、德甲 18)。手動修改後會記住，下次選擇同一賽事自動沿用。")
        c_ls.caption(f"來源：{_ls_src}")
        for _rk in ('h_rank', 'a_rank'):
            if _rk in st.session_state and int(st.session_state[_rk]) > int(league_size):
                st.session_state[_rk] = int(league_size)
        
        _mid_rank = max(1, int(league_size) // 2)
        default_h_rank = st.session_state.get('edit_h_rank', _mid_rank) if is_editing else _mid_rank
        default_a_rank = st.session_state.get('edit_a_rank', _mid_rank) if is_editing else _mid_rank
        home_rank = c_hr.number_input("主隊聯賽排名 (1=最強)", min_value=1, max_value=int(league_size),
                                     value=max(1, min(int(default_h_rank), int(league_size))), key="h_rank", disabled=no_rank,
                                     help="排名1 = 聯賽第一名 (實力最高)；排名越大實力越弱。")
        away_rank = c_ar.number_input("客隊聯賽排名 (1=最強)", min_value=1, max_value=int(league_size),
                                     value=max(1, min(int(default_a_rank), int(league_size))), key="a_rank", disabled=no_rank,
                                     help="排名1 = 聯賽第一名 (實力最高)；排名越大實力越弱。")
        if not no_rank and not is_editing and int(home_rank) == _mid_rank and int(away_rank) == _mid_rank:
            st.caption(f"⚠ 兩隊排名仍是預設值 {_mid_rank}。如果是聯賽請輸入實際排名；如果是盃賽 / 洲際賽請勾選「此賽事沒有聯賽排名」。")
        
        if is_editing:
            _raw_hr = str(st.session_state.get('edit_h_rating_raw', ''))
            _raw_ar = str(st.session_state.get('edit_a_rating_raw', ''))
            if _raw_hr and _parse_rank_int(_raw_hr) is None:
                st.caption(f"⚠ 主隊舊資料為字母評級 ({_raw_hr})，已自動換算為約略排名 {default_h_rank}，請直接修改為實際排名；如不修改，提交時會保留原有字母評級。")
            if _raw_ar and _parse_rank_int(_raw_ar) is None:
                st.caption(f"⚠ 客隊舊資料為字母評級 ({_raw_ar})，已自動換算為約略排名 {default_a_rank}，請直接修改為實際排名；如不修改，提交時會保留原有字母評級。")
        st.caption("💡 排名輸入1代表排名第一 (實力最高)。V9 引擎以去水市場賠率為錨點，結合聯賽排名、近5場戰績、賠率變化、數據庫內兩隊過往攻防數據、對賽往績及賽事環境，自動測試因素組合並以前進式回測驗證，只在 EV 達標時建議投注。")
        # V11：輸入檢查 (同一球隊短時間內出賽、球隊不常參加此賽事)
        try:
            _dbc = st.session_state.df_db
            if home_team and away_team and home_team != "➕ 新增手動輸入..." and not _dbc.empty:
                _md_chk = pd.Timestamp(match_date)
                _mdser = pd.to_datetime(v10_md_series(_dbc), errors='coerce')
                _mname = f"{home_team} vs {away_team}"
                _hh = _dbc['Home_Team'].astype(str).map(v11_clean_name)
                _aa = _dbc['Away_Team'].astype(str).map(v11_clean_name)
                _mm = _dbc['Match'].astype(str).map(v11_clean_name)
                for _tm in (home_team, away_team):
                    _near = _dbc[((_hh == _tm) | (_aa == _tm)) & (_mm != _mname) & ((_mdser - _md_chk).abs() <= pd.Timedelta(days=2))]
                    if not _near.empty:
                        _r0 = _near.iloc[0]
                        st.warning(f"⚠ 「{_tm}」在 {v10_md_series(_near).iloc[0]} 已有另一場比賽 ({_r0['Match']})，相隔不足 2 日。請確認比賽日期及球隊是否正確。")
                    _tt = _dbc[(_hh == _tm) | (_aa == _tm)]['Tournament_Name'].astype(str).map(v11_clean_name).value_counts()
                    if tournament_name and len(_tt) and _tt.sum() >= 3 and _tt.get(tournament_name, 0) == 0:
                        st.warning(f"⚠ 「{_tm}」過往只出現在「{'、'.join(_tt.index[:3])}」，從未出現在「{tournament_name}」。請確認賽事名稱是否選錯。")
        except Exception:
            pass

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
        for _lab, _fm in (('主隊', home_form), ('客隊', away_form)):
            _w, _d, _l = parse_form_str(_fm)
            if _w + _d + _l != 5:
                st.warning(f"⚠ {_lab}近況 {_fm} 合共 {_w + _d + _l} 場，近 5 場勝和負合計應為 5。")

        st.markdown("##### 3. 賽前盤口與賠率走勢紀錄 (JSON結構儲存)")
        if 'odds_history' not in st.session_state:
            st.session_state.odds_history = [{"id": 0, "type": "讓球", "record_time": "", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}]
        render_odds_section(st.session_state.odds_history, "pre")
        _bad_odds = []
        for _h in st.session_state.odds_history:
            try:
                _u, _l = float(_h.get('upper')), float(_h.get('lower'))
                if _u > 1.0 and _l > 1.0 and not v11_valid_pair(_u, _l):
                    _bad_odds.append(f"{_h.get('type')} {float(_h.get('line', 0)):g} @ {_u:g}/{_l:g} (抽水 {1/_u + 1/_l:.2f})")
            except (TypeError, ValueError, AttributeError):
                continue
        if _bad_odds:
            st.warning("⚠ 以下賠率不合理 (正常抽水約 1.03~1.12，兩邊同一數值或 >15 多數是輸入錯誤)，系統分析時會忽略：" + "；".join(_bad_odds[:8]))
        
        st.markdown("---")
        
        if st.button("🚀 賽前數據分析執行", type="primary", use_container_width=True):
            st.session_state.show_analysis = True
            v9_set = v9_get_settings()
            engine, mt_hist, tl_hist = v9_get_engine(st.session_state.df_db)
            df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()

            # 聯賽排名 → 實力分 (1~5，與舊字母尺度一致)；近5場 → 積分
            hr_val = 3.0 if no_rank else rank_to_strength(home_rank, league_size)
            ar_val = 3.0 if no_rank else rank_to_strength(away_rank, league_size)
            _unk_now = bool(no_rank) or v11_is_intl(tournament_name, tournament_category)
            hf_val = float(extract_form_points(home_form))
            af_val = float(extract_form_points(away_form))
            try:
                _md_ts = pd.Timestamp(match_date)
            except Exception:
                _md_ts = pd.Timestamp(get_hkt_now().strftime('%Y-%m-%d'))

            # 依時間排序每個盤口 (類型+盤口線) 的賠率序列：每個時間點只使用「截至該時間點」的賠率 (不用未來數據)
            _mk_sorted = v9_group_markets(st.session_state.odds_history, _md_ts.to_pydatetime())
            _pos_info = {}
            for (_bt, _ln), _snaps in _mk_sorted.items():
                _first = _snaps[0]
                for _rank, _sn in enumerate(_snaps):
                    _pos_info[_sn[1]] = {
                        'drift_up': _sn[2] / _first[2] - 1.0 if _first[2] > 0 else 0.0,
                        'drift_lw': _sn[3] / _first[3] - 1.0 if _first[3] > 0 else 0.0,
                        'nsnap': _rank + 1,
                        'is_latest': _rank == len(_snaps) - 1,
                    }

            candidates_base = []
            nsnap_map = {}
            for idx, r in enumerate(st.session_state.odds_history):
                try:
                    b_type, line_val = str(r['type']), float(r['line'])
                    up_odds, lw_odds = float(r['upper']), float(r['lower'])
                except (KeyError, TypeError, ValueError):
                    continue
                if up_odds <= 1.0 or lw_odds <= 1.0 or not v11_valid_pair(up_odds, lw_odds):
                    continue
                rec_time = r.get('record_time', '')
                pinfo = _pos_info.get(idx, {'drift_up': 0.0, 'drift_lw': 0.0, 'nsnap': 1, 'is_latest': True})
                nsnap_map[(v9_clean_bt(b_type), round(line_val, 3), idx)] = pinfo['nsnap']
                line_str = f"{line_val:g}"
                if b_type == "讓球":
                    label_h, label_a = (f"{line_str}主隊(上盤)", f"{line_str}客隊(下盤)") if line_val <= 0 else (f"{line_str}主隊(下盤)", f"{line_str}客隊(上盤)")
                    candidates_base.extend([
                        {'bet_type': b_type, 'selection': 'Home', 'base_prob': 0.5, 'odds': up_odds, 'line': line_val, 'label': label_h, 'history_idx': idx, 'record_time': rec_time, 'odds_drift': pinfo['drift_up'], 'is_latest': pinfo['is_latest']},
                        {'bet_type': b_type, 'selection': 'Away', 'base_prob': 0.5, 'odds': lw_odds, 'line': line_val, 'label': label_a, 'history_idx': idx, 'record_time': rec_time, 'odds_drift': pinfo['drift_lw'], 'is_latest': pinfo['is_latest']}
                    ])
                else:
                    candidates_base.extend([
                        {'bet_type': b_type, 'selection': 'Over', 'base_prob': 0.5, 'odds': up_odds, 'line': line_val, 'label': f"{line_str}大盤(Over)", 'history_idx': idx, 'record_time': rec_time, 'odds_drift': pinfo['drift_up'], 'is_latest': pinfo['is_latest']},
                        {'bet_type': b_type, 'selection': 'Under', 'base_prob': 0.5, 'odds': lw_odds, 'line': line_val, 'label': f"{line_str}小盤(Under)", 'history_idx': idx, 'record_time': rec_time, 'odds_drift': pinfo['drift_lw'], 'is_latest': pinfo['is_latest']}
                    ])

            # 賽前情境：只使用比賽日期之前的數據庫資料 (球隊攻防數據、對賽往績、賽事環境)
            v9_ctx = v9_match_context(mt_hist, tl_hist, home_team, away_team, tournament_name, _md_ts, hr_val, ar_val, hf_val, af_val)
            scored = v9_score_candidates(engine, candidates_base, v9_ctx, v9_set, nsnap_map) if candidates_base else []

            # V10 看盤引擎：以初盤→收盤的盤口/水位變化 + 實力冷熱，套用樣本外驗證為正 EV 的看盤規則
            kp_engine = v10_get_engine(st.session_state.df_db)
            try:
                kp_map = v10_predict(kp_engine, st.session_state.odds_history, v9_ctx, hr_val, ar_val, hf_val, af_val, _md_ts, v9_set['min_ev'], _unk_now)
            except Exception as _kp_err:
                kp_map = {}
                st.warning(f"看盤引擎分析失敗，已改用 V9 模型: {_kp_err}")
            try:
                arena = v11_get_arena(st.session_state.df_db)
            except Exception as _ar_err:
                arena = {'trained': False, 'msg': f'模型擂台失敗: {_ar_err}', 'active': {}, 'league': {}}
            _league_ctx = {'tourn': tournament_name, 'n_matches': v11_league_match_count(st.session_state.df_db, tournament_name)}
            _mode_now = v9_set['mode'] if (kp_engine.get('trained') or v9_set['mode'] == V11_MODE_AUTO) else '只用 V9 模型'
            scored = v10_apply_to_candidates(scored, kp_map, _mode_now, v9_set, kp_engine, arena, _league_ctx)
            kp_summary = []
            for (_kbt, _kln, _ksd), _kv in kp_map.items():
                kp_summary.append({'盤口類型': _kbt, '方向': {'Home': '主隊', 'Away': '客隊', 'Over': '大', 'Under': '小'}.get(_ksd, _ksd),
                                   '初盤': f"{_kv['line_o']:g} @ {_kv['odds_o']:.2f}", '收盤主盤': f"{_kln:g}",
                                   '隱含值 初→收': f"{_kv['imp_o']:.2f} → {_kv['imp_c']:.2f}",
                                   '盤水移動 (此方向)': f"{_kv['mv_imp']:+.2f}", '冷熱 (負=冷)': f"{_kv['resid']:+.2f}",
                                   '最佳看盤規則': _kv['kp_rule'] or '-', '看盤估計EV': f"{_kv['ev_kp']*100:+.1f}%" if _kv['ev_kp'] is not None else '-',
                                   '觸發規則 (前5)': '；'.join(_kv['kp_fired']) if _kv['kp_fired'] else '-'})

            # 推薦：只考慮每個盤口的最新賠率；每種盤口類型最多一注，總數不超過設定上限
            picks, _used_types = [], set()
            for c in scored:
                if not c.get('is_latest', True) or c.get('status') != '✅ 建議':
                    continue
                _bt = v9_clean_bt(c['bet_type'])
                if _bt in _used_types:
                    continue
                picks.append(c)
                _used_types.add(_bt)
                if len(picks) >= v9_set['max_picks']:
                    break
            # V11 模擬注：未通過驗證 / 新聯賽觀察期的最佳候選 (注碼 $0，只供追蹤)
            paper_picks, _pused = [], set()
            for c in sorted([c for c in scored if c.get('is_latest', True) and str(c.get('status', '')).startswith('📝')], key=lambda x: -x.get('ev', -9)):
                _bt = v9_clean_bt(c['bet_type'])
                if _bt in _pused:
                    continue
                paper_picks.append(c)
                _pused.add(_bt)
                if len(paper_picks) >= v9_set['max_picks']:
                    break

            try:
                _lh, _la, _lc, _lsrc = v9_prematch_lambdas(st.session_state.odds_history, v9_ctx)
            except Exception:
                _lh, _la, _lc, _lsrc = None, None, None, {}

            st.session_state.analysis_result = {
                'v9': True,
                'dims': [v9_dimension_stats(df_settled[df_settled['Tournament_Name'] == tournament_name], "A. 微觀 (賽事名稱)"),
                         v9_dimension_stats(df_settled[df_settled['Tournament_Category'] == tournament_category], "B. 中觀 (賽事分類)"),
                         v9_dimension_stats(df_settled, "C. 宏觀 (全局數據)")],
                'engine': {k: engine.get(k) for k in ('trained', 'msg', 'w', 'kind', 'groups', 'n_matches', 'n_rows', 'll_market', 'll_model')},
                'candidates': scored, 'picks': picks, 'paper_picks': paper_picks, 'kp_summary': kp_summary,
                'arena_msg': arena.get('msg', ''), 'league_ctx': _league_ctx, 'no_rank': bool(no_rank),
                'kp_engine': {k: kp_engine.get(k) for k in ('trained', 'msg', 'n_matches', 'n_rows')}, 'mode': v9_set['mode'],
                'ctx': {'h_n': v9_ctx['hp']['n'], 'a_n': v9_ctx['ap']['n'], 'h2h_n': len(v9_ctx['h2h']), 'league_n': v9_ctx['league']['n'],
                        'h_gf': v9_ctx['hp']['gf'], 'h_ga': v9_ctx['hp']['ga'], 'a_gf': v9_ctx['ap']['gf'], 'a_ga': v9_ctx['ap']['ga'],
                        'h_cf': v9_ctx['hp']['cf'], 'a_cf': v9_ctx['ap']['cf']},
                'lambdas': (_lh, _la, _lc, _lsrc),
                't_name': tournament_name, 't_cat': tournament_category
            }

        if st.session_state.get('show_analysis', False) and not st.session_state.get('analysis_result', {}).get('v9'):
            st.session_state.show_analysis = False

        if st.session_state.get('show_analysis', False):
            res = st.session_state.analysis_result
            v9_set = v9_get_settings()
            eng = res['engine']
            st.success(f"✅ V11 分析完成 (建議引擎：{res.get('mode', V11_MODE_AUTO)})")
            if res.get('arena_msg'):
                st.caption("🏟 " + res['arena_msg'])
            _lc = res.get('league_ctx') or {}
            if _lc and _lc.get('n_matches', 99) < V11_NEW_LEAGUE_MIN:
                st.caption(f"🆕 「{_lc.get('tourn')}」只有 {_lc.get('n_matches')} 場已結算比賽，屬新聯賽觀察期。")
            if res.get('no_rank'):
                st.caption("ℹ 此賽事已標記為沒有聯賽排名：實力差及盤口冷熱不會計入。")

            c1, c2, c3 = st.columns(3)
            for col, dstat in zip([c1, c2, c3], res['dims']):
                with col.container(border=True):
                    st.markdown(f"**{dstat['dim']}**")
                    if dstat['valid']:
                        st.write(f"樣本數: `{dstat['n']}` 張 (系統實注 `{dstat['n_bet']}` 張)")
                        st.write(f"系統實注 ROI: `{dstat['roi']*100:.1f}%`")
                        st.write(f"歷史勝率 (贏半或以上): `{dstat['acc']*100:.1f}%`")
                    else:
                        st.warning(dstat['msg'])

            with st.container(border=True):
                st.markdown("##### 🧠 V9 模型狀態")
                if eng.get('trained'):
                    st.write(f"訓練比賽數: `{eng.get('n_matches', 0)}` 場 | 全盤口樣本: `{eng.get('n_rows', 0)}` 個 | 模型: `{eng.get('kind')}` | 市場錨定權重 w = `{eng.get('w', 0):.1f}`")
                    st.write(f"自動選出的特徵組合: **{' + '.join(eng.get('groups') or ['市場賠率'])}**")
                else:
                    st.write(f"訓練比賽數: `{eng.get('n_matches', 0)}` 場")
                st.caption(eng.get('msg', ''))
                cx = res.get('ctx', {})
                st.caption(f"本場數據庫資料：主隊過往 {cx.get('h_n', 0)} 場、客隊過往 {cx.get('a_n', 0)} 場、對賽往績 {cx.get('h2h_n', 0)} 場、同賽事過往 {cx.get('league_n', 0)} 場 "
                           f"| 主隊場均入球/失球 {cx.get('h_gf', 0):.2f}/{cx.get('h_ga', 0):.2f}，客隊 {cx.get('a_gf', 0):.2f}/{cx.get('a_ga', 0):.2f} (已向聯賽平均收縮)")
                _lh, _la, _lc, _lsrc = res.get('lambdas', (None, None, None, {}))
                if _lh is not None:
                    st.caption(f"市場隱含預期：主隊 {_lh:.2f} 球、客隊 {_la:.2f} 球、總角球 {_lc:.1f} (來源: {', '.join(f'{k}={v}' for k, v in (_lsrc or {}).items())})")

            with st.container(border=True):
                st.markdown("##### 📈 V10 看盤解讀 (初盤 → 收盤，莊家盤口與水位變化)")
                _kpe = res.get('kp_engine') or {}
                st.caption(_kpe.get('msg', ''))
                if res.get('kp_summary'):
                    st.dataframe(pd.DataFrame(res['kp_summary']), use_container_width=True, hide_index=True)
                    st.caption("隱含值：讓球 = 主隊隱含淨勝球，大小盤 = 隱含總數 (已把盤口線與水位一併換算)。「盤水移動」正數 = 莊家調整令此方向更受捧；"
                               "「冷熱」負數 = 市場給此方向的評價低於其排名/近況應得水平。看盤估計 EV = 該規則歷史回報向抽水基準收縮後的估計，並非保證。")
                elif _kpe.get('trained'):
                    st.info("本場賠率紀錄不足以計算看盤特徵 (請至少輸入讓球 / 入球大小 / 角球大小的賽前賠率，最好包括初盤及臨場最新賠率)。")

            st.markdown("### 🧠 AI 預測模型推薦")
            st.markdown("#### 📊 所有盤口評估明細 (完整歷史時間點評估與 EV 運算)")
            cand_list = res.get('candidates', [])

            category_order = ["讓球", "入球大小", "角球大小"]
            all_options = []
            option_map = {}

            if cand_list:
                df_show = pd.DataFrame(cand_list)
                df_show['推薦排序'] = range(1, len(df_show) + 1)
                df_show['最新賠率'] = df_show['is_latest'].apply(lambda v: '✔' if v else '')
                for _c, _d in (('ev_kp', None), ('ev_v9', None), ('kp_rule', '')):
                    if _c not in df_show.columns:
                        df_show[_c] = _d
                df_show['看盤估計EV'] = df_show['ev_kp'].apply(lambda x: f"{x*100:+.1f}%" if x is not None and pd.notna(x) else '-')
                df_show['V9 EV'] = df_show['ev_v9'].apply(lambda x: f"{x*100:+.1f}%" if x is not None and pd.notna(x) else '-')
                df_show['看盤規則'] = df_show['kp_rule'].apply(lambda x: x if x else '-')
                df_show = df_show[['推薦排序', 'status', 'bet_type', 'line', 'label', 'odds', 'record_time', '最新賠率', '看盤規則', '看盤估計EV', 'V9 EV', 'p_mkt', 'prob', 'relative_prob', 'edge', 'ev']]
                df_show.columns = ['推薦排序', '狀態', '盤口類型', '盤口線', '投注方向', '賠率', '時間', '最新賠率', '看盤規則', '看盤估計EV', 'V9 EV', '市場公平勝率', '模型有效勝率', '兩邊相對化', '相對市場優勢', '最終期望值 (EV)']
                for _c in ['市場公平勝率', '模型有效勝率', '兩邊相對化']:
                    df_show[_c] = df_show[_c].apply(lambda x: f"{x*100:.2f}%")
                df_show['相對市場優勢'] = df_show['相對市場優勢'].apply(lambda x: f"{x*100:+.2f}%")
                df_show['最終期望值 (EV)'] = df_show['最終期望值 (EV)'].apply(lambda x: f"{x:+.3f}")

                def highlight_first(row):
                    if str(row['狀態']).startswith('✅'):
                        return ['background-color: rgba(40, 167, 69, 0.2)'] * len(row)
                    if str(row['狀態']).startswith('⛔'):
                        return ['color: #999999'] * len(row)
                    return [''] * len(row)

                st.dataframe(df_show.style.apply(highlight_first, axis=1), use_container_width=True, hide_index=True)
                st.caption("💡 「最終期望值」按側邊欄的建議引擎決定：看盤規則模式使用「看盤估計EV」(只適用於收盤主盤)，V9 模式使用「V9 EV」。"
                           "「模型有效勝率」= (EV+1)/賠率，已計入亞洲盤贏半/輸半/走盤。"
                           f"只有最終 EV ≥ {v9_set['min_ev']*100:.1f}% 且盤口未被熔斷的最新賠率才會被建議。")

                for cat in category_order:
                    for c in [c for c in cand_list if v9_clean_bt(c.get('bet_type')) == cat]:
                        if c.get('history_idx') is None or c.get('is_latest', True):
                            opt_str = f"【{c['bet_type']}】{c['label']} @ {c['odds']}"
                            if opt_str not in option_map:
                                all_options.append(opt_str)
                                option_map[opt_str] = c

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

            picks = res.get('picks', [])
            mc1, mc2, mc3 = st.columns(3)
            if picks:
                bb = picks[0]
                mc1.metric("💡 首選推薦", f"{bb['bet_type']} - {bb['label']} @ {bb['odds']}")
                mc2.metric("🎯 模型有效勝率 / 市場", f"{bb['prob']*100:.1f}% / {bb['p_mkt']*100:.1f}%")
                mc3.metric("📊 EV", f"{bb.get('ev', 0):+.3f}")
                if len(picks) > 1:
                    st.caption("其他建議：" + "；".join(f"{p['bet_type']} {p['label']} @ {p['odds']} (EV {p['ev']:+.3f})" for p in picks[1:]))
                for _p in picks:
                    if _p.get('kp_fired'):
                        st.caption(f"📈 {_p['bet_type']} {_p['label']} 觸發的看盤規則：" + "；".join(_p['kp_fired']))
            else:
                mc1.metric("💡 首選推薦", "⛔ 本場不建議投注")
                _top = cand_list[0] if cand_list else None
                mc2.metric("🎯 最佳候選 EV", f"{_top['ev']:+.3f}" if _top else "-")
                mc3.metric("📏 EV 門檻", f"{v9_set['min_ev']*100:.1f}%")
                st.info("系統未找到 EV 達標且未被熔斷的盤口。不投注本身就是最有效的風險控制：在馬會約 5%~8% 抽水下，沒有優勢的投注長期必然虧損。")
                _pp = res.get('paper_picks', [])
                if _pp:
                    st.markdown("**📝 模擬首選 (注碼 $0，只供追蹤)：** " + "；".join(f"{p['bet_type']} {p['label']} @ {p['odds']} (估計 EV {p.get('ev', 0)*100:+.1f}%，{p.get('status', '')})" for p in _pp))
                    st.caption("模擬注代表模型有傾向，但該策略未通過樣本外驗證 (或屬新聯賽)。你可以把它記錄為系統注碼 $0 的注單，賽果同樣會用作學習；"
                               "當策略在擂台累積足夠的正回報後，系統會自動改為實注。")

            st.markdown("---")
            st.markdown("### 🎯 最終投注決策與注碼配置 (已簡化分組選單)")

            col_sys, col_usr = st.columns(2)

            with col_sys:
                st.markdown("#### 🤖 系統投注 (System)")
                st.caption(f"供模型學習及系統資金策略驗證使用。注碼 = 分數 Kelly ({v9_set['kelly']:.2f}) × 優勢，單注上限 {v9_set['max_pct']*100:.1f}% 本金。")
                default_sys = []
                if is_editing and st.session_state.get('edit_sys_stake', 0) > 0:
                    eb = st.session_state.get('edit_bet_type')
                    el = st.session_state.get('edit_line')
                    es = st.session_state.get('edit_selection')
                    for opt, c in option_map.items():
                        if c['bet_type'] == eb and abs(float(c['line']) - float(el)) < 1e-4 and c['selection'] == es:
                            default_sys.append(opt)
                            break
                elif not is_editing:
                    for p in picks:
                        for opt, c in option_map.items():
                            if c['bet_type'] == p['bet_type'] and abs(float(c['line']) - float(p['line'])) < 1e-4 and c['selection'] == p['selection'] and opt not in default_sys:
                                default_sys.append(opt)
                                break

                sys_selected = st.multiselect("選擇系統投注項目", all_options, default=default_sys, key="sys_multi")
                sys_stakes = {}
                for sel in sys_selected:
                    cand = option_map[sel]
                    sug_stk, raw_stk = calc_suggested_stake(cand, sys_bankroll, sys_max_stake)
                    # 修改模式下，若此項目對應原本下注的項目，還原原本已輸入的系統注碼
                    _stk_val = float(sug_stk)
                    if is_editing and sel in default_sys:
                        _orig_stk = float(st.session_state.get('edit_sys_stake', 0.0) or 0.0)
                        if _orig_stk > 0:
                            _stk_val = _orig_stk
                    sys_stakes[sel] = st.number_input(f"系統建議金額: {sel}", value=_stk_val, disabled=True, key=f"s_stk_{sel}")
                    if _stk_val <= 0:
                        st.caption(f"⚠ `{cand['bet_type']}` {cand.get('status', '')}：EV {cand.get('ev', 0):+.3f}，系統建議注碼 $0 (只作記錄供學習)")

            with col_usr:
                st.markdown("#### 👤 用家投注 (User)")
                st.caption("真實資金決策，不用於系統學習。建議只跟隨「✅ 建議」項目，並使用不高於系統建議的注碼。")
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
                    _uc = option_map[sel]
                    if not str(_uc.get('status', '')).startswith('✅'):
                        st.caption(f"⚠ `{sel}` 不在系統建議內 ({_uc.get('status', '')}，EV {_uc.get('ev', 0):+.3f})，請審慎考慮。")
                    usr_stakes[sel] = st.number_input(f"用家自訂金額 ($): {sel}", min_value=0.0, step=10.0, value=def_val, key=f"u_stk_{sel}")
            st.write("")
            submit_btn_label = f"🔄 確定修改並覆蓋雲端資料庫 (ID: {st.session_state.editing_bet_id})" if is_editing else "✅ 確定投注並寫入雲端資料庫"
            
            if st.button(submit_btn_label, type="primary", use_container_width=True):
                all_keys = set(sys_selected + usr_selected)
                if not all_keys:
                    st.warning("⚠️ 請至少在系統或用家選擇一項投注！")
                else:
                    target_id = st.session_state.get('editing_bet_id')
                    orig_record = None
                    if target_id:
                        orig_rows = st.session_state.df_db[st.session_state.df_db['ID'] == target_id]
                        if not orig_rows.empty:
                            orig_record = orig_rows.iloc[0].to_dict()
                        st.session_state.df_db = st.session_state.df_db[st.session_state.df_db['ID'] != target_id]
                    
                    new_records = []
                    _dup_skipped = []
                    _dbx = st.session_state.df_db
                    _dbx_m = _dbx['Match'].astype(str).map(v11_clean_name) if not _dbx.empty else pd.Series(dtype=str)
                    _dbx_md = v10_md_series(_dbx) if not _dbx.empty else pd.Series(dtype=str)
                    for i, sel in enumerate(all_keys):
                        cand = option_map[sel]
                        s_stk = sys_stakes.get(sel, 0.0)
                        u_stk = usr_stakes.get(sel, 0.0)
                        # V11：防止重複注單 (同一比賽、日期、盤口、方向、盤口線、賠率)
                        if not _dbx.empty:
                            _same = _dbx[(_dbx_m == f"{home_team} vs {away_team}") & (_dbx_md == match_date) &
                                         (_dbx['Bet_Type'].astype(str).str.replace(' (即場)', '', regex=False) == str(cand['bet_type'])) &
                                         (_dbx['Selection'].astype(str) == str(cand['selection'])) &
                                         ((pd.to_numeric(_dbx['Initial_Line'], errors='coerce') - float(cand['line'])).abs() < 1e-6) &
                                         ((pd.to_numeric(_dbx['Initial_Odds'], errors='coerce') - float(cand['odds'])).abs() < 1e-6)]
                            if not _same.empty:
                                _dup_skipped.append(f"{sel} (已存在注單 {', '.join(_same['ID'].astype(str))})")
                                continue
                        
                        # 修改模式下：對應原本項目的注單沿用原 ID；
                        # 若用家改變了投注方向/盤口 (無法對應原本項目) 而只選了一項，
                        # 該項目仍視為對原注單的覆寫 (沿用原 ID 及原有其他欄位)
                        _is_orig = False
                        if target_id:
                            eb = st.session_state.get('edit_bet_type')
                            el = st.session_state.get('edit_line')
                            es = st.session_state.get('edit_selection')
                            if eb and el is not None and es and cand['bet_type'] == eb and abs(float(cand['line']) - float(el)) < 1e-4 and cand['selection'] == es:
                                _is_orig = True
                            elif len(all_keys) == 1:
                                _is_orig = True
                        
                        if _is_orig and target_id:
                            new_id = target_id
                        else:
                            new_id = f"B{get_hkt_now().strftime('%Y%m%d%H%M%S')}{i}"
                        
                        if _is_orig and orig_record is not None:
                            # 修改模式：以原注單完整資料為基礎，只覆寫表單有關欄位，
                            # 保留原有 ID、記錄日期及所有其他欄位 (含即場數據、結算欄位等)
                            _orig_bt = str(orig_record.get('Bet_Type', ''))
                            _keep_live_suffix = ' (即場)' in _orig_bt
                            _final_bt = cand['bet_type'] + (' (即場)' if _keep_live_suffix and ' (即場)' not in cand['bet_type'] else '')
                            new_record = dict(orig_record)
                            new_record.update({
                                'ID': new_id, 'Status': 'Open',
                                'Tournament_Name': res['t_name'], 'Tournament_Category': res['t_cat'], 
                                'Match': f"{home_team} vs {away_team}", 'Match_Date': match_date, 'Home_Team': home_team, 'Away_Team': away_team,
                                'Home_Rating': 'NA' if no_rank else _final_rating_value(st.session_state.get('edit_h_rating_raw'), home_rank, default_h_rank), 'Away_Rating': 'NA' if no_rank else _final_rating_value(st.session_state.get('edit_a_rating_raw'), away_rank, default_a_rank), 'League_Size': int(league_size), 'Home_Form': home_form, 'Away_Form': away_form,
                                'Bet_Type': _final_bt, 'Selection': cand['selection'], 'Initial_Line': cand['line'], 'Initial_Odds': cand['odds'], 
                                'System_Stake': s_stk, 'User_Stake': u_stk,
                                'Odds_History': json.dumps(st.session_state.odds_history, ensure_ascii=False)
                            })
                        else:
                            new_record = {
                                'ID': new_id, 'Date': get_hkt_now().strftime('%Y-%m-%d %H:%M'), 'Status': 'Open',
                                'Tournament_Name': res['t_name'], 'Tournament_Category': res['t_cat'], 
                                'Match': f"{home_team} vs {away_team}", 'Match_Date': match_date, 'Home_Team': home_team, 'Away_Team': away_team,
                                'Home_Rating': 'NA' if no_rank else int(home_rank), 'Away_Rating': 'NA' if no_rank else int(away_rank), 'League_Size': int(league_size), 'Home_Form': home_form, 'Away_Form': away_form,
                                'Bet_Type': cand['bet_type'], 'Selection': cand['selection'], 'Initial_Line': cand['line'], 'Initial_Odds': cand['odds'], 
                                'System_Stake': s_stk, 'User_Stake': u_stk,
                                'Odds_History': json.dumps(st.session_state.odds_history, ensure_ascii=False)
                            }
                        new_records.append(new_record)
                    if _dup_skipped:
                        st.warning("⚠ 以下項目與數據庫現有注單完全相同，已略過以免重複計算：" + "；".join(_dup_skipped))
                    if not new_records:
                        if target_id and orig_record is not None:
                            st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([orig_record])], ignore_index=True)
                        st.stop()
                    v11_save_league_size(res['t_name'], int(league_size), silent=True)
                        
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
        
        if st.session_state.df_db.empty:
            st.info("目前數據庫中尚無任何賽事紀錄。請先至『賽前建檔與投注』新增賽事。")
        else:
            open_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
            if open_df.empty:
                st.info("💡 當前沒有未結算 (Open) 的賽事，您可以選擇歷史賽事進行即場數據更新：")
                match_groups = v10_match_groups(st.session_state.df_db)
            else:
                match_groups = v10_match_groups(open_df)

            # V10：比賽按『賽事 + 對賽 + 比賽日期』區分，同一對球隊在不同日子的比賽不會再被合併
            _ip_opt_map = {f"{r['_md']} | {r['Tournament_Name']} | {r['Match']} (注單數: {r['Bet_Count']} 張)": (r['Tournament_Name'], r['Match'], r['_md'])
                           for _, r in match_groups.iterrows()}
            match_options = list(_ip_opt_map.keys())
            sel_match_str = st.selectbox("⚽ 請選擇要更新即場數據的比賽場次 (按比賽日期區分)：", match_options, key="inplay_match_select")

            if sel_match_str and sel_match_str in _ip_opt_map:
                selected_tourn, selected_match, selected_md = _ip_opt_map[sel_match_str]
                _ip_sel_mask = v10_match_mask(st.session_state.df_db, selected_tourn, selected_match, selected_md)
                matching_bets = st.session_state.df_db[_ip_sel_mask]
                _ip_match_key = f"{selected_tourn}|{selected_match}|{selected_md}"
                # V10：切換比賽時清除上一場的即場輸入值，避免把上一場的賽況誤存到另一場比賽
                if st.session_state.get('_ip_last_match_key') != _ip_match_key:
                    for _k in ['ip_minute', 'ip_hg', 'ip_ag', 'ip_hc', 'ip_ac', 'ip_hred', 'ip_ared', 'ip_hda', 'ip_ada',
                               'ip_hsot', 'ip_asot', 'ip_hsoff', 'ip_asoff', 'ip_hposs', 'ip_aposs',
                               'ip_hd_line', 'ip_hd_upper', 'ip_hd_lower', 'ip_hd_manual', 'ip_ou_line', 'ip_ou_upper', 'ip_ou_lower', 'ip_ou_manual',
                               'ip_cr_line', 'ip_cr_upper', 'ip_cr_lower', 'ip_cr_manual', 'ip_hd_on', 'ip_ou_on', 'ip_cr_on']:
                        st.session_state.pop(_k, None)
                    st.session_state['_ip_last_match_key'] = _ip_match_key
                if st.session_state.get('inplay_analysis_result', {}).get('match_key') not in (None, _ip_match_key):
                    st.session_state.show_inplay_analysis = False

                st.write(f"**該比賽包含的注單列表 ({len(matching_bets)} 張):**")
                st.dataframe(matching_bets[['ID', 'Date', 'Match_Date', 'Status', 'Bet_Type', 'Selection', 'Initial_Line', 'Initial_Odds', 'System_Stake', 'User_Stake']], use_container_width=True)

                last_row = matching_bets.iloc[-1]
                def_minute = int(float(last_row.get('InPlay_Minute', 45))) if pd.notna(last_row.get('InPlay_Minute')) else 45
                def_hg = int(float(last_row.get('Home_Goal', 0))) if pd.notna(last_row.get('Home_Goal')) else 0
                def_ag = int(float(last_row.get('Away_Goal', 0))) if pd.notna(last_row.get('Away_Goal')) else 0
                def_hc = int(float(last_row.get('Home_Corner', 0))) if pd.notna(last_row.get('Home_Corner')) else 0
                def_ac = int(float(last_row.get('Away_Corner', 0))) if pd.notna(last_row.get('Away_Corner')) else 0
                def_hred = int(float(last_row.get('Home_Red', 0))) if pd.notna(last_row.get('Home_Red')) else 0
                def_ared = int(float(last_row.get('Away_Red', 0))) if pd.notna(last_row.get('Away_Red')) else 0
                def_hda = int(float(last_row.get('Home_DA', 0))) if pd.notna(last_row.get('Home_DA')) else 0
                def_ada = int(float(last_row.get('Away_DA', 0))) if pd.notna(last_row.get('Away_DA')) else 0
                def_hsot = int(float(last_row.get('Home_SoT', 0))) if pd.notna(last_row.get('Home_SoT')) else 0
                def_asot = int(float(last_row.get('Away_SoT', 0))) if pd.notna(last_row.get('Away_SoT')) else 0
                def_hsoff = int(float(last_row.get('Home_SoFF', 0))) if pd.notna(last_row.get('Home_SoFF')) else 0
                def_asoff = int(float(last_row.get('Away_SoFF', 0))) if pd.notna(last_row.get('Away_SoFF')) else 0
                def_hposs = int(float(last_row.get('Home_Possession', 50))) if pd.notna(last_row.get('Home_Possession')) else 50

                st.divider()
                st.markdown("##### 1. 實時賽況與進攻數據輸入 (按場次統一套用)")
                
                col_time, col_ip1, col_ip2, col_ip3 = st.columns([2, 3, 3, 3])
                ip_minute = col_time.number_input("比賽時間 (分鐘)", 0, 120, def_minute, key="ip_minute")
                
                h_g = col_ip1.number_input("主隊入球", 0, 50, def_hg, key="ip_hg")
                a_g = col_ip1.number_input("客隊入球", 0, 50, def_ag, key="ip_ag")
                h_c = col_ip2.number_input("主隊角球", 0, 50, def_hc, key="ip_hc")
                a_c = col_ip2.number_input("客隊角球", 0, 50, def_ac, key="ip_ac")
                h_red = col_ip3.number_input("主隊紅牌", 0, 10, def_hred, key="ip_hred")
                a_red = col_ip3.number_input("客隊紅牌", 0, 10, def_ared, key="ip_ared")
                
                c1, c2, c3, c4 = st.columns(4)
                h_da = c1.number_input("主隊危險進攻", 0, 200, def_hda, key="ip_hda")
                a_da = c1.number_input("客隊危險進攻", 0, 200, def_ada, key="ip_ada")
                h_sot = c2.number_input("主隊射正", 0, 50, def_hsot, key="ip_hsot")
                a_sot = c2.number_input("客隊射正", 0, 50, def_asot, key="ip_asot")
                h_soff = c3.number_input("主隊射偏", 0, 50, def_hsoff, key="ip_hsoff")
                a_soff = c3.number_input("客隊射偏", 0, 50, def_asoff, key="ip_asoff")
                h_poss = c4.number_input("主隊控球率 (%)", 0, 100, def_hposs, key="ip_hposs")
                a_poss = 100 - h_poss
                st.caption(f"客隊控球率自動計算為: {a_poss}%")

                # 自動計算指標
                h_total_shots = h_sot + h_soff
                a_total_shots = a_sot + a_soff

                h_conversion = (h_g / h_total_shots * 100.0) if h_total_shots > 0 else 0.0
                a_conversion = (a_g / a_total_shots * 100.0) if a_total_shots > 0 else 0.0

                h_firepower = (h_total_shots / h_da * 100.0) if h_da > 0 else 0.0
                a_firepower = (a_total_shots / a_da * 100.0) if a_da > 0 else 0.0

                h_corner_eff = (h_c / h_da * 100.0) if h_da > 0 else 0.0
                a_corner_eff = (a_c / a_da * 100.0) if a_da > 0 else 0.0

                h_efficiency = (h_firepower / h_poss * 100.0) if h_poss > 0 else 0.0
                a_efficiency = (a_firepower / a_poss * 100.0) if a_poss > 0 else 0.0

                st.markdown("##### 2. 自動計算實時進攻效率與指標看板")
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.markdown("**🏠 主隊 (Home Team)**")
                    st.metric("射球命中率", f"{h_conversion:.2f}%", help="入球數量 / (射正次數 + 射偏次數) * 100%")
                    st.metric("進攻火力", f"{h_firepower:.2f}%", help="(射正次數 + 射偏次數) / 危險進攻次數 * 100%")
                    st.metric("進攻產生角球效率", f"{h_corner_eff:.2f}%", help="角球數量 / 危險進攻數量 * 100%")
                    st.metric("實際進攻效率", f"{h_efficiency:.2f}%", help="進攻火力 / 控球率 * 100%")
                with m_col2:
                    st.markdown("**✈️ 客隊 (Away Team)**")
                    st.metric("射球命中率", f"{a_conversion:.2f}%", help="入球數量 / (射正次數 + 射偏次數) * 100%")
                    st.metric("進攻火力", f"{a_firepower:.2f}%", help="(射正次數 + 射偏次數) / 危險進攻次數 * 100%")
                    st.metric("進攻產生角球效率", f"{a_corner_eff:.2f}%", help="角球數量 / 危險進攻數量 * 100%")
                    st.metric("實際進攻效率", f"{a_efficiency:.2f}%", help="進攻火力 / 控球率 * 100%")

                if st.button("💾 儲存實時賽況至此比賽的所有注單", type="primary", use_container_width=True):
                    mask = v10_match_mask(st.session_state.df_db, selected_tourn, selected_match, selected_md)
                    st.session_state.df_db.loc[mask, 'InPlay_Minute'] = ip_minute
                    st.session_state.df_db.loc[mask, 'Home_Goal'] = h_g
                    st.session_state.df_db.loc[mask, 'Away_Goal'] = a_g
                    st.session_state.df_db.loc[mask, 'Home_Corner'] = h_c
                    st.session_state.df_db.loc[mask, 'Away_Corner'] = a_c
                    st.session_state.df_db.loc[mask, 'Home_DA'] = h_da
                    st.session_state.df_db.loc[mask, 'Away_DA'] = a_da
                    st.session_state.df_db.loc[mask, 'Home_SoT'] = h_sot
                    st.session_state.df_db.loc[mask, 'Away_SoT'] = a_sot
                    st.session_state.df_db.loc[mask, 'Home_SoFF'] = h_soff
                    st.session_state.df_db.loc[mask, 'Away_SoFF'] = a_soff
                    st.session_state.df_db.loc[mask, 'Home_Red'] = h_red
                    st.session_state.df_db.loc[mask, 'Away_Red'] = a_red
                    st.session_state.df_db.loc[mask, 'Home_Possession'] = h_poss
                    st.session_state.df_db.loc[mask, 'Away_Possession'] = a_poss
                    st.session_state.df_db.loc[mask, 'Home_Goal_Conversion'] = round(h_conversion, 2)
                    st.session_state.df_db.loc[mask, 'Away_Goal_Conversion'] = round(a_conversion, 2)
                    st.session_state.df_db.loc[mask, 'Home_Firepower'] = round(h_firepower, 2)
                    st.session_state.df_db.loc[mask, 'Away_Firepower'] = round(a_firepower, 2)
                    st.session_state.df_db.loc[mask, 'Home_Corner_Eff'] = round(h_corner_eff, 2)
                    st.session_state.df_db.loc[mask, 'Away_Corner_Eff'] = round(a_corner_eff, 2)
                    
                    save_db(st.session_state.df_db, db_file, db_table)
                    st.toast(f"✅ 已成功更新『{selected_match}』({selected_md}) 共 {mask.sum()} 筆注單的即場數據！", icon="💾")
                    st.rerun()

                # ==========================================
                # 3. 即場投注 (Live Betting) — 賠率連動與手動/自動切換
                # ==========================================
                st.markdown("---")
                st.markdown("##### 3. 即場投注 (Live Betting)")
                st.caption("💡 當您輸入或修改「大/上賠率」後，系統自動依馬會抽水公式 (Margin=1.085) 計算「小/下賠率」。如需覆蓋，直接修改「小/下賠率」即可，系統將以手動值為準並更新抽水。")

                # --- 初始化即場投注 session_state ---
                _ip_prefixes = ["ip_hd", "ip_ou", "ip_cr"]
                _ip_defaults = {
                    "ip_hd": {"line": 0.0, "upper": 1.90},
                    "ip_ou": {"line": 2.5, "upper": 1.90},
                    "ip_cr": {"line": 9.5, "upper": 1.90},
                }
                for _pf in _ip_prefixes:
                    if f"{_pf}_line" not in st.session_state:
                        st.session_state[f"{_pf}_line"] = _ip_defaults[_pf]["line"]
                    if f"{_pf}_upper" not in st.session_state:
                        st.session_state[f"{_pf}_upper"] = _ip_defaults[_pf]["upper"]
                    if f"{_pf}_lower" not in st.session_state:
                        st.session_state[f"{_pf}_lower"] = _calc_lower_from_upper(_ip_defaults[_pf]["upper"])
                    if f"{_pf}_manual" not in st.session_state:
                        st.session_state[f"{_pf}_manual"] = False

                st.caption("💡 並非每場比賽都有全部即場盤口 (例如沒有即場讓球)。取消勾選即可關閉該盤口的輸入，分析及投注選項亦會自動排除。")
                bet_col1, bet_col2, bet_col3 = st.columns(3)

                with bet_col1:
                    st.markdown("**⚽ 讓球 (Handicap)**")
                    ip_hd_on = st.checkbox("此場有此即場盤口", value=True, key="ip_hd_on")
                    ip_hd_line = st.number_input("讓球盤口線", min_value=-10.0, max_value=10.0, step=0.25, format="%.2f", key="ip_hd_line", disabled=not ip_hd_on)
                    ip_hd_upper = st.number_input("大/上賠率 (主隊)", min_value=1.01, max_value=100.0, step=0.01, format="%.2f", key="ip_hd_upper", disabled=not ip_hd_on, on_change=auto_calc_lower, args=("ip_hd",))
                    ip_hd_lower = st.number_input("小/下賠率 (客隊)", min_value=1.01, max_value=100.0, step=0.01, format="%.2f", key="ip_hd_lower", disabled=not ip_hd_on, on_change=mark_manual_lower, args=("ip_hd",))
                    _hd_mode = "手動" if st.session_state.ip_hd_manual else "自動"
                    st.caption(f"抽水: {calc_live_margin_str(ip_hd_upper, ip_hd_lower)} [{_hd_mode}]")

                with bet_col2:
                    st.markdown("**⚽ 入球大小 (Goals O/U)**")
                    ip_ou_on = st.checkbox("此場有此即場盤口", value=True, key="ip_ou_on")
                    ip_ou_line = st.number_input("入球大小盤口線", min_value=0.0, max_value=20.0, step=0.25, format="%.2f", key="ip_ou_line", disabled=not ip_ou_on)
                    ip_ou_upper = st.number_input("大賠率 (Over)", min_value=1.01, max_value=100.0, step=0.01, format="%.2f", key="ip_ou_upper", disabled=not ip_ou_on, on_change=auto_calc_lower, args=("ip_ou",))
                    ip_ou_lower = st.number_input("小賠率 (Under)", min_value=1.01, max_value=100.0, step=0.01, format="%.2f", key="ip_ou_lower", disabled=not ip_ou_on, on_change=mark_manual_lower, args=("ip_ou",))
                    _ou_mode = "手動" if st.session_state.ip_ou_manual else "自動"
                    st.caption(f"抽水: {calc_live_margin_str(ip_ou_upper, ip_ou_lower)} [{_ou_mode}]")

                with bet_col3:
                    st.markdown("**⚽ 角球大小 (Corners O/U)**")
                    ip_cr_on = st.checkbox("此場有此即場盤口", value=True, key="ip_cr_on")
                    ip_cr_line = st.number_input("角球盤口線", min_value=0.0, max_value=30.0, step=0.5, format="%.2f", key="ip_cr_line", disabled=not ip_cr_on)
                    ip_cr_upper = st.number_input("大賠率 (Over)", min_value=1.01, max_value=100.0, step=0.01, format="%.2f", key="ip_cr_upper", disabled=not ip_cr_on, on_change=auto_calc_lower, args=("ip_cr",))
                    ip_cr_lower = st.number_input("小賠率 (Under)", min_value=1.01, max_value=100.0, step=0.01, format="%.2f", key="ip_cr_lower", disabled=not ip_cr_on, on_change=mark_manual_lower, args=("ip_cr",))
                    _cr_mode = "手動" if st.session_state.ip_cr_manual else "自動"
                    st.caption(f"抽水: {calc_live_margin_str(ip_cr_upper, ip_cr_lower)} [{_cr_mode}]")

                # --- 即場數據分析執行 ---
                _ip_on = {'讓球': ip_hd_on, '入球大小': ip_ou_on, '角球大小': ip_cr_on}
                if not any(_ip_on.values()):
                    st.warning("⚠ 已關閉全部即場盤口，請至少勾選一個有即場投注的盤口才可執行分析。")
                if st.button("🚀 即場數據分析執行", type="primary", use_container_width=True, key="btn_inplay_analysis", disabled=not any(_ip_on.values())):
                    st.session_state.show_inplay_analysis = True
                    v9_set = v9_get_settings()
                    # 即場沒有歷史即場賠率可供驗證，因此提高 EV 門檻 2% 作安全邊際
                    v9_set_ip = dict(v9_set)
                    v9_set_ip['min_ev'] = v9_set['min_ev'] + 0.02
                    engine, mt_hist, tl_hist = v9_get_engine(st.session_state.df_db)

                    # 取得比賽資料 (優先使用賽前注單作為賽前資料來源)
                    _pre_rows = matching_bets[~matching_bets['Bet_Type'].astype(str).str.contains('即場', na=False)]
                    ip_last_row = _pre_rows.iloc[-1] if not _pre_rows.empty else matching_bets.iloc[-1]
                    ip_t_name = str(ip_last_row.get('Tournament_Name', '')) if pd.notna(ip_last_row.get('Tournament_Name')) else ''
                    ip_t_cat = str(ip_last_row.get('Tournament_Category', CATEGORY_OPTIONS[0])) if pd.notna(ip_last_row.get('Tournament_Category')) else CATEGORY_OPTIONS[0]
                    ip_h_team = str(ip_last_row.get('Home_Team', '')) if pd.notna(ip_last_row.get('Home_Team')) else ''
                    ip_a_team = str(ip_last_row.get('Away_Team', '')) if pd.notna(ip_last_row.get('Away_Team')) else ''
                    ip_h_rating = str(ip_last_row.get('Home_Rating', 'C')) if pd.notna(ip_last_row.get('Home_Rating')) else 'C'
                    ip_a_rating = str(ip_last_row.get('Away_Rating', 'C')) if pd.notna(ip_last_row.get('Away_Rating')) else 'C'
                    ip_h_form = str(ip_last_row.get('Home_Form', '3W1D1L')) if pd.notna(ip_last_row.get('Home_Form')) else '3W1D1L'
                    ip_a_form = str(ip_last_row.get('Away_Form', '2W2D1L')) if pd.notna(ip_last_row.get('Away_Form')) else '2W2D1L'

                    # 聯賽排名 → 實力分 (新數據為數字排名，舊數據為字母自動換算)
                    ip_league_size = get_league_size(ip_last_row)
                    ip_hr_val = rating_to_strength(ip_h_rating, ip_league_size)
                    ip_ar_val = rating_to_strength(ip_a_rating, ip_league_size)
                    ip_match_date = get_row_match_date(ip_last_row)
                    try:
                        _ip_md_ts = pd.Timestamp(ip_match_date)
                    except Exception:
                        _ip_md_ts = pd.Timestamp(get_hkt_now().strftime('%Y-%m-%d'))
                    ip_ctx = v9_match_context(mt_hist, tl_hist, ip_h_team, ip_a_team, ip_t_name, _ip_md_ts, ip_hr_val, ip_ar_val,
                                              float(extract_form_points(ip_h_form)), float(extract_form_points(ip_a_form)))

                    # 賽前主盤賠率 → 市場隱含 λ；再按剩餘時間、攻勢強度、紅牌調整為剩餘時間 λ
                    try:
                        _pre_oh = json.loads(str(ip_last_row.get('Odds_History', '[]')))
                        if not isinstance(_pre_oh, list):
                            _pre_oh = []
                    except Exception:
                        _pre_oh = []
                    _lh, _la, _lc, _lsrc = v9_prematch_lambdas(_pre_oh, ip_ctx)
                    _rlh, _rla, _rlc, _ints = v9_inplay_lambdas(_lh, _la, _lc, ip_minute, h_red, a_red,
                                                                h_sot + h_soff, a_sot + a_soff, h_da, a_da)

                    def _ip_cand(bt, sel, line, odds, opp_odds, label, hidx):
                        if bt == '讓球':
                            er = v9_expected_ret_hd(sel, float(line), h_g, a_g, _rlh, _rla, float(odds))
                        elif bt == '入球大小':
                            er = v9_expected_ret_total(sel, float(line), h_g + a_g, _rlh + _rla, float(odds))
                        else:
                            er = v9_expected_ret_total(sel, float(line), h_c + a_c, _rlc, float(odds))
                        p_pois = float(np.clip((er + 1.0) / float(odds), 0.0, 1.0))
                        p_mkt_ip, _ = v9_devig(float(odds), float(opp_odds))
                        # 即場：市場公平機率與 Poisson 模型各佔一半 (市場錨定，避免模型過度自信)
                        p_final = 0.5 * p_mkt_ip + 0.5 * p_pois
                        return {'bet_type': bt, 'selection': sel, 'base_prob': p_pois, 'base_prob_override': p_final, 'p_pois': p_pois,
                                'odds': float(odds), 'line': float(line), 'label': label, 'history_idx': hidx,
                                'record_time': get_hkt_now().strftime('%d-%m %H:%M'), 'odds_drift': 0.0, 'is_latest': True}

                    _hd_line_str = f"{ip_hd_line:g}"
                    if ip_hd_line <= 0:
                        _hd_label_h, _hd_label_a = f"{_hd_line_str}主隊(上盤)", f"{_hd_line_str}客隊(下盤)"
                    else:
                        _hd_label_h, _hd_label_a = f"{_hd_line_str}主隊(下盤)", f"{_hd_line_str}客隊(上盤)"
                    _ou_line_str = f"{ip_ou_line:g}"
                    _cr_line_str = f"{ip_cr_line:g}"
                    ip_candidates_base = [
                        _ip_cand('讓球', 'Home', ip_hd_line, ip_hd_upper, ip_hd_lower, _hd_label_h, 0),
                        _ip_cand('讓球', 'Away', ip_hd_line, ip_hd_lower, ip_hd_upper, _hd_label_a, 0),
                        _ip_cand('入球大小', 'Over', ip_ou_line, ip_ou_upper, ip_ou_lower, f"{_ou_line_str}大盤(Over)", 1),
                        _ip_cand('入球大小', 'Under', ip_ou_line, ip_ou_lower, ip_ou_upper, f"{_ou_line_str}小盤(Under)", 1),
                        _ip_cand('角球大小', 'Over', ip_cr_line, ip_cr_upper, ip_cr_lower, f"{_cr_line_str}大盤(Over)", 2),
                        _ip_cand('角球大小', 'Under', ip_cr_line, ip_cr_lower, ip_cr_upper, f"{_cr_line_str}小盤(Under)", 2),
                    ]
                    ip_candidates_base = [c for c in ip_candidates_base if _ip_on.get(c['bet_type'], True)]
                    ip_scored = v9_score_candidates(engine, ip_candidates_base, ip_ctx, v9_set_ip, {}) if ip_candidates_base else []
                    ip_picks, _ip_used = [], set()
                    for c in ip_scored:
                        if c.get('status') == '✅ 建議' and c['bet_type'] not in _ip_used:
                            ip_picks.append(c)
                            _ip_used.add(c['bet_type'])
                            if len(ip_picks) >= v9_set['max_picks']:
                                break

                    df_settled_ip = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()

                    # 建立即場賠率歷史 JSON
                    _ip_now = get_hkt_now().strftime('%d-%m %H:%M')
                    ip_odds_history = [
                        {"id": 0, "type": "讓球", "record_time": _ip_now, "line": float(ip_hd_line), "upper": float(ip_hd_upper), "lower": float(ip_hd_lower), "unlock": True, "margin": round(1.0/ip_hd_upper + 1.0/ip_hd_lower, 3) if ip_hd_upper > 0 and ip_hd_lower > 0 else 1.085},
                        {"id": 1, "type": "入球大小", "record_time": _ip_now, "line": float(ip_ou_line), "upper": float(ip_ou_upper), "lower": float(ip_ou_lower), "unlock": True, "margin": round(1.0/ip_ou_upper + 1.0/ip_ou_lower, 3) if ip_ou_upper > 0 and ip_ou_lower > 0 else 1.085},
                        {"id": 2, "type": "角球大小", "record_time": _ip_now, "line": float(ip_cr_line), "upper": float(ip_cr_upper), "lower": float(ip_cr_lower), "unlock": True, "margin": round(1.0/ip_cr_upper + 1.0/ip_cr_lower, 3) if ip_cr_upper > 0 and ip_cr_lower > 0 else 1.085},
                    ]
                    ip_odds_history = [h for h in ip_odds_history if _ip_on.get(h['type'], True)]

                    st.session_state.inplay_analysis_result = {
                        'v9': True,
                        'dims': [v9_dimension_stats(df_settled_ip[df_settled_ip['Tournament_Name'] == ip_t_name], "A. 微觀 (賽事名稱)"),
                                 v9_dimension_stats(df_settled_ip[df_settled_ip['Tournament_Category'] == ip_t_cat], "B. 中觀 (賽事分類)"),
                                 v9_dimension_stats(df_settled_ip, "C. 宏觀 (全局數據)")],
                        'candidates': ip_scored, 'picks': ip_picks, 'min_ev': v9_set_ip['min_ev'],
                        'lambdas_pre': (_lh, _la, _lc, _lsrc), 'lambdas_rem': (_rlh, _rla, _rlc, _ints),
                        't_name': ip_t_name, 't_cat': ip_t_cat,
                        'h_team': ip_h_team, 'a_team': ip_a_team,
                        'h_rating': ip_h_rating, 'a_rating': ip_a_rating,
                        'league_size': ip_league_size,
                        'h_form': ip_h_form, 'a_form': ip_a_form,
                        'odds_history': ip_odds_history,
                        'remaining_min': max(0, 90 - ip_minute),
                        'match_date': ip_match_date,
                        'match_key': _ip_match_key,
                        'markets_on': [k for k, v in _ip_on.items() if v],
                    }

                if st.session_state.get('show_inplay_analysis', False) and not st.session_state.get('inplay_analysis_result', {}).get('v9'):
                    st.session_state.show_inplay_analysis = False

                # --- 顯示即場分析結果 ---
                if st.session_state.get('show_inplay_analysis', False):
                    ip_res = st.session_state.inplay_analysis_result
                    st.success("✅ V9 即場 Poisson + 市場錨定分析完成！")
                    _plh, _pla, _plc, _psrc = ip_res['lambdas_pre']
                    _rlh, _rla, _rlc, _ints = ip_res['lambdas_rem']
                    st.info(f"⏱ 剩餘比賽時間: {ip_res['remaining_min']} 分鐘 | 賽前隱含預期: 主 {_plh:.2f} 球 / 客 {_pla:.2f} 球 / 角球 {_plc:.1f} "
                            f"| 剩餘時間預期: 主 +{_rlh:.2f} 球 / 客 +{_rla:.2f} 球 / 角球 +{_rlc:.1f} (攻勢係數 主 {_ints[0]:.2f}、客 {_ints[1]:.2f}、角球 {_ints[2]:.2f})")
                    st.caption("來源：" + ", ".join(f"{k}={v}" for k, v in (_psrc or {}).items()) + "。即場模型 = 50% 去水市場機率 + 50% Poisson 剩餘時間模型；EV 門檻自動提高 2%。")

                    ip_c1, ip_c2, ip_c3 = st.columns(3)
                    for col, dstat in zip([ip_c1, ip_c2, ip_c3], ip_res['dims']):
                        with col.container(border=True):
                            st.markdown(f"**{dstat['dim']}**")
                            if dstat['valid']:
                                st.write(f"樣本數: `{dstat['n']}` 張 (系統實注 `{dstat['n_bet']}` 張)")
                                st.write(f"系統實注 ROI: `{dstat['roi']*100:.1f}%`")
                                st.write(f"歷史勝率: `{dstat['acc']*100:.1f}%`")
                            else:
                                st.warning(dstat['msg'])

                    st.markdown("### 🧠 AI 預測模型推薦 (即場)")
                    st.markdown("#### 📊 所有盤口評估明細 (即場 EV 運算)")
                    ip_cand_list = ip_res.get('candidates', [])

                    ip_all_options = []
                    ip_option_map = {}

                    if ip_cand_list:
                        ip_df_show = pd.DataFrame(ip_cand_list)
                        ip_df_show['推薦排序'] = range(1, len(ip_df_show) + 1)
                        ip_df_show = ip_df_show[['推薦排序', 'status', 'bet_type', 'line', 'label', 'odds', 'p_mkt', 'p_pois', 'prob', 'relative_prob', 'ev']]
                        ip_df_show.columns = ['推薦排序', '狀態', '盤口類型', '盤口線', '投注方向', '賠率', '市場公平勝率', 'Poisson 有效勝率', '最終有效勝率', '兩邊相對化', '期望值 (EV)']
                        for _c in ['市場公平勝率', 'Poisson 有效勝率', '最終有效勝率', '兩邊相對化']:
                            ip_df_show[_c] = ip_df_show[_c].apply(lambda x: f"{x*100:.2f}%")
                        ip_df_show['期望值 (EV)'] = ip_df_show['期望值 (EV)'].apply(lambda x: f"{x:+.3f}")

                        def _highlight_first_ip(row):
                            if str(row['狀態']).startswith('✅'):
                                return ['background-color: rgba(40, 167, 69, 0.2)'] * len(row)
                            return [''] * len(row)

                        st.dataframe(ip_df_show.style.apply(_highlight_first_ip, axis=1), use_container_width=True, hide_index=True)
                        st.caption(f"💡 有效勝率已計入亞洲盤贏半/輸半/走盤，EV = 有效勝率 × 賠率 − 1。即場建議門檻: EV ≥ {ip_res.get('min_ev', 0.05)*100:.1f}%。")

                        for c in ip_cand_list:
                            opt_str = f"【{c['bet_type']}】{c['label']} @ {c['odds']}"
                            if opt_str not in ip_option_map:
                                ip_all_options.append(opt_str)
                                ip_option_map[opt_str] = c

                    ip_picks = ip_res.get('picks', [])
                    ip_mc1, ip_mc2, ip_mc3 = st.columns(3)
                    if ip_picks:
                        ip_bb = ip_picks[0]
                        ip_mc1.metric("💡 首選推薦", f"{ip_bb['bet_type']} - {ip_bb['label']} @ {ip_bb['odds']}")
                        ip_mc2.metric("🎯 有效勝率 / 市場", f"{ip_bb['prob']*100:.1f}% / {ip_bb['p_mkt']*100:.1f}%")
                        ip_mc3.metric("📊 EV", f"{ip_bb.get('ev', 0):+.3f}")
                    else:
                        ip_mc1.metric("💡 首選推薦", "⛔ 現時不建議即場投注")
                        ip_mc2.metric("🎯 最佳候選 EV", f"{ip_cand_list[0]['ev']:+.3f}" if ip_cand_list else "-")
                        ip_mc3.metric("📏 即場 EV 門檻", f"{ip_res.get('min_ev', 0.05)*100:.1f}%")

                    st.markdown("---")
                    st.markdown("### 🎯 最終即場投注決策與注碼配置")

                    ip_col_sys, ip_col_usr = st.columns(2)

                    with ip_col_sys:
                        st.markdown("#### 🤖 系統投注 (System)")
                        st.caption("供模型學習及系統資金策略驗證使用。下注金額將自動從系統可用資金中扣除。")
                        ip_default_sys = []
                        for p in ip_picks:
                            for opt, c in ip_option_map.items():
                                if c['bet_type'] == p['bet_type'] and abs(float(c['line']) - float(p['line'])) < 1e-4 and c['selection'] == p['selection'] and opt not in ip_default_sys:
                                    ip_default_sys.append(opt)
                                    break

                        ip_sys_selected = st.multiselect("選擇系統即場投注項目", ip_all_options, default=ip_default_sys, key="ip_sys_multi")
                        ip_sys_stakes = {}
                        _ip_stake_set = dict(v9_get_settings())
                        _ip_stake_set['min_ev'] = ip_res.get('min_ev', _ip_stake_set['min_ev'] + 0.02)
                        for sel in ip_sys_selected:
                            cand = ip_option_map[sel]
                            sug_stk, raw_stk = v9_calc_stake(cand, sys_bankroll, sys_max_stake, _ip_stake_set)
                            ip_sys_stakes[sel] = st.number_input(f"系統建議金額: {sel}", value=float(sug_stk), disabled=True, key=f"ip_s_stk_{sel}")
                            if sug_stk <= 0:
                                st.caption(f"⚠ `{cand['bet_type']}` {cand.get('status', '')}：EV {cand.get('ev', 0):+.3f}，系統建議注碼 $0")

                    with ip_col_usr:
                        st.markdown("#### 👤 用家投注 (User)")
                        st.caption("真實資金決策，不用於系統學習。下注金額將自動從用家可用資金中扣除。")
                        ip_usr_selected = st.multiselect("選擇用家即場投注項目", ip_all_options, default=[], key="ip_usr_multi")
                        ip_usr_stakes = {}
                        for sel in ip_usr_selected:
                            ip_usr_stakes[sel] = st.number_input(f"用家自訂金額 ($): {sel}", min_value=0.0, step=10.0, value=100.0, key=f"ip_u_stk_{sel}")
                    st.write("")
                    if st.button("✅ 確定即場投注並寫入雲端資料庫", type="primary", use_container_width=True, key="btn_submit_inplay"):
                        ip_all_keys = set(ip_sys_selected + ip_usr_selected)
                        if not ip_all_keys:
                            st.warning("⚠️ 請至少在系統或用家選擇一項即場投注！")
                        else:
                            ip_new_records = []
                            for i, sel in enumerate(ip_all_keys):
                                cand = ip_option_map[sel]
                                s_stk = ip_sys_stakes.get(sel, 0.0)
                                u_stk = ip_usr_stakes.get(sel, 0.0)
                                new_id = f"IP{get_hkt_now().strftime('%Y%m%d%H%M%S')}{i}"

                                new_record = {
                                    'ID': new_id, 'Date': get_hkt_now().strftime('%Y-%m-%d %H:%M'), 'Status': 'Open',
                                    'Tournament_Name': ip_res['t_name'], 'Tournament_Category': ip_res['t_cat'],
                                    'Match': f"{ip_res['h_team']} vs {ip_res['a_team']}", 'Match_Date': ip_res.get('match_date', get_hkt_now().strftime('%Y-%m-%d')), 'Home_Team': ip_res['h_team'], 'Away_Team': ip_res['a_team'],
                                    'Home_Rating': ip_res['h_rating'], 'Away_Rating': ip_res['a_rating'], 'League_Size': ip_res.get('league_size', 20), 'Home_Form': ip_res['h_form'], 'Away_Form': ip_res['a_form'],
                                    'Bet_Type': f"{cand['bet_type']} (即場)", 'Selection': cand['selection'], 'Initial_Line': cand['line'], 'Initial_Odds': cand['odds'],
                                    'System_Stake': s_stk, 'User_Stake': u_stk,
                                    'Odds_History': json.dumps(ip_res['odds_history'], ensure_ascii=False),
                                    'InPlay_Minute': ip_minute,
                                    'Home_DA': h_da, 'Away_DA': a_da, 'Home_SoT': h_sot, 'Away_SoT': a_sot,
                                    'Home_SoFF': h_soff, 'Away_SoFF': a_soff,
                                    'Home_Red': h_red, 'Away_Red': a_red, 'Home_Possession': h_poss, 'Away_Possession': a_poss,
                                    'Home_Goal': h_g, 'Away_Goal': a_g, 'Home_Corner': h_c, 'Away_Corner': a_c,
                                    'Home_Goal_Conversion': round(h_conversion, 2), 'Away_Goal_Conversion': round(a_conversion, 2),
                                    'Home_Firepower': round(h_firepower, 2), 'Away_Firepower': round(a_firepower, 2),
                                    'Home_Corner_Eff': round(h_corner_eff, 2), 'Away_Corner_Eff': round(a_corner_eff, 2),
                                }
                                ip_new_records.append(new_record)

                            st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame(ip_new_records)], ignore_index=True)
                            save_db(st.session_state.df_db, db_file, db_table)
                            st.toast("✅ 即場投注紀錄雲端同步成功！本金已從可用資金中扣除。", icon="⏱️")
                            st.session_state.show_inplay_analysis = False
                            st.rerun()

    with t_settle:
        st.subheader("⚖️ 賽果結算與派彩管理")
        display_cumulative_metrics(st.session_state.df_db)
        
        open_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
        if open_df.empty:
            st.info("目前沒有待結算的注單。")
        else:
            st.write("請選擇要結算的比賽並輸入最終賽果 (系統將自動結算該場比賽的所有注單)：")
            
            match_groups = v10_match_groups(open_df)
            _st_opt_map = {f"{r['_md']} | {r['Tournament_Name']} | {r['Match']} (共 {r['Bet_Count']} 張注單)": (r['Tournament_Name'], r['Match'], r['_md'])
                           for _, r in match_groups.iterrows()}
            settle_opts = list(_st_opt_map.keys())
                
            sel_to_settle = st.selectbox("選擇結算比賽 (按比賽日期區分)", settle_opts, key="sel_settle_match")
            
            if sel_to_settle and sel_to_settle in _st_opt_map:
                sel_tourn, sel_match, sel_md = _st_opt_map[sel_to_settle]
                
                target_bets = open_df[v10_match_mask(open_df, sel_tourn, sel_match, sel_md)]
                
                st.write(f"**待結算注單列表 ({len(target_bets)} 張):**")
                st.dataframe(target_bets[['ID', 'Date', 'Match_Date', 'Bet_Type', 'Selection', 'Initial_Line', 'Initial_Odds', 'System_Stake', 'User_Stake']], use_container_width=True)
                
                s_hg_val = pd.to_numeric(target_bets['Home_Goal'], errors='coerce').max()
                s_ag_val = pd.to_numeric(target_bets['Away_Goal'], errors='coerce').max()
                s_hc_val = pd.to_numeric(target_bets['Home_Corner'], errors='coerce').max()
                s_ac_val = pd.to_numeric(target_bets['Away_Corner'], errors='coerce').max()
                
                # 防呆處理 NA 值
                s_hg_val = 0 if pd.isna(s_hg_val) else int(s_hg_val)
                s_ag_val = 0 if pd.isna(s_ag_val) else int(s_ag_val)
                s_hc_val = 0 if pd.isna(s_hc_val) else int(s_hc_val)
                s_ac_val = 0 if pd.isna(s_ac_val) else int(s_ac_val)
                
                c_sg1, c_sg2, c_sc1, c_sc2 = st.columns(4)
                settle_hg = c_sg1.number_input("結算: 主隊入球", 0, 50, s_hg_val, key=f"shg_{sel_tourn}_{sel_match}_{sel_md}")
                settle_ag = c_sg2.number_input("結算: 客隊入球", 0, 50, s_ag_val, key=f"sag_{sel_tourn}_{sel_match}_{sel_md}")
                settle_hc = c_sc1.number_input("結算: 主隊角球", 0, 50, s_hc_val, key=f"shc_{sel_tourn}_{sel_match}_{sel_md}")
                settle_ac = c_sc2.number_input("結算: 客隊角球", 0, 50, s_ac_val, key=f"sac_{sel_tourn}_{sel_match}_{sel_md}")
                
                if st.button("⚖️ 確定結算此賽事所有注單", type="primary", use_container_width=True):
                    mask = v10_match_mask(st.session_state.df_db, sel_tourn, sel_match, sel_md) & (st.session_state.df_db['Status'] == 'Open')
                    idx_to_settle = st.session_state.df_db[mask].index
                    
                    for idx in idx_to_settle:
                        row = st.session_state.df_db.loc[idx]
                        sys_profit, usr_profit, sys_payout, usr_payout, unit_profit, res_label, diff_val = calculate_settlement(
                            row['Bet_Type'], row['Selection'], row['Initial_Line'], row['Initial_Odds'], 
                            row['System_Stake'], row['User_Stake'], 
                            settle_hg, settle_ag, settle_hc, settle_ac
                        )
                        st.session_state.df_db.at[idx, 'Status'] = 'Settled'
                        st.session_state.df_db.at[idx, 'Home_Goal'] = settle_hg
                        st.session_state.df_db.at[idx, 'Away_Goal'] = settle_ag
                        st.session_state.df_db.at[idx, 'Home_Corner'] = settle_hc
                        st.session_state.df_db.at[idx, 'Away_Corner'] = settle_ac
                        st.session_state.df_db.at[idx, 'System_Profit'] = sys_profit
                        st.session_state.df_db.at[idx, 'User_Profit'] = usr_profit
                        st.session_state.df_db.at[idx, 'System_Payout'] = sys_payout
                        st.session_state.df_db.at[idx, 'User_Payout'] = usr_payout
                        st.session_state.df_db.at[idx, 'Unit_Profit'] = unit_profit
                        st.session_state.df_db.at[idx, 'Result_Label'] = res_label
                        
                    save_db(st.session_state.df_db, db_file, db_table)
                    st.toast(f"✅ {sel_match} 結算完成！", icon="⚖️")
                    st.rerun()

    # 先渲染「時間維度盈虧記錄分析」分頁，再執行「全局模型」的 ML 訓練：
    # 確保分頁資料與圖表優先送出，ML 首次訓練不會阻塞其他分頁顯示 (解決空白分頁問題)。
    # (分頁的視覺順序不變，只是渲染順序調整)
    with t_time_pnl:
        df_settled_tdp = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
        render_time_dimension_pnl_tab(df_settled_tdp, sys_bankroll, sys_max_stake)

    with t_ai:
        st.subheader("🤖 全局模型與績效分析")
        df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
        
        if df_settled.empty:
            st.info("目前尚無已結算的賽事，無法進行績效分析。")
        else:
            # 依照下注金額是否大於 0 來判斷是否將純賠率單位利潤計入該帳戶
            df_settled = add_account_unit_profit_columns(df_settled)

            st.markdown("### 📊 全局整體績效概覽")
            st.caption("時間維度盈虧及相關圖表已合併至『時間維度盈虧記錄分析』板塊；本板塊保留全局績效、各盤口類型績效與深度學習/機器學習盈虧分析。")
            total_bets = len(df_settled)
            sys_pnl_total = pd.to_numeric(df_settled['System_Profit'], errors='coerce').sum()
            usr_pnl_total = pd.to_numeric(df_settled['User_Profit'], errors='coerce').sum()
            
            sys_unit_profit_total = df_settled['System_Unit_Profit'].sum()
            usr_unit_profit_total = df_settled['User_Unit_Profit'].sum()
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("總結算注單", f"{total_bets} 張")
            c2.metric("系統總淨盈虧", f"${sys_pnl_total:,.2f}")
            c3.metric("用家總淨盈虧", f"${usr_pnl_total:,.2f}")
            c4.metric("系統單位利潤", f"{sys_unit_profit_total:.2f} U")
            c5.metric("用家單位利潤", f"{usr_unit_profit_total:.2f} U")

            st.divider()
            st.markdown("### 📈 各盤口類型績效分析")
            st.caption("各盤口的時間維度盈虧 (每日/每周/每季/每年) 與相關折線圖已合併至『時間維度盈虧記錄分析』板塊。")
            bet_types = ["讓球", "入球大小", "角球大小"]
            
            tabs = st.tabs(bet_types)
            performance_data = []
            
            for i, b_type in enumerate(bet_types):
                with tabs[i]:
                    sub_df = df_settled[df_settled['Bet_Type'].astype(str).str.contains(b_type, na=False, regex=False)]
                    if sub_df.empty:
                        st.write(f"暫無 {b_type} 結算紀錄。")
                        performance_data.append({
                            "盤口類型": b_type, "注單數": 0, "勝率 (%)": "0.0%", 
                            "系統淨盈虧": 0.0, "用家淨盈虧": 0.0,
                            "系統單位利潤": 0.0, "用家單位利潤": 0.0,
                        })
                        continue
                        
                    t_bets = len(sub_df)
                    win_bets = len(sub_df[pd.to_numeric(sub_df['Unit_Profit'], errors='coerce') > 0])
                    win_rate = (win_bets / t_bets) * 100 if t_bets > 0 else 0
                    
                    sys_p = pd.to_numeric(sub_df['System_Profit'], errors='coerce').sum()
                    usr_p = pd.to_numeric(sub_df['User_Profit'], errors='coerce').sum()
                    sys_u = sub_df['System_Unit_Profit'].sum()
                    usr_u = sub_df['User_Unit_Profit'].sum()
                    
                    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                    sc1.metric(f"{b_type} 注單", f"{t_bets} 張")
                    sc2.metric("系統淨盈虧", f"${sys_p:,.2f}")
                    sc3.metric("用家淨盈虧", f"${usr_p:,.2f}")
                    sc4.metric("系統單位利潤", f"{sys_u:.2f} U")
                    sc5.metric("用家單位利潤", f"{usr_u:.2f} U")
                    
                    st.write(f"**勝率 (贏半或以上):** `{win_rate:.1f}%` ({win_bets}/{t_bets})")
                    
                    performance_data.append({
                        "盤口類型": b_type,
                        "注單數": t_bets,
                        "勝率 (%)": f"{win_rate:.1f}%",
                        "系統淨盈虧": sys_p,
                        "用家淨盈虧": usr_p,
                        "系統單位利潤": sys_u,
                        "用家單位利潤": usr_u,
                    })

            st.divider()
            st.markdown("### 📋 盤口類型績效對比一覽表")
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                st.dataframe(
                    perf_df,
                    column_config={
                        "系統淨盈虧": st.column_config.NumberColumn("系統淨盈虧", format="$%.2f"),
                        "用家淨盈虧": st.column_config.NumberColumn("用家淨盈虧", format="$%.2f"),
                        "系統單位利潤": st.column_config.NumberColumn("系統單位利潤", format="%.2f U"),
                        "用家單位利潤": st.column_config.NumberColumn("用家單位利潤", format="%.2f U"),
                    },
                    use_container_width=True
                )
            
            # --- 各賽事名稱績效分析 (與各盤口類型績效分析同格式) ---
            st.divider()
            st.markdown("### 🏆 各賽事名稱績效分析")
            st.caption("按賽事名稱 (Tournament_Name) 分類顯示績效，與各盤口類型績效分析格式相同：注單數、系統/用家淨盈虧、系統/用家單位利潤及勝率。分頁按注單數目由多至少排列。")
            
            # 取得賽事名稱清單 (按注單數由多至少；空值/NaN 統一顯示為「未指定賽事」)
            _tourn_series = df_settled['Tournament_Name'].fillna('').astype(str).str.strip()
            _tourn_series = _tourn_series.replace('', '未指定賽事').replace('nan', '未指定賽事').replace('None', '未指定賽事')
            tourn_names = _tourn_series.value_counts().index.tolist()
            
            if not tourn_names:
                st.write("暫無賽事結算紀錄。")
            else:
                tourn_tabs = st.tabs(tourn_names)
                tournament_performance_data = []
                
                for i, t_name in enumerate(tourn_names):
                    with tourn_tabs[i]:
                        sub_df = df_settled[_tourn_series == t_name]
                        
                        t_bets = len(sub_df)
                        win_bets = len(sub_df[pd.to_numeric(sub_df['Unit_Profit'], errors='coerce') > 0])
                        win_rate = (win_bets / t_bets) * 100 if t_bets > 0 else 0
                        
                        sys_p = pd.to_numeric(sub_df['System_Profit'], errors='coerce').sum()
                        usr_p = pd.to_numeric(sub_df['User_Profit'], errors='coerce').sum()
                        sys_u = sub_df['System_Unit_Profit'].sum()
                        usr_u = sub_df['User_Unit_Profit'].sum()
                        
                        tmc1, tmc2, tmc3, tmc4, tmc5 = st.columns(5)
                        tmc1.metric(f"{t_name} 注單", f"{t_bets} 張")
                        tmc2.metric("系統淨盈虧", f"${sys_p:,.2f}")
                        tmc3.metric("用家淨盈虧", f"${usr_p:,.2f}")
                        tmc4.metric("系統單位利潤", f"{sys_u:.2f} U")
                        tmc5.metric("用家單位利潤", f"{usr_u:.2f} U")
                        
                        st.write(f"**勝率 (贏半或以上):** `{win_rate:.1f}%` ({win_bets}/{t_bets})")
                        
                        tournament_performance_data.append({
                            "賽事名稱": t_name,
                            "注單數": t_bets,
                            "勝率 (%)": f"{win_rate:.1f}%",
                            "系統淨盈虧": sys_p,
                            "用家淨盈虧": usr_p,
                            "系統單位利潤": sys_u,
                            "用家單位利潤": usr_u,
                        })
                
                # --- 賽事績效對比一覽表 ---
                st.divider()
                st.markdown("### 📋 賽事績效對比一覽表")
                if tournament_performance_data:
                    tourn_df = pd.DataFrame(tournament_performance_data)
                    st.dataframe(
                        tourn_df,
                        column_config={
                            "系統淨盈虧": st.column_config.NumberColumn("系統淨盈虧", format="$%.2f"),
                            "用家淨盈虧": st.column_config.NumberColumn("用家淨盈虧", format="$%.2f"),
                            "系統單位利潤": st.column_config.NumberColumn("系統單位利潤", format="%.2f U"),
                            "用家單位利潤": st.column_config.NumberColumn("用家單位利潤", format="%.2f U"),
                        },
                        use_container_width=True
                    )
            
            # --- 深度學習與機器學習盈虧分析 (由『時間維度盈虧記錄分析』板塊移至此處) ---
            st.divider()
            render_dl_ml_analysis_section(df_settled)

if __name__ == "__main__":
    main()
