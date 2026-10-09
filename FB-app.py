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
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
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
st.set_page_config(page_title="Actuarial and fund management system by Dr. EdwinPro", page_icon="⚽", layout="wide")

DB_COLUMNS = [
    'ID', 'Date', 'Status', 
    'Tournament_Name', 'Tournament_Category', 'Match', 'Match_Date', 'Home_Team', 'Away_Team', 
    'Home_Rating', 'Away_Rating', 'Home_Form', 'Away_Form',
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
    """增強版特徵工程：加入更多特徵供深度學習模型使用"""
    X, y = [], []
    for _, r in df.iterrows():
        try:
            hr = rating_map.get(r.get('Home_Rating', 'C'), 3)
            ar = rating_map.get(r.get('Away_Rating', 'C'), 3)
            hf = extract_form_points(r.get('Home_Form', '0W0D0L'))
            af = extract_form_points(r.get('Away_Form', '0W0D0L'))
            line = float(r.get('Initial_Line', 0))
            odds = float(r.get('Initial_Odds', 1.90))
            stake = float(r.get('System_Stake', 0))
            # 新特徵
            rating_diff = hr - ar
            form_diff = hf - af
            stake_log = np.log1p(max(0, stake)) if np is not None else 0
            odds_margin = float(r.get('Initial_Odds', 1.90)) - 1.0
            X.append([hr, ar, hf, af, line, odds, rating_diff, form_diff, stake_log, odds_margin])
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
                    x_input = np.array([[h_data['hr'], h_data['ar'], h_data['hf'], h_data['af'], c['line'], c['odds'], 
                                          h_data['hr'] - h_data['ar'], h_data['hf'] - h_data['af'], 
                                          0.0, c['odds'] - 1.0]])
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
    _match_date_str = get_row_match_date(row)
    st.session_state.edit_match_date = _match_date_str
    try:
        st.session_state.match_date_input = datetime.strptime(_match_date_str, "%Y-%m-%d").date()
    except:
        st.session_state.match_date_input = get_hkt_now().date()
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
# 4b. 已完成注單資料修補工具 (開啟已儲存賽事更新數據)
# ==========================================
def is_missing_match_date(v):
    """判斷 Match_Date 是否缺失 (None/NaN/空字串)"""
    if pd.isna(v):
        return True
    s = str(v).strip()
    return s == '' or s.lower() in ('none', 'nan', 'nat')

@st.dialog("🗂 已完成注單資料修補與更新", width="large")
def edit_settled_bets_dialog(db_file, db_table):
    """開啟資料庫中已儲存的注單 (含已完成 Settled) 進行更新：補填或修正比賽日期 (Match_Date)"""
    st.caption("此工具可開啟已完成 (Settled) 的注單進行數據更新：補填或修正比賽日期 (Match_Date)，供時間維度盈虧分析使用。修改後會自動同步雲端數據庫。")
    
    settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
    if settled.empty:
        st.info("目前沒有已完成的注單。")
        return
    
    n_missing = int(settled['Match_Date'].apply(is_missing_match_date).sum())
    
    tab_match, tab_id = st.tabs(["📋 按比賽場次批量補填", "🔍 按注單 ID 單筆修改"])
    
    with tab_match:
        st.write(f"目前共有 **{len(settled)}** 張已完成注單，其中 **{n_missing}** 張缺少比賽日期。")
        
        # 依「賽事 + 對賽 + 記錄日期」分組，避免不同日期的同組合對賽被誤改
        settled_groups = settled.copy()
        settled_groups['_Record_Date'] = pd.to_datetime(settled_groups['Date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('無記錄日期')
        
        groups = settled_groups.fillna({'Tournament_Name': '', 'Match': ''}).groupby(['Tournament_Name', 'Match', '_Record_Date'])
        group_indices = []
        group_opts = []
        for (tname, match, record_date), g in groups:
            group_indices.append(list(g.index))
            existing_dates = sorted({str(d)[:10] for d in g['Match_Date'] if not is_missing_match_date(d)})
            miss_cnt = int(g['Match_Date'].apply(is_missing_match_date).sum())
            date_disp = "、".join(existing_dates) if existing_dates else "未設定"
            group_opts.append(f"{tname} | {match} | 記錄日期: {record_date} | 比賽日期: {date_disp} | 缺日期: {miss_cnt}/{len(g)} 張")
        
        sel_group = st.selectbox("選擇要補填的比賽場次", group_opts, key="sbd_group_sel")
        g_idx = group_opts.index(sel_group)
        
        only_missing = st.checkbox("僅更新目前缺少比賽日期的注單", value=True, key="sbd_only_missing")
        new_date = st.date_input("比賽日期 (Match Date)", value=get_hkt_now().date(), key="sbd_new_date")
        
        if st.button("💾 更新此場次注單的比賽日期", type="primary", use_container_width=True, key="sbd_save_group"):
            selected_indices = group_indices[g_idx]
            if only_missing:
                upd_indices = [idx for idx in selected_indices if is_missing_match_date(st.session_state.df_db.at[idx, 'Match_Date'])]
            else:
                upd_indices = selected_indices
            
            if not upd_indices:
                st.warning("此場次沒有符合條件的注單可更新。")
            else:
                st.session_state.df_db.loc[upd_indices, 'Match_Date'] = new_date.strftime('%Y-%m-%d')
                save_db(st.session_state.df_db, db_file, db_table)
                st.toast(f"✅ 已更新 {len(upd_indices)} 張注單的比賽日期！", icon="🗂")
                st.rerun()
        
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
        
        target_row = settled[settled['ID'] == target_id]
        if target_row.empty:
            st.warning("找不到所選注單。")
            return
        target_row = target_row.iloc[0]
        
        # 預設日期：現有 Match_Date -> Date -> 今天
        _def_date = None
        if not is_missing_match_date(target_row.get('Match_Date')):
            try:
                _def_date = datetime.strptime(str(target_row['Match_Date'])[:10], '%Y-%m-%d').date()
            except:
                _def_date = None
        if _def_date is None:
            _parsed = pd.to_datetime(target_row.get('Date'), errors='coerce')
            if pd.notna(_parsed):
                _def_date = _parsed.date()
        if _def_date is None:
            _def_date = get_hkt_now().date()
        
        single_date = st.date_input("比賽日期 (Match Date)", value=_def_date, key="sbd_single_date")
        
        if st.button("💾 更新此注單的比賽日期", type="primary", use_container_width=True, key="sbd_save_single"):
            st.session_state.df_db.loc[st.session_state.df_db['ID'] == target_id, 'Match_Date'] = single_date.strftime('%Y-%m-%d')
            save_db(st.session_state.df_db, db_file, db_table)
            st.toast(f"✅ 注單 {target_id} 的比賽日期已更新！", icon="🗂")
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
            # 增強：根據模型信心與樣本數調整 Kelly 分數
            model_confidence = cand.get('model_prob', prob)
            sample_penalty = min(1.0, cand.get('sample_count', 15) / 50.0) if cand.get('sample_count', 0) > 0 else 0.3
            confidence_factor = (0.5 + 0.5 * model_confidence) * sample_penalty
            raw_stake = (sys_bankroll * (kelly * 0.5 * confidence_factor))
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

def render_dl_ml_analysis_section(df_settled):
    """渲染深度學習與機器學習盈虧分析：多模型交叉驗證結果 + 時間趨勢 ML 預測建議
    (涵蓋讓球、入球大小、角球大小三種盤口；ML 結果經快取，不會重複訓練)"""
    st.markdown("### 🧠 深度學習與機器學習盈虧分析")
    st.caption(f"本板塊涵蓋讓球、入球大小、角球大小三種盤口；ML 分析最低樣本門檻為 {ML_MIN_SAMPLES} 場。小樣本 (8-14 場) 時模型結果波動較大，僅供參考。")
    
    if not HAS_AI_MODULES:
        st.warning("未安裝 sklearn 套件，無法執行 ML 模型訓練。")
        return
    
    # 呼叫快取的 ML 分析 (資料變更時自動重算，平時直接讀取快取)
    analysis = run_ml_analysis_cached(df_settled)
    
    bet_types = ["讓球", "入球大小", "角球大小"]
    
    ml_col1, ml_col2 = st.columns(2)
    
    with ml_col1:
        st.markdown("##### 多模型交叉驗證結果")
        
        for bt in bet_types:
            entry = analysis.get(bt, {})
            n = entry.get('n', 0)
            
            if not entry.get('trained', False):
                if n < ML_MIN_SAMPLES:
                    st.warning(f"{bt}: 樣本數不足 ({n} < {ML_MIN_SAMPLES})，無法訓練 ML 模型。")
                elif not entry.get('class_ok', True):
                    st.warning(f"{bt}: 目標類別不足，無法訓練 ML 模型。")
                else:
                    st.warning(f"{bt}: 所有模型訓練失敗。")
                continue
            
            st.markdown(f"**{bt}** (樣本數: {n})")
            model_data = []
            for name, res in entry.get('model_results', {}).items():
                model_data.append({
                    '模型': name,
                    '交叉驗證準確率': f"{res['cv_mean']*100:.1f}%",
                    '標準差': f"{res['cv_std']*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)
            
            if entry.get('best_model'):
                st.success(f"最佳模型: {entry['best_model']}")
    
    with ml_col2:
        st.markdown("##### 時間趨勢 ML 預測建議")
        
        any_suggestion = False
        for bt in bet_types:
            entry = analysis.get(bt, {})
            n = entry.get('n', 0)
            
            if not entry.get('trained', False):
                if n < ML_MIN_SAMPLES:
                    st.caption(f"{bt}: 樣本數不足 ({n} < {ML_MIN_SAMPLES})，無法生成 ML 預測建議。")
                continue
            
            any_suggestion = True
            recent_win_rate = entry.get('recent_win_rate', 0.0)
            
            # 建議
            if recent_win_rate >= 0.6:
                trend = "上升趨勢 📈"
                suggestion = f"{bt} 近期勝率 {recent_win_rate*100:.0f}%，建議繼續關注。"
            elif recent_win_rate >= 0.4:
                trend = "平穩 ➡️"
                suggestion = f"{bt} 近期勝率 {recent_win_rate*100:.0f}%，建議謹慎下注。"
            else:
                trend = "下降趨勢 📉"
                suggestion = f"{bt} 近期勝率 {recent_win_rate*100:.0f}%，建議減少下注或反向操作。"
            
            st.markdown(f"**{bt}** ({trend})")
            st.write(f"最佳模型: {entry.get('best_model', 'N/A')}")
            st.write(suggestion)
            st.caption(f"近期平均賠率: {entry.get('recent_odds', 0.0):.2f} | 樣本數: {n}")
        
        if not any_suggestion:
            st.info(f"需要更多已結算注單 (≥{ML_MIN_SAMPLES}場/盤口類型) 才能生成 ML 預測建議。")

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
    if st.sidebar.button("🗂 已完成注單資料修補 (補填比賽日期)", use_container_width=True):
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
            
            res_micro = evaluate_dimension_enhanced(df_micro, "微觀 - 賽事名稱", candidates_base, rating_map, h_data)
            res_meso = evaluate_dimension_enhanced(df_meso, "中觀 - 賽事分類", candidates_base, rating_map, h_data)
            res_macro = evaluate_dimension_enhanced(df_macro, "宏觀 - 總數據", candidates_base, rating_map, h_data)
            
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
            
            # 顯示最佳模型資訊
            if 'best_model_name' in bm:
                st.info(f"最佳 ML 模型: **{bm['best_model_name']}** | 採用『{bm['dim']}』級別的模型進行運算，其歷史準確率與 EV 獲利期望值最高。")
                if 'model_results' in bm and bm['model_results']:
                    model_data = []
                    for name, mr in bm['model_results'].items():
                        model_data.append({
                            '模型': name,
                            '交叉驗證準確率': f"{mr['cv_mean']*100:.1f}%",
                            '標準差': f"{mr['cv_std']*100:.1f}%",
                        })
                    if model_data:
                        st.markdown("##### 📊 多模型交叉驗證對比")
                        st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)
            else:
                st.info(f"系統分析顯示，針對『{res['t_name']}』，採用『{bm['dim']}』級別的模型進行運算，其歷史準確率與 EV 獲利期望值最高，故本次投注策略依據此模型生成。")
            
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
                            new_id = f"B{get_hkt_now().strftime('%Y%m%d%H%M%S')}{i}"
                            
                        new_record = {
                            'ID': new_id, 'Date': get_hkt_now().strftime('%Y-%m-%d %H:%M'), 'Status': 'Open',
                            'Tournament_Name': res['t_name'], 'Tournament_Category': res['t_cat'], 
                            'Match': f"{home_team} vs {away_team}", 'Match_Date': match_date, 'Home_Team': home_team, 'Away_Team': away_team,
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
        
        if st.session_state.df_db.empty:
            st.info("目前數據庫中尚無任何賽事紀錄。請先至『賽前建檔與投注』新增賽事。")
        else:
            open_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
            if open_df.empty:
                st.info("💡 當前沒有未結算 (Open) 的賽事，您可以選擇歷史賽事進行即場數據更新：")
                match_groups = st.session_state.df_db.groupby(['Tournament_Name', 'Match']).size().reset_index(name='Bet_Count')
            else:
                match_groups = open_df.groupby(['Tournament_Name', 'Match']).size().reset_index(name='Bet_Count')

            match_options = [f"{r['Tournament_Name']} | {r['Match']} (注單數: {r['Bet_Count']} 張)" for _, r in match_groups.iterrows()]
            sel_match_str = st.selectbox("⚽ 請選擇要更新即場數據的比賽場次：", match_options, key="inplay_match_select")

            if sel_match_str:
                selected_tourn = sel_match_str.split(" | ")[0]
                selected_match = sel_match_str.split(" | ")[1].split(" (注單數:")[0]

                matching_bets = st.session_state.df_db[
                    (st.session_state.df_db['Tournament_Name'] == selected_tourn) & 
                    (st.session_state.df_db['Match'] == selected_match)
                ]

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
                    mask = (st.session_state.df_db['Tournament_Name'] == selected_tourn) & (st.session_state.df_db['Match'] == selected_match)
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
                    st.toast(f"✅ 已成功更新『{selected_match}』共 {mask.sum()} 筆注單的即場數據！", icon="💾")
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

                bet_col1, bet_col2, bet_col3 = st.columns(3)

                with bet_col1:
                    st.markdown("**⚽ 讓球 (Handicap)**")
                    ip_hd_line = st.number_input("讓球盤口線", min_value=-10.0, max_value=10.0, step=0.25, format="%.2f", key="ip_hd_line")
                    ip_hd_upper = st.number_input("大/上賠率 (主隊)", min_value=1.01, max_value=100.0, step=0.01, format="%.2f", key="ip_hd_upper", on_change=auto_calc_lower, args=("ip_hd",))
                    ip_hd_lower = st.number_input("小/下賠率 (客隊)", min_value=1.01, max_value=100.0, step=0.01, format="%.2f", key="ip_hd_lower", on_change=mark_manual_lower, args=("ip_hd",))
                    _hd_mode = "手動" if st.session_state.ip_hd_manual else "自動"
                    st.caption(f"抽水: {calc_live_margin_str(ip_hd_upper, ip_hd_lower)} [{_hd_mode}]")

                with bet_col2:
                    st.markdown("**⚽ 入球大小 (Goals O/U)**")
                    ip_ou_line = st.number_input("入球大小盤口線", min_value=0.0, max_value=20.0, step=0.25, format="%.2f", key="ip_ou_line")
                    ip_ou_upper = st.number_input("大賠率 (Over)", min_value=1.01, max_value=100.0, step=0.01, format="%.2f", key="ip_ou_upper", on_change=auto_calc_lower, args=("ip_ou",))
                    ip_ou_lower = st.number_input("小賠率 (Under)", min_value=1.01, max_value=100.0, step=0.01, format="%.2f", key="ip_ou_lower", on_change=mark_manual_lower, args=("ip_ou",))
                    _ou_mode = "手動" if st.session_state.ip_ou_manual else "自動"
                    st.caption(f"抽水: {calc_live_margin_str(ip_ou_upper, ip_ou_lower)} [{_ou_mode}]")

                with bet_col3:
                    st.markdown("**⚽ 角球大小 (Corners O/U)**")
                    ip_cr_line = st.number_input("角球盤口線", min_value=0.0, max_value=30.0, step=0.5, format="%.2f", key="ip_cr_line")
                    ip_cr_upper = st.number_input("大賠率 (Over)", min_value=1.01, max_value=100.0, step=0.01, format="%.2f", key="ip_cr_upper", on_change=auto_calc_lower, args=("ip_cr",))
                    ip_cr_lower = st.number_input("小賠率 (Under)", min_value=1.01, max_value=100.0, step=0.01, format="%.2f", key="ip_cr_lower", on_change=mark_manual_lower, args=("ip_cr",))
                    _cr_mode = "手動" if st.session_state.ip_cr_manual else "自動"
                    st.caption(f"抽水: {calc_live_margin_str(ip_cr_upper, ip_cr_lower)} [{_cr_mode}]")

                # --- 即場數據分析執行 ---
                if st.button("🚀 即場數據分析執行", type="primary", use_container_width=True, key="btn_inplay_analysis"):
                    st.session_state.show_inplay_analysis = True

                    # 取得比賽資料
                    ip_last_row = matching_bets.iloc[-1]
                    ip_t_name = str(ip_last_row.get('Tournament_Name', '')) if pd.notna(ip_last_row.get('Tournament_Name')) else ''
                    ip_t_cat = str(ip_last_row.get('Tournament_Category', CATEGORY_OPTIONS[0])) if pd.notna(ip_last_row.get('Tournament_Category')) else CATEGORY_OPTIONS[0]
                    ip_h_team = str(ip_last_row.get('Home_Team', '')) if pd.notna(ip_last_row.get('Home_Team')) else ''
                    ip_a_team = str(ip_last_row.get('Away_Team', '')) if pd.notna(ip_last_row.get('Away_Team')) else ''
                    ip_h_rating = str(ip_last_row.get('Home_Rating', 'C')) if pd.notna(ip_last_row.get('Home_Rating')) else 'C'
                    ip_a_rating = str(ip_last_row.get('Away_Rating', 'C')) if pd.notna(ip_last_row.get('Away_Rating')) else 'C'
                    ip_h_form = str(ip_last_row.get('Home_Form', '3W1D1L')) if pd.notna(ip_last_row.get('Home_Form')) else '3W1D1L'
                    ip_a_form = str(ip_last_row.get('Away_Form', '2W2D1L')) if pd.notna(ip_last_row.get('Away_Form')) else '2W2D1L'

                    _rating_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
                    ip_hr_val = _rating_map.get(ip_h_rating, 3)
                    ip_ar_val = _rating_map.get(ip_a_rating, 3)
                    ip_h_data = {'hr': ip_hr_val, 'ar': ip_ar_val, 'hf': extract_form_points(ip_h_form), 'af': extract_form_points(ip_a_form)}

                    # 建立即場候選盤口 (結合剩餘時間與攻勢危險度計算基礎機率)
                    ip_candidates_base = []

                    # 讓球
                    _hd_line_str = f"{ip_hd_line:g}"
                    _hd_p_home = calc_inplay_base_prob("讓球", "Home", ip_hd_line, h_g, a_g, h_c, a_c, ip_minute, h_firepower, a_firepower, h_corner_eff, a_corner_eff)
                    _hd_p_away = calc_inplay_base_prob("讓球", "Away", ip_hd_line, h_g, a_g, h_c, a_c, ip_minute, h_firepower, a_firepower, h_corner_eff, a_corner_eff)
                    if ip_hd_line <= 0:
                        _hd_label_h, _hd_label_a = f"{_hd_line_str}主隊(上盤)", f"{_hd_line_str}客隊(下盤)"
                    else:
                        _hd_label_h, _hd_label_a = f"{_hd_line_str}主隊(下盤)", f"{_hd_line_str}客隊(上盤)"
                    ip_candidates_base.extend([
                        {'bet_type': '讓球', 'selection': 'Home', 'base_prob': _hd_p_home, 'odds': float(ip_hd_upper), 'line': float(ip_hd_line), 'label': _hd_label_h, 'history_idx': 0, 'record_time': get_hkt_now().strftime('%m-%d %H:%M')},
                        {'bet_type': '讓球', 'selection': 'Away', 'base_prob': _hd_p_away, 'odds': float(ip_hd_lower), 'line': float(ip_hd_line), 'label': _hd_label_a, 'history_idx': 0, 'record_time': get_hkt_now().strftime('%m-%d %H:%M')},
                    ])

                    # 入球大小
                    _ou_line_str = f"{ip_ou_line:g}"
                    _ou_p_over = calc_inplay_base_prob("入球大小", "Over", ip_ou_line, h_g, a_g, h_c, a_c, ip_minute, h_firepower, a_firepower, h_corner_eff, a_corner_eff)
                    _ou_p_under = calc_inplay_base_prob("入球大小", "Under", ip_ou_line, h_g, a_g, h_c, a_c, ip_minute, h_firepower, a_firepower, h_corner_eff, a_corner_eff)
                    ip_candidates_base.extend([
                        {'bet_type': '入球大小', 'selection': 'Over', 'base_prob': _ou_p_over, 'odds': float(ip_ou_upper), 'line': float(ip_ou_line), 'label': f"{_ou_line_str}大盤(Over)", 'history_idx': 1, 'record_time': get_hkt_now().strftime('%m-%d %H:%M')},
                        {'bet_type': '入球大小', 'selection': 'Under', 'base_prob': _ou_p_under, 'odds': float(ip_ou_lower), 'line': float(ip_ou_line), 'label': f"{_ou_line_str}小盤(Under)", 'history_idx': 1, 'record_time': get_hkt_now().strftime('%m-%d %H:%M')},
                    ])

                    # 角球大小
                    _cr_line_str = f"{ip_cr_line:g}"
                    _cr_p_over = calc_inplay_base_prob("角球大小", "Over", ip_cr_line, h_g, a_g, h_c, a_c, ip_minute, h_firepower, a_firepower, h_corner_eff, a_corner_eff)
                    _cr_p_under = calc_inplay_base_prob("角球大小", "Under", ip_cr_line, h_g, a_g, h_c, a_c, ip_minute, h_firepower, a_firepower, h_corner_eff, a_corner_eff)
                    ip_candidates_base.extend([
                        {'bet_type': '角球大小', 'selection': 'Over', 'base_prob': _cr_p_over, 'odds': float(ip_cr_upper), 'line': float(ip_cr_line), 'label': f"{_cr_line_str}大盤(Over)", 'history_idx': 2, 'record_time': get_hkt_now().strftime('%m-%d %H:%M')},
                        {'bet_type': '角球大小', 'selection': 'Under', 'base_prob': _cr_p_under, 'odds': float(ip_cr_lower), 'line': float(ip_cr_line), 'label': f"{_cr_line_str}小盤(Under)", 'history_idx': 2, 'record_time': get_hkt_now().strftime('%m-%d %H:%M')},
                    ])

                    # 三維度分析
                    df_settled_ip = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
                    df_ip_micro = df_settled_ip[df_settled_ip['Tournament_Name'] == ip_t_name]
                    df_ip_meso = df_settled_ip[df_settled_ip['Tournament_Category'] == ip_t_cat]
                    df_ip_macro = df_settled_ip

                    ip_res_micro = evaluate_dimension_enhanced(df_ip_micro, "微觀 - 賽事名稱", ip_candidates_base, _rating_map, ip_h_data)
                    ip_res_meso = evaluate_dimension_enhanced(df_ip_meso, "中觀 - 賽事分類", ip_candidates_base, _rating_map, ip_h_data)
                    ip_res_macro = evaluate_dimension_enhanced(df_ip_macro, "宏觀 - 總數據", ip_candidates_base, _rating_map, ip_h_data)

                    ip_valid_res = [r for r in [ip_res_micro, ip_res_meso, ip_res_macro] if r['valid']]
                    ip_best_model = max(ip_valid_res, key=lambda x: x['score']) if ip_valid_res else ip_res_macro
                    if not ip_valid_res:
                        ip_best_model['msg'] = "所有維度樣本數不足，降級為純基礎期望值運算。"

                    ip_best_bet = ip_best_model['best'] if 'best' in ip_best_model else ip_candidates_base[0]

                    # 建立即場賠率歷史 JSON
                    ip_odds_history = [
                        {"id": 0, "type": "讓球", "record_time": get_hkt_now().strftime('%m-%d %H:%M'), "line": float(ip_hd_line), "upper": float(ip_hd_upper), "lower": float(ip_hd_lower), "unlock": True, "margin": round(1.0/ip_hd_upper + 1.0/ip_hd_lower, 3) if ip_hd_upper > 0 and ip_hd_lower > 0 else 1.085},
                        {"id": 1, "type": "入球大小", "record_time": get_hkt_now().strftime('%m-%d %H:%M'), "line": float(ip_ou_line), "upper": float(ip_ou_upper), "lower": float(ip_ou_lower), "unlock": True, "margin": round(1.0/ip_ou_upper + 1.0/ip_ou_lower, 3) if ip_ou_upper > 0 and ip_ou_lower > 0 else 1.085},
                        {"id": 2, "type": "角球大小", "record_time": get_hkt_now().strftime('%m-%d %H:%M'), "line": float(ip_cr_line), "upper": float(ip_cr_upper), "lower": float(ip_cr_lower), "unlock": True, "margin": round(1.0/ip_cr_upper + 1.0/ip_cr_lower, 3) if ip_cr_upper > 0 and ip_cr_lower > 0 else 1.085},
                    ]

                    st.session_state.inplay_analysis_result = {
                        'micro': ip_res_micro, 'meso': ip_res_meso, 'macro': ip_res_macro,
                        'best_model': ip_best_model, 'best_bet': ip_best_bet,
                        't_name': ip_t_name, 't_cat': ip_t_cat,
                        'h_team': ip_h_team, 'a_team': ip_a_team,
                        'h_rating': ip_h_rating, 'a_rating': ip_a_rating,
                        'h_form': ip_h_form, 'a_form': ip_a_form,
                        'odds_history': ip_odds_history,
                        'remaining_min': max(0, 90 - ip_minute),
                        'match_date': get_row_match_date(ip_last_row),
                    }

                # --- 顯示即場分析結果 ---
                if st.session_state.get('show_inplay_analysis', False):
                    ip_res = st.session_state.inplay_analysis_result
                    st.success("✅ 三維度即場數據分析與 EV 運算完成！")
                    st.info(f"⏱ 剩餘比賽時間: {ip_res['remaining_min']} 分鐘 | 系統結合剩餘時間、進攻火力與角球效率計算即場期望值。")

                    ip_c1, ip_c2, ip_c3 = st.columns(3)
                    for col, r, title in zip([ip_c1, ip_c2, ip_c3], [ip_res['micro'], ip_res['meso'], ip_res['macro']], ["A. 微觀 (賽事名稱)", "B. 中觀 (賽事分類)", "C. 宏觀 (全局數據)"]):
                        with col.container(border=True):
                            st.markdown(f"**{title}**")
                            if r['valid']:
                                st.write(f"樣本數: `{r['n']}` 場")
                                st.write(f"系統策略 ROI: `{r['roi']*100:.1f}%`")
                                st.write(f"歷史勝率: `{r['acc']*100:.1f}%`")
                            else:
                                st.warning(r['msg'])

                    ip_bm = ip_res['best_model']
                    ip_bb = ip_res['best_bet']
                    ip_dim_label_map = {"微觀 - 賽事名稱": "微觀", "中觀 - 賽事分類": "中觀", "宏觀 - 總數據": "宏觀"}
                    ip_dim_short = ip_dim_label_map.get(ip_bm['dim'], "宏觀")

                    st.markdown(f"### 🧠 AI 預測模型推薦 (即場)")
                    # 顯示最佳模型資訊
                    if 'best_model_name' in ip_bm:
                        st.info(f"最佳 ML 模型: **{ip_bm['best_model_name']}** | 採用『{ip_bm['dim']}』級別的模型進行運算。剩餘比賽時間: {ip_res['remaining_min']} 分鐘。")
                    else:
                        st.info(f"系統分析顯示，針對『{ip_res['t_name']}』，採用『{ip_bm['dim']}』級別的模型進行運算，其歷史準確率與 EV 獲利期望值最高，故本次即場投注策略依據此模型生成。")

                    st.markdown("#### 📊 所有盤口評估明細 (即場 EV 運算)")
                    ip_cand_list = ip_bm.get('candidates', [])

                    ip_all_options = []
                    ip_option_map = {}

                    if ip_cand_list:
                        ip_df_show = pd.DataFrame(ip_cand_list)
                        ip_df_show['推薦排序'] = range(1, len(ip_df_show) + 1)
                        ip_df_show = ip_df_show[['推薦排序', 'bet_type', 'line', 'label', 'odds', 'prob', 'ev']]
                        ip_df_show.columns = ['推薦排序', '盤口類型', '盤口線', '投注方向', '賠率', '預期勝率', '期望值 (EV)']
                        ip_df_show['預期勝率'] = ip_df_show['預期勝率'].apply(lambda x: f"{x*100:.2f}%")
                        ip_df_show['期望值 (EV)'] = ip_df_show['期望值 (EV)'].apply(lambda x: f"{x:.3f}")

                        def _highlight_first_ip(row):
                            if row.name == 0:
                                return ['background-color: rgba(40, 167, 69, 0.2)'] * len(row)
                            return [''] * len(row)

                        st.dataframe(ip_df_show.style.apply(_highlight_first_ip, axis=1), use_container_width=True)

                        for c in ip_cand_list:
                            opt_str = f"【{c['bet_type']}】{c['label']} @ {c['odds']}"
                            if opt_str not in ip_option_map:
                                ip_all_options.append(opt_str)
                                ip_option_map[opt_str] = c

                    ip_mc1, ip_mc2, ip_mc3 = st.columns(3)
                    ip_mc1.metric("💡 首選推薦", f"{ip_bb['bet_type']} - {ip_bb['label']}")
                    ip_mc2.metric(f"🎯 預期勝率 ({ip_dim_short}修正)", f"{ip_bb.get('prob', ip_bb.get('base_prob',0))*100:.1f}%")
                    ip_mc3.metric("📊 修正 EV", f"{ip_bb.get('ev', 0):.3f}")

                    st.markdown("---")
                    st.markdown("### 🎯 最終即場投注決策與注碼配置")

                    ip_col_sys, ip_col_usr = st.columns(2)

                    with ip_col_sys:
                        st.markdown("#### 🤖 系統投注 (System)")
                        st.caption("供模型學習及系統資金策略驗證使用。下注金額將自動從系統可用資金中扣除。")
                        ip_default_sys = []
                        if ip_cand_list:
                            ip_best_c = ip_cand_list[0]
                            if ip_best_c.get('ev', 0) > 0:
                                for opt, c in ip_option_map.items():
                                    if c['bet_type'] == ip_best_c['bet_type'] and abs(float(c['line']) - float(ip_best_c['line'])) < 1e-4 and c['selection'] == ip_best_c['selection']:
                                        ip_default_sys.append(opt)
                                        break

                        ip_sys_selected = st.multiselect("選擇系統即場投注項目", ip_all_options, default=ip_default_sys, key="ip_sys_multi")
                        ip_sys_stakes = {}
                        for sel in ip_sys_selected:
                            cand = ip_option_map[sel]
                            sug_stk, raw_stk = calc_suggested_stake(cand, sys_bankroll, sys_max_stake)
                            ip_sys_stakes[sel] = st.number_input(f"系統建議金額: {sel}", value=float(sug_stk), disabled=True, key=f"ip_s_stk_{sel}")
                            if "讓球" in cand['bet_type'] and 0 < raw_stk < 200:
                                if cand.get('ev', 0) >= 0.03 and cand.get('prob', 0) >= 0.50:
                                    st.caption(f"💡 `{cand['bet_type']}` EV/勝率達標，系統自動升級最低注碼 $200")
                                else:
                                    st.caption(f"⚠ `{cand['bet_type']}` EV未達標，系統建議放棄 (注碼 $0)")

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
                                    'Home_Rating': ip_res['h_rating'], 'Away_Rating': ip_res['a_rating'], 'Home_Form': ip_res['h_form'], 'Away_Form': ip_res['a_form'],
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
            
            match_groups = open_df.groupby(['Tournament_Name', 'Match']).size().reset_index(name='Bet_Count')
            settle_opts = [f"{r['Tournament_Name']} | {r['Match']} (共 {r['Bet_Count']} 張注單)" for _, r in match_groups.iterrows()]
                
            sel_to_settle = st.selectbox("選擇結算比賽", settle_opts, key="sel_settle_match")
            
            if sel_to_settle:
                sel_tourn = sel_to_settle.split(" | ")[0]
                sel_match = sel_to_settle.split(" | ")[1].split(" (共")[0]
                
                target_bets = open_df[(open_df['Tournament_Name'] == sel_tourn) & (open_df['Match'] == sel_match)]
                
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
                settle_hg = c_sg1.number_input("結算: 主隊入球", 0, 50, s_hg_val, key="shg")
                settle_ag = c_sg2.number_input("結算: 客隊入球", 0, 50, s_ag_val, key="sag")
                settle_hc = c_sc1.number_input("結算: 主隊角球", 0, 50, s_hc_val, key="shc")
                settle_ac = c_sc2.number_input("結算: 客隊角球", 0, 50, s_ac_val, key="sac")
                
                if st.button("⚖️ 確定結算此賽事所有注單", type="primary", use_container_width=True):
                    mask = (st.session_state.df_db['Tournament_Name'] == sel_tourn) & (st.session_state.df_db['Match'] == sel_match) & (st.session_state.df_db['Status'] == 'Open')
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
