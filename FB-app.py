import streamlit as st
import pandas as pd
import os
import json
import base64
import re
from datetime import datetime

# 嘗試載入機器學習套件
try:
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    HAS_AI_MODULES = True
except ImportError:
    HAS_AI_MODULES = False
    np = None

# ==========================================
# 1. 初始化設定與資料庫 Schema
# ==========================================
st.set_page_config(page_title="足球博彩精算與資金管理系統", page_icon="⚽", layout="wide")

DB_COLUMNS = [
    'ID', 'Date', 'Status', 
    'Tournament_Name', 'Tournament_Category', 'Match', 'Home_Team', 'Away_Team', 
    'Home_Rating', 'Away_Rating', 'Home_Form', 'Away_Form',
    'Bet_Type', 'Selection', 'Initial_Line', 'Initial_Odds', 'Stake', 'Odds_History',
    'InPlay_Minute', 'Home_DA', 'Away_DA', 'Home_SoT', 'Away_SoT', 'Home_SoFF', 'Away_SoFF',
    'Home_Red', 'Away_Red', 'Home_Sub', 'Away_Sub', 'Home_Possession', 'Away_Possession',
    'Home_Goal', 'Away_Goal', 'Home_Corner', 'Away_Corner', 
    'Result_Label', 'Profit', 'Unit_Profit', 'Payout'
]
LOG_COLUMNS = ['ID', 'Date', 'Match', 'Analysis_Content', 'Confidence_Level']
CAPITAL_COLUMNS = ['ID', 'Date', 'Type', 'Amount', 'Note']
CATEGORY_OPTIONS = ["國內聯賽 (Domestic League)", "國際聯賽 (International League)", "國際盃賽 (Cup)", "國內盃賽 (Domestic Cup)", "友誼賽 (Friendly)"]

def load_db(filename, columns):
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            if 'League' in df.columns and 'Tournament_Name' not in df.columns:
                df['Tournament_Name'] = df['League']
                df['Tournament_Category'] = CATEGORY_OPTIONS[0]
            
            for col in columns:
                if col not in df.columns: 
                    df[col] = pd.Series(dtype='object')
            return df[columns]
        except Exception:
            return pd.DataFrame(columns=columns)
    else:
        return pd.DataFrame(columns=columns)

def save_db(df, filename):
    df.to_csv(filename, index=False)

# ==========================================
# 2. 資金、風控與累計算式
# ==========================================
def recalculate_bankroll_from_scratch(df_cap, df_db):
    if df_cap.empty:
        total_deposit, total_withdraw = 0.0, 0.0
    else:
        total_deposit = pd.to_numeric(df_cap[df_cap['Type'] == 'Deposit']['Amount'], errors='coerce').sum()
        total_withdraw = pd.to_numeric(df_cap[df_cap['Type'] == 'Withdraw']['Amount'], errors='coerce').sum()
    
    net_deposit = max(0.0, total_deposit - total_withdraw)
    
    if df_db.empty:
        total_profit = 0.0
    else:
        settled_df = df_db[df_db['Status'] == 'Settled']
        total_profit = pd.to_numeric(settled_df['Profit'], errors='coerce').sum()
        
    current_bankroll = net_deposit + total_profit
    max_single_stake = net_deposit * 0.10 
    
    return round(total_deposit, 2), round(total_withdraw, 2), round(net_deposit, 2), round(total_profit, 2), round(current_bankroll, 2), round(max_single_stake, 2)

def calculate_settlement(bet_type, selection, line, odds, stake, h_g, a_g, h_c=0, a_c=0):
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
        profit = stake * (odds - 1)
        payout = stake + profit
        unit_profit = round(odds - 1.0, 2)
    elif diff == 0.25:
        res_label = "🟢 贏半"
        profit = stake * (odds - 1) / 2
        payout = stake + profit
        unit_profit = round((odds - 1.0) / 2.0, 2)
    elif diff == 0.0:
        res_label = "⚪ 走盤退本"
        profit = 0.0
        payout = stake
        unit_profit = 0.0
    elif diff == -0.25:
        res_label = "🔴 輸半 (退回半本)"
        profit = -stake / 2
        payout = stake / 2
        unit_profit = -0.50
    else:
        res_label = "❌ 全輸"
        profit = -stake
        payout = 0.0
        unit_profit = -1.00

    return round(profit, 2), round(payout, 2), unit_profit, res_label, diff

def display_cumulative_metrics(df):
    total_profit = pd.to_numeric(df['Profit'], errors='coerce').sum()
    total_unit_profit = pd.to_numeric(df['Unit_Profit'], errors='coerce').sum()
    total_payout = pd.to_numeric(df['Payout'], errors='coerce').sum()
    
    st.markdown("### 📊 數據庫累計總額看板 (Cumulative Summary)")
    m1, m2, m3 = st.columns(3)
    m1.metric("累積淨盈虧 (Total Profit)", f"${total_profit:,.2f}", delta=f"{total_profit:,.2f}")
    m2.metric("累積單位平注盈虧 (Total Unit Profit)", f"{total_unit_profit:,.2f} U", delta=f"{total_unit_profit:,.2f} U")
    m3.metric("累積派彩總額 (Total Payout)", f"${total_payout:,.2f}")
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
    if n_samples < 15:
        return {'dim': dim_name, 'valid': False, 'msg': f"樣本數不足 ({n_samples} < 15場)", 'roi': 0, 'acc': 0, 'n': n_samples}
    
    total_stake = pd.to_numeric(df_subset['Stake'], errors='coerce').sum()
    total_profit = pd.to_numeric(df_subset['Profit'], errors='coerce').sum()
    roi = (total_profit / total_stake) if total_stake > 0 else 0
    wins = len(df_subset[pd.to_numeric(df_subset['Unit_Profit'], errors='coerce') > 0])
    acc = wins / n_samples if n_samples > 0 else 0

    candidates = [c.copy() for c in candidates_base]
    
    model_success = False
    if HAS_AI_MODULES:
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
            shift = (acc - 0.5) * 0.2 + (roi * 0.1)
            c['prob'] = max(0.05, min(0.95, c['base_prob'] + shift))

    for c in candidates:
        c['ev'] = c['prob'] * (c['odds'] - 1) - (1 - c['prob'])

    candidates = sorted(candidates, key=lambda x: x['ev'], reverse=True)
    score = (roi * 0.7) + (acc * 0.3)
    
    return {
        'dim': dim_name, 'valid': True, 'msg': "運算成功", 'roi': roi, 'acc': acc, 
        'n': n_samples, 'candidates': candidates, 'best': candidates[0], 'score': score
    }

# ==========================================
# 4. 模組化組件：動態賠率與 Margin 運算 UI
# ==========================================
def render_odds_section(odds_history_state, prefix="pre"):
    for i, row in enumerate(odds_history_state):
        r_id = row['id']
        up_key, type_key, unlock_key = f"{prefix}_up_{r_id}", f"{prefix}_t_{r_id}", f"{prefix}_u_{r_id}"
        margin_key, low_key, line_key = f"{prefix}_m_{r_id}", f"{prefix}_low_{r_id}", f"{prefix}_l_{r_id}"

        # 優先處理型態切換的數值初始化
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

        # Margin 運算核心：並將結果強制寫回 session_state 確保 UI 即時變更
        if not row['unlock']:
            m_val, u_val = float(row.get('margin', 1.085)), float(row.get('upper', 1.90))
            try:
                calc_lower = round(1 / (m_val - (1 / u_val)), 2)
                row['lower'] = calc_lower if calc_lower > 1 else 1.01
            except:
                row['lower'] = 1.90
            st.session_state[low_key] = float(row['lower']) # 強制覆蓋對盤口的舊有狀態
        else:
            try:
                row['margin'] = (1 / float(row.get('upper', 1.90))) + (1 / float(row.get('lower', 1.90)))
            except:
                row['margin'] = 1.085
            st.session_state[margin_key] = float(row['margin']) # 強制覆蓋 margin 的舊有狀態

        c1, c2, c3, c4, c5, c6 = st.columns([2, 1.5, 1.5, 2, 1.5, 1])
        type_idx = ["讓球", "入球大小", "角球大小"].index(row['type']) if row['type'] in ["讓球", "入球大小", "角球大小"] else 0
        row['type'] = c1.selectbox(f"盤口類型 {i+1}", ["讓球", "入球大小", "角球大小"], key=type_key, index=type_idx)
        
        # 根據盤口類型設定動態 step
        line_step = 1.0 if row['type'] == "角球大小" else 0.25
        row['line'] = c2.number_input("盤口線", step=line_step, value=float(row['line']), key=line_key)
        
        up_lbl, low_lbl = ("主隊", "客隊") if row['type'] == "讓球" else ("大盤(Over)", "小盤(Under)")
        row['upper'] = c3.number_input(f"{up_lbl} 賠率", min_value=1.01, value=float(row['upper']), step=0.01, key=up_key)
        row['unlock'] = c4.checkbox("🔓 解鎖", value=row.get('unlock', False), key=unlock_key)
        
        if not row['unlock']:
            row['margin'] = c4.number_input("抽水(Margin)", min_value=1.00, value=float(row.get('margin', 1.085)), step=0.005, format="%.3f", key=margin_key)
            row['lower'] = c5.number_input(f"{low_lbl} 賠率", value=float(row['lower']), disabled=True, key=low_key)
        else:
            row['lower'] = c5.number_input(f"{low_lbl} 賠率", min_value=1.01, step=0.01, value=float(row['lower']), key=low_key)
            c4.caption(f"隱含抽水: **{row['margin']:.3f}**")

        if len(odds_history_state) > 1:
            if c6.button("❌", key=f"{prefix}_d_{r_id}"):
                odds_history_state.pop(i)
                st.rerun()

    ac1, ac2, ac3 = st.columns(3)
    max_id = max([r['id'] for r in odds_history_state]) if odds_history_state else 0
    if ac1.button("➕ 讓球盤 (0.0)", key=f"{prefix}_add_hand", use_container_width=True):
        odds_history_state.append({"id": max_id+1, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}); st.rerun()
    if ac2.button("➕ 入球大小 (2.5)", key=f"{prefix}_add_goal", use_container_width=True):
        odds_history_state.append({"id": max_id+1, "type": "入球大小", "line": 2.5, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}); st.rerun()
    if ac3.button("➕ 角球大小 (9.5)", key=f"{prefix}_add_corn", use_container_width=True):
        odds_history_state.append({"id": max_id+1, "type": "角球大小", "line": 9.5, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}); st.rerun()

# ==========================================
# 5. 主程式 UI 
# ==========================================
def main():
    st.title("⚽ 足球博彩精算與資金管理系統")
    
    st.sidebar.header("⚙️ 系統設定與資金管理")
    mode = st.sidebar.radio("運作模式選擇", ["🧪 測試模式", "🟢 真實模式"])
    
    db_file = "football_betting_db_test.csv" if mode == "🧪 測試模式" else "football_betting_db.csv"
    capital_file = "football_capital_db_test.csv" if mode == "🧪 測試模式" else "football_capital_db.csv"
    
    st.session_state.df_db = load_db(db_file, DB_COLUMNS)
    st.session_state.df_cap = load_db(capital_file, CAPITAL_COLUMNS)

    tot_dep, tot_wit, net_dep, tot_pnl, curr_bankroll, max_stake = recalculate_bankroll_from_scratch(st.session_state.df_cap, st.session_state.df_db)
    
    st.sidebar.divider()
    st.sidebar.subheader("💰 系統本金與盈虧總覽")
    st.sidebar.metric("系統淨存入本金", f"${net_dep:,.2f}")
    st.sidebar.metric("累積總盈虧 (PnL)", f"${tot_pnl:,.2f}", delta=f"${tot_pnl:,.2f}")
    st.sidebar.metric("當前總可用資金", f"${curr_bankroll:,.2f}")
    st.sidebar.caption(f"🛑 **單注上限 (本金 10%)**: `${max_stake:,.2f}`")

    t_pre, t_inplay, t_settle, t_ai = st.tabs(["📝 賽前建檔與投注", "⏱️ 即場動態", "⚖️ 賽果結算與管理", "🤖 全局模型"])

    with t_pre:
        st.subheader("📝 賽事建檔與智能盤口走勢分析")
        if net_dep <= 0: st.warning("⚠️ 目前系統內部尚無存入本金！無法精確計算注碼。")
        
        opts_tournaments = ["➕ 新增手動輸入..."] + sorted(list(set(st.session_state.df_db['Tournament_Name'].dropna().unique())))
        opts_teams = ["➕ 新增手動輸入..."] + sorted(list(set(st.session_state.df_db['Home_Team'].dropna().tolist() + st.session_state.df_db['Away_Team'].dropna().tolist())))
        
        st.markdown("##### 1. 賽事與球隊資料")
        col_t, col_c = st.columns(2)
        sel_tournament = col_t.selectbox("賽事名稱 (Tournament Name)", opts_tournaments)
        tournament_name = col_t.text_input("輸入新賽事名稱") if sel_tournament == "➕ 新增手動輸入..." else sel_tournament
        
        default_cat_idx = 0
        if sel_tournament != "➕ 新增手動輸入...":
            match_rows = st.session_state.df_db[st.session_state.df_db['Tournament_Name'] == tournament_name]
            if not match_rows.empty:
                last_cat = match_rows.iloc[-1]['Tournament_Category']
                if last_cat in CATEGORY_OPTIONS:
                    default_cat_idx = CATEGORY_OPTIONS.index(last_cat)
                    
        tournament_category = col_c.selectbox("賽事分類 (Tournament Category)", CATEGORY_OPTIONS, index=default_cat_idx)

        col_h, col_a = st.columns(2)
        sel_home = col_h.selectbox("主隊名稱", opts_teams, key="sh")
        home_team = col_h.text_input("輸入新主隊") if sel_home == "➕ 新增手動輸入..." else sel_home
        sel_away = col_a.selectbox("客隊名稱", opts_teams, key="sa")
        away_team = col_a.text_input("輸入新客隊") if sel_away == "➕ 新增手動輸入..." else sel_away

        c_hr, c_ar = st.columns(2)
        home_rating = c_hr.selectbox("主隊實力", ["S", "A", "B", "C", "D"])
        away_rating = c_ar.selectbox("客隊實力", ["S", "A", "B", "C", "D"])

        st.markdown("##### 2. 近 5 場狀態 (勝/和/敗)")
        f1, f2, f3, f4, f5, f6 = st.columns(6)
        home_form = f"{f1.number_input('主勝',0,10,3)}W{f2.number_input('主和',0,10,1)}D{f3.number_input('主敗',0,10,1)}L"
        away_form = f"{f4.number_input('客勝',0,10,2)}W{f5.number_input('客和',0,10,2)}D{f6.number_input('客敗',0,10,1)}L"

        st.markdown("##### 3. 賽前盤口與賠率走勢紀錄")
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
            if 'ev' in best_bet and best_bet['ev'] > 0 and curr_bankroll > 0:
                b = best_bet['odds'] - 1
                kelly = max(0.0, min((best_bet['prob'] * b - (1 - best_bet['prob'])) / b, 0.10))
                suggested_stake = max(10.0, min(float(max_stake), float(round((curr_bankroll * (kelly * 0.5)) / 10) * 10)))

            st.session_state.analysis_result = {
                'micro': res_micro, 'meso': res_meso, 'macro': res_macro, 'best_model': best_model,
                'best_bet': best_bet, 'stake': suggested_stake, 't_name': tournament_name, 't_cat': tournament_category
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
                        st.write(f"回測 ROI: `{r['roi']*100:.1f}%`")
                        st.write(f"歷史勝率: `{r['acc']*100:.1f}%`")
                    else:
                        st.warning(r['msg'])

            bm = res['best_model']
            bb = res['best_bet']
            dim_label_map = {"微觀 - 賽事名稱": "微觀", "中觀 - 賽事分類": "中觀", "宏觀 - 總數據": "宏觀"}
            dim_short = dim_label_map.get(bm['dim'], "宏觀")

            st.markdown(f"### 🧠 AI 最佳預測模型推薦")
            st.info(f"系統分析顯示，針對『{res['t_name']}』，採用『{bm['dim']}』級別的模型進行運算，其歷史準確率與 EV 獲利期望值最高，故本次投注策略依據此模型生成。")
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("💡 首選推薦", f"{bb['bet_type']} - {bb['label']}")
            mc2.metric(f"🎯 預期勝率 ({dim_short}修正)", f"{bb.get('prob', bb.get('base_prob',0))*100:.1f}%")
            mc3.metric("📊 修正 EV", f"{bb.get('ev', 0):.3f}")
            st.markdown(f"**建議注碼**：`${res['stake']:,.2f}`")

            with st.form("bet_form"):
                bc1, bc2, bc3 = st.columns(3)
                final_btype = bc1.selectbox("最終投注項目", [c['bet_type'] for c in bm.get('candidates', [bb])], index=0)
                final_sel = bc2.selectbox("最終投注方向", ["Home", "Away", "Over", "Under"], index=["Home", "Away", "Over", "Under"].index(bb['selection']))
                final_stake = bc3.number_input("實際下注金額 ($)", min_value=10.0, step=10.0, value=float(res['stake'] if res['stake'] > 0 else 50.0))
                
                final_row = next((r for r in st.session_state.odds_history if r['type'] == final_btype), st.session_state.odds_history[-1])
                line, odds = float(final_row['line']), float(final_row['upper']) if final_sel in ["Home", "Over"] else float(final_row['lower'])
                
                if st.form_submit_button("✅ 確定投注並寫入資料庫"):
                    new_id = f"B{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    new_record = {
                        'ID': new_id, 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Status': 'Open',
                        'Tournament_Name': res['t_name'], 'Tournament_Category': res['t_cat'], 
                        'Match': f"{home_team} vs {away_team}", 'Home_Team': home_team, 'Away_Team': away_team,
                        'Home_Rating': home_rating, 'Away_Rating': away_rating, 'Home_Form': home_form, 'Away_Form': away_form,
                        'Bet_Type': final_btype, 'Selection': final_sel, 'Initial_Line': line, 'Initial_Odds': odds, 'Stake': final_stake,
                        'Odds_History': json.dumps(st.session_state.odds_history, ensure_ascii=False)
                    }
                    st.session_state.df_db = pd.concat([st.session_state.df_db, pd.DataFrame([new_record])], ignore_index=True)
                    save_db(st.session_state.df_db, db_file)
                    st.session_state.odds_history = [{"id": 0, "type": "讓球", "line": 0.0, "upper": 1.90, "lower": 1.90, "unlock": False, "margin": 1.085}] 
                    st.session_state.show_analysis = False
                    st.toast("✅ 投注紀錄寫入成功！", icon="📝")
                    st.rerun()

    with t_inplay:
        st.subheader("⏱️ 即場賽事實時更新")
        pending_df = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
        if pending_df.empty:
            st.info("目前沒有待結算的進行中賽事 (Status='Open')。")
        else:
            select_idx = st.selectbox("請選擇要更新即場數據的賽事", pending_df.index, format_func=lambda i: pending_df.loc[i, 'Match'])
            row = pending_df.loc[select_idx]
            
            with st.form("inplay_stats_form"):
                minute = st.number_input("比賽時間 (分鐘)", 0, 120, int(row.get('InPlay_Minute', 45)) if pd.notna(row.get('InPlay_Minute')) else 45)
                c1, c2, c3, c4 = st.columns(4)
                h_g = c1.number_input("主隊入球", value=int(row.get('Home_Goal', 0)) if pd.notna(row.get('Home_Goal')) else 0)
                a_g = c2.number_input("客隊入球", value=int(row.get('Away_Goal', 0)) if pd.notna(row.get('Away_Goal')) else 0)
                h_c = c3.number_input("主隊角球", value=int(row.get('Home_Corner', 0)) if pd.notna(row.get('Home_Corner')) else 0)
                a_c = c4.number_input("客隊角球", value=int(row.get('Away_Corner', 0)) if pd.notna(row.get('Away_Corner')) else 0)
                
                if st.form_submit_button("🔄 更新實時比賽數據"):
                    st.session_state.df_db.loc[select_idx, ['InPlay_Minute', 'Home_Goal', 'Away_Goal', 'Home_Corner', 'Away_Corner']] = [minute, h_g, a_g, h_c, a_c]
                    save_db(st.session_state.df_db, db_file)
                    st.success("✅ 實時數據更新成功！")
                    st.rerun()

    with t_settle:
        st.subheader("⚖️ 賽果結算與資料庫維護")
        display_cumulative_metrics(st.session_state.df_db)
        open_bets = st.session_state.df_db[st.session_state.df_db['Status'] == 'Open']
        if not open_bets.empty:
            for idx, row in open_bets.iterrows():
                with st.expander(f"📌 {row['Match']} - {row['Bet_Type']} ({row['Selection']}) | 盤口: {row['Initial_Line']}"):
                    with st.form(f"settle_form_{row['ID']}"):
                        col1, col2 = st.columns(2)
                        h_g = col1.number_input("全場主隊入球數", min_value=0, value=int(row.get('Home_Goal', 0)) if pd.notna(row.get('Home_Goal')) else 0)
                        a_g = col2.number_input("全場客隊入球數", min_value=0, value=int(row.get('Away_Goal', 0)) if pd.notna(row.get('Away_Goal')) else 0)
                        
                        if st.form_submit_button("確認賽果並結算"):
                            prof, payout, u_prof, lbl, diff = calculate_settlement(
                                row['Bet_Type'], row['Selection'], float(row['Initial_Line']), 
                                float(row['Initial_Odds']), float(row['Stake']), h_g, a_g, 0, 0
                            )
                            st.session_state.df_db.loc[idx, ['Home_Goal', 'Away_Goal', 'Result_Label', 'Profit', 'Unit_Profit', 'Payout', 'Status']] = [h_g, a_g, lbl, prof, u_prof, payout, 'Settled']
                            save_db(st.session_state.df_db, db_file)
                            st.success(f"結算完成！結果：{lbl} | 單位盈虧：{u_prof:+.2f} U")
                            st.rerun()

    with t_ai:
        st.header("🤖 全局預測模型監控")
        df_settled = st.session_state.df_db[st.session_state.df_db['Status'] == 'Settled'].copy()
        st.write(f"當前可供訓練的歷史結算數據：**{len(df_settled)}** 筆")
        st.write(f"機器學習模組狀態 (Scikit-Learn): **{'🟢 已啟用' if HAS_AI_MODULES else '🔴 未偵測到，使用啟發式算法'}**")

if __name__ == "__main__":
    main()
