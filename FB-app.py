import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import uuid
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 核心設定與系統常數
# ==========================================
st.set_page_config(page_title="足球博彩精算與 ML 系統", page_icon="⚽", layout="wide")

REAL_DB = "football_betting_db.csv"
TEST_DB = "football_betting_db_test.csv"
MARGIN_HKJC = 1.085

COLUMNS = [
    'id', 'timestamp', 'data_type', 'match_type', 'home_team', 'away_team',
    'home_strength', 'away_strength', 'home_form_w', 'home_form_d', 'home_form_l',
    'away_form_w', 'away_form_d', 'away_form_l', 'initial_odds', 'final_odds',
    'danger_ratio', 'bet_type', 'handicap_line', 'odds', 'expected_ev', 
    'bet_amount', 'status', 'result_h', 'result_a', 'profit', 'unit_profit', 
    'math_prob', 'ml_prob', 'final_prob'
]

# ==========================================
# 初始化 Session State
# ==========================================
if 'is_real_mode' not in st.session_state:
    st.session_state.is_real_mode = False
if 'initial_bankroll' not in st.session_state:
    st.session_state.initial_bankroll = 10000.0

# ==========================================
# 數據庫操作模組
# ==========================================
def get_db_name():
    return REAL_DB if st.session_state.is_real_mode else TEST_DB

def load_db(file_name=None):
    fname = file_name if file_name else get_db_name()
    if os.path.exists(fname):
        return pd.read_csv(fname)
    return pd.DataFrame(columns=COLUMNS)

def save_db(df, file_name=None):
    fname = file_name if file_name else get_db_name()
    df.to_csv(fname, index=False, encoding='utf-8-sig')

def add_record(record_dict):
    df = load_db()
    record_dict['id'] = str(uuid.uuid4())
    record_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record_dict['data_type'] = 'Real' if st.session_state.is_real_mode else 'Test'
    df = pd.concat([df, pd.DataFrame([record_dict])], ignore_index=True)
    save_db(df)

# ==========================================
# 數學精算與 ML 引擎
# ==========================================
def calculate_margin_odds(high_odds):
    if high_odds <= 1.0:
        return 0.0
    try:
        low_odds = 1 / (MARGIN_HKJC - (1 / high_odds))
        return round(max(low_odds, 1.01), 2)
    except ZeroDivisionError:
        return 0.0

def strength_to_val(s):
    return {'強': 3, '中': 2, '弱': 1}.get(s, 2)

def calculate_math_prob(h_str, a_str, hw, hd, hl, aw, ad, al):
    str_diff = strength_to_val(h_str) - strength_to_val(a_str)
    
    h_total = hw + hd + hl
    a_total = aw + ad + al
    h_form = (hw + hd * 0.5) / h_total if h_total > 0 else 0.5
    a_form = (aw + ad * 0.5) / a_total if a_total > 0 else 0.5
    form_diff = h_form - a_form
    
    base = 0.5 + (str_diff * 0.05) + (form_diff * 0.15)
    return float(np.clip(base, 0.05, 0.95))

def train_and_predict_ml(current_features, data_df):
    settled = data_df[(data_df['status'] == 'Settled') & (data_df['data_type'] == 'Real')]
    if len(settled) <= 50:
        return None, "🤖 樣本數不足 50 筆，ML 尚未介入"
    
    # Feature Engineering
    settled = settled.copy()
    settled['str_diff'] = settled['home_strength'].apply(strength_to_val) - settled['away_strength'].apply(strength_to_val)
    settled['h_form'] = (settled['home_form_w'] + settled['home_form_d']*0.5) / (settled['home_form_w'] + settled['home_form_d'] + settled['home_form_l'] + 0.001)
    settled['a_form'] = (settled['away_form_w'] + settled['away_form_d']*0.5) / (settled['away_form_w'] + settled['away_form_d'] + settled['away_form_l'] + 0.001)
    settled['form_diff'] = settled['h_form'] - settled['a_form']
    settled['odds_movement'] = settled['initial_odds'] - settled['final_odds']
    
    # Target: Profit > 0 means Win/Win Half (1), else Loss/Draw (0)
    settled['target'] = (settled['profit'] > 0).astype(int)
    
    X = settled[['str_diff', 'form_diff', 'odds_movement', 'danger_ratio']]
    y = settled['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    X_pred = pd.DataFrame([current_features], columns=['str_diff', 'form_diff', 'odds_movement', 'danger_ratio'])
    prob = model.predict_proba(X_pred)[0][1]
    return prob, "🤖 ML 模型已介入預測"

def calculate_ev_and_kelly(prob, odds, bankroll):
    ev = (prob * odds) - 1
    if ev > 0:
        b = odds - 1
        q = 1 - prob
        kelly_frac = (prob * b - q) / b
        # Apply fractional Kelly for safety (0.25)
        safe_kelly = max(0, kelly_frac * 0.25)
        raw_amount = bankroll * safe_kelly
        bet_amount = round(raw_amount / 10) * 10 
        return ev, max(bet_amount, 10)
    return ev, 0

def ah_5_state_profit(bet_amount, odds, diff):
    # diff = (Home + Line) - Away. 
    # Positive means upper hand won handicap
    if diff >= 0.5:
        profit = bet_amount * (odds - 1)
        unit = odds - 1
    elif diff == 0.25:
        profit = (bet_amount / 2) * (odds - 1)
        unit = (odds - 1) / 2
    elif diff == 0:
        profit = 0
        unit = 0
    elif diff == -0.25:
        profit = - (bet_amount / 2)
        unit = -0.5
    else:
        profit = - bet_amount
        unit = -1.0
    return profit, unit

# ==========================================
# 側邊欄與本金管理
# ==========================================
with st.sidebar:
    st.header("⚙️ 系統控制板")
    
    # 模式切換
    mode = st.radio("系統模式", ["🧪 測試模式", "🟢 真實模式"], index=0 if not st.session_state.is_real_mode else 1)
    st.session_state.is_real_mode = (mode == "🟢 真實模式")
    
    if st.session_state.is_real_mode:
        st.success("🟢 目前處於 **真實模式** (寫入正式數據庫)")
    else:
        st.warning("🧪 目前處於 **測試模式** (不影響正式數據)")
        
    st.divider()
    
    # 本金管理
    st.subheader("💰 本金管理")
    new_bankroll = st.number_input("初始本金設定", min_value=100.0, value=st.session_state.initial_bankroll, step=100.0)
    if st.button("更新初始本金"):
        st.session_state.initial_bankroll = new_bankroll
        
    df_current = load_db()
    # 計算可用本金：初始本金 + 已結算盈虧 - 待結算注碼
    settled_profit = df_current[df_current['status'] == 'Settled']['profit'].sum()
    pending_bets = df_current[df_current['status'] == 'Pending']['bet_amount'].sum()
    available_bankroll = st.session_state.initial_bankroll + settled_profit - pending_bets
    
    st.metric("即時可用本金", f"${available_bankroll:,.2f}", f"待結算: -${pending_bets:,.2f}")
    
    st.divider()
    
    # 系統工具
    st.subheader("🛠️ 數據庫工具")
    
    @st.dialog("🔍 數據庫即時線上預覽", width="large")
    def preview_dialog():
        st.dataframe(load_db(), use_container_width=True)
        
    if st.button("🔍 線上預覽當前庫"):
        preview_dialog()
        
    # Excel 下載
    csv_data = df_current.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button(
        label="📥 下載 Excel 數據庫",
        data=csv_data,
        file_name=f"football_db_{'real' if st.session_state.is_real_mode else 'test'}.csv",
        mime="text/csv"
    )

    if not st.session_state.is_real_mode:
        if st.button("🚀 轉移測試數據至正式庫"):
            df_test = load_db(TEST_DB)
            if not df_test.empty:
                df_test['data_type'] = 'Real'
                df_real = load_db(REAL_DB)
                df_real = pd.concat([df_real, df_test], ignore_index=True)
                save_db(df_real, REAL_DB)
                save_db(pd.DataFrame(columns=COLUMNS), TEST_DB) # 清空測試
                st.success("轉移成功！")
                st.rerun()
                
        if st.button("🗑️ 清空測試數據庫"):
            save_db(pd.DataFrame(columns=COLUMNS), TEST_DB)
            st.success("已清空測試數據庫！")
            st.rerun()

# ==========================================
# 主畫面 Tabs
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 賽前分析 (Pre-Match)", 
    "⚡ 即場分析 (In-Play)", 
    "💰 結算與儀表板 (Dashboard)", 
    "🚀 量化回測引擎 (Backtest)"
])

# ------------------------------------------
# Tab 1: 賽前分析
# ------------------------------------------
with tab1:
    st.header("📋 賽前精算模型")
    
    col1, col2 = st.columns(2)
    with col1:
        match_type = st.selectbox("賽事類別", ["英超", "西甲", "義甲", "德甲", "法甲", "歐冠", "其他"])
        home_team = st.text_input("主隊名稱")
        home_str = st.selectbox("主隊實力", ["強", "中", "弱"], index=1)
        st.markdown("**主隊近5場狀態**")
        c1, c2, c3 = st.columns(3)
        hw = c1.number_input("主勝", min_value=0, max_value=5, value=2, key='hw')
        hd = c2.number_input("主和", min_value=0, max_value=5, value=1, key='hd')
        hl = c3.number_input("主敗", min_value=0, max_value=5, value=2, key='hl')
        
    with col2:
        bet_type = st.selectbox("投注類型", ["亞洲讓球盤 (AH)", "大小球 (O/U)"])
        away_team = st.text_input("客隊名稱")
        away_str = st.selectbox("客隊實力", ["強", "中", "弱"], index=1)
        st.markdown("**客隊近5場狀態**")
        c4, c5, c6 = st.columns(3)
        aw = c4.number_input("客勝", min_value=0, max_value=5, value=2, key='aw')
        ad = c5.number_input("客和", min_value=0, max_value=5, value=1, key='ad')
        al = c6.number_input("客敗", min_value=0, max_value=5, value=2, key='al')

    st.divider()
    st.subheader("📊 盤口與賠率設定")
    
    col_odds1, col_odds2, col_odds3 = st.columns(3)
    with col_odds1:
        handicap_line = st.number_input("讓球/大小盤口線 (主讓為負)", value=-0.5, step=0.25)
    with col_odds2:
        high_odds = st.number_input("大/上盤賠率 (Initial)", min_value=1.01, value=1.90, step=0.01)
    with col_odds3:
        manual_override = st.checkbox("解鎖手動修改下盤賠率")
        auto_low = calculate_margin_odds(high_odds)
        low_odds = st.number_input("小/下盤賠率", value=auto_low, disabled=not manual_override, step=0.01)

    if st.button("🧠 執行精算與預測", type="primary"):
        if not home_team or not away_team:
            st.error("請輸入主客隊名稱")
        else:
            # Meta-Recommendation (雙軌決策引擎)
            df_real = load_db(REAL_DB)
            league_count = len(df_real[df_real['match_type'] == match_type])
            st.info(f"🔍 **雙軌決策掃描**：同賽制樣本數 = {league_count}。{'✅ 樣本充足，優先參考局部特徵' if league_count >= 15 else '⚠️ 樣本不足 (<15)，建議參考全域數據與數學模型。'}")
            
            # Math Prob
            math_prob = calculate_math_prob(home_str, away_str, hw, hd, hl, aw, ad, al)
            
            # ML Prob
            str_diff_val = strength_to_val(home_str) - strength_to_val(away_str)
            h_form = (hw + hd*0.5) / (hw+hd+hl+0.001)
            a_form = (aw + ad*0.5) / (aw+ad+al+0.001)
            form_diff_val = h_form - a_form
            odds_movement = 0.0 # Pre-match initial equals final for calculation moment
            
            ml_prob, ml_msg = train_and_predict_ml([str_diff_val, form_diff_val, odds_movement, 0.5], df_real)
            
            if ml_prob is not None:
                st.success(ml_msg)
                final_prob = (math_prob * 0.4) + (ml_prob * 0.6)
            else:
                st.warning(ml_msg)
                ml_prob = 0.0
                final_prob = math_prob
                
            # Calculation
            ev, kelly_bet = calculate_ev_and_kelly(final_prob, high_odds, available_bankroll)
            
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric("綜合勝率 (Probability)", f"{final_prob*100:.1f}%")
            col_res2.metric("預期期望值 (EV)", f"{ev:.3f}", "EV > 0 適合投注" if ev > 0 else "EV < 0 不建議")
            col_res3.metric("Kelly 建議注碼", f"${kelly_bet:,}")
            
            if ev > 0 and kelly_bet > 0:
                st.write("---")
                if st.button("✅ 確認寫入待結算清單"):
                    record = {
                        'match_type': match_type, 'home_team': home_team, 'away_team': away_team,
                        'home_strength': home_str, 'away_strength': away_str,
                        'home_form_w': hw, 'home_form_d': hd, 'home_form_l': hl,
                        'away_form_w': aw, 'away_form_d': ad, 'away_form_l': al,
                        'initial_odds': high_odds, 'final_odds': high_odds,
                        'danger_ratio': 0.5, 'bet_type': bet_type, 'handicap_line': handicap_line,
                        'odds': high_odds, 'expected_ev': ev, 'bet_amount': kelly_bet,
                        'status': 'Pending', 'math_prob': math_prob, 'ml_prob': ml_prob, 'final_prob': final_prob
                    }
                    add_record(record)
                    st.success("寫入成功！請前往 Tab 3 進行結算。")

# ------------------------------------------
# Tab 2: 即場分析
# ------------------------------------------
with tab2:
    st.header("⚡ 即場動態分析 (In-Play)")
    
    col_ip1, col_ip2 = st.columns(2)
    with col_ip1:
        ip_home = st.text_input("主隊 (In-Play)")
        ip_mins = st.number_input("比賽進行時間 (分鐘)", 1, 120, 45)
        ip_possession = st.slider("主隊控球率 (%)", 0, 100, 50)
        ip_h_danger = st.number_input("主隊危險進攻", 0, 200, 20)
        ip_a_danger = st.number_input("客隊危險進攻", 0, 200, 15)
        
    with col_ip2:
        ip_away = st.text_input("客隊 (In-Play)")
        ip_h_goals = st.number_input("主隊目前進球", 0, 20, 0)
        ip_a_goals = st.number_input("客隊目前進球", 0, 20, 0)
        ip_h_red = st.number_input("主隊紅牌", 0, 5, 0)
        ip_a_red = st.number_input("客隊紅牌", 0, 5, 0)
        
    st.divider()
    c_op1, c_op2, c_op3 = st.columns(3)
    with c_op1:
        ip_line = st.number_input("即場盤口線", value=0.0, step=0.25)
    with c_op2:
        ip_odds = st.number_input("即場大/上賠率", min_value=1.01, value=1.85, step=0.01)
    with c_op3:
        ip_manual = st.checkbox("解鎖小/下賠率")
        ip_low_odds = st.number_input("即場小/下賠率", value=calculate_margin_odds(ip_odds), disabled=not ip_manual)

    if st.button("⚡ 產生即場策略"):
        if not ip_home or not ip_away:
            st.error("請輸入隊伍名稱")
        else:
            # Danger ratio logic
            total_danger = ip_h_danger + ip_a_danger
            danger_ratio = ip_h_danger / total_danger if total_danger > 0 else 0.5
            
            # Simple momentum math prob for in-play
            base = 0.5
            base += (ip_possession - 50) * 0.005
            base += (danger_ratio - 0.5) * 0.2
            base -= ip_h_red * 0.15
            base += ip_a_red * 0.15
            ip_math_prob = float(np.clip(base, 0.05, 0.95))
            
            ev, ip_kelly = calculate_ev_and_kelly(ip_math_prob, ip_odds, available_bankroll)
            
            st.metric("即場預期勝率", f"{ip_math_prob*100:.1f}%")
            st.metric("即場 EV", f"{ev:.3f}")
            st.metric("建議即場注碼", f"${ip_kelly:,}")
            
            if ev > 0 and ip_kelly > 0:
                if st.button("寫入即場待結算"):
                    record = {
                        'match_type': "即場", 'home_team': ip_home, 'away_team': ip_away,
                        'home_strength': "中", 'away_strength': "中",
                        'home_form_w': 0, 'home_form_d': 0, 'home_form_l': 0,
                        'away_form_w': 0, 'away_form_d': 0, 'away_form_l': 0,
                        'initial_odds': ip_odds, 'final_odds': ip_odds,
                        'danger_ratio': danger_ratio, 'bet_type': "In-Play AH", 
                        'handicap_line': ip_line, 'odds': ip_odds, 'expected_ev': ev, 
                        'bet_amount': ip_kelly, 'status': 'Pending', 
                        'math_prob': ip_math_prob, 'ml_prob': 0.0, 'final_prob': ip_math_prob
                    }
                    add_record(record)
                    st.success("已寫入即場注單！")

# ------------------------------------------
# Tab 3: 結算與儀表板
# ------------------------------------------
with tab3:
    st.header("💰 待結算與績效儀表板")
    
    # Dashboard: Strategy Comparison
    df_all = load_db()
    df_settled = df_all[df_all['status'] == 'Settled']
    
    st.subheader("📊 策略績效極限對比 (已結算數據)")
    dash_col1, dash_col2, dash_col3 = st.columns(3)
    
    if not df_settled.empty:
        dyn_profit = df_settled['profit'].sum()
        flat_profit = df_settled['unit_profit'].sum() * 100 # $100 per unit
        
        dash_col1.metric("系統動態注碼 (Kelly) 總盈虧", f"${dyn_profit:,.2f}")
        dash_col2.metric("固定平注策略 ($100/Unit) 總盈虧", f"${flat_profit:,.2f}")
        
        diff = dyn_profit - flat_profit
        adv = "動態策略較優" if diff > 0 else "平注策略較優"
        dash_col3.metric("優劣差異", f"${abs(diff):,.2f}", f"{adv}")
    else:
        st.info("尚無已結算紀錄可供分析。")
        
    st.divider()
    st.subheader("📝 待結算注單")
    
    df_pending = df_all[df_all['status'] == 'Pending']
    
    if df_pending.empty:
        st.write("目前無待結算注單。")
    else:
        for index, row in df_pending.iterrows():
            with st.container():
                cols = st.columns([3, 2, 2, 2, 2, 2])
                cols[0].markdown(f"**{row['home_team']}** vs **{row['away_team']}**")
                cols[1].markdown(f"盤口: {row['handicap_line']}")
                cols[2].markdown(f"賠率: {row['odds']}")
                cols[3].markdown(f"注碼: ${row['bet_amount']}")
                
                res_h = cols[4].number_input("主隊賽果", min_value=0, key=f"rh_{row['id']}")
                res_a = cols[5].number_input("客隊賽果", min_value=0, key=f"ra_{row['id']}")
                
                if st.button("結算此單", key=f"btn_{row['id']}", type="primary"):
                    # 計算亞洲盤5態
                    diff = (res_h + row['handicap_line']) - res_a
                    profit, unit_profit = ah_5_state_profit(row['bet_amount'], row['odds'], diff)
                    
                    # 更新 DataFrame
                    df_all.loc[df_all['id'] == row['id'], 'result_h'] = res_h
                    df_all.loc[df_all['id'] == row['id'], 'result_a'] = res_a
                    df_all.loc[df_all['id'] == row['id'], 'profit'] = profit
                    df_all.loc[df_all['id'] == row['id'], 'unit_profit'] = unit_profit
                    df_all.loc[df_all['id'] == row['id'], 'status'] = 'Settled'
                    
                    save_db(df_all)
                    st.success(f"結算完成！盈虧: ${profit:.2f} (單位盈虧: {unit_profit})。背景 ML 模型參數已準備更新。")
                    st.rerun()
            st.markdown("---")

# ------------------------------------------
# Tab 4: 量化回測引擎
# ------------------------------------------
with tab4:
    st.header("🚀 量化回測與模型優化 (Backtesting Engine)")
    st.markdown("針對歷史已結算數據，模擬不同 EV 門檻過濾下的長線回報率 (ROI)，找出最佳參數。")
    
    backtest_df = load_db()
    backtest_df = backtest_df[backtest_df['status'] == 'Settled']
    
    if backtest_df.empty:
        st.warning("數據庫中無已結算數據可供回測。請先累積結算賽事。")
    else:
        st.markdown(f"**可用回測樣本數**: {len(backtest_df)} 筆")
        
        c_bt1, c_bt2 = st.columns(2)
        with c_bt1:
            min_ev_range = st.slider("選擇 EV 門檻測試上限 (0.01 到上限)", 0.05, 0.50, 0.20, 0.01)
        with c_bt2:
            bt_match_type = st.selectbox("賽制篩選", ["全部"] + list(backtest_df['match_type'].unique()))
            
        if st.button("啟動量化回測 🚀", type="primary"):
            if bt_match_type != "全部":
                bt_data = backtest_df[backtest_df['match_type'] == bt_match_type]
            else:
                bt_data = backtest_df
                
            results = []
            ev_thresholds = np.arange(0.00, min_ev_range + 0.01, 0.01)
            
            best_roi = -999.0
            best_ev = 0.0
            
            for threshold in ev_thresholds:
                filtered_bets = bt_data[bt_data['expected_ev'] >= threshold]
                if len(filtered_bets) > 0:
                    total_invested = filtered_bets['bet_amount'].sum()
                    total_profit = filtered_bets['profit'].sum()
                    roi = (total_profit / total_invested) if total_invested > 0 else 0.0
                else:
                    roi = 0.0
                    total_invested = 0
                    total_profit = 0
                    
                results.append({
                    'EV_Threshold': threshold,
                    'ROI': roi,
                    'Bets_Count': len(filtered_bets)
                })
                
                if roi > best_roi and len(filtered_bets) >= 5: # Require at least 5 bets for statistical relevance
                    best_roi = roi
                    best_ev = threshold
            
            res_df = pd.DataFrame(results)
            
            st.success(f"🎯 **回測分析結論**：歷史數據顯示，當 **EV > {best_ev:.2f}** 時，長線 ROI 達到最高 ({best_roi*100:.2f}%)。")
            
            st.subheader("不同 EV 門檻對應的 ROI 變化圖表")
            res_df.set_index('EV_Threshold', inplace=True)
            st.line_chart(res_df['ROI'])
            
            st.subheader("詳細回測數據")
            st.dataframe(res_df)
