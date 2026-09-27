import pandas as pd
import requests
from datetime import datetime
import os
import json

def fetch_api_data(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.tipsme.hk/",
        "Origin": "https://www.tipsme.hk"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"API Request Error: {e}")
    return None

def main():
    db_file = "football_betting_db.csv"
    print(f"[{datetime.now()}] 開始執行自動同步背景任務...")

    if os.path.exists(db_file):
        df = pd.read_csv(db_file)
    else:
        print("❌ 找不到資料庫檔案，終止同步。")
        return

    today_str = datetime.now().strftime("%Y-%m-%d")
    schedule_url = f"https://tipsme-web.azurewebsites.net/api/Score/schedule/hkjc/{today_str}"
    
    data = fetch_api_data(schedule_url)
    if data and isinstance(data, list):
        print(f"✅ 成功取得今日賽程，共 {len(data)} 場賽事。")
        updated_count = 0
        for item in data:
            match_id = str(item.get('matchId', ''))
            if not match_id: continue
            
            if 'Tipsme_ID' in df.columns:
                mask = df['Tipsme_ID'].astype(str) == match_id
                if mask.any():
                    odds_url = f"https://tipsme-web.azurewebsites.net/api/Score/odds/hkjc/{match_id}"
                    odds_data = fetch_api_data(odds_url)
                    if odds_data:
                        df.loc[mask, 'Odds_History'] = json.dumps(odds_data, ensure_ascii=False)
                        updated_count += 1
                        
        print(f"🔄 已自動更新 {updated_count} 場進行中賽事的賠率走勢。")
    else:
        print("⚠️ 無法取得今日賽程 API 數據或格式不符。")

    df.to_csv(db_file, index=False)
    print(f"[{datetime.now()}] 背景同步任務執行完畢，資料庫已儲存。")

if __name__ == "__main__":
    main()
