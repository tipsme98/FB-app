# FB-app T−30 三盤自動影子分析：本地候選版 RC1

基線是 [GitHub V18.1 固定提交 `1a5b243`](https://github.com/tipsme98/FB-app/commit/1a5b243f5e052217c56cda22fb61ade6d74d2126)。本候選版只供審查和離線驗證；沒有部署、沒有連正式 Supabase、沒有觸發任何模擬或實際注單寫入。凍結配置、手動研究表單、現有 CLV、5:00 及時段結算邏輯不變。

## 審查入口

| 檔案 | 審查重點 |
| --- | --- |
| `FB-app.py` | opt-in／runner 閘門、共用賽程及賠率回應、影子 fragment、共用已完成賽果。正式路徑原有回傳不變；影子錯誤隔離。 |
| `fb_app_t30_shadow.py` | 來源欄位合約、UTC/HKT、候選、盤口資格、取樣節流、獨立 SQLite、防重、結果核對與 CSV。沒有網絡、Streamlit 或正式帳簿依賴。 |
| `fb_app_parameter_lab.py` | 原版引擎原樣附在 ZIP，方便跑真實引擎的合成測試；不在 patch 的修改範圍。 |
| `tests/test_t30_shadow.py` | 無正式資料、秘密或實時 API；包含三盤完整流程、正式路徑隔離及合成資料透過原版引擎。 |

離線重跑，不需啟動 App：

```bash
python -m unittest discover -s tests -p test_t30_shadow.py -v
```

測試用本地 Python 3.14、Streamlit 1.67（既有回歸）、pandas 3.1.1、NumPy、scikit-learn 1.9.1、SciPy 及 threadpoolctl。使用 patch 的話，先在基線的乾淨副本跑 `git apply --check`，再於本地套用，不用改任何 CSV 或執行 SQL 遷移。

## 數據與運作規則

- **啟動限制：** 預設完全關閉；只有 `FB_APP_SHADOW_MODE=observe` 加上指定 runner 才會讀影子設定或寫影子資料。`FB_APP_SHADOW_DIR` 要指向專用、已存在且有持續性的絕對目錄；不要在正式資料目錄、Git repo 或無權限隔離的共享環境試跑。`runner=1` 是網頁角色標記，不是認證。首輪用沒有正式 Secrets 的隔離測試副本，不共用正式 Streamlit 設定或儲存。
- **賽程和賠率：** 與原流程共用賽程刷新及同場賠率回應；每場三盤分開驗證。不要求已有注單。原始回應不足時，只有在 T−45 至 T−30 才補讀。額外讀取每次 tick 最多一場，專用庫全局相隔至少 25 秒；同場一般 240 秒，距截點最後兩分鐘 45 秒。不依賴原快照 `unlock=True`，不以舊清單充當最新，新增的補讀不重試或切換來源。只看首頁賽程實際包含的賽事，跨午夜轉換有驗證、但翌日賽程覆蓋及高峰吞吐未經正式驗收。
- **鎖定：** 來源時間與採集時間都不晚於 T−30，並落在 15 分鐘內；最新觀察封盤、缺雙邊價、錯場或有衝突時逐盤略過。影子 tick 每 30 秒，延誤寫實際秒數及 `delayed`；最多允許截點後 150 秒內完成，超出只記略過。不能把 `delayed` 當嚴格準點資料或與準點組混報。現有 [Streamlit fragment](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment) 需要連線中的工作階段；本版不是常駐排程。
- **推論及版本：** 只重用 session 已載入的已結算歷史，不另讀正式資料庫。原版研究引擎有自己的歷史日期限制；另記錄當次訓練輸入及引擎檔 SHA-256、來源合約 SHA-256。這些雜湊可驗證輸入版本，不代替長期保存原始歷史快照。未達 EV 門檻仍保留預測；影子不即時調參、不回寫研究成果到主模型。
- **賽後：** 只旁聽原流程成功讀到的完整 Tipsme 賽果，不另發賽果請求。賽事 ID、時刻及隊名必須匹配。沒讀到賽果、取消中斷或角球缺值都待核。與先前已核實數字矛盾時隔離，不靜默改寫。五種結算及未達 EV 訊號完整留底；`hypothetical_units` 為有預測、可結算的對照回報，`strategy_units` 只在已達影子門檻才有值。未有收盤 CLV 或自動改模。
- **落盤和回退：** 只寫 `fb-app-t30-shadow.sqlite3`，按 ID／已核對開賽時刻／盤口／研究版本防重；同一 SQLite 檔可跨分頁防重，不聲稱跨機或跨容器。關閉 `FB_APP_SHADOW_MODE` 並重啟候選副本即停用，已有研究檔保持不動。停機期間未發現的賽事不能事後補成前瞻。持久性、備份、保存期及峰值容量需在正式試點另審。

匯出已存在的研究檔，不覆蓋輸出 CSV：

```bash
python fb_app_t30_shadow.py /absolute/path/to/private-shadow-dir t30-research.csv
```

## 實時驗收尚有一個資料閘

未核實 [現有 Tipsme／HKJC 讀取程式](https://raw.githubusercontent.com/tipsme98/FB-app/1a5b243f5e052217c56cda22fb61ade6d74d2126/FB-app.py) 所取的回應，是否有權威「目前盤」物件和能證明開放／封盤的欄位。故不提供猜測的正式 `FB_APP_SHADOW_PROFILE`。未設合約時，已發現賽事會記 `source_unverified`，不發額外賠率請求、不輸出預測。若實際只提供歷史清單而無狀態，該來源不能作可執行報價，需另核對權威盤口來源。

人工核對合約時需比對：`matchStatus` 未開賽值、`isHkjc` 真布林、盤口物件路徑 `path`、其 `status`、`timestamp`、`line`、`upper`、`lower`、區分型別的 `open`／`closed` 枚舉、明確時區及主隊讓球符號。合約鍵是 `evidence`、`schedule_prematch`、`handicap_sign`、`naive_zone`（預設拒絕無時區時間）、可選回應 `fixture_id` 對應路徑及三盤 `markets`。`tests/test_t30_shadow.py` 的合成合約只供離線測試，不能用於真實資料；不接受把陣列尾項、非空賠率或舊 `unlock=True` 翻譯為開放。先用一場合資格新賽事核對時序、盤線與開放狀態，再比較額外請求數及原流程回歸。
