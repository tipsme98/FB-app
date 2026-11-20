# V16.4.11｜自動批次與手動表單分離

2026-11-20 香港時間。以 [V16.4.10 發布提交](https://github.com/tipsme98/FB-app/commit/d9f4b8a50822d33701d2847f99421360b38994b0) 的 `FB-app.py` 為基礎。此版無須 SQL 遷移；未執行正式驗收或更動正式設定。

## 事故紀錄

17:37:47 開始的 CLV 批次三場皆填充成功，然後在 `ANALYZE` 被比賽 ID／主客隊／香港賽日的合併檢查拒絕，最終 0 注、3 場待跟進。批次確實運行，與較早「主裝置卻只觀看」的過期狀態誤判不同（[批次最後公開紀錄](https://raw.githubusercontent.com/tipsme98/FB-app/7adca1476abd9cfa12dc5efdf61ec92f5fbc7878/auto_status.csv)）。

三場是 18:00 的 [鹿島鹿角 vs 清水心跳（116101）](https://tipsme.hk/zh-HK/match/116101/chat)、[柏雷素爾 vs 千葉市原（116099）](https://tipsme.hk/zh-HK/match/116099/chat) 和 [神戶勝利船 vs 水戶蜀葵（116100）](https://tipsme.hk/zh-HK/match/116100/chat)。離線在 FILL 與 ANALYZE 之間注入過期手動欄位，可重現 V16.4.10 的拒絕；公開批次檔未保存當時哪項 widget 值觸發。

## 維護及驗收

- **批次階段：** FILL 保存 `(批次 SID, 佇列位置, Tipsme ID)` 綁定資料；ANALYZE、BET 不讀手動 widget。BET 重驗分析範圍及盤口；缺快照、賽事／賽日不符、過近開賽、正式庫防重查詢失敗仍不寫入。BET 未獲主儲存確認須保留分頁、匯出報表並查正式庫，勿重送。
- **兼容：** 自動注單的 ID 保持 `B` 加數字，避免同秒撞號。結算把 `Half_Score` 及 `Result_Label` 當文字處理，避免全空欄被 pandas 判為數字。沒有更改結算公式或既有注單。批次完成但有失敗會顯示待跟進；主分頁及心跳診斷沿用既有機制。05:00 為[事故當時的設定](https://raw.githubusercontent.com/tipsme98/FB-app/7adca1476abd9cfa12dc5efdf61ec92f5fbc7878/football_schedule_db.csv)，只更正對硬編碼 09:30 的描述。
- **離線回歸：** 18 項全過。三場重播各寫入一筆，賽事 ID、隊伍、香港賽日和賠率逐場相符，沒有登記用家注碼。另測失效快照、錯場及盤口與分析不符、正式庫防重讀取失敗、重複注單、未確認儲存、分頁狀態、手動填充及編輯，以及歷史日期及輪次建檔與結算。
- **上線後：** 核對 Streamlit 側欄版本及一個指定主分頁的近期心跳，再唯讀觀察下一場合資格未開賽賽事的 FILL → ANALYZE → BET 和報表。不補錄上述三場的賽前注單；待正常排程後再查是否有合資格收盤價。排程和 runner 設定不用改。
- **回退：** 以新提交只恢復 [V16.4.10 的 `FB-app.py`](https://github.com/tipsme98/FB-app/blob/d9f4b8a50822d33701d2847f99421360b38994b0/FB-app.py)，不重設 `main`，不還原資料 CSV，不改正式紀錄。BET 是 FB-app 帳簿寫入，不會向馬會提交交易。

<hr>

<div class="kicker">維護及事故交接　／　2026-11-20 香港時間</div>

# FB-app 項目交接手冊

<div class="subtitle">V16.4.10｜賽前自動入庫與主裝置進度</div>

> **狀態：本地修復及離線驗證完成，未部署，未改正式 Supabase、設定或排程。** 用家確認未部署 024 的 V16.4.9；已核對事件時段的公開主檔與上傳 V16.4.8 相同。修復以 024 候選版本為基底，保留獨立的唯讀參數研究。<sup>1</sup>

## 事件：兩個訊號不是同一件事

**16:28 主裝置提示。** 16:28:17 截圖顯示主分頁心跳為 16:27:58，但共享進度仍引用 16:03:05 啟動的 CLV 批次：0/2，只有啟動記錄。舊版先顯示「另一裝置正在自動輸入／本裝置只觀看」，再檢查已逾 5 分鐘；橫幅用了過期批次，不是本分頁被降級。心跳證明當時有更新，不代表前一個批次仍在推進。16:28:16 又有一筆新的 0/3 啟動記錄；公開 CSV 不列各場 ID，也不保證每階段都寫出，不能據此判定當刻佇列、故障步驟或是哪一個分頁。<sup>2,3</sup>

**16:31 悉尼聯對薩瑟蘭鯊魚。** 截圖顯示 32 筆核心賠率及排名，但「比賽日期」空白，畫面明示只是預覽。空日期會阻止分析，並非近況留空所致。Tipsme 現有頁面確認該場 ID **116097**、排定香港時間 16:30；不能反推 16:31 時舊版收到哪個時間欄位，也不能把此場直接對應到 16:03 或 16:28 的批次。已開賽不補 T−30。<sup>4</sup>

**附件報表核對。** `Sheet1` 按 `Tipsme_Match_ID` 和隊名交叉查找：未見 116097；同時段南墨爾本對坎培拉克羅地亞（116096）在 16:28 有三筆 `Open`，系統及用家注碼都是 $0。記錄 ID 為 `B202611201628530`、`B202611201628531`、`B202611201628532`。因此不是所有賽事都沒記錄；`Open` 可同時是 $0 模擬，報表只是快照，不等於正式庫現況。「BET」是 FB-app 帳簿寫入，不會往馬會提交交易。<sup>5</sup>

## 候選版本改了甚麼

- **日期及流程：** 解析 Tipsme UTC／香港時間，包括香港欄的月／日／年格式；明細缺時間時用同 ID 的已載入賽程／今日賽程。賽日矛盾不猜；填充後核對比賽、日期和表單，才可分析與 BET。缺日期或核心盤口會列出原因；開賽前不足 1 分鐘或已開賽不補錄。
- **狀態及防重：** 批次與分頁採獨立 UUID；區分近期活動、過期及未知狀態。逐步進度原子寫入本地共享 CSV，不在每一步額外推送 GitHub。修正 SQL 刷新後用 `DataFrame or ...` 觸發的布林例外，讓防重檢查實際使用更新後的資料。主裝置心跳及 12 分鐘備援不變；5 分鐘狀態判斷不是跨容器鎖。

<div class="page-sources">
<p>1. 事件時段公開版本：<a href="https://raw.githubusercontent.com/tipsme98/FB-app/8acd8ed7302a9963eadf6c2342a54f5421479bed/FB-app.py">https://raw.githubusercontent.com/tipsme98/FB-app/8acd8ed7302a9963eadf6c2342a54f5421479bed/FB-app.py</a></p>
<p>2. 16:03 公開狀態：<a href="https://raw.githubusercontent.com/tipsme98/FB-app/a2d1c800068de83da72cd80ea8bbd0fd98f0280f/auto_status.csv">https://raw.githubusercontent.com/tipsme98/FB-app/a2d1c800068de83da72cd80ea8bbd0fd98f0280f/auto_status.csv</a></p>
<p>3. 16:28 公開狀態：<a href="https://raw.githubusercontent.com/tipsme98/FB-app/782d5481436874b98bfb11182a190018fc759fef/auto_status.csv">https://raw.githubusercontent.com/tipsme98/FB-app/782d5481436874b98bfb11182a190018fc759fef/auto_status.csv</a></p>
<p>4. 悉尼聯賽事明細：<a href="https://tipsme.hk/zh-HK/match/116097/chat">https://tipsme.hk/zh-HK/match/116097/chat</a></p>
<p>5. 南墨爾本賽事明細：<a href="https://tipsme.hk/zh-HK/match/116096/chat">https://tipsme.hk/zh-HK/match/116096/chat</a></p>
</div>

<div class="page-cut"></div>

<div class="kicker">維護及事故交接　／　V16.4.10</div>

# 上線、驗收及復原

<div class="subtitle">交給下一位維護者的操作清單</div>

## 未確認寫入：先核對，不要重送

自動 BET 只有獲主儲存確認才計入寫入數。若回應失敗或不明，該批次停止，同一分頁暫停自動化和後續主注單寫入。保持原頁，下載完整報表，按 `ID`、`Tipsme_Match_ID`、盤口、方向及盤線唯讀核對正式庫；未判定狀態前不要按「合併同步」、重新提交、切換 runner 或重新整理。確認沒有未處理差異並保留備份後，才重新開啟一個 runner 分頁。這是分頁內保護，不是跨裝置或重啟後的持久鎖；有正式 SQL 時也不能把本地備份視為成功。

## 部署前後

1. **備份及部署：** 記下現行版本和設定，備份已部署主檔及現有報表；不要變更 Secrets、資料表、09:30 排程、模型或風控。由用家按部署流程將 `FB-app_V16.4.10.py` 放為 `FB-app.py`，並把同包的 `fb_app_parameter_lab.py` 放同目錄；沿用現有 `requirements.txt`。V16.4.9 不用先裝，不要讓兩版本同時執行批次。
2. **確認角色：** 側欄應顯示 `V16.4.10`；只開一個帶 `?runner=1` 的主分頁，核對香港時間心跳及原有自動設定。另一分頁不帶 `runner=1`；主分頁在線時，只應唯讀。遇到過期 CLV 狀態，由有權限分頁「重新檢查未開賽賽事（不補注）」，不是重跑歷史輪次。背景檢查依賴 Streamlit 分頁工作階段，並非外部排程服務。<sup>6</sup>
3. **驗收下一場合資格賽事：** 記下 ID、香港開賽時間、實際觸發時間及盤口是否可用。觀察 `FILL → ANALYZE → BET` 或清晰的略過原因。以正式庫唯讀查該場 `Open` 注單、賽日、`Initial_Odds` 和同場同盤口／方向／盤線有無重複；核對 `$0` 模擬與非零系統帳面注碼，不要求一定要有非零注碼。之後若出現新同線收盤價，再核對 `Closing_Odds`、CLV 及翌晨結算。不為驗收重送 11 月 20 日已開賽場次。
4. **回退及追查：** 若版本異常，先停自動化並保留報表、批次記錄及 Streamlit 錯誤時間；再還原部署前主檔／提交。回退程式不回退注單，不要清空或用舊 CSV 覆蓋正式庫。如要還原事件因果，補 16:03–16:31 HKT 的 Streamlit 伺服器日誌（隱去密碼及連線字串），對照場次 ID、階段和錯誤。

## 驗證範圍及未結項目

本地 Streamlit AppTest 模擬 Tipsme／儲存，10 項測試通過：覆蓋舊版兩個症狀、UTC 跨日及香港時間、手動預覽不入庫、自動三階段／防重、日期矛盾、缺盤、臨開賽，以及儲存未確認或拋錯後不繼續。未在正式 Supabase、正式多容器或下一場 CLV 驗收。11 月 19 日歷史入庫核對毋須重做；Egress 首份 baseline 已執行，仍待約 24 小時後的第二份 CSV，不能把本事件歸因於用量超額。後面保留 024 及較早版本的原頁和兩份唯讀 SQL；部署說明以本次為準。

<div class="page-sources">
<p>6. Streamlit fragments 官方文件：<a href="https://docs.streamlit.io/develop/concepts/architecture/fragments">https://docs.streamlit.io/develop/concepts/architecture/fragments</a></p>
</div>
