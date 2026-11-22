# FB-app T−30 RC2：本地 Streamlit 啟動與驗收

本版以 [V18.1 固定提交](https://github.com/tipsme98/FB-app/commit/1a5b243f5e052217c56cda22fb61ade6d74d2126) 為基線，保留 RC1 的三盤時點限制，新增 Streamlit 審計面板。這是可在隔離環境運行的候選版，不是正式上線版本，也不包含實時來源設定、正式數據庫、Secrets 或投注紀錄。

## 啟動

用 Python 3.14，在解壓目錄執行以下 Linux／macOS 指令。不要把正式 `.streamlit/secrets.toml` 或正式 CSV 放入試跑環境。

```sh
APP="$(pwd -P)"
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check

QA="$(mktemp -d)"
mkdir -m 700 "$QA/home" "$QA/run" "$QA/shadow"
(
  cd "$QA/run" || exit
  env -i HOME="$QA/home" PATH="$PATH" LANG=C.UTF-8 TZ=Asia/Hong_Kong \
    PYTHONPATH="$APP" FB_APP_SHADOW_MODE=observe FB_APP_SHADOW_DIR="$QA/shadow" \
    "$APP/.venv/bin/python" -m streamlit run "$APP/FB-app.py" \
    --server.headless=true --server.address=127.0.0.1 --server.port=8501 \
    --browser.gatherUsageStats=false
)
```

- **一般分頁：** `http://127.0.0.1:8501` 不執行 T−30。省略 `FB_APP_SHADOW_MODE=observe` 重啟，即停用所有分頁的 T−30；現有影子紀錄保留。
- **測試主分頁：** `http://127.0.0.1:8501/?runner=1`。頁首展開「T−30 三盤本地影子」，核對賽程時間及採集結果，可下載最近 100 場的 CSV。`runner=1` 只是分頁角色，不是認證。不要將隔離測試主分頁連公開網絡。
- **完整 CSV：** 另開終端執行 `"$APP/.venv/bin/python" "$APP/fb_app_t30_shadow.py" "$QA/shadow" "$QA/all-t30.csv"`，請使用同一個 `APP` 和 `QA` 實際絕對路徑。工具不覆蓋已有輸出；以唯讀方式開啟影子庫。CSV 有 UTF-8 BOM，並將可能視為試算表公式的文字當作純文字輸出。
- **紀錄及時效：** 「略過」「對照訊號」仍保留預測，可在核實賽果後計對照回報；只有「影子模擬」計策略回報。T−30 檢查依賴[仍連線的 Streamlit 分頁](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment)，不是背景排程；資料夾需持久化及獨立備份。同一 SQLite 檔可跨分頁防重，不能據此承諾跨容器。

## 驗收和來源

```sh
.venv/bin/python -m unittest discover -s tests -v
```

- **回放：** `tests/streamlit_replay.py` 只用於測試，畫面標示「離線合成測試」。代用網絡輸入及時鐘，其餘沿用候選版的賽程解析、取樣、鎖定、凍結模型、SQLite 和介面。測試使用的是合成訓練紀錄，不是正式數據或盈利證據。不能部署此入口，也不能使用其來源合約。
- **來源閘：** `FB_APP_SHADOW_PROFILE` 必須是經核實的合約。RC2 要求合約提供回應內的明確 `fixture_id` 路徑，不能用請求 URL 作唯一證據。盤口仍須各自明確開放、雙邊有效，且採集時間與來源報價時間均在 T−45 至 T−30（香港時間）。沒有可靠合約時不向賠率端點發出額外請求、不推論、不回填。
- **已知實時限制：** [Tipsme 公開賠率文件](https://tipsme.hk/zh-HK/developers/docs/odds#getmatchhkjcodds) 的三盤歷史陣列未提供所需的開放狀態；[馬會官方讓球](https://bet.hkjc.com/en/football/hdc)、[入球大小](https://bet.hkjc.com/en/football/hil)、[角球大小](https://bet.hkjc.com/en/football/chl) 可核對開放狀態，但尚未核實到每組當前雙邊報價的來源時間。`foPools.updateAt`、HTTP 取回時間、畫面刷新時間都不作替代。觀察模式與已取得合資格實時預測是兩個不同的驗收結果。
- **回退：** 取消 opt-in，重啟測試服務。RC2 未改影子庫的 schema 或舊紀錄；`version` 欄沿用 `t30-shadow-rc1` 作同一批研究的防重鍵，不用 RC2 重跑已鎖定的舊賽事。舊紀錄的策略回報依原 `建議` 欄重算，不將舊對照訊號計入策略。`T30_REVIEW.md` 是 RC1 歷史審查紀錄，以本文件的差異為準。
