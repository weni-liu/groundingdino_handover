# grounding_api說明 (grounding_api.py)

這段程式是一個自動化物件檢測與標註系統，主要使用 GroundingDINO 模型來檢測圖像中的缺陷並生成標註文件。讓我為您詳細解釋：

## 主要功能

這是一個**自動化標註系統**，可以：
1. 監控指定資料夾等待處理請求
2. 使用 AI 模型檢測圖像中的缺陷
3. 自動生成 XML 標註文件

## 參數設定
 parser.add_argument("--input_folder",default='D:\\Uploads\\Groundingdino_input')
 parser.add_argument("--output_folder",default='D:\\Uploads')
 parser.add_argument("--device", default="cuda:0")

 #parser.add_argument("--SAM_checkpoints", default="weights/sam_pb4.pth")
 parser.add_argument("--GD_checkpoints", default="weights/checkpoint.pth")
 parser.add_argument("--GD_cinfig_py", default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
 parser.add_argument("--TEXT_PROMPT", default='defect')
 parser.add_argument("--BOX_TRESHOLD", type=float, default=0.2)
 parser.add_argument("--TEXT_TRESHOLD",  type=float, default=0.2)
    
## 核心流程

### 1. 模型初始化
```python
# 載入 GroundingDINO 模型用於物件檢測
model = load_model(args.GD_cinfig_py, args.GD_checkpoints, device=device)
```

### 2. 檔案監控循環
```python
while True:
    # 持續監控輸入資料夾中的 .txt 檔案
    paths = glob(Input+'\\*.txt')
```

### 3. 處理邏輯

**文字檔案格式：**
- 一般檔案：`圖片路徑,標籤名稱,fab,md_id,id`
- 結束檔案：`*_end.txt`

**處理步驟：**
1. **圖像檢測**：使用 GroundingDINO 檢測指定物件
2. **邊界框轉換**：將模型輸出轉換為像素座標
3. **非極大值抑制**：去除重疊的檢測框
4. **背景過濾**：使用模板匹配去除背景誤檢
5. **生成標註**：建立 XML 格式的標註檔案

## 關鍵函數說明

| 函數 | 功能 |
|------|------|
| `create_xml()` | 生成包含檢測結果的 XML 標註檔案 |
| `bbox_convert()` | 將相對座標轉換為絕對像素座標 |
| `non_maximum_suppression_fast()` | 去除重疊的檢測框 |
| `cv2.matchTemplate()` | 模板匹配，過濾背景誤檢 |

## 輸出結果

- **有檢測到物件**：生成完整的 XML 標註檔案
- **無檢測到物件**：生成空的 XML 檔案（如果 `n_file=True`）

## 資料夾結構

```
output_folder/
├── fab/
│   └── md_id/
│       └── Labeling/
│           └── id/
│               ├── Label/     # XML 標註檔案
│               └── End/       # 處理完成標記
```

這個系統特別適用於工業檢測場景，可以自動化處理大量圖像並生成標準化的標註資料。
