# gdsam說明
gdsam.py
這段程式碼是一個完整的**AI自動標註系統**，結合了 **GroundingDINO** 和 **SAM (Segment Anything Model)** 來進行物件檢測和精確分割。讓我詳細解析：

## 主要功能

這是一個**雙階段自動標註系統**：
1. **第一階段**：使用 GroundingDINO 進行物件檢測，生成邊界框
2. **第二階段**：使用 SAM 進行精確分割，生成像素級別的遮罩

## 系統架構

````python
# 模型初始化
model = load_model(args.GD_cinfig_py, args.GD_checkpoints, device=device)  # GroundingDINO
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)  # SAM
predictor = SamPredictor(sam)
````

## 核心處理流程

### 1. 物件檢測階段 (GroundingDINO)
```python
# 使用文字提示進行物件檢測
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,  # 例如：'defect'
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD)

# 座標轉換 + NMS去重
bounding_boxes = bbox_convert(boxes, image_source)
nms_output = non_maximum_suppression_fast(bounding_boxes, overlapThresh=0.3)
```

### 2. 背景過濾
```python
# 使用模板匹配去除背景誤檢
tmp_image = m_image.copy()
for box in nms_output:
    # 將檢測區域用白色遮罩覆蓋
    mask = np.ones_like(tmp) * 255
    tmp_image[int(box_int[1]):int(box_int[3]), int(box_int[0]):int(box_int[2])] = mask

# 檢查是否為背景（相似度>0.97視為背景）
out_match = cv2.matchTemplate(tmp_image, tmp, cv2.TM_CCOEFF_NORMED)
if len(np.where(out_match>0.97)[0]) == 0:
    # 不是背景，保留此檢測結果
```

### 3. 精確分割階段 (SAM)
```python
predictor.set_image(image)
for obj in root.findall('object'):
    # 提取邊界框座標
    box = [xmin, ymin, xmax, ymax]
    point = np.array([[int((xmin+xmax)/2), int((ymin+ymax)/2)]])  # 中心點
    
    # SAM分割
    masks, a, _ = predictor.predict(
        point_coords=point,
        point_labels=[1],
        box=np.array(box),
        multimask_output=False)
    
    # 品質檢查（IOU > 0.8）
    if a[0] > 0.8:
        # 生成輪廓用於JSON標註
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

## 輸出格式

### XML 標註 (Pascal VOC格式)
```xml
<annotation>
    <object>
        <name>defect</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>200</ymin>
            <xmax>300</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
</annotation>
```

### JSON 標註 (LabelMe格式)
```json
{
    "shapes": [{
        "label": "defect",
        "points": [[x1,y1], [x2,y2], ...],  // 精確輪廓點
        "shape_type": "polygon"
    }]
}
```

## 關鍵改進點

| 功能 | 說明 |
|------|------|
| **雙模型結合** | GroundingDINO檢測 + SAM精確分割 |
| **品質控制** | IOU閾值(0.8)過濾低品質分割 |
| **背景過濾** | 模板匹配去除背景誤檢 |
| **多格式輸出** | 同時生成XML和JSON標註 |
| **錯誤處理** | 完整的異常捕獲和日誌記錄 |

## 應用場景

這個系統特別適合：
- **工業缺陷檢測**：自動標註產品缺陷
- **醫學影像分析**：病灶區域精確分割
- **自動駕駛**：道路物件檢測與分割
- **遙感影像**：地物目標識別與輪廓提取

相比單純的檢測系統，這個版本提供了**像素級別的精確標註**，大幅提升了標註品質和實用性。

您說得對！讓我重新完整說明這個系統的**資料載入**和**輸出資料**部分：

## 資料載入部分

### 1. 輸入監控機制
````python
# 持續監控指定資料夾
Input = args.input_folder  # 預設: 'D:\\Uploads\\Autolabeling_input'
while True:
    time.sleep(0.1)
    paths = glob(Input+'\\*.txt')  # 監控 .txt 檔案
    if len(paths) > 0:
        # 有新檔案時開始處理
````

### 2. 文字檔案格式
````python
# 一般處理檔案格式：
# 圖片路徑,標籤名稱,fab,md_id,id
# 例如：d:\uploads\LCD1\4056\Labeling\8338\pic\3350-1T79CV968AM050011.JPG,HDC/R,LCD1,4056,8338

# 結束檔案格式：
# 檔名以 _end.txt 結尾
# 例如：20240718_1048_LCD1_4056_8338_end.txt
````

### 3. 圖像資料載入
````python
# 從文字檔案中解析圖片路徑
img_path = os.path.join(Input, txt.split(',')[0])

# 載入圖像（兩種格式）
image_source, image = load_image(img_path)  # GroundingDINO格式
m_image = cv2.imread(img_path)              # OpenCV格式
````

### 4. 輸入資料夾結構
```
D:\Uploads\Autolabeling_input\
├── control_file.txt        # 控制檔案：包含圖片路徑和標籤資訊
├── image1.jpg              # 待處理圖片
├── image2.png              # 待處理圖片
└── 20240718_1048_LCD1_4056_8338_end.txt  # 結束標記檔案
```

## 輸出資料部分

### 1. 輸出路徑邏輯
````python
# 根據輸入格式決定輸出路徑
if len(txt.split(',')) == 5:
    # 有完整資訊：fab, md_id, id
    fab = txt.split(',')[2]     # 例如：LCD1
    md_id = txt.split(',')[3]   # 例如：4056
    id = txt.split(',')[4]      # 例如：8338
    out_dir = os.path.join(args.output_folder, fab, md_id, 'Labeling', id, 'Label')
    # 輸出到：D:\Uploads\LCD1\4056\Labeling\8338\Label\
else:
    # 簡化格式
    out_dir = os.path.join(args.output_folder, 'Autolabeling_tmp', 'Label')
    # 輸出到：D:\Uploads\Autolabeling_tmp\Label\
````

### 2. XML 標註檔案輸出
````python
def create_xml(objects, filename_text, folder_text, width_text, height_text, save_path, save_xml=True):
    # 生成 Pascal VOC 格式的 XML
    if save_xml:
        xml_path = os.path.join(save_path, filename_text.split('.')[0]+'.xml')
        tree.write(xml_path)
        print(xml_path)  # 輸出檔案路徑
    return tree

# XML 檔案範例：image1.xml
````

### 3. JSON 標註檔案輸出
````python
def write_json(image, filename_element, all_contours, save_path, label_name):
    # 生成 LabelMe 格式的 JSON
    d = {
        "version": "4.2.9",
        "shapes": [],
        "imagePath": filename_element,
        "imageHeight": image.shape[0],
        "imageWidth": image.shape[1]
    }
    # 添加精確輪廓資料
    json_path = os.path.join(save_path, os.path.splitext(filename_element)[0]+'.json')
    with open(json_path, "w") as f:
        json.dump(d, f)

# JSON 檔案範例：image1.json
````

### 4. 結束檔案處理
````python
if txt_path0.split('_')[-1] == 'end.txt':
    # 解析結束檔案名稱
    txt_ = txt_path0.split('\\')[-1]
    fab, md_id, id = txt_.split('_')[2], txt_.split('_')[3], txt_.split('_')[4]
    
    # 移動到結束資料夾
    end_dir = os.path.join(args.output_folder, fab, md_id, 'Labeling', id, 'End')
    endpoint_path = shutil.move(txt_path0, end_dir)
````

## 完整輸出資料夾結構

```
D:\Uploads\
├── LCD1\
│   └── 4056\
│       └── Labeling\
│           └── 8338\
│               ├── Label\              # 標註檔案輸出
│               │   ├── image1.xml      # 邊界框標註
│               │   ├── image1.json     # 精確輪廓標註
│               │   ├── image2.xml
│               │   └── image2.json
│               └── End\                # 處理完成標記
│                   └── 20240718_1048_LCD1_4056_8338_end.txt
├── Autolabeling_tmp\
│   └── Label\                          # 簡化格式輸出
│       ├── temp_image1.xml
│       └── temp_image1.json
└── log\                                # 系統日誌
    └── process_id.log
```

## 日誌輸出
````python
# 設定日誌系統
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=os.path.join(log_path, str(os.getpid())+'.log'),
    filemode='a')

# 記錄處理過程
logger.error('End_file has already exist', txt_path0)
logger.error(e)  # 記錄異常
````

## 資料流程總結

```
輸入 → 監控 → 載入 → 處理 → 輸出
 ↓       ↓      ↓      ↓      ↓
.txt → 解析 → 圖片 → AI → .xml/.json
```

這個系統提供了完整的資料處理鏈路，從檔案監控到最終標註檔案輸出，包含錯誤處理和日誌記錄功能。
