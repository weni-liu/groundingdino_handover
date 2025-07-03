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
