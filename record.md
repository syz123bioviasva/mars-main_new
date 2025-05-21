## Experiment Record

### Configuration: full
- **Epochs Trained:** 10
- **Loss Weights:** (7.5, 0.5, 1.5) (Original)
- **Resulting mAP (on test set):** 0.051
[2025-05-17T19:40:33.828391|base.py:162] Epoch 10/10: Train Loss = 5.718, Validation Loss = 5.006
---
- **Loss Weights:** (8.0, 1.5, 1.5)
- **Resulting mAP (on test set):** 0.077
[2025-05-17T20:08:33.286367|base.py:162] Epoch 10/10: Train Loss = 10.650, Validation Loss = 9.316 边界框权重从 8.0 稍微降低，同时保持分类和 DFL 权重不变
---
- **Loss Weights:** (7.0, 1.5, 1.5)
- **Resulting mAP (on test set):** 0.074
[2025-05-17T20:36:32.061331|base.py:162] Epoch 10/10: Train Loss = 10.412, Validation Loss = 9.132

---
- **Loss Weights:** (8.0, 1.5, 1.5)
- **Base Learning Rate:** 1e-3
- **Resulting mAP (on test set):** 0.027
[2025-05-17T20:58:21.804779|base.py:162] Epoch 10/10: Train Loss = 13.554, Validation Loss = 11.798

---
- **Loss Weights:** (7.0, 1.5, 1.5)
- **Base Learning Rate:** 5e-3
- **Resulting mAP (on test set):** 0.068
[2025-05-17T21:22:38.910350|base.py:162] Epoch 10/10: Train Loss = 10.622, Validation Loss = 9.312

---
### Configuration: full
- **Epochs Trained:** 200
- **Loss Weights:** (8.0, 1.5, 1.5)
- **Base Learning Rate:** 5e-3
- **Backbone Freeze Epochs:** First 100 epochs
- **Resulting mAP (on test set):** 0.782

---
### Configuration: full (Data Augmentation Experiment 1)
- **Epochs Trained:** 10
- **Loss Weights:** (8.0, 1.5, 1.5)
- **Base Learning Rate:** 5e-3
- **Backbone Freeze Epochs:** First 5 epochs
- **Data Augmentation (dl/vocdataset.py):**
    - `rescalef`: (0.4, 1.8)
    - `huef`: 0.2
    - `satf`: 0.8
    - `valf`: 0.5
- **Resulting mAP (on test set):** 0.073
[2025-05-17T22:34:01.288001|base.py:162] Epoch 10/10: Train Loss = 10.742, Validation Loss = 9.382

---
### Configuration: full (Data Augmentation Experiment 2 - Mosaic)
- **Epochs Trained:** 10
- **Loss Weights:** (8.0, 1.5, 1.5)
- **Base Learning Rate:** 5e-3
- **Backbone Freeze Epochs:** First 5 epochs
- **Data Augmentation (dl/aug.py):**
    - `rescalef`: (0.4, 1.8)
    - `huef`: 0.2
    - `satf`: 0.8
    - `valf`: 0.5
    - `mosaicProb`: 0.5 (新增 Mosaic 增强)
- **Resulting mAP (on test set):** 0.054
[2025-05-20T00:03:56.770403|base.py:162] Epoch 10/10: Train Loss = 13.417, Validation Loss = 10.242

性能下降原因分析：
1. 训练轮数不足：Mosaic 增强增加了数据复杂度，需要更多轮次才能收敛
2. 边界框处理问题：Mosaic 增强可能导致边界框坐标计算不准确
3. 增强强度过大：同时使用 Mosaic 和其他增强可能导致数据变化过大
4. 类别不平衡：从 AP 结果看，只有 A1、A14、A16、A19 有检测结果，其他类别完全失效

建议改进方向：
1. 降低 Mosaic 增强概率（从 0.5 降至 0.3）
2. 增加训练轮数（至少 50 轮）
3. 调整 Mosaic 增强参数（减小拼接区域的变化范围）
4. 考虑分阶段训练：先不使用 Mosaic 训练几轮，再逐步引入

// ... existing code ...

---
### Configuration: full (Data Augmentation Experiment 2 - Mosaic)
- **Epochs Trained:** 200
- **Loss Weights:** (8.0, 1.5, 1.5)
- **Base Learning Rate:** 5e-3
- **Backbone Freeze Epochs:** First 100 epochs
- **Data Augmentation (dl/aug.py):**
    - `rescalef`: (0.4, 1.8)
    - `huef`: 0.2
    - `satf`: 0.8
    - `valf`: 0.5
    - `mosaicProb`: 0.5 (新增 Mosaic 增强)
- **Resulting mAP (on test set):** 0.435
[2025-05-21T09:45:25.625657|engine.py:36] mAP=0.435

各类别 AP 结果：
- A1: 0.499679
- A2: 0.654335
- A3: 0.432334
- A4: 0.531271
- A5: 0.316237
- A6: 0.259854
- A7: 0.443880
- A8: 0.475563
- A9: 0.339202
- A10: 0.599208
- A11: 0.458553
- A12: 0.241467
- A13: 0.554335
- A14: 0.507797
- A15: 0.189459
- A16: 0.717221
- A17: 0.466241
- A18: 0.151531
- A19: 0.511835
- A20: 0.353664