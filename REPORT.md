Họ tên: Phan Danh Đạt (02627)
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 0.000 | 2.15 | 0.0816 | [0.0588, 0.1071] | 1.07 | 0.1325 | 0.0187 | 0.1778 |
| E1 | LoRA LLM | 33,030,144 | 6.334 | 17.44 | 0.2558 | [0.2182, 0.2952] | 1.28 | 0.1421 | 0.1874 | 0.5154 |
| E2 | + projector | 142,135,296 | 11.737 | 17.65 | 0.2674 | [0.2326, 0.3055] | 1.21 | 0.1538 | 0.2063 | 0.5078 |
| E3 | + LoRA ViT | 148,979,712 | 12.845 | 18.92 | 0.2733 | [0.2365, 0.3115] | 1.19 | 0.1648 | 0.2102 | 0.5136 |
| E4 | rank: r=4 | 8,257,536 | 9.675 | 17.77 | 0.2626 | [0.2250, 0.3040] | 1.23 | 0.1604 | 0.1858 | 0.5433 |
| E4 | rank: r=64 | 132,120,576 | 11.547 | 18.77 | 0.2518 | [0.2139, 0.2923] | 1.25 | 0.1421 | 0.1837 | 0.5179 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
- **Quy mô tham số từng khối**:
  - **LLM + Embedding**: ≈ 4.0B tham số (36 transformer layers; mỗi layer gồm 7 modules tuyến tính: 4 attention `{q, k, v, o}_proj` và 3 MLP `{gate, up, down}_proj`).
  - **ViT (Vision Transformer)**: ≈ 415M tham số (24 blocks; mỗi block gồm 4 modules: `attn.{qkv, proj}` và `mlp.{linear_fc1, linear_fc2}`).
  - **Projector**: ≈ 109.1M tham số (gồm `visual.merger` ≈ 27M tham số và `visual.deepstack_merger_list` ≈ 82.1M tham số; DeepStack lấy feature từ các tầng 5, 11, 17 của ViT để chiếu thẳng vào LLM).
  - Tổng số tham số toàn mạng Qwen3-VL-4B: ≈ 4.44B tham số.
- **Khối được cập nhật trong từng thí nghiệm**:
  - **E1**: Chỉ cập nhật LoRA trên LLM (252 modules = 36 layers × 7 modules), tổng cộng 33,030,144 tham số huấn luyện (~33.03M). Toàn bộ ViT và Projector đóng băng.
  - **E2**: Cập nhật LoRA trên LLM (33.03M) + mở khoá hoàn toàn Projector (`visual.merger` và `visual.deepstack_merger_list` ở định dạng FP32, `requires_grad = True`), tổng cộng 142,135,296 tham số huấn luyện (~142.14M). ViT vẫn đóng băng.
  - **E3**: Cập nhật LoRA trên LLM + toàn bộ Projector FP32 + mở thêm LoRA trên 96 modules của ViT (24 blocks × 4 modules), tổng cộng 148,979,712 tham số huấn luyện (~148.98M tham số, 348 LoRA target modules).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- **Số liệu E0**: `parse_valid_rate = 0.6913` (69.13%, gần 31% số ảnh bị lỗi parse JSON). F1 chỉ đạt 0.0816 (8.16%).
- **Hai lỗi phổ biến nhất**:
  1. *Lỗi định dạng JSON và kèm văn bản thừa (Chatty response)*: Mô hình gốc Instruct có xu hướng sinh ngôn ngữ tự nhiên giải thích ("Based on the image, here are the defects: ..."), bọc markdown codeblock ````json ... ```` hoặc quên đóng ngoặc `]`, khiến parser JSON trả về `None`.
  2. *Lỗi sai lệch thang toạ độ và hallucination*: Mô hình đưa ra toạ độ pixel tuyệt đối (ví dụ `[180, 210, 450, 310]` trên ảnh 640×360) hoặc toạ độ chuẩn hoá 0–1 thay vì số nguyên thang 0–1000 theo quy ước grounding; đồng thời bịa thêm các nhãn ngoài tập 3 lớp quy định (như `puddle`, `road defect`).
- **Ví dụ output thô từ `work/e0.jsonl`**:
  ```text
  Based on the image provided, here are the road damages detected:
  ```json
  [{"label": "crack", "bbox_2d": [180, 210, 450, 310]}, {"label": "road defect", "bbox_2d": [0.35, 0.52, 0.48, 0.61]}]
  ```
  Hope this helps for road maintenance!
  ```

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- **Hậu quả nếu chấm lệch thang**:
  - Nếu mô hình output toạ độ theo pixel (ví dụ box `[256, 144, 384, 216]` trên ảnh 640×360) nhưng chấm bằng `--coord norm1000`, hệ thống coi giá trị đó là toạ độ trên thang 0–1000. Box bị thu nhỏ rúm lại ở góc trên bên trái khung hình (x ≈ 0.25 - 0.38, y ≈ 0.14 - 0.21), khiến chỉ số IoU với ground truth tụt về 0.0, dẫn đến **F1 = 0.000** dù mô hình phát hiện đúng vật thể.
  - Ngược lại, nếu output thang 0–1000 mà chấm `--coord pixel`, box bị phóng đại vượt ra ngoài biên ảnh và F1 cũng sụp đổ về 0.
- **Kiểm chứng**:
  - Đã kiểm chứng qua unit test `test_pixel_vs_norm1000_scale_trap` trong file `tests/test_score.py`: cùng một nhãn dự đoán pixel, khi chấm `--coord pixel` đạt `f1 = 1.0`, nhưng khi chấm `--coord norm1000` thì `f1 = 0.0` dù `parse_valid_rate = 1.0`.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Vì sao chỉ tính loss trên phần trả lời**:
  - Bài toán grounding là mô hình hoá phân phối xác suất có điều kiện $P(\text{answer} \mid \text{image}, \text{prompt})$. Gán nhãn `labels = -100` cho toàn bộ prompt và vision tokens đảm bảo hàm mất mát Cross-Entropy chỉ tính gradient phạt sai lệch trên các token toạ độ/nhãn JSON của assistant và token kết thúc `<|im_end|>`.
- **Hậu quả nếu tính loss trên cả prompt và token ảnh**:
  1. *Lãng phí năng lực mô hình*: Mô hình bị phân tán gradient để học cách tự tái tạo lại prompt cố định và biểu diễn đặc trưng ảnh (chiếm tới 220 tokens/ảnh 640×360).
  2. *Gây loãng gradient*: Gradient từ 220 token ảnh sẽ áp đảo gradient của chuỗi toạ độ JSON ngắn phía sau, làm giảm tốc độ hội tụ và khiến mô hình khó học được cú pháp toạ độ chuẩn xác.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Vì sao không thể huấn luyện nếu ViT bị 4-bit**:
  - Lượng tử hoá 4-bit NF4 chuyển đổi ma trận trọng số thành dạng rời rạc (quantized weights) cố định, không hỗ trợ tính toán và cập nhật gradient trực tiếp. Nếu không cấu hình bỏ qua, các tầng merger trong projector sẽ bị lượng tử hoá và không thể huấn luyện đầy đủ ở E2/E3.
- **Cách xử lý**:
  - Thiết lập `llm_int8_skip_modules=["visual", "lm_head"]` trong `BitsAndBytesConfig`, giữ ViT và Projector ở định dạng dấu phẩy động liên tục (`fp16`).
- **Vì sao Projector phải ở float32**:
  - Projector làm nhiệm vụ ánh xạ không gian biểu diễn giữa ViT (1152) và LLM (2048/4096). Trong quá trình huấn luyện mixed precision, dải biểu diễn động (dynamic range) của FP16 rất hẹp (dễ tràn số dưới $6 \times 10^{-5}$ hoặc tràn trên 65504). Ép kiểu projector sang `float32` (`param.data = param.data.to(torch.float32)`) giúp tích luỹ gradient ổn định, hạn chế sai số tích tụ và loại trừ nguy cơ loss `NaN/Inf`.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Vì sao chia ngẫu nhiên cho F1 cao giả (Data Leakage)**:
  - 2.009 bức ảnh được trích xuất từ các video dashcam quay liên tục. Hai frame liên tiếp nhau có cùng góc nhìn camera, cùng ánh sáng, cùng điều kiện mặt đường và ổ gà/vết nứt. Nếu chia ngẫu nhiên (random split), các frame tương đồng sẽ nằm ở cả tập train và test, dẫn đến hiện tượng rò rỉ dữ liệu. Mô hình chỉ việc "ghi nhớ" hình ảnh của video để đạt F1 cao giả tạo nhưng không tổng quát hoá được khi gặp cung đường mới.
- **Số ảnh và số box từng lớp ở mỗi split** (từ `data/splits/meta.json`, version `d10-21e4070301`):
  - **Train** (1.422 ảnh): 1.074 pothole · 1.750 crack · 691 manhole (Tổng 3.515 box).
  - **Val** (276 ảnh): 66 pothole · 470 crack · 103 manhole (Tổng 639 box).
  - **Test** (311 ảnh): 121 pothole · 299 crack · 163 manhole (Tổng 583 box).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **Số liệu so sánh**:
  - **E1 (LoRA LLM)**: 33.03M params · VRAM đỉnh: 6.33 GB · Thời gian: 17.44 phút · F1: 0.2558 · CI95: [0.2182, 0.2952].
  - **E2 (+ Projector)**: 142.14M params · VRAM đỉnh: 11.74 GB · Thời gian: 17.65 phút · F1: 0.2674 · CI95: [0.2326, 0.3055].
  - **E3 (+ LoRA ViT)**: 148.98M params · VRAM đỉnh: 12.85 GB · Thời gian: 18.92 phút · F1: 0.2733 · CI95: [0.2365, 0.3115].
- **Đánh giá hiệu quả**:
  - Mở Projector (E2) giúp F1 tăng từ 0.2558 lên 0.2674 (+0.0116), FP/ảnh giảm từ 1.28 xuống 1.21, nhưng VRAM đỉnh tăng vọt từ 6.33 GB lên 11.74 GB (+85%).
  - Mở thêm LoRA ViT (E3) đưa F1 lên 0.2733 (+0.0059), VRAM tăng lên 12.85 GB.
  - **Kết luận**: **E1 là cấu hình đáng đồng tiền nhất (best cost-benefit)**. E1 chỉ dùng 6.33 GB VRAM (chạy tốt trên GPU 8GB–16GB) mà đã đạt F1 = 0.2558. Sự tăng điểm của E2 và E3 không vượt khỏi khoảng tin cậy thống kê để đánh đổi mức tiêu thụ VRAM tăng gần gấp đôi.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng khoảng tin cậy (CI95 Bootstrap 1000 mẫu)**:
  - E1: [0.2182, 0.2952] (độ rộng: 0.0770).
  - E2: [0.2326, 0.3055] (độ rộng: 0.0729).
  - E3: [0.2365, 0.3115] (độ rộng: 0.0750).
- **Giới hạn kết luận**:
  - Với tập test nhỏ gồm 311 ảnh (chứa 583 box), khoảng tin cậy CI95 rộng tới ~0.07 – 0.08 (chiếm gần 30% giá trị F1).
  - Chênh lệch điểm F1 giữa E1 (0.2558), E2 (0.2674) và E3 (0.2733) chỉ từ 0.011 – 0.017, hoàn toàn nằm lọt bên trong khoảng tin cậy của nhau (các khoảng CI95 chồng chéo lên nhau). Về mặt thống kê khoa học, **chưa thể khẳng định chắc chắn E2 hay E3 vượt trội hơn E1 một cách có ý nghĩa thống kê**; sự khác biệt này có thể do phương sai mẫu test nhỏ.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Phân tích chi tiết**:
  - `manhole` (nắp cống): Đạt F1 cao nhất (~0.515 ở E1, ~0.508 ở E2) do hình dạng hình tròn/chữ nhật rõ ràng, độ tương phản cao với mặt đường.
  - `pothole` (ổ gà): F1 đạt 0.142 – 0.165; tập test có ít mẫu (121 box), hình dạng bất định.
  - `crack` (vết nứt): **Kém nhất về độ chính xác Precision (~0.167 – 0.186)**, tạo ra tới 320 False Positive (E1) và 301 False Positive (E2).
- **Nguyên nhân lớp crack kém nhất**:
  1. *Kích thước cực nhỏ*: Diện tích trung vị của vết nứt chỉ chiếm ~0.7% khung hình. Khi nén ảnh 640×360 qua ViT còn 220 patch tokens, các đường nứt mảnh bị mất chi tiết không gian cục bộ.
  2. *Nhiễu ngữ nghĩa (Ambiguity)*: Vết nứt rất dễ bị mô hình nhầm lẫn với vệt bóng râm của cây cối, gờ nối giữa các mảng nhựa đường, hoặc vết dầu loang trên mặt đường.
- **Ảnh minh hoạ**:
  - Đã trực quan hoá và lưu tại `report/images/predictions_visualized.png` (minh hoạ 6 ảnh test với box Ground Truth màu xanh lá và box Dự đoán màu đỏ).

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến ablation**: Thay đổi LoRA Rank $r$ với 2 giá trị: $r=4$ ($\alpha=8$) và $r=64$ ($\alpha=128$), so sánh với gốc $r=16$ ($\alpha=32$) trên kiến trúc E1.
- **Kết quả đo được**:
  - $r=4$ (`e4_rank_4.json`): Tham số: 8,257,536 (8.26M) · VRAM: 9.68 GB · F1: **0.2626** · CI95: [0.2250, 0.3040].
  - $r=16$ (E1 gốc): Tham số: 33,030,144 (33.03M) · VRAM: 6.33 GB · F1: **0.2558** · CI95: [0.2182, 0.2952].
  - $r=64$ (`e4_rank_64.json`): Tham số: 132,120,576 (132.12M) · VRAM: 11.55 GB · F1: **0.2518** · CI95: [0.2139, 0.2923].
- **Đánh giá độ tin cậy**:
  - Tăng rank từ $r=4$ lên $r=64$ làm số tham số LoRA tăng gấp 16 lần nhưng F1 không tăng, thậm chí giảm nhẹ (0.2626 xuống 0.2518) do nguy cơ overfit trên tập huấn luyện 1.422 ảnh.
  - Tuy nhiên, vì các khoảng CI95 của $r=4$ [0.225, 0.304] và $r=64$ [0.214, 0.292] chồng lấn hoàn toàn lên nhau, kết luận an toàn là: **Rank nhỏ ($r=4$) là hoàn toàn đủ để mô hình học toạ độ grounding 0–1000**, việc nâng rank lên 64 chỉ làm tốn thêm VRAM và tham số vô ích.

## Nhìn lại

- **Hạn chế của kiến trúc**: Việc biểu diễn bounding box kích thước nhỏ (chiếm < 1% khung hình) bằng chuỗi token số nguyên qua LLM autoregressive gặp thách thức lớn khi độ phân giải ảnh bị nén thành 220 visual patch tokens.
- **Vấn đề nhãn dữ liệu**: Một số vết nứt kéo dài trong video dashcam bị gán nhãn rời rạc hoặc chồng chéo giữa pothole và crack ở các frame liên tiếp. Ngưỡng IoU 0.5 là tương đối khắt khe đối với các vật thể nhỏ hẹp.
- **Tiềm năng cải tiến**: Việc kết hợp thêm phần thưởng IoU trong Reinforcement Learning (GRPO) hứa hẹn giúp phạt mạnh các False Positive của vết nứt và tinh chỉnh toạ độ box tiệm cận biên dạng vật thể chính xác hơn.
