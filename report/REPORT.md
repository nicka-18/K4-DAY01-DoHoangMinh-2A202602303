# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**
11/9/2026

**Runtime Colab:** CPU/GPU
CPU

**Python / PyTorch / Ultralytics:**
Python: 3.13.15
PyTorch: 2.11.0+cpu
Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`
PASS
**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
<<<<<<< HEAD
    "score": 0.510915,
    "taxonomy_name": "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào?
   Ảnh có mã định danh: 
=======
    "score": 0.510915, (51,09%)
    "taxonomy_name": "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào?
   Record này mô tả toàn ảnh (`traffic`) với duy nhất một lớp xếp hạng cao nhất: "cab" (xe taxi) có score 51.09%. Nó không mô tả riêng từng đối tượng trong ảnh mà là xác suất của toàn bộ bức ảnh thuộc lớp nào. Đây là cách tiếp cận của **single-label image classification** - mỗi ảnh nhận một nhãn duy nhất từ model. 
>>>>>>> 6c7f027 (Update REPORT.md)
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
   Danh sách lớp do tổ chức/tác giả sáng lập nên tập dữ liệu ImageNet-1K định nghĩa.
   Checkpoint yolo11n-cls.pt chỉ dự đoán trong phạm vi $1.000$ lớp đã được thiết lập sẵn này.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
<<<<<<< HEAD
  ID (class_id: 468): Mã định danh số chuẩn xác giúp máy tính truy vấn và lập trình không bị lỗi nhầm lẫn hay lệch ký tự.
  Tên lớp (class_name: "cab"): Giúp con người đọc hiểu ngay đối tượng dự đoán là "xe taxi" mà không cần tra cứu bảng mã.Tên taxonomy (taxonomy_name: "ImageNet-1K"): 
  Xác định rõ hệ quy chiếu dữ liệu, giúp phân biệt mã $468$ của ImageNet với mã $468$ của các tập dữ liệu khác (như COCO hay Open Images).
=======
  - **ID (class_id: 468)**: Mã định danh số chuẩn xác giúp máy tính truy vấn và lập trình không bị lỗi nhầm lẫn hay lệch ký tự. Nó là khóa chính để tham chiếu trong database hoặc hệ thống automated pipeline.
  
  - **Tên lớp (class_name: "cab")**: Giúp con người đọc hiểu ngay đối tượng dự đoán là "xe taxi" mà không cần tra cứu bảng mã. Đây là dạng human-readable, quan trọng cho reviewer/QC.
  
  - **Tên taxonomy (taxonomy_name: "ImageNet-1K")**: Xác định rõ hệ quy chiếu dữ liệu, giúp phân biệt mã 468 của ImageNet với mã 468 của các tập dữ liệu khác (như COCO hay Open Images). Khi chuyển sang model hoặc checkpoint khác, class_id có thể thay đổi nhưng taxonomy_name ghi lại nguồn gốc.
>>>>>>> 6c7f027 (Update REPORT.md)
  
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  Guideline cần quy định rõ tiêu chí chọn nhãn duy nhất cho tác vụ image_classification: chọn đối tượng chiếm diện tích lớn nhất (dominant object), đối tượng nằm ở trung tâm bức ảnh, hay chọn theo độ ưu tiên của bài toán kinh doanh (ví dụ: ưu tiên phân loại phương tiện giao thông hơn cảnh quan).
- Vì sao model score không phải ground truth?
Mức score $0.510915$ ($51.09\%$) chỉ đại diện cho độ tự tin bằng xác suất của mô hình YOLOv11 tại thời điểm suy luận. Đây vẫn là một khoảng xác suất khá thấp (gần như $50/50$), mô hình có thể đoán sai do nhiễu góc quay hoặc ảnh chứa nhiều xe khác nhau; nhãn Ground Truth chuẩn xác bắt buộc phải do con người kiểm duyệt và xác nhận.
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
<<<<<<< HEAD
- Diễn giải vị trí box bằng lời:
- So sánh số prediction ở hai threshold:
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
- Đề xuất một quy tắc box chặt:
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
=======
    "class_name": "person",
    "score": 0.912625, 
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
- Diễn giải vị trí box bằng lời:
   Khu vực trong ảnh: Bounding box nằm ở nửa bên phải của bức ảnh (theo trục ngang, từ $385.33$ px đến $498.92$ px trên tổng chiều rộng $640$ px).
   Độ cao chiếm dụng: Khung kéo dài từ gần mép trên bức ảnh ($69.24$ px) xuống đến gần sát phần giữa-dưới của ảnh ($348.92$ px trên tổng chiều cao $427$ px).
   Kích thước & Hình dáng: Đối tượng có hình dáng dọc/đứng (chiều cao $279.68$ px gấp khoảng $2.5$ lần chiều rộng $113.58$ px), tương ứng với dáng đứng hoặc ngồi thẳng của một người (person) ở góc bên phải căn bếp
- So sánh số prediction ở hai threshold:
  prediction: số lượng dự đoán
  Mối quan hệ tỉ lệ nghịch: Ngưỡng score càng thấp thì số lượng dự đoán giữ lại càng nhiều và ngược lại.
  TH threshold thấp (0.25; 0.35)
   -Số lượng: cho ra nhiều dự đoán hơn, tăng độ bao phủ
   -Đặc điểm: Giữ lại đối tượng bị che mờ, bị khuất hoặc có độ tự tin thấp, nhưng chấp nhận rủi ro, tăng số lượng dự đoán sai.
  TH threshold cao ( 0.50; 0.70)
   -Số lượng cho ra ít hơn, tăng độ chính xác
   -Đặc điểm: Chỉ giữ lại những dự đoán mà mô hình cực kỳ tự tin, loại bỏ hầu hết các box có độ bấp bấp 

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  - **Độ bao phủ (recall)**: Threshold thấp → recall cao (bao phủ nhiều object), threshold cao → recall thấp (bỏ sót object).
  - **Khối lượng QC**: Threshold thấp → reviewer phải xem **nhiều false positive** (dự đoán sai), tăng công việc QC. Threshold cao → reviewer xem ít hơn nhưng có nguy cơ **miss object thực sự** (object bị loại nhưng tồn tại).
  - **Trade-off**: Chọn threshold là cân bằng giữa độ phủ (không bỏ sót) và chi phí QC (không quá nhiều sai).

- Đề xuất một quy tắc box chặt:
  **Quy tắc box chặt được đề xuất**: "Bounding box phải sát biên đối tượng; margin trống quanh object ≤ 5% chiều rộng/cao của object. Nếu box chứa ≥ 30% nền/background không liên quan, request rework."
  - Giúp các label phục vụ training model có độ chính xác cao hơn.
  - Dễ phân biệt object từ nền (background).

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  - **Nếu object bị che khuất < 30%**: Annotator vẫn vẽ box bao toàn bộ phần nhìn thấy, không dự đoán phần bị che.
  - **Nếu object bị che khuất 30-70%**: **Escalation** → Reviewer quyết định: vẽ box theo phần nhìn thấy hay bao toàn bộ object dự kiến?
  - **Nếu object bị che khuất > 70% hoặc cắt mép quá**: **Escalation** → Có thể bỏ qua object hoặc ghi nhận với flag "occluded" tùy quy trình.
>>>>>>> 6c7f027 (Update REPORT.md)

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
<<<<<<< HEAD
- Polygon bổ sung chi tiết gì so với box?
- `instance_id` dùng để làm gì và không phải loại ID nào?
- Đề xuất một quy tắc biên mask:
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
=======
  ```json
  {
    "instance_id": "kitchen-002",
    "class_name": "person",
    "score": 0.895,
    "polygon_xy": [
      [385.5, 69.2], [398.3, 71.1], [410.2, 85.5], ..., [380.2, 68.9]
    ]
  }
  ```
  - Record này chứa danh sách **điểm tọa độ** (x, y) của đa giác bao quanh instance.
  - `score` 0.895 = 89.5% độ tự tin model.
  - Số điểm polygon thường từ 50-500 tùy độ phức tạp biên object.

- Polygon bổ sung chi tiết gì so với box?
  - **Box (xyxy)**: Hình chữ nhật axis-aligned, bao quanh object. Đơn giản, nhanh, nhưng **mất chi tiết** phần không là object trong box (ví dụ: khoảng trống giữa hai chân người).
  - **Polygon**: Đa giác **sát biên object**, theo từng cạnh cơ thể. Giữ lại hình dáng chi tiết, giúp model học **ranh giới chính xác hơn**.
  - **Ứng dụng**: Polygon tốt cho segmentation (chia ảnh theo instance), trong khi box phù hợp với detection (dựng vật thể).

- `instance_id` dùng để làm gì và không phải loại ID nào?
  - **Mục đích**: Phân biệt giữa hai object **cùng lớp** (ví dụ: hai người khác nhau trong bức ảnh).
  - **Định dạng**: `instance_id` = `<sample_id>-NNN` (ví dụ: `kitchen-001`, `kitchen-002`). Là ID **trong phạm vi bức ảnh hiện tại**, không phải ID toàn cầu.
  - **Không phải**:
    - `class_id`: (class_id là số lớp, ở đây cả hai person có class_id = 0).
    - `tracking_id`: (tracking_id dùng cho video, theo dõi object qua frame; instance_id chỉ áp dụng cho ảnh tĩnh).

- Đề xuất một quy tắc biên mask:
  **Quy tắc biên mask được đề xuất**: "Polygon phải bám sát biên object; khoảng cách từ điểm đa giác đến pixel thực của object ≤ 3 pixel. Nếu có vùng ngoài object bị bao trong polygon > 10% diện tích, request rework."
  - Đảm bảo precision trong training (mô hình học biên chính xác).
  - Tránh annotator "vẽ tắt" quanh biên mơ hồ.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  - **Biên rõ ràng**: Annotator vẽ chính xác theo edge.
  - **Biên mơ hồ/mịn (soft edge)**: **Guideline**: Vẽ theo điểm cuối cùng clear của object, không cố dự đoán phần gradient mịn.
  - **Tiếp xúc giữa hai object (ví dụ: người chạm tay vào người khác)**: **Guideline**: Vẽ mỗi object riêng; tại điểm tiếp xúc, dùng đường thẳng giữa hai polygon, không đoán phần chồng lấn.
  - **Che khuất từ vật khác**: Nếu object bị khuất ≥ 40%, **Escalation** → Reviewer quyết định: vẽ phần nhìn thấy hay dự đoán toàn bộ object.
>>>>>>> 6c7f027 (Update REPORT.md)

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
<<<<<<< HEAD
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |
=======
| **Phân loại ảnh** | Single label (class_id + class_name) cho toàn ảnh | Model xấp xỉ: ảnh có 2+ chủ thể → model chọn 1; score thấp < 60% → kém chắc chắn. Ảnh đa chủ thể hay mơ hồ cần guideline rõ. | Chọn 1 nhãn theo guideline (dominant object, priority, hoặc central object). Ghi chú nếu ảnh có 2+ chủ thể. | Kiểm tra: (1) Nhãn có đúng guideline? (2) Nếu 2+ chủ thể, có ghi note? (3) Nhãn có phù hợp phạm vi (in-scope) của dự án? Nếu không, mark "rework". |
| **Phát hiện vật thể** | Danh sách box: [class_id, bbox_xyxy, confidence_flag] cho mỗi instance | Model bỏ sót object (false negative) hay dự đoán sai vị trí. Object bị che ≥ 30% hay cắt mép → box chặt vs. full predicted. Threshold filter → số box khác nhau. | Vẽ box chặt sát từng object; ghi lại class từ guideline. Nếu object bị khuất/cắt mép > quy định, flag "escalation". | Kiểm tra: (1) Box chặt sát? (2) Class đúng? (3) Có miss object nào không? (4) Có false positive (box trên nền)? Nếu > 5% lỗi, mark "rework". |
| **Instance segmentation** | Danh sách polygon: [instance_id, class_id, polygon_xy] cho mỗi instance | Model: biên mơ hồ → polygon "vẽ tắt"; tiếp xúc object → không tách rõ đa giác. Vùng che khuất → dự đoán hay chỉ vẽ phần nhìn? | Vẽ polygon sát biên từng object riêng. Biên mơ hồ: vẽ theo phần rõ nhất. Tiếp xúc: vẽ ranh giới nét tách. Nếu che ≥ 40%, flag escalation. | Kiểm tra: (1) Polygon sát biên (≤ 3px)? (2) Tách object rõ ràng? (3) Có polygon ngoài vùng object không (> 10% diện tích)? Nếu lỗi > 8%, mark "rework". |
>>>>>>> 6c7f027 (Update REPORT.md)

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
<<<<<<< HEAD
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
=======
  **Quy tắc**: "Tất cả ảnh, JSON, và output phải từ source công khai được xác nhận (ví dụ: COCO, ImageNet, OpenImages công khai; hoặc dataset được phía GV/dự án cấp). Không được thêm ảnh cá nhân, ảnh từ máy ảnh riêng, hoặc dữ liệu nội bộ công ty. Nếu có dữ liệu khách hàng/VinFast, phải được redact hoặc bỏ trước khi upload."
  - Mục đích: Bảo vệ quyền riêng tư, tuân thủ GDPR/luật bảo vệ dữ liệu cá nhân, tránh rò rỉ IP/KD nội bộ.
  - Kiểm tra: Reviewer phải xác nhận source ảnh trước khi approve labeling task.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  **Danh sách liên hệ escalation**:
  - Lab Coach / Mentor (kênh hỏi hỗ trợ chính của lớp): Nếu không rõ guideline hoặc dữ liệu mơ hồ.
  - GV giảng dạy: Nếu phát hiện dữ liệu không công khai / chứa thông tin nhạy cảm.
  - IT/Data Protection Team: Nếu nghi ngờ rò rỉ dữ liệu khách hàng hoặc IP nội bộ.
  - **Hành động**: Không xóa/sửa dữ liệu riêng; chụp screenshot lỗi, ghi rõ vị trí, submit ticket rồi dừng công việc cho đến khi được hướng dẫn tiếp.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json` – Record rank 1 với class_id 468, class_name "cab", score 0.510915, taxonomy ImageNet-1K.
- [x] `detection_predictions.json` – Record person với score 0.912625, bbox [385.33, 69.24, 498.92, 348.92], kích thước 113.58×279.68 px.
- [x] `segmentation_predictions.json` – Record instance_id kitchen-002, class_name person, score 0.895, polygon_xy với điểm tọa độ.
- [x] `IMAGE_ATTRIBUTION.md` – Ghi rõ nguồn 3 ảnh công khai từ COCO dataset (traffic, kitchen×2).
- [x] `visuals/classification_top5.png` – Biểu đồ top-5 lớp cho ảnh traffic; "cab" xếp hạng 1.
- [x] `visuals/detection_predictions.png` – Ảnh kitchen với bounding box người ở nửa phải, score 0.91.
- [x] `visuals/segmentation_prediction.png` – Ảnh kitchen với polygon bao quanh người, instance_id kitchen-002.
- [x] Ô validation cuối notebook báo `PASS` – Checksum 3 model ✓, 3 ảnh ✓, 3 JSON ✓, 3 PNG ✓.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output – Báo cáo chỉ ghi mã code (class_id), không ghi cá nhân hoặc khách hàng.
>>>>>>> 6c7f027 (Update REPORT.md)
