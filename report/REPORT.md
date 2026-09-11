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
    "score": 0.510915,
    "taxonomy_name": "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào?
   Ảnh có mã định danh: 
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
   Danh sách lớp do tổ chức/tác giả sáng lập nên tập dữ liệu ImageNet-1K định nghĩa.
   Checkpoint yolo11n-cls.pt chỉ dự đoán trong phạm vi $1.000$ lớp đã được thiết lập sẵn này.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  ID (class_id: 468): Mã định danh số chuẩn xác giúp máy tính truy vấn và lập trình không bị lỗi nhầm lẫn hay lệch ký tự.
  Tên lớp (class_name: "cab"): Giúp con người đọc hiểu ngay đối tượng dự đoán là "xe taxi" mà không cần tra cứu bảng mã.Tên taxonomy (taxonomy_name: "ImageNet-1K"): 
  Xác định rõ hệ quy chiếu dữ liệu, giúp phân biệt mã $468$ của ImageNet với mã $468$ của các tập dữ liệu khác (như COCO hay Open Images).
  
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  Guideline cần quy định rõ tiêu chí chọn nhãn duy nhất cho tác vụ image_classification: chọn đối tượng chiếm diện tích lớn nhất (dominant object), đối tượng nằm ở trung tâm bức ảnh, hay chọn theo độ ưu tiên của bài toán kinh doanh (ví dụ: ưu tiên phân loại phương tiện giao thông hơn cảnh quan).
- Vì sao model score không phải ground truth?
Mức score $0.510915$ ($51.09\%$) chỉ đại diện cho độ tự tin bằng xác suất của mô hình YOLOv11 tại thời điểm suy luận. Đây vẫn là một khoảng xác suất khá thấp (gần như $50/50$), mô hình có thể đoán sai do nhiễu góc quay hoặc ảnh chứa nhiều xe khác nhau; nhãn Ground Truth chuẩn xác bắt buộc phải do con người kiểm duyệt và xác nhận.
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
- Diễn giải vị trí box bằng lời:
- So sánh số prediction ở hai threshold:
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
- Đề xuất một quy tắc box chặt:
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
- Polygon bổ sung chi tiết gì so với box?
- `instance_id` dùng để làm gì và không phải loại ID nào?
- Đề xuất một quy tắc biên mask:
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
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
