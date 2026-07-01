# Báo cáo Nghiên cứu: Tokenization của ERNIE-Image trên Tiếng Việt & Lỗi Byte Fallback

## 1. Tổng quan
Trong quá trình huấn luyện mô hình **Vietnamese Calligraphy Qwen LoRA** sử dụng text encoder từ **ERNIE-Image**, việc bảo toàn tính chính xác của các ký tự tiếng Việt (đặc biệt là các nguyên âm có dấu thanh) đóng vai trò quyết định đến chất lượng sinh chữ thư pháp.

Báo cáo này tài liệu hóa kết quả nghiên cứu và kiểm tra tokenization trên 2 nguồn dữ liệu:
1. Danh sách **7,184 âm tiết tiếng Việt** (`vietnamesesyllable_7184.txt`).
2. Dataset thực tế **Phase A** gồm 14,600 prompt (`data/dataset_7714_phase_a/metadata.jsonl`).

---

## 2. So sánh Tokenization: Viết thường vs Viết hoa chữ đầu (Capitalized)

Phân tích định lượng trên bộ dữ liệu 7,184 âm tiết tiếng Việt cho thấy sự khác biệt đáng kể khi thay đổi định dạng chữ thường sang viết hoa chữ cái đầu tiên (ví dụ: `và` ➡️ `Và`):

### 📊 Bảng thống kê so sánh

| Chỉ số | Viết thường (Lowercase) | Viết hoa chữ cái đầu |
| :--- | :---: | :---: |
| **Tổng số âm tiết đánh giá** | 7,184 | 7,184 |
| **Số âm tiết là Single-token** | 562 (7.82%) | 246 (3.42%) |
| **Số lượng Token trung bình / Âm tiết** | 2.1434 | 2.2163 |
| **Tỷ lệ âm tiết thay đổi độ dài token** | - | 8.41% (604 âm tiết) |

### 🔍 Các phát hiện chính

#### ⚠️ Hiện tượng Lỗi Byte Fallback (Nguy hiểm nhất)
Khi viết hoa các nguyên âm có dấu phụ hoặc dấu thanh ở đầu câu (như `Ở`, `Ông`, `Ảnh`, `Ý`, `Ứng`, `Ấy`, `Ước`, `Ủng`, `Ơn`, `Ấm`, `Ôm`...), tokenizer của ERNIE-Image không thể ánh xạ trực tiếp sang token chữ cái tương ứng mà sẽ phân rã thành **Byte Fallback** (mã byte thô UTF-8, được giải mã đơn lẻ thành ký tự thay thế `\ufffd` hay ``).
* **Ví dụ**:
  * `ở` ➡️ `['ở']` (Token ID: `[7878]`) - **Không lỗi**
  * `Ở` ➡️ `['', '']` (Token ID: `[1589, 1158]`) - **Lỗi Byte Fallback**
  * `ông` ➡️ `['ông']` (Token ID: `[5811]`) - **Không lỗi**
  * `Ông` ➡️ `['', '', 'ng']` (Token ID: `[1195, 1148, 1716]`) - **Lỗi Byte Fallback**

#### 📉 Hiện tượng phân rã Token (Fragmentation)
Nhiều âm tiết rất phổ biến có dạng single-token khi viết thường, nhưng khi viết hoa chữ đầu sẽ bị xé nhỏ thành các token cấp ký tự:
* `qua` (`['qua']`) ➡️ `Qua` (`['Q', 'ua']`)
* `xe` (`['xe']`) ➡️ `Xe` (`['X', 'e']`)
* `anh` (`['anh']`) ➡️ `Anh` (`['A', 'nh']`)
* `gia` (`['gia']`) ➡️ `Gia` (`['G', 'ia']`)

#### 📈 Hiện tượng gộp Token (Merging)
Trái lại, có khoảng 8.41% trường hợp các âm tiết khi viết thường bị tách nhỏ nhưng viết hoa chữ cái đầu lại được gộp thành single-token:
* `các` (`['c', 'ác']`) ➡️ `Các` (`['Các']`)
* `trong` (`['tr', 'ong']`) ➡️ `Trong` (`['Trong']`)
* `một` (`['m', 'ột']`) ➡️ `Một` (`['Một']`)

---

## 3. Phân tích lỗi Byte Fallback trên Dataset Phase A (`dataset_7714_phase_a`)

Thực hiện quét toàn bộ 14,600 prompt trong file `metadata.jsonl`, thu được **100 trường hợp** xảy ra lỗi Byte Fallback, phân loại như sau:

### 3.1. Lỗi do nguyên âm viết hoa có dấu (2 trường hợp)
Ký tự viết hoa **`Ơ`** trong chữ `Ơn` thuộc từ ghép `Biết Ơn` kích hoạt lỗi byte fallback:
* **Dòng 938 & Dòng 4114**: Prompt `Vietnamese calligraphy: "Biết Ơn"`
  * **Tokens**: `[..., 1066, 34254, 1032, 1198, 1160, 1110, 1034]`
  * **Giải mã lỗi**: Cụm token `[1198, 1160]` giải mã đơn lẻ thành `['', '']` thay vì biểu diễn ký tự `Ơ`.

### 3.2. Lỗi do ký tự viết thường cụ thể (98 trường hợp)
Tokenizer của ERNIE-Image bị lỗi phân rã byte fallback trên một số ký tự viết thường đặc thù:
* **Tổ hợp nguyên âm `ỵ`**:
  * Các từ chứa `ỵ` như `quỵt`, `uỵch`, `khuỵ`, `lỵ`, `vỵ`, `kỵ`, `nguỵ`, `tuỵ`, `luỵ`, `quỵ`, `thuỵ` đều bị phân rã ký tự `ỵ` thành cụm token fallback `[1589, 1181]` (`['', '']`).
* **Tổ hợp chữ `gi` kết hợp nguyên âm có dấu**:
  * Các từ như `giờ`, `giớ`, `giống`, `giỡn`, `giọt`, `giục`, `giỏ`, `giốc`, `giọng`, `giỗ`, `giữa` bị lỗi phân tách byte ở token ID `4151` (`i`).

---

## 4. Khuyến nghị và Giải pháp cho Training Pipeline (V8 & V9)

Từ các phát hiện thực nghiệm trên, nhóm nghiên cứu đề xuất các phương án tối ưu hóa huấn luyện mô hình thư pháp như sau:

1. **Khuyến nghị Case Normalization (Chuyển đổi Chữ thường)**:
   * **Bắt buộc**: Chuẩn hóa toàn bộ phần text nội dung chữ trong prompt huấn luyện và suy diễn về dạng **chữ viết thường (lowercase)** trước khi đưa qua Text Encoder của ERNIE-Image.
   * Điều này giúp loại bỏ 100% các lỗi Byte Fallback nguy hiểm do ký tự viết hoa có dấu gây ra (ví dụ: chuyển `Biết Ơn` thành `biết ơn`).
   
2. **Cập nhật Tiêu chuẩn Prompt (Prompt Standard V8/V9)**:
   * Chuẩn hóa cấu trúc prompt huấn luyện:
     * Dạng cũ: `Vietnamese calligraphy: "Biết Ơn"`
     * Dạng mới (đề xuất): `Vietnamese calligraphy: "biết ơn"`
     * Dạng chuẩn hóa lowercase toàn bộ: `vietnamese calligraphy: "biết ơn"`

3. **Xử lý đặc biệt đối với ký tự `ỵ` và cụm `gi`**:
   * Với 98 trường hợp lỗi viết thường, do đây là giới hạn vật lý của Tokenizer ERNIE-Image (không có vocabulary phù hợp cho ký tự `ỵ` hoặc cách tổ hợp dấu tiếng Việt), mô hình cần được tăng cường huấn luyện (augmentation) thông qua các mặt nạ diacritic nâng cao (Diacritic Masked Loss) tại các vùng chữ này để bù đắp sự thiếu hụt thông tin ngữ nghĩa từ text encoder.
