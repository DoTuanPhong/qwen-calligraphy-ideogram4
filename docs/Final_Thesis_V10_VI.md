BỘ GIÁO DỤC VÀ ĐÀO TẠO

TRƯỜNG ĐẠI HỌC FPT

Fine-tuning Qwen Image for Generating Vietnamese Calligraphy Images with Accurate Diacritics

Tác giả

Đỗ Tuấn Phong

Luận văn được nộp nhằm đáp ứng yêu cầu
cho học vị Thạc sĩ Kỹ thuật phần mềm

Giảng viên hướng dẫn:

TS. Nguyễn Bích Thủy

© Bản quyền thuộc về Đỗ Tuấn Phong 2026

---

# Fine-tuning Qwen Image for Generating Vietnamese Calligraphy Images with Accurate Diacritics

Đỗ Tuấn Phong

Chương trình: Thạc sĩ Kỹ thuật phần mềm

Chuyên ngành: Trí tuệ nhân tạo

Trường Đại học FPT

2026

---

## Tóm tắt

Thư pháp chữ Quốc ngữ là một hình thức nghệ thuật đặc thù của Việt Nam, trong đó hệ chữ viết Latin của tiếng Việt được thể hiện bằng bút pháp thư pháp truyền thống. Trong khi các mô hình sinh ảnh hiện đại có thể tạo ra hình ảnh giàu tính thẩm mỹ, chúng vẫn gặp khó khăn lớn khi phải hiển thị chính xác tiếng Việt, do tiếng Việt phụ thuộc chặt chẽ vào dấu thanh và dấu phụ của nguyên âm. Một ảnh sinh ra có thể nhìn đẹp nhưng vẫn thất bại về mặt nội dung nếu biến `Cữu` thành `Cưu`, `Chưởng` thành `Chưỡng`, hoặc `Gẫy` thành `Gậy`. Trong thư pháp tiếng Việt, các dấu này không phải chi tiết trang trí; chúng là thành phần ngôn ngữ bắt buộc và đồng thời phải hòa vào hình học nét bút.

Luận văn này được đăng ký chính thức dưới hướng đề tài fine-tuning Qwen Image, và ban đầu đi theo tuyến triển khai dựa trên Qwen-Image. Trong quá trình nghiên cứu, bản triển khai cuối đã chuyển sang Ideogram4, một mô hình sinh ảnh Diffusion Transformer có năng lực hiển thị chữ mạnh hơn. Quyết định này xuất phát từ hai nguyên nhân thực nghiệm. Thứ nhất, Qwen Image yêu cầu VRAM quá lớn cho chu trình thử nghiệm cục bộ lặp lại nhiều lần. Thứ hai, khả năng hiển thị tiếng Việt zero-shot của Qwen Image quá yếu, thường sai ở cấp độ ký tự trước khi có thể tách riêng bài toán dấu thanh tinh vi hơn. ERNIE Image được xem xét như một hướng trung gian và nhìn chung tốt hơn Qwen Image: mô hình này thường giữ được gần đúng số lượng ký tự và đôi khi sinh đúng một vài cụm từ tiếng Việt. Tuy nhiên, ERNIE Image vẫn sai ký tự nhiều và các ghi chú từ bước ERNIE Image trước đó cho thấy đường mã hóa văn bản dựa trên Mistral3/Ministral3 với byte-level BPE gây khó khăn lớn cho việc căn chỉnh âm tiết tiếng Việt với token. Do đó, Ideogram4 được chọn làm nền tảng thực nghiệm cuối cùng.

Đóng góp trung tâm của luận văn là một bản triển khai Ideogram4 DiT-LoRA cho mục tiêu đăng ký fine-tuning sinh ảnh thư pháp tiếng Việt, được tổ chức như một pipeline hướng glyph-binding nhằm cải thiện độ chính xác dấu tiếng Việt. Pipeline này chẩn đoán xem tín hiệu dấu còn tồn tại hay suy yếu ở đâu, fine-tune các module DiT được chọn bằng LoRA, ổn định các checkpoint có phương sai cao, và cuối cùng huấn luyện trực tiếp trên bố cục nhiều từ để tiến gần hơn tới mục tiêu sinh câu dài.

Các thử nghiệm LoRA chỉ trên attention trước đó bị mắc ở vùng plateau khoảng 32-39 từ đúng trên 60 trong panel một từ khó. Probe hidden-state và projection cho thấy Qwen3-VL không mù dấu tiếng Việt toàn cục; tín hiệu dấu vẫn tồn tại trong conditioning. Điểm nghẽn chính nằm ở việc DiT không luôn gắn được tín hiệu dấu yếu đó vào hình học glyph đúng trong quá trình sinh ảnh. Dựa trên chẩn đoán này, luận văn mở rộng mục tiêu LoRA từ attention-only sang tập module DiT rộng hơn gồm `attention.qkv`, `attention.o`, `feed_forward.w1`, `feed_forward.w2`, `feed_forward.w3`, và `adaln_modulation`.

Thiết lập wide-target giúp vượt plateau một từ, đạt 48/60 ở checkpoint đơn tốt nhất và 52/60 sau khi trung bình hóa các checkpoint LoRA tương thích. Tuy nhiên, kết quả một từ không tự động chuyển tốt sang ảnh nhiều từ. Checkpoint đơn-từ tốt nhất vẫn tạo nhiều lỗi khi viết các ảnh thư pháp hai dòng. Vì vậy, luận văn xây dựng tập dữ liệu compound gồm ảnh 4/5/7/8 từ, chia hai dòng căn giữa, render từ font thư pháp mục tiêu và ghép với prompt Ideogram4 không dùng bounding box. Tập dữ liệu này bao phủ toàn bộ 406 token ID tiếng Việt có dấu được liệt kê ở cấp tokenizer trên cả chữ hoa và chữ thường.

Trên panel compound Eval28 gồm 28 ảnh và tổng cộng 168 từ, checkpoint gold một từ `soup567` tạo 56 lỗi. Sau compound bridge fine-tuning và trung bình hóa checkpoint tương thích, mốc `soup_e4r2r3r4` giảm lỗi xuống còn 6 lỗi / 168 từ. Một nhánh follow-up từ mốc này với learning rate `3e-5`, sau đó trung bình hóa nhẹ theo tỉ lệ 90% nhánh mới + 10% mốc cũ, tiếp tục giảm lỗi xuống còn 4 lỗi / 168 từ, tương đương khoảng 97,6% độ đúng ở cấp từ trên panel này. Kết quả cho thấy bài toán sinh thư pháp tiếng Việt chính xác không chỉ cần một text-to-image base model mạnh, mà còn cần chẩn đoán đúng vị trí tín hiệu ngôn ngữ, điều chỉnh đúng module DiT, ổn định checkpoint, và huấn luyện trực tiếp trên phân phối bố cục nhiều từ.

**Từ khóa:** thư pháp tiếng Việt, Qwen Image, Ideogram4, fine-tuning, LoRA, Diffusion Transformer, sinh ảnh từ văn bản, dấu tiếng Việt, glyph binding.

---

## Lời cảm ơn

Em xin gửi lời cảm ơn sâu sắc tới TS. Nguyễn Bích Thủy, giảng viên hướng dẫn của em, vì sự chỉ dẫn, góp ý và kiên nhẫn trong suốt quá trình nghiên cứu. Bài toán trong luận văn này đã thay đổi nhiều lần: từ các thử nghiệm Qwen-Image ban đầu, sang ERNIE Image, rồi tới Ideogram4, từ kiểm tra lỗi dấu tiếng Việt đơn lẻ tới chẩn đoán tín hiệu conditioning và cuối cùng là huấn luyện trên bố cục nhiều từ. Những góp ý của cô giúp nghiên cứu giữ được định hướng rõ ràng khi kết quả thực nghiệm nhiều nhiễu và khi một số giả thuyết hấp dẫn phải bị loại bỏ bởi bằng chứng.

Em cũng xin cảm ơn Trường Đại học FPT và chương trình MSE-AI đã tạo điều kiện để em thực hiện một đề tài kết hợp giữa kỹ thuật trí tuệ nhân tạo và di sản văn hóa Việt Nam. Đây là một bài toán vừa có ý nghĩa kỹ thuật, vừa có ý nghĩa ứng dụng trong thiết kế, giáo dục và bảo tồn thư pháp chữ Quốc ngữ.

Em biết ơn các cộng đồng mã nguồn mở và các nhóm kỹ thuật đã phát triển các công cụ được sử dụng trong nghiên cứu này, bao gồm Ideogram AI, nhóm Qwen, DiffSynth-Studio, PyTorch, HuggingFace Transformers, safetensors và hệ sinh thái Python machine learning. Các thí nghiệm trong luận văn phụ thuộc nhiều vào script tái lập, quản lý checkpoint, chuyển đổi LoRA và đánh giá trên GPU.

Cuối cùng, em xin cảm ơn gia đình và bạn bè vì sự động viên và kiên nhẫn. Nhiều kết quả quan trọng của luận văn đến từ việc đánh giá thủ công từng ảnh thư pháp, một công việc tốn thời gian nhưng cần thiết để bảo đảm độ tin cậy của kết luận.

---

# Mục lục

Lời cảm ơn

Danh mục bảng

Danh mục hình

Danh mục phụ lục

1. Giới thiệu

1.1. Phát biểu bài toán

1.2. Mục tiêu và phạm vi nghiên cứu

1.2.1. Sự thay đổi phạm vi từ Qwen-Image sang Ideogram4

1.2.2. Mục tiêu và ranh giới nghiên cứu

1.3. Tổng quan quy trình và thách thức miền bài toán

1.3.1. Pipeline nghiên cứu tổng thể

1.3.2. Các thách thức đặc thù

1.4. Tổng quan tài liệu

1.4.1. Các phương pháp dựa trên GAN

1.4.2. Các mô hình diffusion

1.4.3. Diffusion Transformer và text encoder đa phương thức

1.4.4. Fine-tuning hiệu quả tham số

1.4.5. Mô hình thương mại và khoảng trống nghiên cứu

1.5. Phương pháp đề xuất và đóng góp

1.6. Cấu trúc luận văn

2. Cơ sở lý thuyết

2.1. Sinh ảnh bằng AI: từ GAN tới Diffusion Transformer

2.2. Kiến trúc Ideogram4

2.3. Kỹ thuật fine-tuning hiệu quả

2.4. Thư pháp tiếng Việt: đặc điểm thị giác và ngôn ngữ

3. Triển khai và đánh giá

3.1. Thiết lập hệ thống

3.2. Dữ liệu và tiền xử lý

3.3. Các probe chẩn đoán

3.4. Cấu hình fine-tuning

3.5. Giao thức đánh giá

3.6. Kết quả

4. Kết luận

4.1. Giá trị lý thuyết và thực tiễn

4.2. Tóm tắt kết quả chính

4.3. Hạn chế hiện tại

4.4. Hướng phát triển tương lai

Tài liệu tham khảo

Phụ lục

---

# Danh mục bảng

**Bảng 1.1:** So sánh đối thủ và baseline cho sinh ảnh thư pháp tiếng Việt

**Bảng 2.1:** Tổng quan ba thành phần chính trong pipeline Ideogram4

**Bảng 3.1:** Cấu hình phần cứng

**Bảng 3.2:** Môi trường phần mềm

**Bảng 3.3:** Nhóm module LoRA trong DiT

**Bảng 3.4:** Kết quả probe Qwen3-VL signal

**Bảng 3.5:** Kết quả panel một từ

**Bảng 3.6:** Tiến trình kết quả compound Eval28

---

# Danh mục hình

**Hình 1.1:** Pipeline tổng thể cho sinh ảnh thư pháp tiếng Việt

**Hình 1.2:** So sánh trực quan giữa các phương pháp đối thủ và checkpoint đề xuất
Tên file dự kiến: `docs/thesis/figures/fig_1_2_competitor_baseline_comparison.png`

**Hình 1.3:** So sánh năng lực gốc của Qwen Image, ERNIE Image và Ideogram4 trước fine-tuning
Tên file dự kiến: `docs/thesis/figures/fig_1_3_base_model_capability_comparison.png`

**Hình 2.1:** Kiến trúc Ideogram4 dùng trong luận văn

**Hình 2.2:** Conditioning Qwen3-VL nhiều layer đưa vào DiT

**Hình 3.1:** Các điểm chèn wide-target LoRA trong Ideogram4 DiT

**Hình 3.2:** Bố cục compound hai dòng căn giữa

**Hình 3.3:** Tiến trình từ plateau một từ tới checkpoint soup

**Hình 3.4:** Tiến trình từ baseline compound tới compound bridge

**Hình 3.5:** Ví dụ trước/sau trên các từ tiếng Việt có dấu khó
Tên file dự kiến: `docs/thesis/figures/fig_3_5_single_word_before_after_examples.png`

**Hình 3.6:** So sánh compound Eval28 trước/sau giữa `soup567` và checkpoint cuối `soup_lr3e5_gold4_9to1`
Tên file dự kiến: `docs/thesis/figures/fig_3_6_compound_eval28_before_after.png`

**Hình 3.7:** Các lỗi còn lại của checkpoint compound gold hiện tại
Tên file dự kiến: `docs/thesis/figures/fig_3_7_remaining_error_cases.png`

**Hình 3.8:** Kiểm tra checkpoint fine-tune khi kết hợp chữ thư pháp với năng lực sinh ảnh nền của Ideogram4
Tên file dự kiến: `docs/thesis/figures/fig_3_8_calligraphy_with_base_model_capability.png`

---

# Danh mục phụ lục

**Phụ lục A:** Lệnh training wide-target chính

**Phụ lục B:** Lệnh tạo dữ liệu compound

**Phụ lục C:** Script hậu xử lý và trung bình hóa checkpoint

**Phụ lục D:** Registry checkpoint

**Phụ lục E:** Thư mục đánh giá

**Phụ lục F:** Các lỗi còn lại của checkpoint compound gold hiện tại

**Phụ lục G:** Báo cáo probe nội bộ

**Phụ lục H:** Danh sách ảnh so sánh dự kiến

**Phụ lục I:** Prompt dùng cho các hình so sánh

---

# 1. Giới thiệu

## 1.1. Phát biểu bài toán

Thư pháp chữ Quốc ngữ là hình thức thể hiện hệ chữ Latin tiếng Việt bằng bút pháp thư pháp. Khác với chữ Latin thông thường, tiếng Việt có hệ thống dấu thanh và dấu phụ dày đặc. Một thay đổi rất nhỏ ở dấu có thể biến một từ thành từ khác. Ví dụ, `Cưu`, `Cừu`, `Cứu`, `Cửu`, `Cữu`, và `Cựu` có hình thức thị giác gần nhau nhưng mang nghĩa khác nhau.

Trong thư pháp, bài toán còn phức tạp hơn vì dấu không chỉ cần đúng chính tả. Dấu phải hòa vào tổng thể nét bút, giữ nhịp điệu, độ dày mỏng, độ nghiêng và cảm giác thủ công. Dấu nặng không được giống một chấm mực ngẫu nhiên; dấu ngã không được biến thành dấu hỏi; dấu mũ và dấu thanh phải đặt đúng vị trí nhưng vẫn tự nhiên trong bố cục nét.

Các mô hình sinh ảnh thương mại có thể tạo ảnh thư pháp nhìn đẹp, nhưng thường không cho phép fine-tune trên một font cụ thể, không bảo đảm tái lập, và không kiểm soát tốt từng dấu tiếng Việt. Ngược lại, render trực tiếp từ font máy cho kết quả chính tả đúng nhưng thiếu độ sống của nét bút: không có biến thiên mực, độ khô, độ nhấn, hay tương tác tự nhiên giữa các nét. Khoảng trống của nghiên cứu này nằm giữa hai cực đó: học một phong cách thư pháp cụ thể nhưng vẫn giữ được chính xác dấu tiếng Việt.

Đề tài ban đầu được đăng ký theo hướng Qwen Image. Tuy nhiên, trong quá trình nghiên cứu, Qwen Image cho thấy hai hạn chế lớn: yêu cầu VRAM quá cao cho quy trình lặp cục bộ và khả năng render tiếng Việt ban đầu yếu, thường sai ngay ở cấp ký tự. ERNIE Image được xem xét như một hướng thay thế và tốt hơn Qwen Image ở một số điểm: mô hình thường giữ được gần đúng số lượng ký tự và đôi khi sinh đúng một vài cụm từ tiếng Việt. Dù vậy, ERNIE Image vẫn chưa đủ tốt vì còn nhiều lỗi ký tự; hơn nữa, các ghi chú từ bước ERNIE Image trước đó cho thấy tokenizer/đường mã hóa Mistral3/Ministral3 theo byte-level BPE làm việc căn chỉnh dấu theo âm tiết trở nên khó khăn.

Vì vậy, bản triển khai cuối của luận văn chuyển sang Ideogram4. Sự chuyển đổi này không làm thay đổi mục tiêu nghiên cứu: vẫn là fine-tune một mô hình sinh ảnh hiện đại để tạo thư pháp tiếng Việt có dấu chính xác. Điểm thay đổi là nền tảng thực nghiệm được chọn theo bằng chứng thay vì cố giữ một backbone kém phù hợp.

## 1.2. Mục tiêu và phạm vi nghiên cứu

### 1.2.1. Sự thay đổi phạm vi từ Qwen-Image sang Ideogram4

Tên luận văn chính thức giữ cụm Qwen Image vì đây là tên đã đăng ký khi hướng nghiên cứu ban đầu tập trung vào Qwen-Image. Trong luận văn này, title được hiểu là title hành chính và title của mục tiêu nghiên cứu: fine-tune mô hình text-to-image để sinh ảnh thư pháp tiếng Việt với dấu chính xác.

Trong quá trình thực nghiệm, phạm vi triển khai đã thay đổi. Qwen Image không thuận lợi cho chu trình nghiên cứu vì tốn VRAM và có baseline tiếng Việt yếu. ERNIE Image cho thấy một số tín hiệu khả quan hơn Qwen Image, nhưng vẫn bị giới hạn bởi lỗi ký tự và vấn đề tokenizer byte-level BPE. Ideogram4 được chọn vì có nền tảng text-rendering mạnh hơn và thực tế hơn cho huấn luyện LoRA cục bộ. Pipeline Ideogram4 cũng dùng Qwen3-VL-based text conditioning, do đó vẫn liên quan trực tiếp đến câu hỏi ban đầu về tín hiệu conditioning họ Qwen.

Vì vậy, các chương thực nghiệm của luận văn báo cáo bản triển khai cuối đã được xác thực: pipeline Ideogram4 DiT-LoRA cho sinh ảnh thư pháp tiếng Việt có dấu chính xác.

### 1.2.2. Mục tiêu và ranh giới nghiên cứu

Mục tiêu chính của luận văn là xây dựng một pipeline fine-tuning để sinh ảnh thư pháp tiếng Việt có dấu chính xác. Bản triển khai cuối sử dụng Ideogram4 và LoRA trong phạm vi đề tài Qwen Image đã đăng ký.

Ở mức cụ thể hơn, nghiên cứu phân tích các kiểu lỗi của Ideogram4 khi sinh ảnh thư pháp tiếng Việt có dấu, sau đó xác định lỗi chủ yếu đến từ text encoder Qwen3-VL, prompt, thiếu capacity của LoRA, hay hành vi glyph-binding của DiT. Từ chẩn đoán đó, luận văn thiết kế cấu hình LoRA để cải thiện render dấu tiếng Việt mà không cần full fine-tuning, xây dựng dữ liệu và panel đánh giá cho cả một từ lẫn bố cục nhiều từ, đồng thời kiểm tra xem cải thiện ở một từ có chuyển được sang bố cục compound và câu giống thực tế hay không. Kết quả cuối của nghiên cứu không chỉ là một checkpoint, mà còn là pipeline có thể tái lập cho training, chuyển đổi checkpoint, render đánh giá và chấm thủ công.

Phạm vi thực nghiệm được giới hạn có chủ đích để kết quả dễ diễn giải. Bản triển khai cuối dùng Ideogram4 làm base model và giữ text encoder Qwen3-VL frozen. Quá trình fine-tuning sử dụng LoRA adapter trên DiT backbone, với font Thu Phap Thanh Cong Unicode làm phong cách mục tiêu chính và độ phân giải 1024×1024. Đánh giá tập trung vào độ đúng ở cấp từ do người kiểm tra thủ công, vì OCR hiện chưa đủ tin cậy cho thư pháp tiếng Việt có phong cách nét bút mạnh. Nội dung chính gồm một từ tiếng Việt và ảnh compound 4/5/7/8 từ chia hai dòng.

Luận văn không tuyên bố giải quyết mọi phong cách thư pháp hoặc mọi dạng render tiếng Việt. Trọng tâm là một pipeline thực nghiệm cho một font thư pháp cụ thể, với mục tiêu chính là độ chính xác dấu tiếng Việt.

## 1.3. Tổng quan quy trình và thách thức miền bài toán

### 1.3.1. Pipeline nghiên cứu tổng thể

Pipeline nghiên cứu cuối bắt đầu bằng việc quan sát baseline của Ideogram4 trên các prompt thư pháp tiếng Việt, nhằm xác định những kiểu lỗi thường gặp. Sau đó, nghiên cứu chuyển sang chẩn đoán tín hiệu bằng hidden-state probe, token-span probe và projection probe để kiểm tra liệu tín hiệu dấu tiếng Việt có còn tồn tại trong Qwen3-VL và có tới được cửa vào DiT hay không. Khi bottleneck đã rõ hơn, LoRA được huấn luyện trên các từ tiếng Việt khó. Thiết lập attention-only được thử trước, nhưng sau khi đạt plateau, target LoRA được mở rộng sang một nhóm module DiT rộng hơn.

Sau giai đoạn một từ, pipeline xử lý vấn đề bất ổn checkpoint. Các LoRA checkpoint hiệu quả được trung bình hóa nếu chúng tương thích và có vẻ nằm trong cùng vùng tối ưu. Cách này giúp ổn định kết quả mà không tăng chi phí inference. Giai đoạn cuối chuyển từ một từ sang bố cục compound: dữ liệu nhiều từ hai dòng được tạo từ font thư pháp mục tiêu, checkpoint một từ tốt nhất được dùng làm điểm khởi đầu, và mô hình được đánh giá trên một panel compound cố định.

### 1.3.2. Các thách thức đặc thù

Thách thức đầu tiên đến từ chính hệ thống chữ viết tiếng Việt. Tiếng Việt có sáu thanh và nhiều biến thể nguyên âm như `â`, `ă`, `ê`, `ô`, `ơ`, `ư`; một lỗi rất nhỏ ở dấu có thể làm sai hoàn toàn từ. Trong thư pháp, dấu còn phải được tích hợp vào nét bút, không chỉ đặt đúng vị trí như trong font in. Nếu dấu quá tách rời, ảnh mất chất thư pháp; nếu dấu hòa vào nét quá mạnh, người đọc có thể hiểu sai.

Thách thức thứ hai nằm ở quá trình fine-tuning và bố cục. Attention-only LoRA cải thiện được một phần nhưng nhanh chóng mắc ở vùng 32-39/60, cho thấy chỉ điều chỉnh attention không đủ để sửa hình học glyph. Ngoài ra, cùng một công thức training có thể tạo checkpoint tốt hoặc sụp mạnh, ví dụ một vòng warm-continue bị nhiều lỗi dấu nặng. Khi chuyển sang ảnh nhiều từ, khó khăn lại tăng lên vì mô hình phải đồng thời xử lý căn giữa, khoảng cách chữ, cỡ chữ và cạnh tranh tín hiệu giữa nhiều từ trong cùng một ảnh.

## 1.4. Tổng quan tài liệu

### 1.4.1. Các phương pháp dựa trên GAN

GAN huấn luyện một generator và discriminator đối kháng. Trong bài toán thư pháp, GAN từng được dùng để chuyển đổi glyph in sang phong cách thư pháp. Tuy nhiên, GAN dễ mode collapse và khó đảm bảo bao phủ đầy đủ các tổ hợp dấu tiếng Việt.

### 1.4.2. Các mô hình diffusion

Diffusion model sinh ảnh bằng quá trình khử nhiễu lặp. So với GAN, diffusion ổn định hơn và tạo ảnh đa dạng hơn. Latent Diffusion tiếp tục giảm chi phí bằng cách khử nhiễu trong latent space thay vì pixel space, giúp sinh ảnh độ phân giải cao khả thi hơn.

### 1.4.3. Diffusion Transformer và text encoder đa phương thức

Diffusion Transformer thay U-Net bằng transformer block trong mạng khử nhiễu. Transformer thuận lợi cho việc tích hợp token văn bản và token ảnh, nhưng render chữ chính xác vẫn khó vì mô hình phải biến tín hiệu ký hiệu thành hình học không gian. Với tiếng Việt, tín hiệu quan trọng thường rất nhỏ nhưng có ý nghĩa ngôn ngữ lớn.

Các mô hình như Qwen-Image, ERNIE Image, FLUX, Ideogram4 thể hiện xu hướng dùng DiT và text encoder mạnh. Tuy nhiên, mỗi mô hình có điểm nghẽn khác nhau. Qwen Image nặng VRAM và yếu baseline tiếng Việt; ERNIE Image tốt hơn Qwen Image nhưng gặp rủi ro căn chỉnh do byte-level BPE; Ideogram4 cung cấp nền tảng thực tế nhất cho pipeline này.

Điểm chung về kiến trúc ba hướng mã nguồn mở được khảo sát. Ba mô hình trong bảng so sánh đều thuộc họ text-to-image dựa trên Diffusion Transformer, nhưng có những lựa chọn kiến trúc khác nhau đáng chú ý:

- Qwen-Image (Qwen Team, công bố 2025-08-04, arXiv 2508.02324) là foundation model sinh ảnh trong họ Qwen với 20B tham số, được huấn luyện từ đầu và phát hành dưới giấy phép Apache 2.0. Kiến trúc là MMDiT (Multimodal Diffusion Transformer) — biến thể DiT xử lý đồng thời token văn bản và token ảnh trong các stream song song. Text encoder được xác nhận dùng họ Qwen2.5-VL (multimodal LLM cùng hệ với Qwen3-VL mà Ideogram4 dùng). Model hỗ trợ đa tỉ lệ khung hình từ 1:1 đến 16:9, sinh ảnh ở 50 inference steps với `true_cfg_scale=4.0` theo công thức công bố. Trên benchmark nội bộ của Qwen (GenEval, DPG, OneIG-Bench cho generation; GEdit, ImgEdit, GSO cho editing; LongText-Bench, ChineseWord, TextCraft cho text rendering), Qwen-Image đạt state-of-the-art, đặc biệt cho Chinese text rendering [21].

  Hành vi trên tiếng Việt — quan sát thực nghiệm trong luận văn này. Dù Qwen-Image mạnh trên chữ Hán và tiếng Anh, các thử nghiệm zero-shot trong luận văn cho thấy mô hình này có *rất ít kiến thức hình ảnh về ký tự tiếng Việt có dấu*. Cụ thể, ở cấp baseline chưa qua fine-tuning, Qwen-Image thường mắc lỗi *trước khi* có thể tách riêng bài toán dấu thanh tinh vi hơn:

  (i) Sai ở cấp ký tự — nhiều từ tiếng Việt bị sinh ra với nguyên âm sai (`ư/u`, `ơ/o`, `â/ă/a`), phụ âm cuối bị mất, hoặc cả từ bị biến dạng không nhận diện được.

  (ii) Sai ở cấp từ — với cùng một từ đơn lặp lại nhiều lần với các seed khác nhau, Qwen-Image thường cho ra các cách viết không thống nhất; số lượng ký tự sinh ra cũng dao động mạnh quanh số ký tự đúng.

  (iii) Yếu ở cấp dấu thanh và dấu phụ — ngay cả khi giữ được gần đúng phần gốc của từ, các dấu (`sắc`, `huyền`, `hỏi`, `ngã`, `nặng`) và dấu mũ trên nguyên âm (`â`, `ê`, `ô`) thường bị bỏ sót, đổi chỗ, hoặc thêm thừa.

  Hệ quả là Qwen-Image không thể dùng làm baseline đánh giá trực tiếp cho bài toán dấu tiếng Việt: tỉ lệ lỗi đã cao ngay ở cấp ký tự, nên metric "độ đúng dấu" gần như không mang thông tin phân biệt. Kết hợp với chi phí VRAM rất lớn (chỉ chạy được với `torch.bfloat16` + GPU cao cấp), đây là hai nguyên nhân thực nghiệm chính khiến luận văn chuyển sang Ideogram4 làm backbone thực nghiệm cuối cùng.

- ERNIE-Image (Baidu) là mô hình sinh ảnh đa phương thức kết hợp text encoder dựa trên Mistral3 / Ministral3 với đường mã hoá byte-level BPE [17, 21]. Đặc điểm byte-level BPE có lợi thế về tính tổng quát và khả năng biểu diễn các ngôn ngữ ít tài nguyên, nhưng với tiếng Việt — nơi thông tin ngữ nghĩa quan trọng nằm ở cấp âm tiết và dấu thanh — cách mã hoá này khiến việc căn chỉnh âm tiết với token trở nên khó khăn hơn so với BPE cấp chữ hoặc SentencePiece truyền thống.

  Trong các thử nghiệm ERNIE Image trước khi chuyển sang Ideogram4, một báo cáo tokenizer nội bộ cho thấy đường mã hóa này không ổn định với một phần ký tự tiếng Việt có dấu, đặc biệt ở dạng viết hoa đầu câu. Một số nguyên âm tiếng Việt viết hoa có dấu thanh hoặc dấu phụ, chẳng hạn `Ở`, `Ảnh`, `Ý`, `Ước`, `Nguyễn`, có thể bị phân rã thành byte thô thay vì được ánh xạ thành subword token ổn định. Khi decode lại, các trường hợp này có thể sinh ký tự thay thế hoặc chuỗi không còn giữ đúng hình thức ban đầu. Trên bộ 7.184 âm tiết tiếng Việt dùng trong dự án, thử nghiệm tokenizer ghi nhận các nhóm lỗi đáng kể ở cả một số dạng viết thường và nhiều dạng viết hoa đầu câu.

  Hệ quả thực tế là ERNIE Image không chỉ sai dấu theo nghĩa thị giác, mà còn có rủi ro mất thông tin ngay từ đường mã hóa văn bản. Khi token không giữ ổn định âm tiết tiếng Việt, mô hình khó học quan hệ giữa prompt và glyph thư pháp đúng. Trong một giai đoạn thử nghiệm trước, việc chuẩn hóa dữ liệu về chữ thường có thể giảm một phần lỗi tokenizer, nhưng cách này không phù hợp với mục tiêu ứng dụng vì thư pháp trang trọng thường cần chữ viết hoa đầu câu hoặc chữ hoa trong các cụm như `An Khang Thịnh Vượng`, `Nhẫn`, `Tâm`.

  So với hướng ERNIE Image đó, đường mã hóa văn bản trong bản triển khai Ideogram4 cuối ổn định hơn trên cùng bộ âm tiết tiếng Việt được kiểm tra. Kết quả này là một trong các lý do khiến luận văn không tiếp tục chọn ERNIE Image làm backbone chính, dù ERNIE Image nhìn chung tốt hơn Qwen Image ở khả năng giữ số lượng ký tự và đôi khi sinh đúng cụm từ tiếng Việt ngắn.

- Ideogram4 (Ideogram AI, công bố 2026-06-03) là mô hình *text-to-image mã nguồn mở đầu tiên* của Ideogram, được huấn luyện từ đầu chứ không phải fine-tune từ checkpoint có sẵn. Mô hình có 9.3B tham số và sử dụng kiến trúc fully single-stream Diffusion Transformer (34 lớp transformer) trong đó token văn bản và token ảnh được nối vào cùng một chuỗi thống nhất và xử lý chung bởi cùng một transformer — không có nhánh text/image tách rời. Text encoder là Qwen3-VL-8B-Instruct (đã xác nhận), trong đó hidden state được trích từ 13 lớp trung gian rồi concat lại để cung cấp đặc trưng ngữ nghĩa đa cấp. Mô hình dùng flow-matching thay vì DDPM thuần, và hỗ trợ độ phân giải bản địa 256–2048 [15]. Với 9.3B tham số, Ideogram4 đạt chất lượng render chữ tốt nhất trong số các mô hình open-weight được benchmark — vượt Qwen-Image (20B), FLUX.2 [dev] (32B), HunyuanImage 3.0 (80B MoE) — và thiết kế pipeline FP8 base + BF16 LoRA giúp fine-tune cục bộ khả thi. Đây là lý do Ideogram4 được chọn làm backbone thực nghiệm cuối cùng của luận văn.

Điểm chung đáng lưu ý nhất: cả ba mô hình đều sử dụng text encoder họ Qwen3-VL/Mistral3 (hoặc tương đương) làm nguồn conditioning, có nghĩa câu hỏi nghiên cứu ban đầu của luận văn về *tín hiệu ngôn ngữ trong họ Qwen* vẫn có giá trị bất kể backbone sinh ảnh nào được chọn.

### 1.4.4. Fine-tuning hiệu quả tham số

Full fine-tuning mô hình text-to-image lớn đắt đỏ và dễ phá hỏng năng lực base model. LoRA học các cập nhật hạng thấp trên một số module, trong khi giữ trọng số gốc frozen. Đây là lựa chọn phù hợp cho bài toán học phong cách thư pháp cụ thể nhưng vẫn muốn giữ khả năng sinh ảnh nền và prior thị giác của model gốc.

Nghiên cứu này cho thấy target module của LoRA rất quan trọng. Attention-only LoRA không đủ vượt plateau dấu tiếng Việt. Cần mở rộng sang feed-forward và adaLN modulation để tác động trực tiếp hơn tới hình học glyph.

### 1.4.5. Mô hình thương mại và khoảng trống nghiên cứu

Mô hình thương mại có thể tạo ảnh thư pháp đẹp nhưng thường không cho phép kiểm soát font, dữ liệu riêng, checkpoint, seed, hoặc pipeline đánh giá. Trong bản luận văn đã được duyệt trước đây, các công cụ như Nano Banana 2 và GPT Image 1.5 High được dùng như hệ thống so sánh thương mại: chúng có thể tạo ảnh thư pháp tiếng Việt nhìn hấp dẫn, nhưng hoạt động như black-box với phong cách mặc định và không có cơ chế fine-tune trên một font thư pháp riêng. Vì vậy, chúng là đối thủ tốt để so sánh trực quan, nhưng không thay thế được một pipeline fine-tuning có thể tái lập.

Font máy là một baseline quan trọng khác. Cách này đúng chính tả và dấu vì chữ được render trực tiếp từ file font. Tuy nhiên, ảnh font máy mang tính tĩnh: không có biến thiên mực, lực nhấn, hiệu ứng khô bút, hay tương tác hữu cơ giữa các nét. Do đó, font máy mạnh về độ đúng văn bản nhưng yếu về độ tự nhiên thư pháp.

Các mô hình open-weight cũng là một phần của bức tranh đối thủ. Qwen Image là hướng đăng ký ban đầu của luận văn, nhưng trong thực nghiệm tốn VRAM và yếu ở baseline tiếng Việt. ERNIE Image tốt hơn Qwen Image ở một số điểm, thường giữ được gần đúng số ký tự và đôi khi sinh đúng cụm từ tiếng Việt ngắn. Tuy nhiên, ERNIE Image vẫn sai nhiều ở cấp ký tự và bước ERNIE Image trước đó cho thấy đường mã hóa Mistral3/Ministral3 byte-level BPE gây khó căn chỉnh âm tiết với dấu. Ideogram4 gốc là nền tảng open mạnh nhất trong dự án này, nhưng vẫn nhầm các dấu tiếng Việt khó và không tự giải quyết tốt ảnh nhiều từ.

Bảng 1.1 tóm tắt các đối thủ và baseline chính.

| Hệ thống / phương pháp | Điểm mạnh | Hạn chế với luận văn này |
|---|---|---|
| Render bằng font máy | Đúng chữ và dấu tuyệt đối | Glyph tĩnh; thiếu biến thiên nét bút tự nhiên |
| Công cụ thương mại black-box | Chất lượng hình ảnh cao; prompt tiện | Phong cách mặc định; không fine-tune dữ liệu riêng; khó tái lập |
| Qwen Image | Hướng đăng ký ban đầu; thuộc họ model mạnh | Tốn VRAM; baseline tiếng Việt yếu trong dự án này |
| ERNIE Image | Tốt hơn Qwen Image về số lượng ký tự; đôi khi đúng cụm từ | Vẫn sai ký tự nhiều; byte-level BPE làm khó căn chỉnh âm tiết/dấu |
| Ideogram4 gốc | Nền tảng open text-rendering tốt nhất trong các hướng thử | Vẫn sai dấu khó và chưa tốt ở bố cục nhiều từ |
| Pipeline Ideogram4 DiT-LoRA đề xuất | Học font cụ thể, tái lập được, cải thiện một từ và compound | Hiện còn giới hạn ở một font chính và panel chấm thủ công |

Khoảng trống nghiên cứu là xây dựng một pipeline fine-tuning có thể tái lập, học phong cách font cụ thể, và giữ độ chính xác dấu tiếng Việt trong cả một từ lẫn bố cục nhiều từ.

Trong bản luận văn hoàn thiện, Hình 1.2 so sánh trực quan cùng một cụm từ tiếng Việt qua các hướng đại diện: font máy, Qwen Image, ERNIE Image, Ideogram4 gốc và checkpoint đề xuất.

<!-- Figure placeholder: copy the final competitor comparison panel to docs/thesis/figures/fig_1_2_competitor_baseline_comparison.png -->
![Hình 1.2. So sánh trực quan giữa các phương pháp đối thủ và checkpoint đề xuất.](figures/fig_1_2_competitor_baseline_comparison.png)

**Hình 1.2.** So sánh trực quan giữa các phương pháp đối thủ và checkpoint đề xuất. Trình bày kết quả render cùng một cụm từ tiếng Việt qua font máy chữ số, công cụ thương mại black-box, Qwen Image, ERNIE Image, Ideogram4 gốc và checkpoint LoRA đề xuất để làm nổi bật độ chính xác dấu và chất lượng nét bút.

Ngoài hình so sánh hệ sinh thái đó, Hình 1.3 tách riêng năng lực gốc của ba mô hình open-weight được khảo sát: Qwen Image, ERNIE Image và Ideogram4 trước fine-tuning. Hình này dùng hai prompt giống nhau cho cả ba mô hình: một prompt không có chữ để so sánh năng lực dựng cảnh/nền/thiết kế, và một prompt thư pháp tiếng Việt để so sánh khả năng text rendering zero-shot. Cách tách riêng này giúp làm rõ rằng Ideogram4 không chỉ được chọn vì viết tiếng Việt tốt hơn, mà còn vì nó cân bằng hơn giữa chất lượng ảnh, khả năng render chữ và tính khả thi của fine-tuning cục bộ.

<!-- Figure placeholder: copy the final base model comparison to docs/thesis/figures/fig_1_3_base_model_capability_comparison.png -->
![Hình 1.3. So sánh năng lực gốc của Qwen Image, ERNIE Image và Ideogram4 trước fine-tuning.](figures/fig_1_3_base_model_capability_comparison.png)

**Hình 1.3.** So sánh năng lực gốc của Qwen Image, ERNIE Image và Ideogram4 trước fine-tuning. Panel trên hiển thị năng lực sinh ảnh nền không có chữ, panel dưới hiển thị năng lực render chữ zero-shot trên prompt thư pháp tiếng Việt.

## 1.5. Phương pháp đề xuất và đóng góp

Đóng góp trung tâm của luận văn là:

> Một bản triển khai Ideogram4 DiT-LoRA cho mục tiêu đăng ký Qwen Image về sinh ảnh thư pháp tiếng Việt, được tổ chức như pipeline glyph-binding nhằm cải thiện độ chính xác dấu.

Đóng góp này được trình bày như một pipeline tích hợp, không phải danh sách nhiều mẹo rời rạc. Các thành phần như probe tín hiệu, chọn target LoRA, checkpoint averaging, compound training và đánh giá thủ công chỉ có ý nghĩa khi kết hợp với nhau để giải quyết cùng một bottleneck.

<!-- Figure placeholder: copy the final pipeline diagram to docs/thesis/figures/fig_1_1_research_pipeline.png -->
![Hình 1.1. Pipeline tổng thể cho sinh ảnh thư pháp tiếng Việt.](figures/fig_1_1_research_pipeline.png)

**Hình 1.1.** Pipeline tổng thể cho sinh ảnh thư pháp tiếng Việt. Biểu đồ tóm tắt quy trình từ khảo sát baseline, probe tín hiệu, huấn luyện wide-target LoRA, checkpoint souping, xây dựng dữ liệu compound và đánh giá Eval28.

So với hướng Qwen-Image ban đầu, đóng góp này khác ở một số điểm quan trọng. Backbone cuối cùng là Ideogram4 thay vì Qwen-Image, bottleneck chính được xác định là glyph binding của DiT thay vì text encoder mù dấu toàn cục, và mục tiêu cuối được mở rộng từ đọc đúng một từ riêng lẻ sang hành vi thư pháp nhiều từ/câu dài.

## 1.6. Cấu trúc luận văn

Chương 2 trình bày cơ sở lý thuyết về GAN, diffusion, Diffusion Transformer, Ideogram4, LoRA, FP8/BF16 và đặc điểm thư pháp tiếng Việt. Chương 3 mô tả triển khai, dữ liệu, probe chẩn đoán, cấu hình fine-tuning và kết quả đánh giá. Chương 4 tổng kết giá trị lý thuyết/thực tiễn, hạn chế và hướng phát triển.

---

# 2. Cơ sở lý thuyết

## 2.1. Sinh ảnh bằng AI: từ GAN tới Diffusion Transformer

GAN mở ra hướng học sinh ảnh qua đối kháng, nhưng khó kiểm soát độ chính xác chữ. DDPM và các mô hình diffusion cải thiện độ ổn định bằng cách học đảo ngược quá trình thêm nhiễu. Latent Diffusion giảm chi phí tính toán bằng cách vận hành trong không gian latent. Diffusion Transformer tiếp tục thay thế backbone tích chập bằng transformer, phù hợp hơn với các nhiệm vụ cần tích hợp văn bản và ảnh.

Trong bài toán thư pháp tiếng Việt, Diffusion Transformer phải làm nhiều việc cùng lúc: hiểu prompt, giữ phong cách thư pháp, đặt chữ trong bố cục, và render dấu nhỏ chính xác. Đây là lý do lỗi không thể chỉ được hiểu là lỗi OCR hoặc lỗi font; nó là lỗi binding giữa conditioning ngôn ngữ và hình học thị giác.

## 2.2. Kiến trúc Ideogram4

Pipeline Ideogram4 trong luận văn gồm ba thành phần chính (Bảng 2.1).

**Bảng 2.1.** Tổng quan về ba thành phần chính trong pipeline Ideogram4.

| Thành phần | Vai trò | Ý nghĩa với thư pháp tiếng Việt |
|---|---|---|
| Qwen3-VL text encoder | Mã hóa prompt và structured prompt | Cung cấp conditioning có hiểu tiếng Việt |
| Ideogram4 DiT | Sinh latent ảnh theo conditioning | Mục tiêu LoRA chính |
| VAE | Giải mã latent thành ảnh 1024×1024 | Tạo ảnh cuối |

<!-- Figure placeholder: copy the final architecture diagram to docs/thesis/figures/fig_2_1_ideogram4_architecture.png -->
![Hình 2.1. Kiến trúc Ideogram4 dùng trong luận văn.](figures/fig_2_1_ideogram4_architecture.png)

**Hình 2.1.** Kiến trúc Ideogram4 dùng trong luận văn. Pipeline fine-tuning đóng băng text encoder Qwen3-VL và trọng số base Ideogram4, chèn adapter LoRA vào các module DiT được chọn, và giải mã ảnh latent sinh ra qua VAE.

Text encoder Qwen3-VL được giữ frozen. Ideogram4 lấy tín hiệu từ nhiều layer tap của text encoder, concat thành vector conditioning lớn rồi chiếu qua `llm_cond_norm + llm_cond_proj` vào DiT. Probe của luận văn kiểm tra cả không gian tap lẫn không gian projection để xác định tín hiệu dấu còn tồn tại tới đâu.

<!-- Figure placeholder: copy the final conditioning diagram to docs/thesis/figures/fig_2_2_qwen3vl_multilayer_conditioning.png -->
![Hình 2.2. Conditioning Qwen3-VL nhiều layer đưa vào DiT.](figures/fig_2_2_qwen3vl_multilayer_conditioning.png)

**Hình 2.2.** Conditioning Qwen3-VL nhiều layer đưa vào DiT. Hidden state từ nhiều layer Qwen3-VL được concat, chuẩn hóa và chiếu trước khi vào DiT Ideogram4. Hình này hỗ trợ cho câu hỏi chẩn đoán xem tín hiệu dấu tiếng Việt đã yếu sẵn ở giao diện conditioning hay bị mất sau đó trong quá trình glyph binding.

DiT là nơi LoRA được chèn. Các module quan trọng gồm attention, feed-forward và adaLN modulation. Khi attention-only LoRA không đủ, việc mở rộng target cho thấy DiT cần được tác động ở nhiều kênh hơn để sửa hình học chữ.

## 2.3. Kỹ thuật fine-tuning hiệu quả

LoRA giả sử cập nhật trọng số có thể xấp xỉ bằng tích hai ma trận hạng thấp. Nếu trọng số gốc là `W`, LoRA thêm cập nhật:

```text
W' = W + B A * scale
```

Trong luận văn, base weight FP8 được giữ frozen, còn nhánh LoRA hoạt động ở BF16. Điều này cho phép training trong giới hạn VRAM nhưng vẫn tạo gradient ổn định cho adapter. Sau training, checkpoint cần được chuyển đổi sang định dạng inference đúng scale.

Checkpoint averaging được dùng khi nhiều checkpoint LoRA cùng vùng tối ưu nhưng có lỗi khác nhau. Trung bình hóa adapter có thể giảm phương sai và tạo checkpoint ổn định hơn mà không tăng chi phí inference.

## 2.4. Thư pháp tiếng Việt: đặc điểm thị giác và ngôn ngữ

Thư pháp chữ Quốc ngữ kết hợp bảng chữ cái Latin, hệ thống dấu tiếng Việt và bút pháp thư pháp. Một từ có thể có nhiều tầng dấu: dấu mũ hoặc dấu râu trên nguyên âm, cộng thêm dấu thanh. Các lỗi thường gặp bao gồm mất dấu thanh, đổi dấu hỏi/ngã, thêm dấu nặng, đổi nguyên âm như `ư/u`, `ơ/o`, `â/ă/a`, hoặc sai phụ âm và mất ký tự khi ảnh chứa nhiều từ.

Về Unicode, tiếng Việt có thể được biểu diễn ở dạng composed hoặc decomposed. Tokenizer cũng có thể xử lý chữ hoa và chữ thường khác nhau. Vì vậy, luận văn không chỉ dựa vào ký tự nhìn thấy mà kiểm tra coverage ở cấp token ID. Tập compound cuối bao phủ 406 token ID tiếng Việt có dấu.

---

# 3. Triển khai và đánh giá

## 3.1. Thiết lập hệ thống

### 3.1.1. Cấu hình phần cứng

| Thành phần | Cấu hình |
|---|---|
| GPU | GPU có VRAM đủ cho Ideogram4 FP8 và LoRA training |
| Precision | FP8 base weights, BF16 LoRA |
| Storage | Lưu checkpoint, dữ liệu render, artifact probe và kết quả đánh giá |
| Runtime | Python, PyTorch, DiffSynth-Studio, safetensors |

### 3.1.2. Môi trường phần mềm

| Thành phần | Vai trò |
|---|---|
| DiffSynth-Studio | Load và inference Ideogram4 |
| PyTorch | Training LoRA |
| safetensors | Lưu checkpoint |
| HuggingFace Transformers | Tokenizer/text encoder |
| Scripts thí nghiệm | Build dataset, train, convert, render, probe |

### 3.1.3. Quy trình thí nghiệm tái lập

Mỗi nhánh thí nghiệm được quản lý bằng tên checkpoint riêng, log riêng và folder render riêng. Các panel đánh giá dùng cùng seed base để tránh nhầm cải thiện checkpoint với biến thiên seed. Khi một checkpoint tốt được phát hiện, nó không ghi đè checkpoint trước mà được giữ như một candidate độc lập.

## 3.2. Dữ liệu và tiền xử lý

### 3.2.1. Dữ liệu một từ

Giai đoạn đầu dùng danh sách từ tiếng Việt có dấu để tạo ảnh một từ. Panel fragile 60 được dùng để đo các từ dễ sai trong quá trình fine-tuning. Người đánh giá thủ công quyết định từ nào đúng/sai dựa trên ảnh sinh ra, ưu tiên đúng chính tả và dấu hơn các chỉ số tự động như SSIM.

### 3.2.2. Token coverage

Token coverage được kiểm tra ở cấp tokenizer vì cùng một ký tự có thể map khác nhau ở chữ hoa và chữ thường. Enumeration trên lexicon cho thấy cần bao phủ 406 token ID tiếng Việt có dấu. Điều này dẫn tới việc xây dựng dữ liệu compound bao phủ cả uppercase và lowercase, thay vì giả định chỉ cần một dạng.

### 3.2.3. Tập dữ liệu compound

Tập compound gồm ảnh 4/5/7/8 từ, chia hai dòng và căn giữa. Dữ liệu được render từ font Thu Phap Thanh Cong Unicode để tạo target ổn định. Tổng tập compound chính gồm:

```text
2808 ảnh compound
147 ảnh single-word bổ trợ
2955 bản ghi metadata
406/406 token ID tiếng Việt có dấu được bao phủ
```

Thiết kế compound bắt nguồn từ quan sát rằng model sinh nhiều từ tốt hơn khi được train trực tiếp trên nhiều từ. Đây là pivot quan trọng của luận văn: thay vì chỉ tối ưu một từ rồi hy vọng tự generalize sang câu, pipeline đưa bố cục nhiều từ vào training distribution.

### 3.2.4. Prompt không dùng bounding box

Các thử nghiệm bbox cho thấy bbox hẹp theo cell hoặc bbox dòng không luôn được model tuân thủ tốt; một số từ bị trùng vị trí hoặc vẫn sai. Trong khi đó, prompt không bbox với ảnh target căn giữa cho phép model học quy luật bố cục tự nhiên hơn. Vì vậy compound bridge dùng no-bounding-box prompt và để dữ liệu target dạy model cách căn dòng.

<!-- Figure placeholder: copy the final layout comparison to docs/thesis/figures/fig_3_2_no_bbox_vs_bbox_layout_comparison.png -->
![Hình 3.2. So sánh bố cục có bounding box và không dùng bounding box cho prompt compound.](figures/fig_3_2_no_bbox_vs_bbox_layout_comparison.png)

**Hình 3.2.** So sánh bố cục có bounding box và không dùng bounding box cho prompt compound. Cùng một prompt sáu từ được render dưới các điều kiện không bbox, wide-bbox, và row-bbox nhằm minh họa lý do dữ liệu compound cuối cùng dựa trên ảnh target căn giữa hơn là các chỉ dẫn bounding box cứng nhắc.

<!-- Figure placeholder: copy the final font comparison to docs/thesis/figures/fig_3_3_font_reference_vs_model_generation.jpg -->
![Hình 3.3. So sánh giữa ảnh target render từ font máy và bố cục do model tự sinh.](figures/fig_3_3_font_reference_vs_model_generation.jpg)

**Hình 3.3.** So sánh giữa ảnh target render từ font máy và bố cục do model tự sinh. So sánh xác nhận rằng dữ liệu target render từ font thư pháp đủ gần với bố cục căn giữa hai dòng tự nhiên của model để dùng làm dữ liệu giám sát trong quá trình fine-tuning, đồng thời đảm bảo ký tự và dấu tiếng Việt chính xác.

## 3.3. Các probe chẩn đoán

### 3.3.1. Qwen3-VL signal probe

Mục tiêu của probe là kiểm tra liệu Qwen3-VL có làm mất tín hiệu dấu tiếng Việt trước khi DiT nhận conditioning hay không. Probe dùng danh sách canonical 990 từ từ `manifest_words.json`, so với lexicon candidate 7165 từ từ bộ 7184. Kết quả:

```text
990 vs 7165
elapsed = 81.1s
span_miss = 0
very_hard = 42
hard = 225
medium = 567
easy = 156
```

Kết luận: Qwen3-VL không mù dấu toàn cục. Tuy nhiên, có một cụm hard/very-hard đủ lớn để dùng Qwen signal như bộ lọc rủi ro.

### 3.3.2. Probe họ Cưu/Cừu/Cữu

Do các vòng train có hiện tượng `Cưu` và `Cữu` bị sinh thành `Cừu`, một probe nhỏ trên 6 thanh được chạy:

```text
Cưu, Cừu, Cứu, Cửu, Cữu, Cựu
```

Probe đo cả tap-space và proj-space sau `llm_cond_norm + llm_cond_proj`. Kết quả không ủng hộ mạnh giả thuyết `Cừu` là attractor toàn cục trong Qwen conditioning. Điều này làm nghiêng kết luận về DiT/glyph-binding hoặc visual prior sâu hơn.

### 3.3.3. Diễn giải probe

Các probe cho thấy không nên chuyển toàn bộ chiến lược sang train text encoder. Một số từ có tín hiệu conditioning gần nhau và có thể dùng để mining, nhưng nhiều lỗi ảnh xảy ra dù tín hiệu Qwen vẫn phân biệt được. Với các từ đó, hướng phù hợp hơn là sửa DiT/glyph binding bằng LoRA và dữ liệu visual đúng phân phối.

## 3.4. Cấu hình fine-tuning

### 3.4.1. Attention-only baseline

Attention-only LoRA được thử trước vì ít rủi ro và giữ base model tốt. Tuy nhiên, kết quả mắc plateau:

```text
Khoảng 32-39/60 trên fragile panel
```

Điều này cho thấy lỗi dấu tiếng Việt không chỉ nằm ở luồng attention, mà cần can thiệp vào các phần tác động mạnh hơn tới hình học nét.

### 3.4.2. Wide-target DiT-LoRA

Wide-target LoRA mở rộng target sang:

```text
attention.qkv
attention.o
feed_forward.w1
feed_forward.w2
feed_forward.w3
adaln_modulation
```

<!-- Figure placeholder: copy the final LoRA target diagram to docs/thesis/figures/fig_3_1_widetarget_lora_injection_points.png -->
![Hình 3.1. Các điểm chèn wide-target LoRA trong Ideogram4 DiT.](figures/fig_3_1_widetarget_lora_injection_points.png)

**Hình 3.1.** Các điểm chèn wide-target LoRA trong Ideogram4 DiT. Các module đích bao gồm attention projections (`qkv`, `o`), feed-forward layers (`w1`, `w2`, `w3`), và adaLN modulation.

Kết quả tốt nhất của checkpoint đơn đạt 48/60. Các vòng warm-continue có phương sai lớn: có vòng tăng, có vòng tụt mạnh như wide8.

### 3.4.3. Gentle warm-continue

Gentle warm-continue dùng learning rate nhẹ và warm từ checkpoint tốt. Tuy nhiên, kết quả cho thấy không nên chain dài mù quáng. Thay vào đó, mỗi nhánh nên được xem như một sample độc lập trong vùng tốt; nếu đạt gate thì đưa vào candidate soup.

### 3.4.4. Checkpoint averaging

Checkpoint averaging tạo `soup567` từ các checkpoint tốt r5, r6, r7:

```text
soup567 = mean(r5, r6, r7)
```

Kết quả `soup567` đạt 52/60, vượt mọi checkpoint đơn trước đó. Thử thêm r9 tạo `soup5679` vẫn đạt 52/60, không cải thiện nhưng giữ ổn định.

Với compound bridge, checkpoint averaging tiếp tục được dùng:

```text
soup_e4r2r3
soup_e4r2r3r4
compound_lr3e5_from_gold4
soup_lr3e5_gold4_9to1
```

Checkpoint `soup_e4r2r3r4` là mốc Gold4 ổn định, đạt 6 lỗi trên Eval28. Từ mốc này, nhánh follow-up learning rate `3e-5` đạt 5 lỗi; soup nhẹ 90% nhánh `lr3e5` + 10% Gold4 đạt 4 lỗi và được chọn làm checkpoint compound cuối.

## 3.5. Giao thức đánh giá

### 3.5.1. Đánh giá thủ công cấp từ

Vì OCR và metric tự động không đủ tin cậy cho thư pháp có dấu, đánh giá chính là thủ công. Một từ được coi là đúng nếu người đọc thấy đúng ký tự, đúng dấu, và không có dấu thừa làm đổi nghĩa. SSIM có thể dùng tham khảo nhưng không phải metric quyết định.

### 3.5.2. Fragile panel một từ

Fragile panel 60 gồm các từ khó thường sai dấu hoặc sai ký tự. Panel này được dùng để đo tiến triển single-word. Seed base được cố định để tránh seed confound.

### 3.5.3. Compound Eval28

Compound Eval28 gồm 28 ảnh, mỗi ảnh chứa nhiều từ, tổng cộng 168 từ. Panel này phù hợp với mục tiêu cuối của dự án hơn một từ vì nó kiểm tra layout, spacing, cỡ chữ, và khả năng giữ dấu khi có nhiều token cạnh tranh trong cùng ảnh.

### 3.5.4. Kiểm tra khả năng giữ năng lực sinh ảnh nền

Ngoài độ đúng chữ, luận văn cũng cần kiểm tra xem LoRA có làm suy giảm năng lực gốc của Ideogram4 hay không. Vì vậy, bản luận văn hoàn thiện nên có một panel prompt giàu ngữ cảnh, trong đó chữ thư pháp tiếng Việt được đặt trong các cảnh như thiệp Tết, tranh mực, banner lễ hội hoặc bố cục poster. Panel này không thay thế Eval28, nhưng bổ sung một góc nhìn thực dụng: checkpoint tốt phải vừa cải thiện chữ, vừa không làm nghèo nền ảnh, bố cục, ánh sáng và chất liệu so với base model.

## 3.6. Kết quả

### 3.6.1. Kết quả một từ

| Checkpoint / giai đoạn | Số từ đúng / 60 |
|---|---:|
| Attention-only plateau | 32-39 |
| Wide-target r3 | 43 |
| Wide-target r4 | 41 |
| Wide-target r5 | 46 |
| Wide-target r6 | 48 |
| Wide-target r7 | 47 |
| Wide-target r8 | 38 |
| `soup567` | **52** |
| `soup5679` | **52** |

Kết quả cho thấy wide-target đã phá plateau attention-only, nhưng warm-continue có phương sai cao. Checkpoint soup là hướng ổn định hơn checkpoint đơn.

<!-- Figure placeholder: copy the final progression chart to docs/thesis/figures/fig_3_4_result_progression.png -->
![Hình 3.4. Tiến trình từ baseline compound tới compound bridge.](figures/fig_3_4_result_progression.png)

**Hình 3.4.** Tiến trình từ baseline compound tới compound bridge. Đồ thị thể hiện tỉ lệ chính xác trên panel một từ fragile-60 (tăng từ 32-39 lên 52/60 đúng) và số lượng lỗi trên panel compound Eval28 (giảm từ 56/168 lỗi xuống 4/168 lỗi).

### 3.6.2. Hành vi phương sai cao

Wide8 là trường hợp quan trọng vì dùng cùng công thức chung nhưng tụt mạnh xuống 38/60 và xuất hiện nhiều dấu nặng thừa. Log không cho thấy checkpoint hỏng hoặc NaN. Cách giải thích hợp lý hơn là subsystem dấu nằm gần biên phương sai cao; một update bình thường có thể đẩy model sang vùng xấu. Điều này củng cố lý do dùng soup và gate thủ công.

### 3.6.3. Kết quả compound bridge

| Checkpoint | Lỗi / 168 |
|---|---:|
| `soup567` baseline | 56 |
| `compound_bridge` e4 | 26 |
| `compound_bridge_r2` | 19 |
| `compound_bridge_r3` | 18 |
| `soup_e4r2r3` | 13 |
| `r4_from_soup` | 15 |
| `soup_e4r2r3r4` / Gold4 | 6 |
| `compound_lr3e5_from_gold4` | 5 |
| `soup_lr3e5_gold4_3to1` | 6 |
| `soup_lr3e5_gold4_9to1` | **4** |

Quá trình sau Gold4 cho thấy learning rate và tỉ lệ soup đều quan trọng. Nhánh `5e-5` trước đó học quá mạnh và tạo nhiều lỗi mới; `2e-5` giữ mức 6 lỗi nhưng không vượt mốc; `3e-5` là vùng tốt hơn, giảm solo xuống 5 lỗi. Khi soup với Gold4, tỉ lệ 75/25 kéo một số lỗi cũ quay lại, còn tỉ lệ 90/10 giữ phần lớn cải thiện của nhánh mới và chỉ dùng Gold4 như regularizer nhẹ.

<!-- Figure placeholder: copy the final compound comparison to docs/thesis/figures/fig_3_6_compound_eval28_before_after.png -->
![Hình 3.6. So sánh compound Eval28 trước/sau giữa `soup567` và checkpoint cuối `soup_lr3e5_gold4_9to1`.](figures/fig_3_6_compound_eval28_before_after.png)

**Hình 3.6.** So sánh compound Eval28 trước/sau giữa `soup567` và checkpoint cuối `soup_lr3e5_gold4_9to1`. Các mẫu prompt Eval28 cố định seed được hiển thị cho `soup567` và checkpoint compound cuối cùng `soup_lr3e5_gold4_9to1`, minh họa việc giảm từ 56/168 lỗi xuống còn 4/168 lỗi trên ảnh thư pháp nhiều từ.

Checkpoint compound cuối:

```text
experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/step-soup_infer.safetensors
```

Các lỗi còn lại trên Eval28:

```text
Hấn -> Hẩn
Chịt -> Chút
Huyên -> Huyện
Dôi -> Dồi
```

<!-- Figure placeholder: copy the final remaining errors to docs/thesis/figures/fig_3_7_remaining_error_cases.png -->
![Hình 3.7. Các lỗi còn lại của checkpoint compound gold hiện tại.](figures/fig_3_7_remaining_error_cases.png)

**Hình 3.7.** Các lỗi còn lại của checkpoint compound gold hiện tại. Bốn lỗi còn lại của Eval28 tập trung vào các biến thể nguyên âm hoặc dấu tiếng Việt gần nhau, không phải là collapse toàn cục.

### 3.6.4. Checkpoint gold hiện tại

Single-word gold:

```text
experiments/checkpoints/coverage_v10_widetarget_soup567/step-soup_infer.safetensors
```

<!-- Figure placeholder: copy the final single word examples to docs/thesis/figures/fig_3_5_single_word_before_after_examples.png -->
![Hình 3.5. Ví dụ trước/sau trên các từ tiếng Việt có dấu khó.](figures/fig_3_5_single_word_before_after_examples.png)

**Hình 3.5.** Ví dụ trước/sau trên các từ tiếng Việt có dấu khó. Sáu ví dụ một từ ở seed 7000 (`Bậc`, `Cữu`, `Chưởng`, `Gẫy`, `Ghế`, `Huyên`) minh họa cải tiến giữa các giai đoạn và checkpoint soup.

Compound gold:

```text
experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/step-soup_infer.safetensors
```

Cả hai checkpoint gold đã được đăng tải công khai trên Hugging Face tại [phong09021998/vietnamese-calligraphy-ideogram4-lora](https://huggingface.co/phong09021998/vietnamese-calligraphy-ideogram4-lora) để hỗ trợ cộng đồng truy cập và tái lập nghiên cứu.

<!-- Figure placeholder: copy the final base model scene capability comparison to docs/thesis/figures/fig_3_8_calligraphy_with_base_model_capability.png -->
![Hình 3.8. Kiểm tra checkpoint fine-tune khi kết hợp chữ thư pháp với năng lực sinh ảnh nền của Ideogram4.](figures/fig_3_8_calligraphy_with_base_model_capability.png)

**Hình 3.8.** Kiểm tra checkpoint fine-tune khi kết hợp chữ thư pháp với năng lực sinh ảnh nền của Ideogram4. Hai prompt tĩnh vật Tết so sánh Ideogram4 base, checkpoint soup một từ `soup567` và checkpoint compound cuối cùng `soup_lr3e5_gold4_9to1`. Hình kiểm tra xem độ chính xác chữ viết tiếng Việt có thể cải thiện trong khi vẫn giữ nguyên bố cục, ánh sáng, kết cấu giấy và các yếu tố nền trang trí hay không.

---

# 4. Kết luận

## 4.1. Giá trị lý thuyết và thực tiễn

### 4.1.1. Giá trị lý thuyết

Luận văn cho thấy lỗi dấu tiếng Việt trong thiết lập Ideogram4 không nên bị đơn giản hóa thành lỗi text encoder mù dấu. Qwen3-VL vẫn giữ tín hiệu dấu ở nhiều trường hợp. Vấn đề chính là DiT không luôn gắn tín hiệu đó vào hình học glyph đúng. Phân biệt này quan trọng vì nó thay đổi chiến lược: thay vì train text encoder toàn cục, hướng hiệu quả hơn là điều chỉnh DiT ở các module có ảnh hưởng tới hình học chữ.

Luận văn cũng cho thấy một từ và nhiều từ là hai bài toán liên quan nhưng không đồng nhất. Một checkpoint có thể viết khá tốt khi chỉ có một từ, nhưng thất bại khi cần viết nhiều từ trong hai dòng. Do đó, nếu mục tiêu cuối là câu dài, training trực tiếp trên bố cục nhiều từ là cần thiết.

### 4.1.2. Giá trị thực tiễn

Luận văn cung cấp một pipeline kỹ thuật có thể tái lập cho sinh ảnh thư pháp tiếng Việt. Pipeline này bao gồm tạo dữ liệu từ font mục tiêu, kiểm tra coverage ở cấp tokenizer, fine-tune LoRA, chuyển đổi checkpoint, trung bình hóa checkpoint, render đánh giá bằng seed cố định và kiểm tra thủ công ở cấp từ. Các thành phần này được giữ tái lập vì bài toán rất nhạy với seed, checkpoint và định dạng prompt.

Kết quả giảm lỗi compound từ 56/168 xuống 4/168 cho thấy pipeline có khả năng tạo checkpoint thực dụng hơn cho mục tiêu sinh thư pháp nhiều từ. Giá trị của nó không chỉ nằm ở một thí nghiệm riêng lẻ, mà còn ở việc tạo nền tảng cho các ứng dụng thiết kế đồ họa, ảnh chúc mừng cá nhân hóa, bảo tồn văn hóa số và giáo dục thư pháp.

## 4.2. Tóm tắt kết quả chính

Các thí nghiệm cho thấy một tiến trình cải thiện rõ ràng. Attention-only LoRA mắc plateau khoảng 32-39/60, chứng tỏ các module attention riêng lẻ không đủ để giải quyết bottleneck dấu tiếng Việt. Khi mở rộng target LoRA sang nhiều module DiT hơn, checkpoint đơn tốt nhất tăng lên 48/60. Sau đó, trung bình hóa các checkpoint tương thích tạo ra `soup567`, đạt 52/60.

Kết quả quan trọng tiếp theo là hiệu năng một từ không tự động chuyển sang ảnh nhiều từ. Checkpoint gold một từ vẫn tạo nhiều lỗi khi viết bố cục hai dòng. Tuy nhiên, sau compound bridge training, follow-up learning rate `3e-5`, và soup nhẹ 90/10 với mốc Gold4, lỗi trên Eval28 giảm từ 56/168 xuống 4/168. Các lỗi còn lại tập trung ở một số cặp dấu khó, không còn là collapse toàn cục.

## 4.3. Hạn chế hiện tại

Công trình hiện tại vẫn có một số hạn chế. Hạn chế gần nhất là chi phí đánh giá thủ công: metric chính dựa vào người kiểm tra ảnh, đáng tin cậy trong miền thư pháp nhưng chậm. Panel Eval28 đủ hữu ích cho quá trình lặp nhanh, và seed 7000 giúp so sánh công bằng sau khi phát hiện seed confound, nhưng các kết luận thống kê mạnh hơn vẫn cần benchmark lớn hơn và nhiều seed hơn.

Phạm vi phong cách cũng còn hẹp. Thí nghiệm tập trung vào Thu Phap Thanh Cong Unicode, nên khả năng generalization sang các font khác cần được kiểm tra thêm. Dữ liệu training hiện được sinh từ font thư pháp số; cách này rất hữu ích để kiểm soát chính tả và dấu tiếng Việt, nhưng chưa thể bao quát đầy đủ chất liệu của thư pháp viết tay thật, như kết cấu giấy, độ loang mực, lực nhấn không hoàn hảo và bố cục cá nhân của nghệ nhân.

Phạm vi nội dung cũng cần mở rộng. Compound 4/5/7/8 từ gần mục tiêu cuối hơn một từ, nhưng các câu tiếng Việt tự nhiên vẫn cần panel riêng. Ngoài ra, dù LoRA giúp giảm rủi ro phá base model bằng cách giữ trọng số gốc frozen, vẫn cần kiểm tra rộng hơn khả năng sinh ảnh nền và hành vi của adapter trên các prompt ngoài thư pháp.

## 4.4. Hướng phát triển tương lai

Vì pipeline đề xuất đã đạt kết quả mạnh trên panel compound hiện tại, hướng phát triển tiếp theo không nên chỉ là tiếp tục tìm checkpoint tốt hơn trên cùng một panel. Thay vào đó, các bước sau luận văn nên tập trung vào mở rộng phạm vi, chuẩn hóa đánh giá và đưa hệ thống gần hơn tới ứng dụng thực tế.

Một hướng quan trọng là mở rộng sang nhiều font thư pháp. Luận văn hiện tập trung vào Thu Phap Thanh Cong Unicode, nhưng một hệ thống thư pháp thực dụng cần hỗ trợ nhiều phong cách chữ Quốc ngữ khác nhau. Việc xây dựng dữ liệu cho các font bổ sung sẽ cho phép kiểm tra liệu cùng pipeline DiT-LoRA có thể học nhiều phong cách nét bút mà vẫn giữ chính xác dấu tiếng Việt hay không. Đây là bước chuyển từ mô hình một phong cách sang hệ thống sinh thư pháp có thể điều khiển theo font.

Hướng thứ hai là học từ ảnh thư pháp thật. Dữ liệu font máy cung cấp giám sát chính tả ổn định, nhưng tác phẩm do nghệ nhân viết thật chứa động lực nét bút phong phú hơn, độ loang mực, kết cấu giấy, lực nhấn không hoàn hảo và bố cục tự nhiên hơn. Một chiến lược thực tế là dùng dữ liệu render từ font để ổn định chữ và dấu, sau đó bổ sung ảnh thư pháp thật đã tuyển chọn để tăng chất nghệ thuật mà vẫn giữ độ đúng văn bản.

Hệ thống cũng cần được mở rộng sang nội dung tiếng Việt dài và tự nhiên hơn. Dữ liệu compound đã vượt khỏi bài toán một từ, nhưng ứng dụng thực tế thường cần lời chúc, tên riêng, khẩu hiệu hoặc câu thơ ngắn. Vì vậy, nghiên cứu tiếp theo nên đánh giá các cụm 9-16 từ, có xuống dòng và quy tắc bố cục gần hơn với tác phẩm thư pháp thật.

Về đánh giá và triển khai, cần xây dựng benchmark lớn hơn, nhiều seed hơn và có khả năng phản ánh nhiều kiểu câu hơn. Một evaluator nhẹ có thể giúp phát hiện mất dấu, đổi dấu, thừa dấu hoặc thay ký tự, từ đó giảm chi phí chấm tay; tuy nhiên, đánh giá thủ công vẫn nên là chuẩn cuối cho chất lượng thẩm mỹ và các trường hợp mơ hồ. Đồng thời, inference cần rẻ hơn trước khi hệ thống có thể dùng tương tác. Các hướng như quantization, attention kernel tối ưu, biên dịch mô hình hoặc adapter nhỏ hơn đều phù hợp với mục tiêu đưa mô hình vào công cụ thiết kế, web service hoặc workflow sinh ảnh hàng loạt.

Cuối cùng, các lỗi còn lại mở ra hướng chống hallucination trong render chữ. Mô hình không còn thất bại rộng, nhưng vẫn nhầm một số cặp dấu và nguyên âm khó. Hard-word mining, dữ liệu replay có glyph đúng và adapter chuyên cho layout-binding có thể xử lý nhóm lỗi này. Khi các thành phần đó trưởng thành, hệ thống có thể được tích hợp thành công cụ cho người dùng nhập tiếng Việt, chọn phong cách thư pháp, chọn nền ảnh và nhận ảnh độ phân giải cao phục vụ thiết kế, bảo tồn văn hóa hoặc giáo dục.

Tóm lại, luận văn xây dựng một pipeline fine-tuning Ideogram4 thực nghiệm và có bằng chứng cho sinh ảnh thư pháp tiếng Việt. Kết quả cho thấy muốn render tiếng Việt chính xác không chỉ cần base model mạnh; cần biết tín hiệu dấu tồn tại ở đâu, tác động đúng module DiT, ổn định checkpoint và train trực tiếp trên phân phối bố cục nhiều từ mà ứng dụng thực tế cần.

---

# Tài liệu tham khảo

[1] I. Goodfellow et al., "Generative Adversarial Nets," *NeurIPS*, 2014.

[2] Z. Lyu, X. Bai, B. Shi, and C. Yao, "CalliGAN: Style and Structure-aware Chinese Calligraphy Character Generator," *CVPR Workshops*, 2020.

[3] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks," *ICCV*, 2017.

[4] J. Ho, A. Jain, and P. Abbeel, "Denoising Diffusion Probabilistic Models," *NeurIPS*, 2020.

[5] J. Song, C. Meng, and S. Ermon, "Denoising Diffusion Implicit Models," *ICLR*, 2021.

[6] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models," *CVPR*, 2022.

[7] W. Peebles and S. Xie, "Scalable Diffusion Models with Transformers," *ICCV*, 2023.

[8] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," *ICLR*, 2022.

[9] N. Ruiz et al., "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation," *CVPR*, 2023.

[10] M. Chen et al., "TextDiffuser: Diffusion Models as Text Painters," *NeurIPS*, 2023.

[11] A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," *ICML*, 2021.

[12] Y. Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models," *ICML*, 2023.

[13] T. Wortsman et al., "Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time," *ICML*, 2022.

[14] P. Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization," *UAI*, 2018.

[15] Ideogram AI, "Ideogram 4.0 Technical Details," technical documentation, 2026.

[16] Qwen Team, "Qwen3-VL Technical Report," *arXiv:2511.21631*, 2025.

[17] Baidu ERNIE-Image Team, "ERNIE-Image Technical Report," technical report, 2026.

[18] ModelScope Contributors, "DiffSynth-Studio: A Diffusion Engine," software repository, 2024-2026.

[19] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, "Flow Matching for Generative Modeling," *ICLR*, 2023.

[20] P. Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis," *ICML*, 2024.

[21] Qwen Team, "Qwen-Image Technical Report," *arXiv:2508.02324*, 2025.

[22] Black Forest Labs, "FLUX," technical report and software repository, 2024.

[23] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image Quality Assessment: From Error Visibility to Structural Similarity," *IEEE Transactions on Image Processing*, 2004.

[24] Y. Tuo, W. Xiang, J.-Y. He, Y. Geng, and X. Xie, "AnyText: Multilingual Visual Text Generation and Editing," *ICLR*, 2024.

[25] Y. Yang et al., "GlyphControl: Glyph Conditional Control for Visual Text Generation," *NeurIPS*, 2023.

[26] Z. Liu et al., "Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering," *ECCV*, 2024.

[27] D. Kalajdzievski, "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA," *arXiv:2312.03732*, 2023.

[28] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs," *NeurIPS*, 2023.

---

**Ghi chú về tính hợp lệ của một số reference**

- `[15]` Ideogram AI 2026 — trỏ đến bài technical blog công bố Ideogram 4 (xem `ideogram.ai/blog/ideogram-4.0/`). Đây là nguồn chính thức duy nhất được Ideogram AI công bố tính đến thời điểm viết; *technical report dạng arXiv chưa được phát hành*.
- `[16]` Qwen3-VL Technical Report (arXiv:2511.21631, submitted 2025-11-26) — bài báo arXiv chính thức của Qwen Team mô tả chi tiết kiến trúc Qwen3-VL. Đã được cập nhật sau khi viết bản thảo đầu. Trước đây reference này được ghi chung chung là "technical report, 2026" vì bản arXiv chưa xuất hiện khi viết; nay đã có arXiv ID chính thức, hội đồng có thể tra cứu trực tiếp. Qwen3-VL mô tả các biến thể dense 2B/4B/8B/32B và MoE 30B-A3B/235B-A22B, hỗ trợ interleaved context 256K token. Phiên bản Ideogram4 dùng là Qwen3-VL-8B-Instruct (biến thể dense 8B).
- `[17]` Baidu ERNIE-Image Team — dẫn đến bài blog chính thức "Introducing ERNIE-Image" trên `yiyan.baidu.com/blog/posts/ernie-image`. Đây là nguồn Baidu công bố thông tin về ERNIE-Image, mặc dù technical report dạng PDF chưa được phát hành rộng rãi.
- `[21]` Qwen-Image Technical Report (arXiv:2508.02324) — đây là reference được cite cho phần kiến trúc Qwen-Image, có arXiv chính thức và là nguồn được peer-review khả thi nhất.
- `[18]` DiffSynth-Studio — tham chiếu phần mềm mã nguồn mở; nên ghi rõ URL repository (`github.com/modelscope/DiffSynth-Studio`) và commit hash cụ thể (`6d103c0`, 2026-06-05) đã dùng để load Ideogram4.

Các reference `[1]`–`[14]` và `[19]`–`[28]` là công trình được công bố rộng rãi, có thể tra cứu qua Google Scholar hoặc arXiv; không có vấn đề xác thực.

**Đối chiếu thân bài ↔ danh sách:** ba citation `[15]`, `[16]`, `[17]` được dùng trong mục 1.4.3 đã được đối chiếu với danh sách tài liệu tham khảo và đảm bảo mỗi mô hình cite đúng mục tương ứng (Qwen-Image → `[21]`, ERNIE-Image → `[17, 21]`, Ideogram4 → `[15]`). Trong các vòng sửa sau này, reference `[16]` đã được cập nhật từ dạng "technical report, 2026" chung chung sang arXiv ID chính thức `2511.21631` sau khi Qwen Team công bố bản technical report đầy đủ.

---

# Phụ lục

## Phụ lục A: Lệnh training wide-target chính

```bash
bash experiments/scripts/v10/run_widetarget_gentle.sh \
  <warm_checkpoint> \
  <output_dir> \
  3 \
  5e-5 \
  <metadata_jsonl>
```

## Phụ lục B: Lệnh tạo dữ liệu compound

```bash
python3 experiments/scripts/v10/build_compound_dataset.py \
  --seed 27 \
  --img_sub images_compound_r5 \
  --out metadata_compound_2808_r5.jsonl
```

## Phụ lục C: Hậu xử lý compound branch

```bash
BASE_WEIGHT=0.1111111111 experiments/scripts/v10/postprocess_compound_branch.sh \
  no_training_session_for_manual_soup \
  experiments/checkpoints/coverage_v10_compound_lr3e5_from_gold4/step-11820.safetensors \
  experiments/checkpoints/coverage_v10_compound_soup_e4r2r3r4/step-soup.safetensors \
  experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1 \
  experiments/results/coverage_v10_eval/compound_eval28_soup_lr3e5_gold4_9to1
```

Script tính soup theo công thức `(BASE_WEIGHT * base + new) / (BASE_WEIGHT + 1)`, nên `BASE_WEIGHT=0.1111111111` tương ứng xấp xỉ 90% nhánh `lr3e5` và 10% Gold4.

## Phụ lục D: Registry checkpoint

```text
Single-word gold:
experiments/checkpoints/coverage_v10_widetarget_soup567/step-soup_infer.safetensors

Compound gold:
experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/step-soup_infer.safetensors
```

## Phụ lục E: Thư mục đánh giá

```text
experiments/results/coverage_v10_eval/compound_eval28_soup567_baseline
experiments/results/coverage_v10_eval/compound_eval28_bridge_e4
experiments/results/coverage_v10_eval/compound_eval28_bridge_r2
experiments/results/coverage_v10_eval/compound_eval28_bridge_r3
experiments/results/coverage_v10_eval/compound_eval28_soup_e4r2r3
experiments/results/coverage_v10_eval/compound_eval28_soup_e4r2r3r4
experiments/results/coverage_v10_eval/compound_eval28_lr3e5_from_gold4
experiments/results/coverage_v10_eval/compound_eval28_soup_lr3e5_gold4_9to1
```

## Phụ lục F: Lỗi còn lại của compound gold

```text
Hấn -> Hẩn
Chịt -> Chút
Huyên -> Huyện
Dôi -> Dồi
```

## Phụ lục G: Báo cáo probe nội bộ

```text
docs/deep_research/v11 Curing the Vietnamese Diacritic Plateau: A Lever Matrix for DiT LoRA SFT/_QWEN3_VL_SIGNAL_PROBE_2026-06-23.md
docs/deep_research/v11 Curing the Vietnamese Diacritic Plateau: A Lever Matrix for DiT LoRA SFT/_QWEN3_VL_CUU_FAMILY_PROBE_2026-06-24.md
```

## Phụ lục H: Danh sách ảnh so sánh

Các tên file sau được giữ sẵn cho các hình nên chèn vào bản luận văn hoàn thiện:

| Hình | Tên file | Nội dung dự kiến |
|---|---|---|
| Hình 1.2 | `docs/thesis/figures/fig_1_2_competitor_baseline_comparison.png` | Cùng một prompt qua font máy, công cụ thương mại black-box, Qwen Image, ERNIE Image, Ideogram4 gốc và checkpoint đề xuất |
| Hình 1.3 | `docs/thesis/figures/fig_1_3_base_model_capability_comparison.png` | So sánh Qwen Image, ERNIE Image và Ideogram4 gốc trên prompt dựng cảnh và prompt thư pháp tiếng Việt trước fine-tuning |
| Hình 3.5 | `docs/thesis/figures/fig_3_5_single_word_before_after_examples.png` | Ví dụ một từ khó trước và sau wide-target/checkpoint soup |
| Hình 3.6 | `docs/thesis/figures/fig_3_6_compound_eval28_before_after.png` | Ví dụ compound Eval28 so sánh `soup567` với checkpoint cuối `soup_lr3e5_gold4_9to1` |
| Hình 3.7 | `docs/thesis/figures/fig_3_7_remaining_error_cases.png` | Các lỗi còn lại: `Hấn`, `Chịt`, `Huyên`, `Dôi` |
| Hình 3.8 | `docs/thesis/figures/fig_3_8_calligraphy_with_base_model_capability.png` | Prompt giàu ngữ cảnh để kiểm tra checkpoint fine-tune có giữ được năng lực sinh ảnh nền của Ideogram4 khi thêm chữ thư pháp tiếng Việt hay không |

## Phụ lục I: Prompt dùng cho các hình so sánh

Hình 1.3 dùng cùng một prompt ngắn cho các mô hình base để so sánh khả năng sinh thư pháp tiếng Việt trước fine-tuning:

```text
Traditional Vietnamese calligraphy written in black ink on white paper.
The text says "An Khang Thịnh Vượng".
```

Hình 3.8 dùng prompt JSON giàu ngữ cảnh để kiểm tra khả năng kết hợp chữ thư pháp với năng lực sinh ảnh nền. Bộ ảnh được render bằng lệnh:

```bash
python3 experiments/scripts/v10/render_thesis_scene_candidates.py \
  --preset phuc_duc \
  --seed 7002
```

Hai text element được dùng trong Hình 3.8 là:

```text
Prompt A:
Phúc Thọ
An Khang

Prompt B:
Tâm Đức Trí Nhân
Phúc Thọ Khang Vượng
```

Phần mô tả cảnh chung của prompt yêu cầu một ảnh tĩnh vật Tết Việt Nam trên bàn sơn đỏ, gồm giấy dó/giấy gạo màu kem, chữ thư pháp mực đen, hoa mai, quả quất, bút lông, nghiên mực và ấn đỏ. Prompt cũng yêu cầu giữ chi tiết ảnh nền như ánh sáng, chất liệu giấy, phản chiếu sơn mài và bokeh, đồng thời giữ đúng dấu tiếng Việt và tránh sinh chữ Hán.

Hai prompt JSON đầy đủ được lưu tại:

```text
experiments/results/coverage_v10_eval/thesis_scene_candidates/phuc_duc_seed7002/prompt_phuc_tho_an_khang.json
experiments/results/coverage_v10_eval/thesis_scene_candidates/phuc_duc_seed7002/prompt_tam_duc_tri_nhan.json
```

Các câu này được chọn sau khi thử một số biến thể khác. Cụm `Lộc` không được dùng trong hình promote vì dễ mất mũ `ô`, trong khi `Vượng` cho kết quả chấp nhận được hơn ở cùng bối cảnh.

---

# Ghi nhận bản quyền và công cụ sử dụng

Luận văn ghi nhận các mô hình, công cụ mã nguồn mở và tài liệu nghiên cứu được sử dụng trong quá trình thực nghiệm, bao gồm Ideogram4, Qwen3-VL, các tài liệu tham chiếu ERNIE Image, DiffSynth-Studio, PyTorch, HuggingFace Transformers, safetensors và các công trình liên quan tới sinh ảnh. Tất cả dữ liệu sinh, checkpoint, probe và báo cáo đánh giá được nhắc tới trong luận văn được sử dụng cho mục đích nghiên cứu học thuật.
