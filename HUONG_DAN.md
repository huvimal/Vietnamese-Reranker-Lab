# Dự án 17 — Vietnamese Reranker Lab

Mục tiêu: học cách huấn luyện và đo hiệu quả mô hình xếp hạng tài liệu tiếng Việt.
Đã có sẵn toàn bộ code và dữ liệu. Không cần API key, GPU hay cài thư viện bên ngoài.
Đây là mô hình tuyến tính học từ các đặc trưng văn bản, chưa phải mô hình transformer.

## 1. Vào đúng thư mục

Mở PowerShell và chạy:

```powershell
cd "C:\Users\LOQ\Documents\Codex\2026-11-06\cj\outputs\Vietnamese-Reranker-Lab"
python --version
python -m venv .venv
```

Các lệnh phía sau dùng Python trong `.venv`, không cần chạy Activate.ps1.

## 2. Kiểm tra code

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kỳ vọng: `Ran 11 tests` và `OK`. Thời gian chạy có thể khác nhau.

## 3. Huấn luyện

```powershell
.\.venv\Scripts\python.exe lab.py train
```

Kỳ vọng dòng đầu:

```text
Train: 216 | Dev: 72 | Pairs: 1080 | Missed positives: 0
```

- Train: dữ liệu dùng học trọng số.
- Dev: dữ liệu dùng chọn cấu hình.
- Pairs: cặp tài liệu đúng/sai lấy từ kết quả BM25 để học.
- Missed positives: số câu hỏi train không tìm được tài liệu đúng trong top 20.

Code chọn cấu hình bằng dev, chưa dùng nhãn test. Model lưu tại `models/reranker.json`.
Không đổi tham số chỉ để đạt điểm test đẹp hơn.

## 4. Đánh giá

```powershell
.\.venv\Scripts\python.exe lab.py evaluate --split test
$LASTEXITCODE
```

Kỳ vọng 72 câu hỏi test và mã thoát `0`. Kết quả mặc định đã kiểm tra:

| Mô hình | nDCG@5 | MRR@5 | Recall@5 |
| --- | ---: | ---: | ---: |
| BM25 | 0.8494 | 0.7986 | 1.0000 |
| lexical | 0.8801 | 0.8403 | 1.0000 |
| full | 1.0000 | 1.0000 | 1.0000 |

- BM25: thứ tự tìm kiếm ban đầu.
- lexical: mô hình học chỉ dùng nhóm đặc trưng từ vựng.
- full: mô hình học thêm đặc trưng mã sản phẩm và ý định câu hỏi.
- nDCG@5: tài liệu phù hợp có được đưa lên đầu trong năm kết quả không; càng cao càng tốt.
- MRR@5: tài liệu đúng đầu tiên đứng càng gần hạng 1 thì điểm càng cao.
- Recall@5: tỷ lệ tài liệu đúng nằm trong năm kết quả.
- CandidateRecall: tài liệu đúng có trong danh sách 20 ứng viên ban đầu không.

Điểm 1.0 chỉ đúng trên bộ giả lập này. Các câu hỏi test dùng sản phẩm khác nhưng
vẫn có chung mẫu câu với train. Bộ này chưa chứng minh khả năng trên dữ liệu thực tế.
Khoảng tin cậy bootstrap có thể trùng hai đầu vì các nhóm dùng cùng mẫu câu.

## 5. Thử tìm kiếm

Chạy từng lệnh:

```powershell
.\.venv\Scripts\python.exe lab.py search --query "SP005 được đổi trả trong mấy ngày?"
.\.venv\Scripts\python.exe lab.py search --query "Mua SP012 cần bao nhiêu tiền để miễn phí giao hàng?"
.\.venv\Scripts\python.exe lab.py search --query "Tôi có thể trả tiền mặt lúc nhận SP020 không?"
```

Mỗi lệnh in hai danh sách `BM25` và `RERANKED`. Kiểm tra mã sản phẩm và nội dung của
tài liệu đầu tiên: lần lượt là đổi trả SP005, vận chuyển SP012, thanh toán SP020.
Hai danh sách có thể giống nhau nếu BM25 đã xếp đúng. Điểm số không phải xác suất.
Đây là ứng dụng terminal; không cần mở localhost hay trình duyệt.

## 6. Đọc báo cáo và code

```powershell
Get-Content reports/training.json -Encoding utf8
Get-Content reports/test.json -Encoding utf8
```

`reports/test.json` chứa `metrics`, `paired_delta`, `failures`, `per_query`.
Mục `failures` rỗng khi full đã xếp đúng tất cả câu hỏi trong fixture; điều đó hợp lệ.
Tài liệu đúng không được tự chèn vào tập ứng viên khi đánh giá.

Đọc code theo thứ tự:

1. `make_data.py`: cách tạo dữ liệu và chia sản phẩm thành train/dev/test.
2. `ranking.py`: BM25, trích đặc trưng, học cặp đúng–sai và tính điểm đánh giá.
3. `lab.py`: ghép các phần thành lệnh train/evaluate/search; lưu model và báo cáo.
4. `tests/test_ranking.py`: kiểm tra Unicode, metric, rò rỉ dữ liệu và ứng viên.

## 7. Khi nào hoàn tất bản đầu?

- 11 tests đạt.
- Huấn luyện tạo model và báo cáo.
- Đánh giá chạy thành công, mã thoát 0.
- Ba câu hỏi tìm kiếm trả về đúng sản phẩm và chính sách ở hạng đầu.
- Có thể giải thích sự khác nhau giữa train/dev/test, BM25 và reranker.

Để phát triển thành dự án portfolio mạnh hơn, phần quan trọng tiếp theo là thay
dữ liệu mẫu bằng câu hỏi được viết và gán nhãn độc lập. README có schema và các
giới hạn cần xử lý. Không dùng dữ liệu nội bộ của khách hàng làm dữ liệu công khai.
