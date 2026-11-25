# Đồ án 2 – Dự báo ngắn hạn số khách và doanh thu

Dự báo **số khách** (`Customers`) và **doanh thu** (`Sales`) theo ngày cho từng cửa hàng Rossmann, cho **ngày mai** (h = 1) và **tuần tới** (h = 7). Mục đích là hỗ trợ **xếp lịch nhân viên** và **chuẩn bị lượng hàng**. Đồ án không nhằm dự báo dài hạn; phần đó là đồ án 1 (dự báo trước 6 tuần cho Kaggle).

## Cấu trúc thư mục

```
PredictRevenue/
├── train.csv, store.csv, test.csv    ← dữ liệu gốc, dùng chung với đồ án 1
└── du_bao_ngan_han/
    ├── 1. Tien_Xu_Ly_Du_Lieu.ipynb     đọc dữ liệu, tạo lịch đầy đủ, chia tập
    ├── 2. Kham_Pha_Du_Lieu.ipynb       EDA số khách, khuyến mãi, chu kỳ, tự tương quan
    ├── 3. Khai_Thac_Dac_Trung.ipynb    đặc trưng thường + độ trễ theo horizon, thí nghiệm
    ├── 4. Huan_Luyen_Mo_Hinh.ipynb     baseline, LightGBM, LightGBM P80, đóng gói mô hình
    ├── 5. Danh_Gia_Mo_Hinh.ipynb       chỉ số, h=1 vs h=7, P80, ứng dụng xếp nhân viên
    ├── features.py                      code tạo đặc trưng, DÙNG CHUNG cho notebook và web
    ├── app.py                           web demo (Streamlit)
    ├── requirements.txt                 phiên bản thư viện
    ├── models/                          4 gói mô hình .joblib (tạo bởi notebook 4)
    ├── data/processed/                  dữ liệu trung gian và dự đoán (tạo bởi notebook 1–4)
    └── figures/                         hình cho báo cáo
```

Chạy notebook theo thứ tự 1 → 5, từ trong thư mục `du_bao_ngan_han/`. Notebook 4 mất khoảng 12 phút và cần khoảng 4–5 GB RAM trống.

---

## Những gì đã làm

### 1. Tiền xử lý
- Giữ lại cột `Customers` (đồ án 1 đã xoá vì coi là rò rỉ dữ liệu).
- Phát hiện 180 cửa hàng thiếu trọn 184 ngày (01/07–31/12/2014). Tạo **lịch đầy đủ** cho từng cửa hàng (`make_full_calendar`), để `lag_7` luôn là "7 ngày lịch trước", không bị nhảy qua khoảng trống.
- Ghép thêm các ngày của `test.csv`, để web biết lịch khuyến mãi của những ngày tương lai.
- Giữ đúng mốc chia tập của đồ án 1: huấn luyện đến 24/04/2015, kiểm định đến 12/06/2015, kiểm tra 13/06–31/07/2015.

### 2. Khám phá dữ liệu: các phát hiện chính
- Số khách và doanh thu tương quan 0,82, nhưng giá trị mỗi lượt mua khác nhiều giữa các loại cửa hàng (loại b: 4,9; loại d: 11,4). Vì vậy cần dự báo riêng hai đại lượng.
- Khuyến mãi tăng doanh thu 39%, trong đó **số khách +21%** và **giá trị mỗi lượt mua +14%**. Ngày khuyến mãi cần thêm cả người lẫn hàng.
- **Khuyến mãi chạy xen kẽ từng tuần**: chỉ 9,9% số ngày có cùng trạng thái khuyến mãi với tuần trước, so với 78,8% khi so với 2 tuần trước. Tự tương quan của doanh thu ở lag 14 (0,58) cao hơn hẳn lag 7 (0,15).

### 3. Đặc trưng (`features.py`)
- **Đặc trưng thường**: giữ đúng bộ đặc trưng cuối cùng của đồ án 1 (lịch, đối thủ, khuyến mãi, ngày lễ). Sửa thêm `PromoPrev`/`PromoNext` để tính trong từng cửa hàng.
- **Độ trễ**, với quy tắc **độ trễ nhỏ nhất ≥ horizon**, nên lúc dự báo mọi độ trễ đều là số thật, không cần dự báo cuốn chiếu:

  | Mô hình | Độ trễ | Đặc trưng tổng hợp (tính từ ngày ≤ t − h) |
  |---|---|---|
  | h = 1 (ngày mai) | 1, 2, 7, 14 | `roll7`, `roll28`, `dow_mean4`, `same_promo` |
  | h = 7 (tuần tới) | 7, 14, 21, 28 | như trên |

  - Có độ trễ cho **cả** `Sales` và `Customers`, nhưng **không** dùng giá trị cùng ngày của đại lượng kia.
  - `same_promo`: giá trị của ngày cùng thứ, cùng trạng thái khuyến mãi gần nhất, rút ra từ phát hiện khuyến mãi xen kẽ.
  - Ngày đóng cửa được coi là NaN, không phải 0.
- Thí nghiệm trên tập kiểm định: độ trễ h = 1 giảm RMSPE khoảng 12% (số khách) và 15% (doanh thu) so với không dùng độ trễ.

### 4. Mô hình
Có 4 bài toán = {số khách, doanh thu} × {h = 1, h = 7}. Mỗi bài toán gồm:

| Mô hình | Ý nghĩa |
|---|---|
| Baseline "Tuần trước" | Giống cùng thứ tuần trước (`lag_7`) |
| Baseline "Cùng KM gần nhất" | Giống ngày cùng thứ, cùng trạng thái khuyến mãi gần nhất |
| LightGBM | Dự đoán mức trung bình (học trên `log1p`) |
| LightGBM P80 | Hồi quy phân vị `alpha = 0.8`: mức an toàn, 80% số ngày thực tế không vượt quá |

### 5. Kết quả trên tập kiểm tra (13/06–31/07/2015)

| Bài toán | Tuần trước | Cùng KM gần nhất | **LightGBM** | MAE LightGBM |
|---|---|---|---|---|
| Số khách, ngày mai | 0,222 | 0,120 | **0,082** | 43 khách (5,9%) |
| Số khách, tuần tới | 0,222 | 0,120 | **0,092** | 49 khách (6,6%) |
| Doanh thu, ngày mai | 0,392 | 0,156 | **0,106** | 533 (7,5%) |
| Doanh thu, tuần tới | 0,392 | 0,156 | **0,114** | 583 (8,2%) |

(Chỉ số: RMSPE; trong ngoặc là MAE so với giá trị trung bình. Chi tiết: MAE, RMSE, R², sai số theo thứ và theo tuần, độ tin cậy của P80, mô phỏng xếp nhân viên, xem notebook 5.)

- Baseline "Tuần trước" sai rất nhiều, nhất là với doanh thu, vì khuyến mãi xen kẽ từng tuần.
- LightGBM giảm MAE 23–34% so với baseline tốt nhất; R² đạt 0,92–0,97. Kết quả trên tập kiểm định và tập kiểm tra gần như bằng nhau, nên mô hình ổn định.
- Dự báo ngày mai chính xác hơn dự báo tuần tới, nhờ có số liệu của hôm qua.
- Đặc trưng quan trọng nhất là `same_promo` (48–77% mức đóng góp).
- **P80**: trên dữ liệu mới, 76–78% số ngày thực tế không vượt mức này (mục tiêu 80%). P80 chỉ cao hơn mức trung bình 5–7%.
- **Xếp nhân viên** (giả định 150 khách/người): xếp theo P80 thì số ngày thiếu người còn 5% (theo mức trung bình là 13,6%, theo baseline là 15,4%), đổi lại thừa người nhiều hơn. Chọn mức nào là quyết định của quản lý.

### Giới hạn
- Dữ liệu chỉ có **theo ngày**: biết ngày nào cần nhiều người hơn, chưa xếp được ca theo giờ cao điểm.
- Doanh thu là **tổng**: ước lượng được tổng lượng hàng, không chỉ ra mặt hàng cụ thể.
- Số khách mỗi nhân viên phục vụ (mặc định 150) là **giả định**, vì dữ liệu không có số nhân viên.
- Ngoài thực tế, web phải nối với hệ thống bán hàng để có số liệu mới nhất mỗi ngày. Trong đồ án, "hôm nay" được giả lập bằng một ngày trong dữ liệu.

---

## Đưa mô hình lên web demo

### Bước 1 – Đóng gói mô hình (notebook 4, mục 4.6)
Mỗi bài toán được lưu thành `models/<bài toán>.joblib` (`customers_h1`, `customers_h7`, `sales_h1`, `sales_h7`), gồm **mọi thứ cần để dự đoán**:

```python
package = {
    'model': ..., 'model_p80': ...,          # mô hình trung bình và mô hình P80
    'features': [...],                       # danh sách cột đúng thứ tự lúc học
    'categories': {...},                     # danh mục của cột category
    'target': 'Customers', 'horizon': 1, 'log_target': True,
    'store_median': ...,                     # để chạy baseline
    'trained_until': '2015-04-24', 'test_rmspe': ..., 'versions': {...},
}
joblib.dump(package, 'models/customers_h1.joblib', compress=3)
```

### Bước 2 – Dùng chung code tạo đặc trưng
Mô hình chỉ hiểu đúng các cột đã học (`Customers_lag_1`, `PromoRun`, …). Toàn bộ code tạo đặc trưng nằm trong **`features.py`**; notebook và `app.py` cùng gọi file này. **Notebook 4, mục 4.7** kiểm tra lại: tạo đặc trưng kiểu web, xoá số liệu tương lai rồi dự đoán. Kết quả phải trùng khớp với dự đoán trong notebook.

### Bước 3 – Web demo bằng Streamlit (`app.py`)
Quy trình của app:
1. Lúc khởi động (một lần, khoảng 20 giây): đọc dữ liệu gốc, tạo đặc trưng thường, nạp 4 gói mô hình (có cache).
2. Người dùng chọn **cửa hàng** và **"hôm nay"** (01/01–31/07/2015).
3. App **xoá số khách/doanh thu sau "hôm nay"** để giả lập chưa biết tương lai, tạo độ trễ cho cửa hàng đó rồi dự đoán (dưới 1 giây). Ngày mai dùng mô hình h = 1, cả tuần tới dùng mô hình h = 7. Mô hình trung bình và mô hình P80 được học riêng nên thỉnh thoảng P80 thấp hơn mức trung bình; app lấy P80 = max(P80, trung bình).
4. Hiển thị:
   - Số khách và doanh thu dự kiến cùng mức an toàn P80.
   - **Số nhân viên gợi ý** (chỉnh được số khách/nhân viên, số người tối thiểu, xếp theo P80 hay theo mức trung bình).
   - Bảng lịch 7 ngày và biểu đồ 4 tuần gần nhất kèm dự báo.
   - Nếu ngày đó đã có số liệu thật thì hiện thêm để so sánh.

### Cách chạy
```bash
cd du_bao_ngan_han
pip install -r requirements.txt
# chạy notebook 1 → 4 (nếu chưa có thư mục models/)
streamlit run app.py
```
Trình duyệt sẽ mở `http://localhost:8501`.

### Lưu ý
- **Phiên bản thư viện** (LightGBM, scikit-learn, pandas) phải giống lúc huấn luyện, nếu không file `.joblib` có thể không mở được. Phiên bản được ghi trong `requirements.txt` và trong từng gói mô hình (`versions`).
- Chọn "hôm nay" **trước 24/04/2015** thì app cảnh báo, vì đó là giai đoạn mô hình đã học nên sai số trông thấp hơn thực tế. Nên demo với ngày **sau 12/06/2015**.
- Với "hôm nay" = 31/07/2015, app dự báo cho tuần 01–07/08/2015, tuần không có số liệu thật, giống tình huống dùng thực tế. Các cửa hàng không có trong `test.csv` sẽ hiện "Chưa có lịch".
- Muốn đưa lên mạng: có thể dùng Streamlit Community Cloud (miễn phí), nhưng phải xử lý chuyện `train.csv` (37 MB) và các file mô hình. Ví dụ: lưu sẵn một file đặc trưng gọn hơn, hoặc dùng Git LFS.
