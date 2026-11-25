# Dự báo doanh thu và lưu lượng khách – Rossmann Store Sales

Kho này gồm 2 đồ án dùng chung bộ dữ liệu Rossmann (`train.csv`, `store.csv`, `test.csv`).

| | Đồ án 1 – Dự báo 6 tuần (hiện tại) | Đồ án 2 – Dự báo ngắn hạn (mới) |
|---|---|---|
| Mục đích | Bài thi Kaggle: dự báo doanh thu trước 6 tuần | Vận hành: xếp nhân viên, chuẩn bị hàng |
| Mục tiêu | `Sales` | `Customers` và `Sales` |
| Horizon | 42 ngày | Ngày mai (h = 1) và tuần tới (h = 7) |
| Lag dùng được | 49, 56, 63 | 1, 7, 8, 14 + trung bình trượt |
| Đánh giá | RMSPE, nộp Kaggle | So với baseline `lag_7`, kiểm định trượt theo tuần, mức P80 |
| Kết quả | File nộp Kaggle | Bảng dự báo + gợi ý số nhân viên |

---

## Trạng thái hiện tại

- **Đồ án 1** vẫn nằm ở thư mục gốc (chưa chuyển vào `du_bao_6_tuan/`). Notebook đánh giá `5. Danh_Gia_Mo_Hinh.ipynb` đã được làm lại cho cả 6 mô hình. Notebook cũ `05_Danh_gia_mo_hinh.ipynb` đã lỗi thời (đọc file `results.pkl` không còn tồn tại, chỉ so 3 mô hình).
- **Đồ án 2** đã làm xong trong [`du_bao_ngan_han/`](du_bao_ngan_han/README.md): 5 notebook, `features.py`, mô hình đóng gói và web demo `app.py`. Dữ liệu gốc đọc từ thư mục gốc (`../train.csv`…).

## Cấu trúc thư mục (dự kiến khi tách hẳn)

```
PredictRevenue/
├── data/raw/            ← train.csv, store.csv, test.csv (dùng chung, không sao chép)
├── du_bao_6_tuan/       ← đồ án 1 (notebook 1–5, data/processed, figures, submissions)
├── du_bao_ngan_han/     ← đồ án 2 (notebook 1–5, data/processed, figures)
│   ├── features.py      ← code tạo đặc trưng, dùng chung cho notebook và web
│   ├── models/          ← mô hình đã đóng gói (.joblib)
│   └── app.py           ← web demo (Streamlit)
└── README.md
```

- Dữ liệu gốc chỉ để một chỗ; notebook đọc `../data/raw/...`. `data/processed` và `figures` để riêng trong từng đồ án.
- Mỗi đồ án tự đủ (chấp nhận lặp vài hàm như `rmspe`, `save_fig`) để có thể nộp riêng.
- Chuyển file bằng `git mv` để giữ lịch sử; dọn notebook trùng `05_Danh_gia_mo_hinh.ipynb`.

---

## Đồ án 2 – Ý tưởng chính

### Bài toán
Dự báo **số khách** (để xếp nhân viên) và **doanh thu** (để ước lượng lượng hàng cần chuẩn bị) theo ngày cho từng cửa hàng, cho **ngày mai** và **tuần tới**. Không nhằm giải quyết horizon dài.

### Không cần dự báo cuốn chiếu
Quy định lag nhỏ nhất ≥ horizon, nên mọi lag đều là giá trị thật:

| Mô hình | Dùng để | Lag được phép |
|---|---|---|
| h = 1 | Dự báo ngày mai, điều chỉnh ca | `lag_1`, `lag_7`, `lag_8`, `roll7` |
| h = 7 | Dự báo cả tuần tới, xếp lịch, đặt hàng | `lag_7`, `lag_14`, `roll7`/`roll28` lùi 7 ngày |

Mô hình h = 7 dùng chung cho cả 7 ngày của tuần tới.

### Tham số trong notebook
```python
TARGET  = 'Customers'   # 'Sales' hoặc 'Customers'
HORIZON = 1             # 1 hoặc 7
RUN     = f'{TARGET.lower()}_h{HORIZON}'   # lưu vào figures/<RUN>/, predictions/<RUN>/
```

### Quy tắc về đặc trưng
- Không dùng giá trị **cùng ngày** của biến kia (mô hình Customers không dùng `Sales` cùng ngày và ngược lại). Lag của cả hai biến thì dùng được.
- Không xóa cột `Customers` ở notebook 3 nữa (là mục tiêu hoặc nguồn tạo lag). Luôn loại cả `Sales` và `Customers` khỏi X:
  ```python
  def split_xy(df, start, end):
      part = df[(df['Date'] > start) & (df['Date'] <= end)]
      return part.drop(columns=['Sales', 'Customers', 'Date']), part[TARGET]
  ```
- **Tạo lag theo ngày lịch**, không theo ngày mở cửa. Hiện `groupby('Store').shift(k)` chạy trên dữ liệu đã bỏ ngày đóng cửa, nên không khớp với PACF (tính theo ngày lịch). Cách làm: chèn đủ ngày cho từng cửa hàng (`asfreq('D')`, ngày đóng cửa = NaN) rồi mới `shift`.
- Vẽ PACF riêng cho `Customers` trước khi chốt danh sách lag.

### Thay đổi theo từng notebook
1. **Tiền xử lý:** gần như giữ nguyên; sửa đường dẫn đọc dữ liệu gốc.
2. **Khám phá dữ liệu:** thêm EDA cho `Customers` (phân phối, theo thứ, theo Promo, theo StoreType); phân tích `Sales / Customers` (giá trị mỗi lượt mua); chạy lại xu hướng/mùa vụ/PACF cho `Customers`, lưu `y_deseason_customers.csv`.
3. **Khai thác đặc trưng:** giữ `Customers`, dùng `split_xy`, tạo lag ngắn theo lịch đầy đủ cho cả 2 biến, chạy lại các thí nghiệm chọn đặc trưng cho từng mục tiêu.
4. **Huấn luyện:** `y = X[TARGET]`, danh sách đặc trưng theo `LAGS` thay vì viết cứng; thêm baseline `lag_7`; thêm mô hình phân vị P80. Bỏ phần nộp Kaggle.
5. **Đánh giá:** đọc dự đoán theo `RUN`; so với baseline `lag_7`; MAE của Customers tính bằng "số khách"; so sánh Sales h = 1 với Sales h = 42 của đồ án 1.

### Đánh giá
- `test.csv` không có `Sales`/`Customers` → không nộp Kaggle; đánh giá trên phần tách ra từ `train.csv`.
- Nên dùng **kiểm định trượt theo tuần**: huấn luyện đến tuần t → dự đoán tuần t+1 → lặp lại.
- **Baseline nghiệp vụ:** "bằng cùng thứ tuần trước" (`lag_7`). Mô hình phải thắng rõ baseline này mới đáng dùng.

### Đoán thiếu và đoán thừa gây hậu quả khác nhau
Đoán thiếu thì thiếu nhân viên, hết hàng, mất khách; đoán thừa chỉ tốn thêm chi phí. Ngoài dự đoán trung bình, đưa ra thêm mức an toàn P80:
```python
model_p80 = lgb.LGBMRegressor(objective='quantile', alpha=0.8, ...)
```
Quy ra con số người quản lý dùng được, ví dụ: *"Thứ Hai tuần tới dự kiến ~780 khách (mức an toàn 850) → cần 6 nhân viên nếu mỗi người phục vụ ~150 khách."* (Tỷ lệ khách/nhân viên là **giả định**, phải ghi rõ vì dữ liệu không có số nhân viên.)

### Hướng mở rộng
Dự đoán `Customers` và **basket** (= `Sales / Customers`), rồi tính `Sales = Customers × basket`:
- Doanh thu và số khách luôn khớp nhau.
- Basket ổn định hơn nên dễ dự đoán hơn.
- Phân tích được khuyến mãi làm tăng số khách hay tăng giá trị mỗi lượt mua.

### Giới hạn cần ghi trong báo cáo
- Dữ liệu chỉ có **theo ngày**, không theo giờ → chỉ trả lời "ngày nào cần nhiều người hơn", chưa xếp được ca theo giờ cao điểm.
- Doanh thu là **tổng**, không chia theo sản phẩm → chỉ ước lượng tổng lượng hàng, không chỉ ra mặt hàng cụ thể.

---

## Đóng gói mô hình và web demo (đồ án 2)

Mục đích: đưa mô hình ra khỏi notebook để quản lý cửa hàng dùng được qua trình duyệt, không cần mở Jupyter.

### Bước 1 – Đóng gói (lưu mô hình ra file)
Lưu **mọi thứ cần để dự đoán**, không chỉ mô hình:
```python
import joblib
joblib.dump({
    'model':      model_lgb,                 # mô hình dự đoán trung bình
    'model_p80':  model_p80,                 # mô hình mức an toàn
    'features':   list(X_train.columns),     # đúng thứ tự cột lúc học
    'categories': {c: X_train[c].cat.categories for c in ['StoreType', 'Assortment']},
    'target': 'Customers', 'horizon': 1, 'log_target': True,
}, 'models/customers_h1.joblib')
```
- Chuyển phần tạo đặc trưng từ notebook ra **`features.py`**, để notebook và web dùng **chung một đoạn code**. Nếu hai nơi viết hai kiểu, kết quả sẽ sai mà không báo lỗi.
- Chỉ lưu mô hình tốt nhất (không lưu Random Forest vì file rất nặng).

### Bước 2 – Nhúng vào web
| Cách | Mô tả | Hợp với |
|---|---|---|
| **Streamlit** | Viết giao diện web bằng Python, ~50 dòng | Đồ án, demo khi bảo vệ (**chọn cách này**) |
| FastAPI/Flask + giao diện riêng | Backend có API `/predict`, giao diện HTML/JS gọi API | Hệ thống thật, nhiều ứng dụng cùng dùng |

Chức năng trang demo: chọn cửa hàng, ngày dự báo, có khuyến mãi hay không, số khách mỗi nhân viên phục vụ → hiện số khách dự kiến, mức an toàn P80, doanh thu dự kiến, số nhân viên gợi ý và biểu đồ 4 tuần gần nhất + dự báo.

Chạy: `streamlit run app.py`

### Lưu ý
1. **Lag cần dữ liệu mới nhất.** Trong đồ án giả lập "hôm nay" bằng một ngày trong dữ liệu (ví dụ 31/07/2015). Thực tế phải nối với hệ thống bán hàng → ghi vào phần giới hạn.
2. **Phiên bản thư viện** (LightGBM, scikit-learn) phải giống lúc huấn luyện → ghi vào `requirements.txt`.
3. **Cột category** phải có cùng danh mục như lúc học → lưu kèm `categories` trong file đóng gói.
4. Muốn đưa lên mạng: Streamlit Community Cloud (miễn phí), nhưng phải xử lý chuyện file dữ liệu lớn.

---

## Ghi chú: nếu sau này cần dự báo cuốn chiếu (horizon dài)
Không thuộc phạm vi đồ án 2, ghi lại để tham khảo:
- Sai số cộng dồn và lan giữa 2 biến (Customers sai → lag của Sales sai → …).
- Lúc học dùng lag thật, lúc dự đoán dùng lag tự đoán (exposure bias).
- Biến đổi `log1p` làm dự đoán hơi thấp, cuốn chiếu làm lệch thêm.
- Giải pháp: chiến lược trực tiếp (mỗi horizon một mô hình), giới hạn số bước cuốn chiếu, hoặc dùng lag dài như đồ án 1.
