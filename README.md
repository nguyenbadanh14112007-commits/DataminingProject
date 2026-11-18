# Dự báo doanh thu Rossmann Store Sales

Đồ án môn Khai phá dữ liệu: dự báo doanh thu theo ngày của 1.115 cửa hàng Rossmann (dữ liệu Kaggle *Rossmann Store Sales*), so sánh **Baseline**, **mô hình lai Boosted Hybrid** (Hồi quy tuyến tính + LightGBM) và **LightGBM**.

## Cấu trúc

| File | Nội dung | Đầu ra |
|---|---|---|
| `01_Tien_xu_ly_va_phan_tich_du_lieu.ipynb` | 01 – Đọc, ghép, làm sạch dữ liệu | `data/processed/rossmann_clean.csv` |
| `02_Kham_pha_du_lieu.ipynb` | 02 – Khám phá dữ liệu (EDA) | hình trong `figures/` |
| `03_Khai_thac_dac_trung.ipynb` | 03 – Đánh giá & tạo đặc trưng, phân cụm, chia tập | `data/processed/features.pkl` |
| `04_Huan_luyen_mo_hinh.ipynb` | 04 – Huấn luyện Hybrid, LightGBM, Baseline | `data/processed/results.pkl` |
| `05_Danh_gia_mo_hinh.ipynb` | 05 – Đánh giá, phân tích phần dư | `data/processed/metrics.csv` |
| `rossmann-store-sales.ipynb` | Notebook gốc ban đầu (chỉ để tham khảo, không dùng) | — |

Mỗi notebook đọc file do notebook trước lưu, nên **chạy độc lập được** (chỉ cần chạy đúng thứ tự 01 → 05 một lần đầu).
Các đoạn code đã sửa hoặc thêm mới được đánh dấu `# [UPDATED]` kèm giải thích.

## Cách chạy

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

for nb in 01_Tien_xu_ly_va_phan_tich_du_lieu 02_Kham_pha_du_lieu 03_Khai_thac_dac_trung 04_Huan_luyen_mo_hinh 05_Danh_gia_mo_hinh; do
  jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb"
done
```

Thời gian chạy toàn bộ khoảng 7–8 phút (CPU 8 luồng); phần lâu nhất là huấn luyện LightGBM ở notebook 04.

## Dữ liệu

`store.csv`, `train.csv`, `test.csv` lấy từ https://www.kaggle.com/competitions/rossmann-store-sales.
`test.csv` không có cột doanh thu nên không dùng; tập kiểm tra được cắt từ `train.csv` (05–07/2015).


Khoe dataset
Khoe model (thuat toan lien quan) - lam dc gi, do chinh xac
Q&A