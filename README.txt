ROSSMANN STORE SALES - CHIA THÀNH 5 NOTEBOOK

Chỉ tách file; toàn bộ code, markdown, output, execution_count và metadata
của cell được giữ nguyên. Không thêm cell và không sửa đường dẫn.
Ghép các cell của 5 notebook theo thứ tự sẽ khôi phục đúng danh sách cell gốc.

NỘI DUNG VÀ RANH GIỚI CELL (đánh số từ 1)
01_Tien_xu_ly_va_phan_tich_du_lieu.ipynb: cell gốc 1-25, 25 cell (13 code).
02_Kham_pha_du_lieu.ipynb: cell gốc 26-37, 12 cell (4 code).
03_Khai_thac_dac_trung.ipynb: cell gốc 38-121, 84 cell (33 code).
04_Huan_luyen_mo_hinh.ipynb: cell gốc 122-138, 17 cell (7 code).
05_Danh_gia_mo_hinh.ipynb: cell gốc 139-185, 47 cell (16 code).

01: Đọc/ghép dữ liệu, kiểm tra cấu trúc, giá trị thiếu và làm sạch.
02: Phân phối, biến đổi log và quan hệ giữa các biến.
03: RF/MI chọn đặc trưng; tạo lịch, xu hướng, Fourier, lag; chuẩn bị các
    tập dữ liệu, định nghĩa lớp hybrid, mã hóa và phân cụm tạo đặc trưng.
04: Huấn luyện hybrid/LightGBM, dự báo và biểu đồ trong phần huấn luyện.
05: Chỉ số, phân tích sai số, so sánh mô hình và kết luận.

LƯU Ý CHẠY
Các phần phụ thuộc biến được tạo ở phần trước; KHÔNG chạy độc lập.
Chạy theo thứ tự 01 -> 02 -> 03 -> 04 -> 05 trong CÙNG MỘT KERNEL Python.
Nếu môi trường tự tạo kernel riêng cho mỗi notebook, biến không tự truyền.
Có thể chọn cùng kernel đang chạy nếu giao diện hỗ trợ, hoặc dùng IPython
%run -i với từng notebook .ipynb theo đúng thứ tự (tên file đặt trong dấu nháy).
Không restart kernel giữa các phần. Khi chạy lại toàn bộ, restart rồi bắt
đầu từ 01 vì code gốc có các biến đổi tại chỗ.

Đường dẫn /kaggle/input/... và yêu cầu thư viện vẫn giống notebook gốc.
Gói này không kèm CSV. Những bước fit phục vụ tạo/chọn đặc trưng được giữ
ở phần 03 để bảo toàn trình tự code; các cell pha trộn nhiều thao tác
không bị tách nhỏ hoặc sửa.

KIỂM TRA
Đã đọc lại JSON, kiểm tra cú pháp Python và đối chiếu chính xác toàn bộ
185 cell: không thay đổi, không bỏ sót, không lặp cell.
Chưa chạy lại mô hình; output hiện có là kết quả đã lưu trong notebook gốc.
Các lỗi/hạn chế của code gốc được giữ nguyên theo yêu cầu.
