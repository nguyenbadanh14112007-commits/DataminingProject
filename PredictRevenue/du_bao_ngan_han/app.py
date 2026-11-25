"""Web demo: dự báo số khách và doanh thu cho ngày mai và tuần tới.

Chạy (trong thư mục du_bao_ngan_han):
    streamlit run app.py

Cách hoạt động:
    1. Lúc khởi động: đọc dữ liệu gốc, tạo đặc trưng thường (features.py) và nạp 4 gói mô hình.
    2. Người dùng chọn cửa hàng và "hôm nay".
    3. Xoá số khách / doanh thu sau "hôm nay" (giả lập chưa biết tương lai), tạo độ trễ, dự đoán.
       Ngày mai dùng mô hình h = 1, cả tuần tới dùng mô hình h = 7.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import features as F

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent                 # train.csv, store.csv, test.csv ở thư mục gốc
MODEL_DIR = APP_DIR / 'models'
DAY_NAMES = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN']
TRAIN_END = pd.Timestamp('2015-04-24')    # mô hình đã thấy dữ liệu đến ngày này
LAST_DATE = pd.Timestamp('2015-07-31')    # ngày cuối có số liệu thật

st.set_page_config(page_title='Dự báo khách và doanh thu', layout='wide')


def fmt(x):
    """Định dạng số kiểu Việt Nam: 12.345"""
    return f'{x:,.0f}'.replace(',', '.')


# ------------------------------------------------------------------ Nạp dữ liệu và mô hình (một lần)
@st.cache_resource(show_spinner='Đang nạp mô hình...')
def load_models():
    return {f'{t.lower()}_h{h}': joblib.load(MODEL_DIR / f'{t.lower()}_h{h}.joblib')
            for t in F.TARGETS for h in F.HORIZONS}


@st.cache_data(show_spinner='Đang đọc dữ liệu và tạo đặc trưng (chỉ làm một lần, khoảng 20 giây)...')
def load_base():
    raw = F.load_raw(DATA_DIR, include_test=True)
    return F.add_features(F.make_full_calendar(raw))


# ------------------------------------------------------------------ Dự báo
def forecast(base, models, store_id, today):
    """Dự báo 7 ngày sau `today` cho một cửa hàng, bằng cả mô hình h = 1 và h = 7."""
    one = base[base['Store'] == store_id].copy()
    truth = one.set_index('Date')[['Customers', 'Sales']]                 # giữ lại số thật để so sánh
    one.loc[one['Date'] > today, ['Customers', 'Sales']] = np.nan         # chưa biết tương lai

    days = pd.date_range(today + pd.Timedelta(days=1), periods=7, freq='D')
    out = pd.DataFrame(index=days)
    for h in F.HORIZONS:
        feat = F.add_lag_features(one, h).set_index('Date').reindex(days)
        for target in F.TARGETS:
            pkg = models[f'{target.lower()}_h{h}']
            X = feat[pkg['features']]
            mean = np.expm1(pkg['model'].predict(X))
            p80 = np.expm1(pkg['model_p80'].predict(X))
            out[f'{target}_h{h}'] = mean
            # Hai mô hình học riêng nên thỉnh thoảng P80 thấp hơn trung bình → mức an toàn không được thấp hơn trung bình
            out[f'{target}_h{h}_p80'] = np.maximum(p80, mean)
        if h == 7:
            out['Open'] = feat['Open'].fillna(0).astype(int)
            out['Promo'] = feat['Promo'].fillna(0).astype(int)
            out['NoSchedule'] = feat['InCalendarGap'].fillna(1).astype(int)   # không có lịch của ngày này
    for target in F.TARGETS:
        out[f'{target}_actual'] = truth[target].reindex(days)
    history = truth.loc[today - pd.Timedelta(days=27):today]
    return out, history, one


# ------------------------------------------------------------------ Giao diện
models = load_models()
base = load_base()

st.title('Dự báo lưu lượng khách và doanh thu')
st.caption('Đồ án 2: dự báo theo ngày cho từng cửa hàng Rossmann, hỗ trợ xếp lịch nhân viên và chuẩn bị hàng.')

with st.sidebar:
    st.header('Thiết lập')
    store_id = st.selectbox('Cửa hàng', sorted(base['Store'].unique()), index=261)
    today = pd.Timestamp(st.date_input('Hôm nay là ngày', value=LAST_DATE.date(),
                                       min_value=pd.Timestamp('2015-01-01').date(),
                                       max_value=LAST_DATE.date(), format='DD/MM/YYYY'))
    st.divider()
    st.subheader('Quy đổi ra nhân viên')
    per_staff = st.number_input('Số khách mỗi nhân viên phục vụ/ngày', min_value=50, max_value=500,
                                value=150, step=10, help='Giả định, vì dữ liệu không có số nhân viên.')
    min_staff = st.number_input('Số nhân viên tối thiểu khi mở cửa', min_value=1, max_value=10, value=2)
    use_p80 = st.toggle('Xếp nhân viên theo mức an toàn P80', value=True,
                        help='P80: 80% số ngày, số khách thực tế không vượt quá mức này. '
                             'Tắt đi để xếp theo mức trung bình (ít người hơn, dễ thiếu người hơn).')

if today <= TRAIN_END:
    st.warning(f'Ngày {today:%d/%m/%Y} nằm trong giai đoạn huấn luyện (đến {TRAIN_END:%d/%m/%Y}): '
               'mô hình đã thấy dữ liệu này, nên sai số sẽ thấp hơn thực tế. '
               'Chọn ngày sau 12/06/2015 để xem đúng chất lượng dự báo.')

out, history, one = forecast(base, models, store_id, today)
info = one.iloc[0]
st.markdown(f'**Cửa hàng {store_id}** · loại {info["StoreType"]} · mặt hàng {info["Assortment"]} · '
            f'đối thủ cách {fmt(info["CompetitionDistance"])} m · '
            f'"hôm nay": **{DAY_NAMES[today.dayofweek]} {today:%d/%m/%Y}**')

# ---- Ngày mai (mô hình h = 1)
tomorrow = out.iloc[0]
st.subheader(f'Ngày mai – {DAY_NAMES[out.index[0].dayofweek]} {out.index[0]:%d/%m/%Y}')
if tomorrow['NoSchedule']:
    st.info('Chưa có lịch mở cửa của cửa hàng này cho ngày mai (cửa hàng không có trong test.csv).')
elif not tomorrow['Open']:
    st.info('Cửa hàng **đóng cửa** ngày mai.')
else:
    staff_basis = tomorrow['Customers_h1_p80'] if use_p80 else tomorrow['Customers_h1']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Số khách dự kiến', fmt(tomorrow['Customers_h1']),
              help=f'Mức an toàn P80: {fmt(tomorrow["Customers_h1_p80"])} khách')
    c2.metric('Mức an toàn P80', fmt(tomorrow['Customers_h1_p80']))
    c3.metric('Doanh thu dự kiến', fmt(tomorrow['Sales_h1']),
              help=f'Mức an toàn P80: {fmt(tomorrow["Sales_h1_p80"])}')
    c4.metric('Nhân viên gợi ý', int(F.staff_needed(staff_basis, per_staff, min_staff)))
    notes = ['có **khuyến mãi**' if tomorrow['Promo'] else 'không có khuyến mãi']
    if pd.notna(tomorrow['Customers_actual']):
        notes.append(f'thực tế: {fmt(tomorrow["Customers_actual"])} khách, doanh thu {fmt(tomorrow["Sales_actual"])}')
    st.caption(' · '.join(notes))

# ---- Tuần tới (mô hình h = 7)
st.subheader('Tuần tới – lịch gợi ý')
rows = []
for day, r in out.iterrows():
    label = f'{DAY_NAMES[day.dayofweek]} {day:%d/%m}'
    if r['NoSchedule'] or not r['Open']:
        rows.append({'Ngày': label, 'Trạng thái': 'Chưa có lịch' if r['NoSchedule'] else 'Đóng cửa'})
        continue
    basis = r['Customers_h7_p80'] if use_p80 else r['Customers_h7']
    row = {'Ngày': label, 'Trạng thái': 'Khuyến mãi' if r['Promo'] else 'Bình thường',
           'Khách dự kiến': round(r['Customers_h7']), 'Khách P80': round(r['Customers_h7_p80']),
           'Doanh thu dự kiến': round(r['Sales_h7']), 'Doanh thu P80': round(r['Sales_h7_p80']),
           'Nhân viên gợi ý': int(F.staff_needed(basis, per_staff, min_staff))}
    if pd.notna(r['Customers_actual']):
        row['Khách thực tế'] = round(r['Customers_actual'])
        row['Nhân viên lẽ ra cần'] = int(F.staff_needed(r['Customers_actual'], per_staff, min_staff))
    rows.append(row)
week = pd.DataFrame(rows)
st.dataframe(week, hide_index=True, width='stretch')
open_days = week[week['Trạng thái'].isin(['Khuyến mãi', 'Bình thường'])]
if len(open_days):
    st.caption(f'Tổng tuần: khoảng {fmt(open_days["Khách dự kiến"].sum())} khách, '
               f'doanh thu khoảng {fmt(open_days["Doanh thu dự kiến"].sum())}, '
               f'{int(open_days["Nhân viên gợi ý"].sum())} ngày-công.')

# ---- Biểu đồ
st.subheader('4 tuần gần nhất và dự báo tuần tới')
tab_c, tab_s = st.tabs(['Số khách', 'Doanh thu'])
for tab, target in [(tab_c, 'Customers'), (tab_s, 'Sales')]:
    is_open = (out['Open'] == 1) & (out['NoSchedule'] == 0)
    chart = pd.DataFrame({'Thực tế': pd.concat([history[target].where(history[target] > 0),
                                                out[f'{target}_actual'].where(is_open)]),
                          'Dự báo (h = 7)': out[f'{target}_h7'].where(is_open),
                          'Mức an toàn P80': out[f'{target}_h7_p80'].where(is_open)})
    tab.line_chart(chart, color=['#888888', '#2a78b5', '#e07b39'])

# ---- Thông tin mô hình
with st.expander('Mô hình hoạt động thế nào?'):
    st.markdown('''
- **Mô hình:** LightGBM, học trên dữ liệu 01/01/2013 – 24/04/2015. Mỗi bài toán (số khách / doanh thu × ngày mai / tuần tới)
  có một mô hình dự đoán mức trung bình và một mô hình phân vị 80% (mức an toàn).
- **Không nhìn trước tương lai:** mọi số liệu sau "hôm nay" bị xoá trước khi dự đoán. Mô hình ngày mai chỉ dùng số liệu đến
  hôm nay; mô hình tuần tới chỉ dùng số liệu cách ít nhất 7 ngày.
- **Giới hạn:** dữ liệu chỉ có theo ngày (chưa xếp được ca theo giờ) và doanh thu là tổng (không chia theo mặt hàng).
  Số khách mỗi nhân viên phục vụ là giả định.
''')
    perf = pd.DataFrame([{'Bài toán': run, 'RMSPE (tập kiểm tra)': round(p['test_rmspe'], 4),
                          'MAE (tập kiểm tra)': round(p['test_mae'], 1), 'Học đến ngày': p['trained_until']}
                         for run, p in models.items()])
    st.dataframe(perf, hide_index=True)
