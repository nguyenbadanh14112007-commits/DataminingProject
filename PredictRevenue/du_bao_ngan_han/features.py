"""Tạo đặc trưng cho đồ án dự báo ngắn hạn (ngày mai và tuần tới).

File này được dùng chung cho các notebook VÀ web demo (app.py), để lúc huấn luyện
và lúc dự đoán thật tạo đặc trưng theo đúng một cách.

Các bước chính:
    1. load_raw()             đọc train.csv / test.csv / store.csv và ghép lại
    2. make_full_calendar()   chèn đủ mọi ngày cho từng cửa hàng (lịch đầy đủ)
    3. add_features()         đặc trưng lịch, khuyến mãi, ngày lễ, đối thủ (giống đồ án 1)
    4. add_lag_features()     độ trễ của Sales và Customers theo horizon
"""
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ['Customers', 'Sales']           # hai đại lượng cần dự báo
HORIZONS = [1, 7]                          # 1 = ngày mai, 7 = tuần tới

# Lag nhỏ nhất luôn >= horizon, nên lúc dự báo mọi lag đều là giá trị THẬT (không phải tự đoán)
LAGS = {1: [1, 2, 7, 14],
        7: [7, 14, 21, 28]}

# Đặc trưng không phải độ trễ: giữ đúng bộ đặc trưng cuối cùng của đồ án 1
BASE_FEATURES = ['Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitorLongevity',
                 'Promo2', 'Promo2SinceWeek', 'Promo2_Active',
                 'DayOfWeek', 'Day', 'Month',
                 'Promo', 'PromoPrev', 'PromoNext', 'PromoRun', 'LastPromo',
                 'DaysSinceHoliday', 'DaystoHoliday']

CATEGORIES = {'StoreType': ['a', 'b', 'c', 'd'], 'Assortment': ['a', 'b', 'c']}   # cố định để train và web giống nhau
MONTH_NAME = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
              7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}


# ---------------------------------------------------------------- 1. Đọc dữ liệu
def load_raw(data_dir='..', include_test=True):
    """Đọc dữ liệu gốc và ghép thông tin cửa hàng.

    include_test=True: thêm các ngày của test.csv (01/08–17/09/2015) với Sales, Customers = NaN.
    Web demo cần các dòng này để biết lịch khuyến mãi / ngày lễ của những ngày tương lai.
    """
    data_dir = Path(data_dir)
    store = pd.read_csv(data_dir / 'store.csv')
    train = pd.read_csv(data_dir / 'train.csv', parse_dates=['Date'], dtype={'StateHoliday': str})
    parts = [train]
    if include_test:
        test = pd.read_csv(data_dir / 'test.csv', parse_dates=['Date'], dtype={'StateHoliday': str})
        test['Open'] = test['Open'].fillna(1)               # vài dòng thiếu Open → coi như mở cửa
        parts.append(test.drop(columns=['Id']))
    daily = pd.concat(parts, ignore_index=True)
    return daily.merge(store, on='Store', how='left')


# ---------------------------------------------------------------- 2. Lịch đầy đủ
def make_full_calendar(df):
    """Mỗi cửa hàng có đủ MỌI ngày từ ngày đầu đến ngày cuối của dữ liệu.

    Tại sao cần: train.csv thiếu 6 tháng (07–12/2014) của 180 cửa hàng. Nếu shift trên
    dữ liệu bị thiếu thì 'lag_7' có thể là 7 tháng trước chứ không phải 7 ngày trước.
    Các ngày được chèn thêm có InCalendarGap = 1 và Sales/Customers = NaN.
    """
    stores = df['Store'].unique()
    dates = pd.date_range(df['Date'].min(), df['Date'].max(), freq='D')
    grid = pd.MultiIndex.from_product([stores, dates], names=['Store', 'Date']).to_frame(index=False)
    full = grid.merge(df, on=['Store', 'Date'], how='left')
    full['InCalendarGap'] = full['Open'].isna().astype(int)

    # Thông tin cố định của cửa hàng: lấy lại từ store.csv cho các ngày được chèn
    store_cols = ['StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                  'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
    static = df.drop_duplicates('Store').set_index('Store')[store_cols]
    for col in store_cols:
        full[col] = full['Store'].map(static[col])

    # Lịch khuyến mãi gần như chung cho cả chuỗi → ngày bị thiếu lấy theo cửa hàng khác cùng ngày
    promo_by_date = df.groupby('Date')['Promo'].max()
    full['Promo'] = full['Promo'].fillna(full['Date'].map(promo_by_date)).fillna(0).astype(int)
    full['StateHoliday'] = full['StateHoliday'].fillna('0').astype(str)
    full['SchoolHoliday'] = full['SchoolHoliday'].fillna(0).astype(int)
    full['Open'] = full['Open'].fillna(0).astype(int)
    full['DayOfWeek'] = full['Date'].dt.dayofweek + 1               # 1 = thứ Hai ... 7 = Chủ nhật
    return full.sort_values(['Store', 'Date']).reset_index(drop=True)


# ---------------------------------------------------------------- 3. Đặc trưng thường
def add_features(df, competition_median=2325.0):
    """Đặc trưng lịch, đối thủ, khuyến mãi, ngày lễ — cùng ý tưởng với notebook 3 của đồ án 1.

    df phải là kết quả của make_full_calendar (đã sắp theo Store, Date).
    competition_median: trung vị CompetitionDistance trong store.csv, dùng để điền khuyết.
    """
    df = df.copy()
    g = df.groupby('Store')

    # Lịch
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month

    # Số tháng đối thủ đã mở cửa (không biết → -1)
    yr, mo = df['CompetitionOpenSinceYear'], df['CompetitionOpenSinceMonth']
    months_open = 12 * (df['Date'].dt.year - yr) + (df['Month'] - mo)
    df['CompetitorLongevity'] = months_open.where((yr > 0) & (mo > 0)).fillna(-1)
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(competition_median)

    # Ngày đó có nằm trong chuỗi khuyến mãi Promo2 không
    iso = df['Date'].dt.isocalendar()
    since = df['Promo2SinceYear'].fillna(9999) * 100 + df['Promo2SinceWeek'].fillna(0)   # chưa tham gia → năm 9999
    started = (iso['year'].astype(int) * 100 + iso['week'].astype(int)) >= since
    cur_month = df['Month'].map(MONTH_NAME)
    in_month = [m in str(s).split(',') for m, s in zip(cur_month, df['PromoInterval'])]
    df['Promo2_Active'] = ((df['Promo2'] == 1) & started & np.array(in_month)).astype(int)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(-1)

    # Khoảng cách tới ngày lễ gần nhất (không có → 30)
    hol_date = df['Date'].where(df['StateHoliday'] != '0')
    df['DaysSinceHoliday'] = (df['Date'] - hol_date.groupby(df['Store']).ffill()).dt.days.fillna(30)
    df['DaystoHoliday'] = (hol_date.groupby(df['Store']).bfill() - df['Date']).dt.days.fillna(30)

    # Khuyến mãi hôm qua / ngày mai (tính trong từng cửa hàng)
    df['PromoPrev'] = g['Promo'].shift(1).fillna(0).astype(int)
    df['PromoNext'] = g['Promo'].shift(-1).fillna(0).astype(int)

    # Hôm nay là ngày khuyến mãi liên tiếp thứ mấy
    block = (df['Promo'] != g['Promo'].shift()).cumsum()
    df['PromoRun'] = df.groupby(['Store', block]).cumcount() + 1
    df.loc[df['Promo'] == 0, 'PromoRun'] = 0

    # Số ngày kể từ ngày khuyến mãi gần nhất (chưa từng có → -1)
    date_promo = df['Date'].where(df['Promo'] == 1)
    df['LastPromo'] = (df['Date'] - date_promo.groupby(df['Store']).ffill()).dt.days.fillna(-1)

    for col, cats in CATEGORIES.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    return df


# ---------------------------------------------------------------- 4. Độ trễ
def lag_columns(horizon):
    """Tên các cột độ trễ được phép dùng cho một horizon."""
    cols = []
    for col in TARGETS:
        cols += [f'{col}_lag_{k}' for k in LAGS[horizon]]
        cols += [f'{col}_roll7', f'{col}_roll28', f'{col}_dow_mean4', f'{col}_same_promo']
    return cols


def feature_columns(horizon):
    return BASE_FEATURES + lag_columns(horizon)


def add_lag_features(df, horizon):
    """Thêm độ trễ của CẢ Sales và Customers (số khách hôm qua cũng giúp đoán doanh thu hôm nay).

    - Ngày đóng cửa / ngày bị thiếu → NaN (không coi là 0). LightGBM tự xử lý NaN.
    - {col}_lag_k     : giá trị cách đây k ngày lịch
    - {col}_roll7/28  : trung bình 7/28 ngày, tính đến ngày (t - horizon)
    - {col}_dow_mean4 : trung bình cùng thứ của 4 tuần trước (lag 7, 14, 21, 28)
    - {col}_same_promo: giá trị của ngày cùng thứ, CÙNG trạng thái Promo, gần nhất trong 4 tuần trước.
                        Cần vì khuyến mãi chạy xen kẽ từng tuần: tuần trước thường ngược trạng thái.
    Mọi giá trị đều lấy từ ngày <= t - horizon, nên không nhìn thấy tương lai.
    """
    df = df.copy()
    promo_g = df.groupby('Store')['Promo']
    same_promo = {k: promo_g.shift(k) == df['Promo'] for k in (7, 14, 21, 28)}
    for col in TARGETS:
        value = df[col].where((df['Open'] == 1) & (df[col] > 0))
        g = value.groupby(df['Store'])
        for k in LAGS[horizon]:
            df[f'{col}_lag_{k}'] = g.shift(k)

        shifted = g.shift(horizon).groupby(df['Store'])
        df[f'{col}_roll7'] = shifted.rolling(7, min_periods=3).mean().reset_index(level=0, drop=True)
        df[f'{col}_roll28'] = shifted.rolling(28, min_periods=10).mean().reset_index(level=0, drop=True)

        same_dow = pd.concat([g.shift(k) for k in (7, 14, 21, 28)], axis=1)
        df[f'{col}_dow_mean4'] = same_dow.mean(axis=1)

        same = pd.Series(np.nan, index=df.index)
        for k in (7, 14, 21, 28):                       # từ gần đến xa, lấy ngày phù hợp đầu tiên
            same = same.fillna(g.shift(k).where(same_promo[k]))
        df[f'{col}_same_promo'] = same

    new_cols = lag_columns(horizon)
    df[new_cols] = df[new_cols].astype('float32')       # nhẹ bộ nhớ; notebook và web cùng một kiểu số
    return df


def is_trainable(df):
    """Dòng dùng để huấn luyện/đánh giá: cửa hàng mở cửa, có doanh thu, không phải ngày được chèn."""
    return (df['Open'] == 1) & (df['Sales'] > 0) & (df['InCalendarGap'] == 0)


# ---------------------------------------------------------------- Baseline và nghiệp vụ
def baseline(df, target, fallback, method='same_promo'):
    """Cách quản lý hay đoán khi không có mô hình.

    method='lag7'       : 'giống cùng thứ tuần trước'
    method='same_promo' : 'giống ngày cùng thứ, cùng trạng thái khuyến mãi gần nhất'
    Thiếu giá trị → trung bình cùng thứ 4 tuần trước → trung vị của cửa hàng
    (fallback: Series theo Store, tính trên tập train).
    """
    col = f'{target}_lag_7' if method == 'lag7' else f'{target}_same_promo'
    pred = df[col].fillna(df[f'{target}_dow_mean4'])
    return pred.fillna(df['Store'].map(fallback)).to_numpy()


def staff_needed(customers, customers_per_staff=150, min_staff=2):
    """Quy số khách ra số nhân viên. customers_per_staff là GIẢ ĐỊNH (dữ liệu không có số nhân viên)."""
    return np.maximum(np.ceil(np.asarray(customers, dtype=float) / customers_per_staff), min_staff).astype(int)
