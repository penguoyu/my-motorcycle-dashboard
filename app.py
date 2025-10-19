import streamlit as st
import pandas as pd
import glob
import numpy as np
import altair as alt # 我們會更常用到它

# --- 網頁的基本設定 (Layout 設為 "wide" 讓儀表板更寬) ---
st.set_page_config(
    page_title="113年機車事故儀表板",
    page_icon="🏍️",
    layout="wide" 
)

# --- 1. 讀取「超快速」的 Parquet 檔案 ---
@st.cache_data
def load_fast_data():
    try:
        df_full = pd.read_parquet("all_accidents_data.parquet")
        return df_full
    except FileNotFoundError:
        st.error("錯誤：找不到 'all_accidents_data.parquet' 檔案。")
        st.error("請先執行 convert.py 檔案來產生快速讀取檔。")
        st.stop()


# --- 2. 資料分析與處理 (V8) ---
@st.cache_data
def analyze_motorcycle_data(df):
    # --- 篩選機車 ---
    df_motorcycle = df[df['當事者區分-類別-大類別名稱-車種'].astype(str).str.contains('機車', na=False)].copy()

    # --- 處理時間相關 ---
    df_motorcycle['發生時間_num'] = pd.to_numeric(df_motorcycle['發生時間'], errors='coerce')
    df_motorcycle = df_motorcycle.dropna(subset=['發生時間_num'])
    df_motorcycle['發生小時'] = (df_motorcycle['發生時間_num'] // 10000).astype(int)
    
    df_motorcycle['發生日期_dt'] = pd.to_datetime(df_motorcycle['發生日期'], errors='coerce')
    df_motorcycle = df_motorcycle.dropna(subset=['發生日期_dt']) 
    df_motorcycle['發生月份'] = df_motorcycle['發生日期_dt'].dt.month
    df_motorcycle['發生星期'] = df_motorcycle['發生日期_dt'].dt.dayofweek + 1  # 1(一) - 7(日)
    
    # --- Feature Engineering ---
    df_motorcycle['時段 (週末/平日)'] = df_motorcycle['發生星期'].apply(lambda x: '週末 (六/日)' if x >= 6 else '平日 (一至五)')

    # --- 處理「年齡」 (使用 V7 的短標籤) ---
    df_motorcycle['年齡'] = pd.to_numeric(df_motorcycle['當事者事故發生時年齡'], errors='coerce')
    df_motorcycle = df_motorcycle[df_motorcycle['年齡'].between(0, 99)]
    bins = [0, 17, 24, 34, 44, 54, 64, 99]
    labels = ['0-17歲', '18-24歲', '25-34歲', '35-44歲', '45-54歲', '55-64歲', '65+歲']
    df_motorcycle['年齡層'] = pd.cut(df_motorcycle['年齡'], bins=bins, labels=labels, right=True)
    
    # --- 處理其他描述欄位 ---
    df_motorcycle['肇因'] = df_motorcycle['肇因研判子類別名稱-主要'].fillna('未知')
    df_motorcycle = df_motorcycle[
        ~df_motorcycle['肇因'].isin(['無(非車輛駕駛人因素)', '尚未發現肇事因素'])
    ]
    
    df_motorcycle['天候'] = df_motorcycle['天候名稱'].fillna('其他')
    df_motorcycle['道路類型'] = df_motorcycle['道路型態大類別名稱'].fillna('其他')
    df_motorcycle['性別'] = df_motorcycle['當事者屬-性-別名稱'].fillna('未知')
    df_motorcycle['縣市'] = df_motorcycle['處理單位名稱警局層'].fillna('未知')
    df_motorcycle['發生地點'] = df_motorcycle['發生地點'].fillna('未知')
    df_motorcycle['號誌種類'] = df_motorcycle['號誌-號誌種類名稱'].fillna('未知')
    df_motorcycle['安全帽'] = df_motorcycle['保護裝備名稱'].fillna('未知或無')
    df_motorcycle['事故型態(子類別)'] = df_motorcycle['事故類型及型態子類別名稱'].fillna('未知')
    df_motorcycle['事故型態(大類別)'] = df_motorcycle['事故類型及型態大類別名稱'].fillna('未知') 

    # --- 處理經緯度 ---
    df_motorcycle['lat'] = pd.to_numeric(df_motorcycle['緯度'], errors='coerce')
    df_motorcycle['lon'] = pd.to_numeric(df_motorcycle['經度'], errors='coerce')
    df_motorcycle = df_motorcycle.dropna(subset=['lat', 'lon'])

    return df_motorcycle

# --- 3. 建立網頁介面 (Dashboard) ---
st.title("🏍️ (113年度機車事故)") 

with st.spinner('正在讀取 all_accidents_data.parquet 檔案...'):
    df_full_data = load_fast_data() 

with st.spinner('正在分析機車事故資料...'):
    df_motorcycle_data = analyze_motorcycle_data(df_full_data)

st.success(f"資料載入完成！共分析了 {len(df_motorcycle_data)} 筆機車事故。")

# --- 側邊欄篩選器 ---
st.sidebar.header("🔍 全域篩選器")
all_cities = sorted(df_motorcycle_data['縣市'].unique())
default_city = ['臺北市政府警察局'] if '臺北市政府警察局' in all_cities else [all_cities[0]]
selected_cities = st.sidebar.multiselect("選擇縣市", options=all_cities, default=default_city)
month_range = st.sidebar.slider("選擇月份範圍", 1, 12, (1, 12))
weather_options = sorted(df_motorcycle_data['天候'].unique())
selected_weather = st.sidebar.multiselect("天候狀況", options=weather_options, default=weather_options)
selected_period = st.sidebar.radio("選擇時段 (平日/週末)", options=['全部', '平日 (一至五)', '週末 (六/日)'], index=0)
map_points_slider = st.sidebar.slider("地圖抽樣點數", 1000, 20000, 5000, 1000)

# --- 套用所有篩選條件 ---
filtered_data = df_motorcycle_data[
    (df_motorcycle_data['縣市'].isin(selected_cities)) &
    (df_motorcycle_data['發生月份'].between(month_range[0], month_range[1])) &
    (df_motorcycle_data['天候'].isin(selected_weather))
]
if selected_period != '全部':
    filtered_data = filtered_data[filtered_data['時段 (週末/平日)'] == selected_period]
if filtered_data.empty:
    st.warning("在目前的篩選條件下，找不到任何資料！")
    st.stop()
st.sidebar.info(f"篩選出 **{len(filtered_data)}** 筆資料")

# --- 建立分頁 ---
tab_names = ["📊 事故趨勢", "🗺️ 地理分布", "👥 人口統計", "🔍 肇事分析", "🔥 交叉分析"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_names)

# --- Tab 1: 事故趨勢分析 ---
with tab1:
    st.header(f"事故趨勢分析 ({selected_period})")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("每月事故統計")
        monthly_accidents = filtered_data.groupby('發生月份').size().reset_index(name='件數').set_index('發生月份')
        st.line_chart(monthly_accidents) # 折線圖的標籤通常沒問題，保留 st.line_chart
        
    with col2:
        st.subheader("每週事故分布")
        
        # --- (V8 修正) ---
        # 1. 準備資料 (不再 set_index)
        weekly_accidents = filtered_data.groupby('發生星期').size().reset_index(name='件數')
        weekly_accidents['星期標籤'] = weekly_accidents['發生星期'].map({1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '日'})
        
        # 2. 畫圖
        weekly_chart = alt.Chart(weekly_accidents).mark_bar().encode(
            x=alt.X('星期標籤:O', title='星期', sort=['一', '二', '三', '四', '五', '六', '日'], 
                    axis=alt.Axis(labelAngle=0)), # <-- 強制水平
            y=alt.Y('件數:Q', title='事故件數'),
            tooltip=['星期標籤', '件數']
        ).interactive()
        st.altair_chart(weekly_chart, use_container_width=True) # <-- 改用 st.altair_chart
    
    st.subheader(f"事故發生時段分析 ({selected_period}) (0-23點)")
    hourly_accidents = filtered_data.groupby('發生小時').size().reset_index(name='件數').set_index('發生小時')
    st.line_chart(hourly_accidents)

# --- Tab 2: 地理分布 ---
with tab2:
    st.header(f"地理分布分析 ({selected_period})")
    st.subheader(f"事故熱點地圖 (隨機抽樣 {map_points_slider} 點)")
    
    if len(filtered_data) > map_points_slider:
        map_data = filtered_data[['lat', 'lon']].sample(map_points_slider)
    else:
        map_data = filtered_data[['lat', 'lon']]
    st.map(map_data, zoom=10)
    
    col1, col2 = st.columns(2)
    with col1:
        if len(selected_cities) > 1:
            city_accidents = filtered_data.groupby('縣市').size().reset_index(name='件數').set_index('縣市').sort_values(by='件數', ascending=False)
            st.subheader("各縣市事故件數")
            st.bar_chart(city_accidents)
    with col2:
        st.subheader(f"最常發生事故路段 (Top 10)")
        dangerous_roads = filtered_data['發生地點'].value_counts().head(10)
        st.table(dangerous_roads) 

# --- Tab 3: 人口統計 ---
with tab3:
    st.header(f"人口統計分析 ({selected_period})")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("年齡層分布")
        
        # --- (V8 修正) ---
        # 1. 準備資料
        age_dist = filtered_data.groupby('年齡層').size().reset_index(name='件數')
        age_labels = ['0-17歲', '18-24歲', '25-34歲', '35-44歲', '45-54歲', '55-64歲', '65+歲']
        
        # 2. 畫圖
        age_chart = alt.Chart(age_dist).mark_bar().encode(
            x=alt.X('年齡層:O', title='年齡層', sort=age_labels, 
                    axis=alt.Axis(labelAngle=0)), # <-- 強制水平
            y=alt.Y('件數:Q', title='事故件數'),
            tooltip=['年齡層', '件數']
        ).interactive()
        st.altair_chart(age_chart, use_container_width=True) # <-- 改用 st.altair_chart
        
    with col2:
        st.subheader("性別分布")
        
        # --- (V8 修正) ---
        # 1. 準備資料
        gender_dist = filtered_data.groupby('性別').size().reset_index(name='件數')
        
        # 2. 畫圖
        gender_chart = alt.Chart(gender_dist).mark_bar().encode(
            x=alt.X('性別:O', title='性別', 
                    axis=alt.Axis(labelAngle=0)), # <-- 強制水平
            y=alt.Y('件數:Q', title='事故件數'),
            tooltip=['性別', '件數']
        ).interactive()
        st.altair_chart(gender_chart, use_container_width=True) # <-- 改用 st.altair_chart

    st.divider() 
    st.subheader("⛑️ 保護裝備 (安全帽) 分析")
    helmet_data = filtered_data[filtered_data['安全帽'].str.contains('帽', na=False) | (filtered_data['安全帽'] == '未戴')]
    helmet_counts = helmet_data['安全帽'].value_counts().head(5)
    st.bar_chart(helmet_counts, horizontal=True) # 水平長條圖的標籤通常OK，保留 st.bar_chart

# --- Tab 4: 肇事原因 ---
with tab4:
    st.header(f"肇事原因與事故型態 ({selected_period})")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("事故碰撞對象 (大類別)")
        type_major_counts = filtered_data['事故型態(大類別)'].value_counts().head(5)
        st.bar_chart(type_major_counts, horizontal=True) # 水平長條圖
    with col2:
        st.subheader("事故型態 (怎麼撞的？)")
        type_minor_counts = filtered_data['事故型態(子類別)'].value_counts().head(5)
        st.bar_chart(type_minor_counts, horizontal=True) # 水平長條圖
    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("號誌種類分析")
        signal_counts = filtered_data['號誌種類'].value_counts().head(5)
        st.bar_chart(signal_counts, horizontal=True) # 水平長條圖
    with col4:
        st.subheader("天候狀況分析")
        weather_accidents = filtered_data['天候'].value_counts().head(5) 
        st.bar_chart(weather_accidents, horizontal=True) # 水平長條圖

# --- (新功能) Tab 5: 交叉分析 ---
with tab5:
    st.header(f"🔥 交叉分析 ({selected_period})")
    
    st.subheader("肇事原因 vs 年齡層 (熱力圖)")
    st.info("這張圖顯示了：在特定年齡層中，各種肇事原因所佔的「件數」。")
    
    top_5_causes = filtered_data['肇因'].value_counts().head(5).index
    df_top5_causes = filtered_data[filtered_data['肇因'].isin(top_5_causes)]
    
    crosstab_df = pd.crosstab(
        df_top5_causes['年齡層'],
        df_top5_causes['肇因']
    )
    crosstab_melted = crosstab_df.reset_index().melt(id_vars='年齡層', var_name='肇因', value_name='件數')
    
    # (V7 修正)
    age_labels_heatmap = ['0-17歲', '18-24歲', '25-34歲', '35-44歲', '45-54歲', '55-64歲', '65+歲']
    heatmap = alt.Chart(crosstab_melted).mark_rect().encode(
        x=alt.X('年齡層:O', title='年齡層', axis=alt.Axis(labelAngle=0), sort=age_labels_heatmap), # <-- 修正！
        y=alt.Y('肇因:O', title='肇事原因'),
        color=alt.Color('件數:Q', title='事故件數'),
        tooltip=['年齡層', '肇因', '件數']
    ).interactive()
    
    st.altair_chart(heatmap, use_container_width=True)
    
    st.divider()
    
    st.subheader("肇事原因 vs 星期 (熱力圖)")
    st.info("這張圖顯示了：在一週的哪一天，哪些肇事原因特別多。")

    crosstab_weekday_df = pd.crosstab(
        filtered_data['發生星期'],
        filtered_data['肇因']
    )
    crosstab_weekday_df = crosstab_weekday_df[top_5_causes]
    
    weekday_map = {1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '日'}
    crosstab_weekday_df.index = crosstab_weekday_df.index.map(weekday_map)
    
    crosstab_weekday_melted = crosstab_weekday_df.reset_index().melt(id_vars='發生星期', var_name='肇因', value_name='件數')
    
    heatmap_weekday = alt.Chart(crosstab_weekday_melted).mark_rect().encode(
        x=alt.X('發生星期:O', title='星期', sort=['一', '二', '三', '四', '五', '六', '日'], 
                axis=alt.Axis(labelAngle=0)), # <-- 我也幫這張圖加上了！
        y=alt.Y('肇因:O', title='肇事原因'),
        color=alt.Color('件數:Q', title='事故件數'),
        tooltip=['發生星期', '肇因', '件數']
    ).interactive()
    
    st.altair_chart(heatmap_weekday, use_container_width=True)