import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FAST Data Processor", page_icon="📊", layout="wide")
st.title("📊 FAST Data Cleaner & Report Generator")
st.write("Upload your raw FAST file (CSV or Excel) to automatically clean, unpivot, and build benchmark pivot tables.")

uploaded_file = st.file_uploader("Choose a raw FAST file", type=["csv", "xlsx", "xls"])


def clean_benchmark_code(val):
    """Dynamically cleans both Math and ELA standard codes."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if not s or s.lower() in ['nan', 'none', 'n/a']:
        return ""

    if '|' in s:
        s = s.split('|')[-1].strip()

    parts = s.split('.')
    subject_prefixes = ['MA', 'ELA', 'MATH', 'SC', 'SS', 'FL']

    if len(parts) >= 3 and parts[0].upper() in subject_prefixes:
        if parts[1].isdigit() or parts[1].upper() in ['K', 'PK', 'HS', 'KG', '912']:
            subject = parts[0].upper()
            rest = '.'.join(parts[2:])
            if subject == 'ELA':
                return rest
            else:
                return f"{subject}.{rest}"

    return s


@st.cache_data(show_spinner="Processing and reshaping file...")
def process_fast_file(file_bytes, file_name):
    # 1. Load File
    if file_name.endswith('.csv'):
        df_raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df_raw = pd.read_excel(io.BytesIO(file_bytes))

    # 2. Disability Map
    disability_codes = [
        'K - Specific Learning Disability', 'J - Emotional/Behavioral Disability',
        'P - Autism Spectrum Disorder', 'L - Gifted', 'V - Other Health Impaired',
        'H - Deaf or Hard of Hearing', 'G - Language Impaired', 'I - Visually Impaired',
        'F - Speech Impaired', 'W - Intellectual Disabilities', 'M - Hospital Homebound',
        'C - Orthopedic Impairment', 'O - Dual Sensory Impaired', 'U - Established Conditions',
        'Z - Other'
    ]
    disability_map = {'N - N/A': 'No', **{code: 'Yes' for code in disability_codes}}
    df_raw = df_raw.replace(disability_map)

    # 3. Separate Metadata vs Benchmark Columns
    meta_col_names = [
        "Student Name", "504", "Primary Exceptionality", "English Learner",
        "Scale Score", "Level Score", "Percentile Rank"
    ]

    static_cols_to_delete = [
        'Student ID', 'Student DOB', 'Ethnicity', 'Enrolled Grade', 'Sex', 'Enrolled School',
        'Enrolled District', 'Local ID', 'Testing Location', 'District Use', 'Test Reason',
        'Test OppNumber', 'Date Taken', 'Test Completion Date'
    ]

    bench_cols = [
        c for c in df_raw.columns 
        if c not in meta_col_names 
        and c not in static_cols_to_delete 
        and 'Performance' not in str(c) 
        and 'Category' not in str(c)
    ]

    num_triplets = len(bench_cols) // 3
    valid_bench_cols = bench_cols[:num_triplets * 3]

    # 4. Fast Vectorized Unpivoting via NumPy
    meta_vals = df_raw[meta_col_names].values
    bench_vals = df_raw[valid_bench_cols].values

    if num_triplets > 0:
        meta_tiled = np.repeat(meta_vals[:, np.newaxis, :], num_triplets, axis=1)
        bench_reshaped = bench_vals.reshape(len(df_raw), num_triplets, 3)
        flat_matrix = np.concatenate([meta_tiled, bench_reshaped], axis=2).reshape(-1, 10)
    else:
        flat_matrix = np.empty((0, 10))

    headers = meta_col_names + ["Benchmark", "Points Earned", "Points Possible"]
    df_long = pd.DataFrame(flat_matrix, columns=headers)

    # Convert numeric types safely
    df_long['Points Earned'] = pd.to_numeric(df_long['Points Earned'], errors='coerce')
    df_long['Points Possible'] = pd.to_numeric(df_long['Points Possible'], errors='coerce')

    # Drop blank benchmarks
    df_long['Benchmark'] = df_long['Benchmark'].apply(clean_benchmark_code)
    df_long = df_long[(df_long['Benchmark'] != '') & df_long['Points Earned'].notna() & df_long['Points Possible'].notna()].copy()

    # 5. Fast Pivot Table Creation
    pivot_earned = df_long.pivot_table(index='Benchmark', columns='Student Name', values='Points Earned', aggfunc='sum', fill_value=0)
    pivot_possible = df_long.pivot_table(index='Benchmark', columns='Student Name', values='Points Possible', aggfunc='sum', fill_value=0)
    pivot_pct = (pivot_earned / pivot_possible * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(1)

    student_meta = df_long.groupby('Student Name').first()
    students = sorted(df_long['Student Name'].unique())

    # Build MultiIndex dictionary directly (Instant performance)
    combined_data = {}
    for student in students:
        s_504 = student_meta.loc[student, '504']
        s_ese = student_meta.loc[student, 'Primary Exceptionality']
        s_esol = student_meta.loc[student, 'English Learner']
        s_scale = student_meta.loc[student, 'Scale Score'] if pd.notna(student_meta.loc[student, 'Scale Score']) else ""
        s_level = student_meta.loc[student, 'Level Score'] if pd.notna(student_meta.loc[student, 'Level Score']) else ""
        
        p_rank = student_meta.loc[student, 'Percentile Rank']
        s_perc = f"{int(p_rank)}%" if pd.notna(p_rank) and str(p_rank).replace('.','',1).isdigit() else (str(p_rank) if pd.notna(p_rank) else "")

        combined_data[(student, 'Points Earned')] = [s_504, s_ese, s_esol, s_perc, s_scale, s_level] + pivot_earned[student].tolist()
        combined_data[(student, 'Points Possible')] = ["", "", "", "", "", ""] + pivot_possible[student].tolist()
        combined_data[(student, '% Score')] = ["", "", "", "", "", ""] + (pivot_pct[student].astype(str) + '%').tolist()

    row_indices = [
        '504 Status',
        'ESE (Primary Exceptionality)',
        'ESOL (English Learner)',
        'Percentile Rank',
        'Scale Score',
        'Performance Level'
    ] + pivot_earned.index.tolist()

    combined_pivot = pd.DataFrame(combined_data, index=row_indices)
    combined_pivot.columns = pd.MultiIndex.from_tuples(combined_pivot.columns)

    return df_long, combined_pivot


if uploaded_file is not None:
    try:
        df_long, combined_pivot = process_fast_file(uploaded_file.getvalue(), uploaded_file.name.lower())

        st.success("✅ File processed successfully!")

        tab1, tab2 = st.tabs(["📋 Long Data Format", "📊 Benchmark & Demographic Summary"])
        with tab1:
            st.dataframe(df_long.head(20))
        with tab2:
            st.write("### Combined Benchmark & Student Demographic Summary")
            st.dataframe(combined_pivot)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_long.to_excel(writer, sheet_name='Data', index=False)
            combined_pivot.to_excel(writer, sheet_name='Benchmark Summary')

        st.download_button(
            label="📥 Download Cleaned Excel File",
            data=output.getvalue(),
            file_name="FAST_Data_Processed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
