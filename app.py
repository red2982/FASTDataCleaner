import io
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
                return f"{parts[0]}.{rest}"

    return s


if uploaded_file is not None:
    try:
        # Load CSV or Excel
        file_name = uploaded_file.name.lower()
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # 1. Delete Unwanted Columns
        static_cols_to_delete = [
            'Student ID', 'Student DOB', 'Ethnicity', 'Enrolled Grade', 'Sex', 'Enrolled School',
            'Enrolled District', 'Local ID', 'Testing Location', 'District Use', 'Test Reason',
            'Test OppNumber', 'Date Taken', 'Test Completion Date'
        ]

        cols_to_drop = [
            c for c in df.columns 
            if c in static_cols_to_delete or 'Performance' in str(c) or 'Category' in str(c)
        ]
        df = df.drop(columns=cols_to_drop)

        # 2. Map Disability Codes
        disability_codes = [
            'K - Specific Learning Disability', 'J - Emotional/Behavioral Disability',
            'P - Autism Spectrum Disorder', 'L - Gifted', 'V - Other Health Impaired',
            'H - Deaf or Hard of Hearing', 'G - Language Impaired', 'I - Visually Impaired',
            'F - Speech Impaired', 'W - Intellectual Disabilities', 'M - Hospital Homebound',
            'C - Orthopedic Impairment', 'O - Dual Sensory Impaired', 'U - Established Conditions',
            'Z - Other'
        ]
        disability_map = {'N - N/A': 'No', **{code: 'Yes' for code in disability_codes}}
        df = df.replace(disability_map)

        # 3. Convert Wide to Long Format
        long_data = []
        for _, row in df.iterrows():
            meta = [row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5], row.iloc[6]]
            bench_values = row.iloc[7:].values

            for j in range(0, len(bench_values) - 1, 3):
                bm = bench_values[j]
                pe = bench_values[j + 1] if j + 1 < len(bench_values) else None
                pp = bench_values[j + 2] if j + 2 < len(bench_values) else None

                if pd.notna(bm) and pd.notna(pe) and pd.notna(pp) and str(bm).strip().upper() not in ["", "N/A", "NAN"]:
                    long_data.append(meta + [bm, pe, pp])

        headers = [
            "Student Name", "504", "Primary Exceptionality", "English Learner",
            "Scale Score", "Level Score", "Percentile Rank",
            "Benchmark", "Points Earned", "Points Possible"
        ]

        df_long = pd.DataFrame(long_data, columns=headers)
        df_long['Points Earned'] = pd.to_numeric(df_long['Points Earned'], errors='coerce')
        df_long['Points Possible'] = pd.to_numeric(df_long['Points Possible'], errors='coerce')

        # 4. Clean Benchmark Standard Codes
        df_long['Benchmark'] = df_long['Benchmark'].apply(clean_benchmark_code)
        df_long = df_long[df_long['Benchmark'] != ''].copy()

        # 5. Generate Pivot Summaries with Metadata Rows
        pivot_earned = df_long.pivot_table(index='Benchmark', columns='Student Name', values='Points Earned', aggfunc='sum', fill_value=0)
        pivot_possible = df_long.pivot_table(index='Benchmark', columns='Student Name', values='Points Possible', aggfunc='sum', fill_value=0)
        pivot_pct = (pivot_earned / pivot_possible * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(1)

        # Extract student metadata values
        student_meta = df_long.groupby('Student Name').first()

        combined_dfs = []
        students = sorted(df_long['Student Name'].unique())

        for student in students:
            s_504 = student_meta.loc[student, '504']
            s_ese = student_meta.loc[student, 'Primary Exceptionality']
            s_esol = student_meta.loc[student, 'English Learner']
            s_scale = student_meta.loc[student, 'Scale Score'] if pd.notna(student_meta.loc[student, 'Scale Score']) else ""
            s_level = student_meta.loc[student, 'Level Score'] if pd.notna(student_meta.loc[student, 'Level Score']) else ""
            
            p_rank = student_meta.loc[student, 'Percentile Rank']
            s_perc = f"{int(p_rank)}%" if pd.notna(p_rank) and str(p_rank).replace('.','',1).isdigit() else (str(p_rank) if pd.notna(p_rank) else "")

            # Metadata top rows
            meta_rows = pd.DataFrame({
                (student, 'Points Earned'): [s_504, s_ese, s_esol, s_perc, s_scale, s_level],
                (student, 'Points Possible'): ["", "", "", "", "", ""],
                (student, '% Score'): ["", "", "", "", "", ""]
            }, index=[
                '504 Status',
                'ESE (Primary Exceptionality)',
                'ESOL (English Learner)',
                'Percentile Rank',
                'Scale Score',
                'Performance Level'
            ])

            # Benchmark performance rows
            bench_earned = pivot_earned[student]
            bench_possible = pivot_possible[student]
            bench_pct = pivot_pct[student].astype(str) + '%'

            bench_rows = pd.DataFrame({
                (student, 'Points Earned'): bench_earned,
                (student, 'Points Possible'): bench_possible,
                (student, '% Score'): bench_pct
            })

            full_st_df = pd.concat([meta_rows, bench_rows])
            combined_dfs.append(full_st_df)

        combined_pivot = pd.concat(combined_dfs, axis=1)
        combined_pivot.columns = pd.MultiIndex.from_tuples(combined_pivot.columns)

        st.success("✅ File processed successfully!")

        # Display preview tabs
        tab1, tab2 = st.tabs(["📋 Long Data Format", "📊 Benchmark & Demographic Summary"])
        with tab1:
            st.dataframe(df_long.head(20))
        with tab2:
            st.write("### Combined Benchmark & Student Demographic Summary")
            st.dataframe(combined_pivot)

        # Build Excel File with 2 worksheets
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
