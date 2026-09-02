import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FAST Data Processor", page_icon="📊", layout="wide")
st.title("📊 FAST Data Cleaner & Report Generator")
st.write("Upload your raw FAST file (CSV or Excel) to automatically clean, unpivot, and build benchmark pivot tables.")

# Allow both CSV and Excel file types
uploaded_file = st.file_uploader("Choose a raw FAST file", type=["csv", "xlsx", "xls"])


def clean_benchmark_code(val):
    """Dynamically cleans both Math and ELA standard codes.

    Examples:
    - PRR|MA.7.AR.4|MA.7.AR.4.2 -> MA.AR.4.2
    - NSOAR|MA.7.AR.2|MA.7.AR.2.2 -> MA.AR.2.2
    - RP|ELA.7.R.1.1|ELA.7.R.1.1 -> R.1.1
    """
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
        # Check file extension and load appropriately
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

        # 5. Generate Combined MultiIndex Pivot Summary
        pivot_earned = df_long.pivot_table(index='Benchmark', columns='Student Name', values='Points Earned', aggfunc='sum', fill_value=0)
        pivot_possible = df_long.pivot_table(index='Benchmark', columns='Student Name', values='Points Possible', aggfunc='sum', fill_value=0)
        pivot_pct = (pivot_earned / pivot_possible * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(1)

        # Group Earned, Possible, and % metrics under each Student Name
        combined_dfs = []
        students = sorted(df_long['Student Name'].unique())

        for student in students:
            st_df = pd.DataFrame({
                (student, 'Points Earned'): pivot_earned[student],
                (student, 'Points Possible'): pivot_possible[student],
                (student, '% Score'): pivot_pct[student].astype(str) + '%'
            })
            combined_dfs.append(st_df)

        combined_pivot = pd.concat(combined_dfs, axis=1)
        combined_pivot.columns = pd.MultiIndex.from_tuples(combined_pivot.columns)

        st.success("✅ File processed successfully!")

        # Display preview tabs
        tab1, tab2 = st.tabs(["📋 Long Data Format", "📊 Benchmark Summary"])
        with tab1:
            st.dataframe(df_long.head(20))
        with tab2:
            st.write("### Combined Benchmark Performance Summary")
            st.dataframe(combined_pivot)

        # Build Excel File with 2 worksheets (Data + Consolidated Pivot)
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
