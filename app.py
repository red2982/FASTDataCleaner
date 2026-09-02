import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FAST Data Processor", page_icon="📊", layout="wide")
st.title("📊 FAST Data Cleaner & Report Generator")
st.write("Upload your raw FAST CSV file to automatically clean, unpivot, and build benchmark pivot tables.")

uploaded_file = st.file_uploader("Choose a raw FAST CSV file", type=["csv"])


def clean_benchmark_code(val):
    """Dynamically parses and cleans standard codes for both Math and ELA.

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

    # Step 1: Extract final segment after the last pipe '|'
    if '|' in s:
        s = s.split('|')[-1].strip()

    # Step 2: Remove grade identifier (e.g. 7, 8, 912, K, PK)
    parts = s.split('.')
    subject_prefixes = ['MA', 'ELA', 'MATH', 'SC', 'SS', 'FL']

    if len(parts) >= 3 and parts[0].upper() in subject_prefixes:
        if parts[1].isdigit() or parts[1].upper() in ['K', 'PK', 'HS', 'KG', '912']:
            subject = parts[0].upper()
            rest = '.'.join(parts[2:])
            if subject == 'ELA':
                return rest  # Standard ELA format (e.g., R.1.1)
            else:
                return f"{parts[0]}.{rest}"  # Standard Math format (e.g., MA.AR.4.2)

    return s


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # 1. Delete Unwanted Columns
        cols_to_delete = [
            'Student ID', 'Student DOB', 'Ethnicity', 'Enrolled Grade', 'Sex', 'Enrolled School',
            'Enrolled District', 'Local ID', 'Testing Location', 'District Use', 'Test Reason',
            'Test OppNumber', 'Date Taken', 'Test Completion Date', '1. Reading Prose and Poetry Performance',
            '2. Reading Informational Text Performance', '3. Reading Across Genres & Vocabulary Performance', 'Category'
        ] + [f'{i}. Category' for i in range(1, 73)]

        df = df.drop(columns=[c for c in cols_to_delete if c in df.columns])

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

        # 4. Clean Benchmark Standard Codes Dynamically
        df_long['Benchmark'] = df_long['Benchmark'].apply(clean_benchmark_code)
        df_long = df_long[df_long['Benchmark'] != ''].copy()

        # 5. Generate Pivot Summaries (Earned, Possible, and Percentage)
        pivot_earned = df_long.pivot_table(index='Benchmark', columns='Student Name', values='Points Earned', aggfunc='sum', fill_value=0)
        pivot_possible = df_long.pivot_table(index='Benchmark', columns='Student Name', values='Points Possible', aggfunc='sum', fill_value=0)
        
        # Calculate Percentage Score safely
        pivot_pct = (pivot_earned / pivot_possible * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(1)

        st.success("✅ File processed successfully!")

        # Display preview tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Long Data Format", "📊 Earned Points", "🎯 Possible Points", "📈 Percentage Summary (%)"])
        with tab1:
            st.dataframe(df_long.head(20))
        with tab2:
            st.write("### Earned Points by Benchmark")
            st.dataframe(pivot_earned)
        with tab3:
            st.write("### Possible Points by Benchmark")
            st.dataframe(pivot_possible)
        with tab4:
            st.write("### Score Percentage (%) by Benchmark")
            st.dataframe(pivot_pct.astype(str) + '%')

        # Build Excel File with all 4 worksheets
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_long.to_excel(writer, sheet_name='Data', index=False)
            pivot_earned.to_excel(writer, sheet_name='Earned Points')
            pivot_possible.to_excel(writer, sheet_name='Possible Points')
            pivot_pct.to_excel(writer, sheet_name='Percentage')

        st.download_button(
            label="📥 Download Cleaned Excel File",
            data=output.getvalue(),
            file_name="FAST_Data_Processed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
