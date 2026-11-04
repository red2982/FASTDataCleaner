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


def clean_ese_status(val):
    """Converts ESE Exceptionality codes to Yes/No safely."""
    if pd.isna(val):
        return "No"
    s = str(val).strip()
    if not s or s.upper().startswith("N -") or s.upper() in ["N/A", "NONE", "NO", "N", "NAN"]:
        return "No"
    return "Yes"


def get_domain(bm):
    """Maps benchmarks into Math Strands or ELA FAST Reporting Categories."""
    s = str(bm).strip()
    s_up = s.upper()
    parts = s_up.split('.')

    # 1. Math Standards (e.g., MA.NSO, MA.AR, MA.GR, MA.DP)
    if s_up.startswith('MA.'):
        if len(parts) >= 2:
            return f"MA.{parts[1]}"
        return "MATH"

    # 2. ELA Standards mapped to FAST Reporting Categories
    if 'R.1' in s_up:
        return "RPP - Reading Prose & Poetry"
    elif 'R.2' in s_up:
        return "RIT - Reading Informational Text"
    elif 'R.3' in s_up:
        return "RAG - Reading Across Genres"
    elif 'V.' in s_up or s_up.startswith('V') or 'VOCAB' in s_up:
        return "VOC - Vocabulary"
    elif 'C.' in s_up or s_up.startswith('C'):
        return "COM - Communication"
    elif 'F.' in s_up or s_up.startswith('F'):
        return "FND - Foundational Skills"
    elif len(parts) >= 1:
        return parts[0]

    return "OTHER"


@st.cache_data(show_spinner="Processing and reshaping FAST file...")
def process_fast_file(file_bytes, file_name):
    # 1. Load File
    if file_name.endswith('.csv'):
        df_raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df_raw = pd.read_excel(io.BytesIO(file_bytes))

    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # 2. Map Exceptionality / Disability Codes flexibly
    ese_col = next((c for c in df_raw.columns if 'exceptionality' in c.lower()), None)
    if ese_col:
        df_raw[ese_col] = df_raw[ese_col].apply(clean_ese_status)

    # 3. Dynamic Column Mapping
    cols = list(df_raw.columns)

    scale_col = next((c for c in cols if 'scale score' in c.lower()), None)
    level_col = next((c for c in cols if 'achievement level' in c.lower() or 'level score' in c.lower() or c.lower().endswith(' level')), None)
    rank_col = next((c for c in cols if 'percentile rank' in c.lower() or 'percentile' in c.lower()), None)

    meta_mapping = {
        "Student Name": next((c for c in cols if 'student name' in c.lower()), None),
        "504": next((c for c in cols if '504' in c.lower()), None),
        "Primary Exceptionality": ese_col,
        "English Learner": next((c for c in cols if 'ell' in c.lower() or 'english' in c.lower()), None),
        "Scale Score": scale_col,
        "Level Score": level_col,
        "Percentile Rank": rank_col
    }

    used_meta_cols = {v for k, v in meta_mapping.items() if v is not None}

    meta_order = [
        "Student Name", "504", "Primary Exceptionality", "English Learner",
        "Scale Score", "Level Score", "Percentile Rank"
    ]

    meta_df = pd.DataFrame(index=df_raw.index)
    for target in meta_order:
        orig_col = meta_mapping[target]
        if orig_col and orig_col in df_raw.columns:
            meta_df[target] = df_raw[orig_col]
        else:
            meta_df[target] = ""

    # 4. Filter Benchmark Columns
    static_cols_to_delete = [
        'student id', 'fleid', 'student dob', 'ethnicity', 'enrolled grade', 'sex', 'enrolled school',
        'enrolled district', 'local id', 'testing location', 'district use', 'test reason',
        'test oppnumber', 'date taken', 'test completion date'
    ]

    bench_cols = []
    for c in df_raw.columns:
        c_lower = str(c).strip().lower()
        if c in used_meta_cols:
            continue
        if any(s in c_lower for s in static_cols_to_delete):
            continue
        if 'performance' in c_lower or 'category' in c_lower:
            continue
        bench_cols.append(c)

    num_triplets = len(bench_cols) // 3
    valid_bench_cols = bench_cols[:num_triplets * 3]

    # 5. Fast Vectorized Unpivoting
    meta_vals = meta_df.values
    bench_vals = df_raw[valid_bench_cols].values

    if num_triplets > 0:
        meta_tiled = np.repeat(meta_vals[:, np.newaxis, :], num_triplets, axis=1)
        bench_reshaped = bench_vals.reshape(len(df_raw), num_triplets, 3)
        flat_matrix = np.concatenate([meta_tiled, bench_reshaped], axis=2).reshape(-1, 10)
    else:
        flat_matrix = np.empty((0, 10))

    headers = meta_order + ["Benchmark", "Points Earned", "Points Possible"]
    df_long = pd.DataFrame(flat_matrix, columns=headers)

    df_long['Points Earned'] = pd.to_numeric(df_long['Points Earned'], errors='coerce')
    df_long['Points Possible'] = pd.to_numeric(df_long['Points Possible'], errors='coerce')

    df_long['Benchmark'] = df_long['Benchmark'].apply(clean_benchmark_code)
    df_long = df_long[(df_long['Benchmark'] != '') & df_long['Points Earned'].notna() & df_long['Points Possible'].notna()].copy()

    # Assign category / domain
    df_long['Domain'] = df_long['Benchmark'].apply(get_domain)

    # 6. Pivot Calculations for Benchmarks, Domains, and Overall Totals
    pivot_earned = df_long.pivot_table(index='Benchmark', columns='Student Name', values='Points Earned', aggfunc='sum', fill_value=0)
    pivot_possible = df_long.pivot_table(index='Benchmark', columns='Student Name', values='Points Possible', aggfunc='sum', fill_value=0)

    domain_earned = df_long.pivot_table(index='Domain', columns='Student Name', values='Points Earned', aggfunc='sum', fill_value=0)
    domain_possible = df_long.pivot_table(index='Domain', columns='Student Name', values='Points Possible', aggfunc='sum', fill_value=0)

    overall_earned = df_long.groupby('Student Name')['Points Earned'].sum()
    overall_possible = df_long.groupby('Student Name')['Points Possible'].sum()

    student_meta = df_long.groupby('Student Name').first()
    students = sorted(df_long['Student Name'].unique())
    domains = sorted(df_long['Domain'].unique())

    combined_data = {}

    # CLASS / GROUP TOTAL column first
    combined_data[('CLASS / GROUP TOTAL', 'Points Earned')] = ["Total Students", len(students), "", "", "", ""]
    combined_data[('CLASS / GROUP TOTAL', 'Points Possible')] = ["", "", "", "", "", ""]
    combined_data[('CLASS / GROUP TOTAL', '% Score')] = ["", "", "", "", "", ""]

    # Student metadata headers
    for student in students:
        s_504 = student_meta.loc[student, '504']
        s_ese = student_meta.loc[student, 'Primary Exceptionality']
        s_esol = student_meta.loc[student, 'English Learner']
        s_scale = student_meta.loc[student, 'Scale Score'] if pd.notna(student_meta.loc[student, 'Scale Score']) else ""
        s_level = student_meta.loc[student, 'Level Score'] if pd.notna(student_meta.loc[student, 'Level Score']) else ""
        p_rank = student_meta.loc[student, 'Percentile Rank']
        s_perc = f"{int(p_rank)}%" if pd.notna(p_rank) and str(p_rank).replace('.','',1).isdigit() else (str(p_rank) if pd.notna(p_rank) else "")

        combined_data[(student, 'Points Earned')] = [s_504, s_ese, s_esol, s_perc, s_scale, s_level]
        combined_data[(student, 'Points Possible')] = ["", "", "", "", "", ""]
        combined_data[(student, '% Score')] = ["", "", "", "", "", ""]

    row_labels = [
        '504 Status',
        'ESE (Primary Exceptionality)',
        'ESOL (English Learner)',
        'Percentile Rank',
        'Scale Score',
        'Performance Level'
    ]

    # Populate Hybrid Rows (Domain Subtotal -> Individual Standards)
    for domain in domains:
        subtotal_label = f"TOTAL: {domain}"
        row_labels.append(subtotal_label)

        # Domain Class / Group Total
        c_d_e = domain_earned.loc[domain].sum() if domain in domain_earned.index else 0
        c_d_p = domain_possible.loc[domain].sum() if domain in domain_possible.index else 0
        c_d_pct = f"{round((c_d_e / c_d_p * 100), 1)}%" if c_d_p > 0 else "0.0%"
        combined_data[('CLASS / GROUP TOTAL', 'Points Earned')].append(c_d_e)
        combined_data[('CLASS / GROUP TOTAL', 'Points Possible')].append(c_d_p)
        combined_data[('CLASS / GROUP TOTAL', '% Score')].append(c_d_pct)

        # Domain Student Values
        for student in students:
            d_e = domain_earned.loc[domain, student] if (domain in domain_earned.index and student in domain_earned.columns) else 0
            d_p = domain_possible.loc[domain, student] if (domain in domain_possible.index and student in domain_possible.columns) else 0
            d_pct = f"{round((d_e / d_p * 100), 1)}%" if d_p > 0 else "0.0%"
            combined_data[(student, 'Points Earned')].append(d_e)
            combined_data[(student, 'Points Possible')].append(d_p)
            combined_data[(student, '% Score')].append(d_pct)

        # Individual Benchmarks within Domain
        bms = sorted(df_long[df_long['Domain'] == domain]['Benchmark'].unique())
        for bm in bms:
            row_labels.append(f"  └─ {bm}")

            c_b_e = pivot_earned.loc[bm].sum() if bm in pivot_earned.index else 0
            c_b_p = pivot_possible.loc[bm].sum() if bm in pivot_possible.index else 0
            c_b_pct = f"{round((c_b_e / c_b_p * 100), 1)}%" if c_b_p > 0 else "0.0%"
            combined_data[('CLASS / GROUP TOTAL', 'Points Earned')].append(c_b_e)
            combined_data[('CLASS / GROUP TOTAL', 'Points Possible')].append(c_b_p)
            combined_data[('CLASS / GROUP TOTAL', '% Score')].append(c_b_pct)

            for student in students:
                b_e = pivot_earned.loc[bm, student] if (bm in pivot_earned.index and student in pivot_earned.columns) else 0
                b_p = pivot_possible.loc[bm, student] if (bm in pivot_possible.index and student in pivot_possible.columns) else 0
                b_pct = f"{round((b_e / b_p * 100), 1)}%" if b_p > 0 else "0.0%"
                combined_data[(student, 'Points Earned')].append(b_e)
                combined_data[(student, 'Points Possible')].append(b_p)
                combined_data[(student, '% Score')].append(b_pct)

    # Overall Test Total Row
    row_labels.append("🏆 OVERALL TEST TOTAL")
    c_o_e = overall_earned.sum()
    c_o_p = overall_possible.sum()
    c_o_pct = f"{round((c_o_e / c_o_p * 100), 1)}%" if c_o_p > 0 else "0.0%"
    combined_data[('CLASS / GROUP TOTAL', 'Points Earned')].append(c_o_e)
    combined_data[('CLASS / GROUP TOTAL', 'Points Possible')].append(c_o_p)
    combined_data[('CLASS / GROUP TOTAL', '% Score')].append(c_o_pct)

    for student in students:
        o_e = overall_earned.loc[student] if student in overall_earned.index else 0
        o_p = overall_possible.loc[student] if student in overall_possible.index else 0
        o_pct = f"{round((o_e / o_p * 100), 1)}%" if o_p > 0 else "0.0%"
        combined_data[(student, 'Points Earned')].append(o_e)
        combined_data[(student, 'Points Possible')].append(o_p)
        combined_data[(student, '% Score')].append(o_pct)

    combined_pivot = pd.DataFrame(combined_data, index=row_labels)
    combined_pivot.columns = pd.MultiIndex.from_tuples(combined_pivot.columns)

    return df_long, combined_pivot


if uploaded_file is not None:
    try:
        df_long, combined_pivot = process_fast_file(uploaded_file.getvalue(), uploaded_file.name.lower())

        st.success("✅ File processed successfully!")

        # Preview tabs
        tab1, tab2 = st.tabs(["📊 Benchmark & Demographic Summary", "📋 Long Data Format (Preview Only)"])
        with tab1:
            st.write("### Combined Benchmark & Student Demographic Summary")
            st.dataframe(combined_pivot)
        with tab2:
            st.dataframe(df_long.head(20))

        # Build Excel File with Summary sheet
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            combined_pivot.to_excel(writer, sheet_name='Benchmark Summary')

        st.download_button(
            label="📥 Download Cleaned Excel File",
            data=output.getvalue(),
            file_name="FAST_Data_Processed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
