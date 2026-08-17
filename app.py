import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="FAST Data Processor", page_icon="📊", layout="wide")
st.title("📊 FAST Data Cleaner & Report Generator")
st.write("Upload your raw FAST CSV file to automatically clean, unpivot, and build benchmark pivot tables.")

uploaded_file = st.file_uploader("Choose a raw FAST CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # 1. Delete Unwanted Columns
        cols_to_delete = [
            'Student ID', 'Student DOB','Ethnicity','Enrolled Grade','Sex','Enrolled School',
            'Enrolled District', 'Local ID','Testing Location','District Use','Test Reason',
            'Test OppNumber','Date Taken','Test Completion Date','1. Reading Prose and Poetry Performance',
            '2. Reading Informational Text Performance','3. Reading Across Genres & Vocabulary Performance','Category'
        ]
        df = df.drop(columns=[c for c in cols_to_delete if c in df.columns])

        # 2. Bulk Find and Replace Exact
        data_map = {
            'N - N/A': 'No', 'K - Specific Learning Disability': 'Yes', 'J - Emotional/Behavioral Disability': 'Yes',
            'P - Autism Spectrum Disorder': 'Yes', 'L - Gifted': 'Yes', 'V - Other Health Impaired': 'Yes',
            'H - Deaf or Hard of Hearing': 'Yes', 'G - Language Impaired': 'Yes', 'I - Visually Impaired': 'Yes',
            'F - Speech Impaired': 'Yes', 'W - Intellectual Disabilities': 'Yes', 'M - Hospital Homebound': 'Yes',
            'C - Orthopedic Impairment': 'Yes', 'O - Dual Sensory Impaired': 'Yes', 'U - Established Conditions': 'Yes',
            'Z - Other': 'Yes',
            'RP|ELA.7.R.1.1': 'R.1.1', 'RP|ELA.7.R.1.2': 'R.1.2', 'RP|ELA.7.R.1.3': 'R.1.3', 'RP|ELA.7.R.1.4': 'R.1.4',
            'RI|ELA.7.R.2.1': 'R.2.1', 'RI|ELA.7.R.2.2': 'R.2.2', 'RI|ELA.7.R.2.3': 'R.2.3', 'RI|ELA.7.R.2.4': 'R.2.4',
            'RGV|ELA.7.R.3.1': 'R.3.1', 'RGV|ELA.7.R.3.2': 'R.3.2', 'RGV|ELA.7.R.3.3': 'R.3.3', 'RGV|ELA.7.R.3.4': 'R.3.4',
            'RGV|ELA.7.V.1.3': 'V.1.3', 'RGV|ELA.7.V.1.2': 'V.1.2',
            'RP|ELA.6.R.1.1': 'R.1.1', 'RP|ELA.6.R.1.2': 'R.1.2', 'RP|ELA.6.R.1.3': 'R.1.3', 'RP|ELA.6.R.1.4': 'R.1.4',
            'RI|ELA.6.R.2.1': 'R.2.1', 'RI|ELA.6.R.2.2': 'R.2.2', 'RI|ELA.6.R.2.3': 'R.2.3', 'RI|ELA.6.R.2.4': 'R.2.4',
            'RGV|ELA.6.R.3.1': 'R.3.1', 'RGV|ELA.6.R.3.2': 'R.3.2', 'RGV|ELA.6.R.3.3': 'R.3.3', 'RGV|ELA.6.R.3.4': 'R.3.4',
            'RGV|ELA.6.V.1.3': 'V.1.3', 'RGV|ELA.6.V.1.2': 'V.1.2',
            'RP|ELA.8.R.1.1': 'R.1.1', 'RP|ELA.8.R.1.2': 'R.1.2', 'RP|ELA.8.R.1.3': 'R.1.3', 'RP|ELA.8.R.1.4': 'R.1.4',
            'RI|ELA.8.R.2.1': 'R.2.1', 'RI|ELA.8.R.2.2': 'R.2.2', 'RI|ELA.8.R.2.3': 'R.2.3', 'RI|ELA.8.R.2.4': 'R.2.4',
            'RGV|ELA.8.R.3.1': 'R.3.1', 'RGV|ELA.8.R.3.2': 'R.3.2', 'RGV|ELA.8.R.3.3': 'R.3.3', 'RGV|ELA.8.R.3.4': 'R.3.4',
            'RGV|ELA.8.V.1.3': 'V.1.3', 'RGV|ELA.8.V.1.2': 'V.1.2',
            'N/A': ''
        }
        df = df.replace(data_map)

        # 3. Convert Wide to Long Format
        long_data = []
        for _, row in df.iterrows():
            meta = [row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5], row.iloc[6]]
            bench_values = row.iloc[7:].values
            
            for j in range(0, len(bench_values) - 1, 3):
                bm = bench_values[j]
                pe = bench_values[j+1] if j+1 < len(bench_values) else None
                pp = bench_values[j+2] if j+2 < len(bench_values) else None
                
                if pd.notna(bm) and pd.notna(pe) and pd.notna(pp) and str(bm).strip() != "":
                    long_data.append(meta + [bm, pe, pp])

        headers = ["Student Name", "504", "Primary Exceptionality", "English Learner", 
                   "Scale Score", "Level Score", "Percentile Rank", 
                   "Benchmark", "Points Earned", "Points Possible"]
        
        df_long = pd.DataFrame(long_data, columns=headers)
        df_long['Points Earned'] = pd.to_numeric(df_long['Points Earned'], errors='coerce')
        df_long['Points Possible'] = pd.to_numeric(df_long['Points Possible'], errors='coerce')

        # 4. Generate Pivot Summary
        pivot_earned = df_long.pivot_table(index='Benchmark', columns='Student Name', values='Points Earned', aggfunc='sum', fill_value=0)
        pivot_possible = df_long.pivot_table(index='Benchmark', columns='Student Name', values='Points Possible', aggfunc='sum', fill_value=0)

        st.success("✅ File processed successfully!")
        
        # Display preview tabs
        tab1, tab2 = st.tabs(["📋 Long Data Format", "📊 Benchmark Pivot Summary"])
        with tab1:
            st.dataframe(df_long.head(20))
        with tab2:
            st.write("### Earned Points by Benchmark")
            st.dataframe(pivot_earned)

        # Build Excel File in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_long.to_excel(writer, sheet_name='Data', index=False)
            pivot_earned.to_excel(writer, sheet_name='Earned Points')
            pivot_possible.to_excel(writer, sheet_name='Possible Points')

        st.download_button(
            label="📥 Download Cleaned Excel File",
            data=output.getvalue(),
            file_name="FAST_Data_Processed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
