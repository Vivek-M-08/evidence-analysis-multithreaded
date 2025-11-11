import csv
import pandas as pd

# ==== CONFIG ====
csv1_path = "/Users/user/Documents/AI/parallel-process/output [pre-processor]/preprocessed_data.csv"
csv2_path = "/Users/user/Documents/AI/OUPUT/MAIN-SPLITS/merged_output.csv"
output_missing_rows = "missing_rows.csv"

# ==== STEP 0: Print row counts (excluding header) ====
print("📊 Row counts for the given files (excluding header):")

# CSV 1
with open(csv1_path, newline="") as fp:
    csv1_count = sum(1 for _ in csv.reader(fp)) - 1
print(f"{csv1_path}: {csv1_count}")

# CSV 2
with open(csv2_path, newline="") as fp:
    csv2_count = sum(1 for _ in csv.reader(fp)) - 1
print(f"{csv2_path}: {csv2_count}")

# ==== STEP 1: Read both CSVs ====
df_csv1 = pd.read_csv(csv1_path, dtype=str)
df_csv2 = pd.read_csv(csv2_path, dtype=str)

# ==== STEP 2: Limit columns up to 'Project Evidence' ====
if 'Project Evidence' not in df_csv1.columns:
    raise ValueError("'Project Evidence' column not found in first CSV")
if 'Project Evidence' not in df_csv2.columns:
    raise ValueError("'Project Evidence' column not found in second CSV")

# Get column names up to and including 'Project Evidence'
cols_to_check = list(df_csv1.columns[:df_csv1.columns.get_loc('Project Evidence') + 1])

df_csv1 = df_csv1[cols_to_check].astype(str)
df_csv2 = df_csv2[cols_to_check].astype(str)

# ==== STEP 3: Identify missing rows (in CSV1 but not in CSV2) ====
merged = df_csv1.merge(df_csv2.drop_duplicates(), how='left', indicator=True)
missing_rows = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

if missing_rows.empty:
    print("✅ No missing rows found.")
else:
    print(f"⚠️ Found {len(missing_rows)} missing rows in CSV2 that exist in CSV1.")

# ==== STEP 4: Write missing rows to CSV ====
missing_rows.to_csv(output_missing_rows, index=False)
print(f"💾 Missing rows saved to: {output_missing_rows}")

# ==== STEP 5: Generate Comparison Report ====
print("\n" + "="*80)
print(" COMPARISON REPORT ".center(80, "="))
print("="*80)

print(f"\n📁 FILES COMPARED:")
print(f"   • CSV 1 (Source): {csv1_path}")
print(f"   • CSV 2 (Target): {csv2_path}")

print(f"\n📊 ROW STATISTICS:")
print(f"   • Rows in CSV 1: {csv1_count}")
print(f"   • Rows in CSV 2: {csv2_count}")
print(f"   • Missing rows: {len(missing_rows)}")
if csv1_count > 0:
    print(f"   • Match rate: {((csv1_count - len(missing_rows)) / csv1_count * 100):.2f}%")

print(f"\n📋 COLUMNS COMPARED:")
print(f"   • Total columns checked: {len(cols_to_check)}")
if len(cols_to_check) > 5:
    print(f"   • Columns: {', '.join(cols_to_check[:5])}...")
else:
    print(f"   • Columns: {', '.join(cols_to_check)}")

if not missing_rows.empty:
    print(f"\n⚠️  MISSING ROWS ANALYSIS:")
    print(f"   • Total missing: {len(missing_rows)}")
    if csv1_count > 0:
        print(f"   • Percentage missing: {(len(missing_rows) / csv1_count * 100):.2f}%")
    print(f"   • Output file: {output_missing_rows}")
    
    # Show first few missing rows as sample
    print(f"\n📋 SAMPLE MISSING ROWS (first 3):")
    for idx, row in missing_rows.head(3).iterrows():
        print(f"   Row {idx + 2}:")  # +2 because of 0-indexing and header
        for col in cols_to_check[:3]:  # Show first 3 columns
            value = str(row[col])[:50]  # Truncate long values
            print(f"      • {col}: {value}")
else:
    print(f"\n✅ RESULT: All rows from CSV 1 exist in CSV 2")

print("\n" + "="*80)
print(" COMPARISON COMPLETED ".center(80, "="))
print("="*80 + "\n")