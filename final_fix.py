import json

with open('Paper_Code_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix the two problematic cells:
# Cell 2: lines 28-48 - the setup cell with UPLOAD_DIR
# Cell 4: lines 58-110 - the load_raw/build_aggregated functions

for cell in nb['cells']:
    source = cell.get('source', [])
    if not source:
        continue
    
    text = ''.join(source) if isinstance(source, list) else source
    
    # Fix 1: Change Unix path to current directory path (appears in cell 1)
    if 'UPLOAD_DIR = "/mnt/user-data/uploads"' in text:
        text = text.replace('UPLOAD_DIR = "/mnt/user-data/uploads"', 'UPLOAD_DIR = "./data_cache"')
    
    # Fix 2: Change pickle loading to CSV loading (appears in cell 4)
    if 'pd.read_pickle(os.path.join(UPLOAD_DIR, "cache_lmp' in text:
        old_load = '''def load_raw():
    lmp = pd.read_pickle(os.path.join(UPLOAD_DIR, "cache_lmp_20230101_20251231_0c3247.pkl"))
    as_p = pd.read_pickle(os.path.join(UPLOAD_DIR, "cache_as_20230101_20251231.pkl"))
    return lmp, as_p'''
        new_load = '''def load_raw():
    lmp = pd.read_csv(os.path.join(UPLOAD_DIR, "merged_df_clean.csv.gz"))
    as_p = lmp.copy()
    lmp['datetime'] = pd.to_datetime(lmp['datetime'])
    as_p['datetime'] = lmp['datetime']
    return lmp, as_p'''
        text = text.replace(old_load, new_load)
    
    # Fix 3: Fix the build_aggregated to not try to merge identical dataframes
    if 'def build_aggregated():' in text and 'pd.merge(lmp, as_p' in text:
        old_build = '''def build_aggregated():
    lmp, as_p = load_raw()
    df = pd.merge(lmp, as_p, on="datetime", how="inner")
    agg = (df.groupby("datetime")
             .agg({"SP15": "mean", "NonSpin": "mean", "RegDown": "mean", "RegUp": "mean", "Spin": "mean"})
             .reset_index()
             .sort_values("datetime")
             .reset_index(drop=True))
    return agg'''
        new_build = '''def build_aggregated():
    lmp, as_p = load_raw()
    agg = (lmp.groupby("datetime")
             .agg({"SP15": "mean", "NonSpin": "mean", "RegDown": "mean", "RegUp": "mean", "Spin": "mean"})
             .reset_index()
             .sort_values("datetime")
             .reset_index(drop=True))
    return agg'''
        text = text.replace(old_build, new_build)
    
    # Update source
    if text != (''.join(source) if isinstance(source, list) else source):
        if isinstance(source, list):
            cell['source'] = [line + '\n' if i < len(text.split('\n')) - 1 else line 
                              for i, line in enumerate(text.split('\n'))]
        else:
            cell['source'] = text

# Clear execution data
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []

with open('Paper_Code_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("✓ Fixed")
