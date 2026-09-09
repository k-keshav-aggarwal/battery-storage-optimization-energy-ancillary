import json

# Load the notebook
with open('Paper_Code_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix all cells with the problematic paths
for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    
    source = cell.get('source', [])
    if not source:
        continue
    
    # Join lines if it's a list
    text = ''.join(source) if isinstance(source, list) else source
    
    # Replace Unix paths with Windows paths
    text = text.replace('UPLOAD_DIR = "/mnt/user-data/uploads"', 'UPLOAD_DIR = "./data_cache"')
    text = text.replace('OUT_DIR = "/home/claude/work/pipeline/artifacts"', 'OUT_DIR = "./artifacts"')
    
    # Replace pickle loading with CSV loading
    if 'pd.read_pickle' in text:
        text = text.replace(
            'lmp = pd.read_pickle(os.path.join(UPLOAD_DIR, "cache_lmp_20230101_20251231_0c3247.pkl"))\n    as_p = pd.read_pickle(os.path.join(UPLOAD_DIR, "cache_as_20230101_20251231.pkl"))\n    return lmp, as_p',
            'lmp = pd.read_csv(os.path.join(UPLOAD_DIR, "merged_df_clean.csv.gz"))\n    as_p = lmp.copy()\n    lmp["datetime"] = pd.to_datetime(lmp["datetime"])\n    as_p["datetime"] = lmp["datetime"]\n    return lmp, as_p'
        )
    
    # Update cell source
    if isinstance(source, list):
        cell['source'] = text.split('\n')
        # Add newlines back
        cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line 
                          for i, line in enumerate(cell['source'])]
    else:
        cell['source'] = text

# Save the fixed notebook
with open('Paper_Code_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("✓ Notebook fixed successfully!")
