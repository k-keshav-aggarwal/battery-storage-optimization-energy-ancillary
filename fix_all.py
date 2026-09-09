import json
import re

# Load the notebook
with open('Paper_Code_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix all problematic code patterns
for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    
    source = cell.get('source', [])
    if not source:
        continue
    
    # Join lines if it's a list
    text = ''.join(source) if isinstance(source, list) else source
    
    # Fix any remaining cache/pickle references
    text = re.sub(r'cache_lmp.*\.pkl', '"merged_df_clean.csv.gz"', text)
    text = re.sub(r'cache_as.*\.pkl', '"merged_df_clean.csv.gz"', text)
    text = re.sub(r'pd\.read_pickle\s*\(', 'pd.read_csv(', text)
    
    # Replace Unix paths
    text = text.replace('UPLOAD_DIR = "/mnt/user-data/uploads"', 'UPLOAD_DIR = "./data_cache"')
    text = text.replace('OUT_DIR = "/home/claude/work/pipeline/artifacts"', 'OUT_DIR = "./artifacts"')
    
    # Update cell source
    if isinstance(source, list):
        cell['source'] = text.split('\n')
        cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line 
                          for i, line in enumerate(cell['source'])]
    else:
        cell['source'] = text

# Clear all execution counts and outputs since we're fixing the code
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []

# Save the fixed notebook
with open('Paper_Code_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("✓ Notebook fully fixed - all pickle/path references replaced!")
