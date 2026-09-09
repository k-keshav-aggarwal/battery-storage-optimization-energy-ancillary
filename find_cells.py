import json

with open('Paper_Code_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
    
print(f"Total cells: {len(nb['cells'])}")
print("\nFirst 10 cell IDs:")
for i, cell in enumerate(nb['cells'][:10]):
    cell_id = cell.get('id', 'NO ID')
    cell_type = cell.get('cell_type', 'UNKNOWN')
    print(f"  {i}: ID={cell_id}, Type={cell_type}")
    
# Find the cell right before cell 5 (ID #VSC-53cb9431)
found = False
for i, cell in enumerate(nb['cells']):
    cell_id = cell.get('id', '')
    if cell_id == 'VSC-53cb9431' or cell_id == '#VSC-53cb9431':
        print(f"\nCell 5 found at index {i}")
        if i > 0:
            print(f'Previous cell ID: {nb["cells"][i-1].get("id", "NO ID")}')
        found = True
        break
        
if not found:
    print("\nCell with ID #VSC-53cb9431 not found, searching by execution count...")
    exec_counts = []
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'code':
            outputs = cell.get('outputs', [])
            exec_count = cell.get('execution_count', None)
            if outputs:
                print(f"  Cell {i}: ID={cell.get('id', 'NO ID')}, exec_count={exec_count}, has_outputs=True")
