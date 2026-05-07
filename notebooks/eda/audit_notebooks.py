import json

import os
base = r'F:\footagent\notebooks\eda'
for nb_name in ['01_eda_statsbomb_360.ipynb', '02_eda_mvfouls.ipynb', '03_eda_soccernet_tracking.ipynb']:
    nb_file = os.path.join(base, nb_name)
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    sep = "=" * 70
    print()
    print(sep)
    print("NOTEBOOK:", nb_file)
    print(sep)
    
    for i, cell in enumerate(nb['cells']):
        cell_type = cell['cell_type']
        source = ''.join(cell.get('source', []))[:80].replace('\n', ' ')
        
        if cell_type == 'markdown':
            print(f"  Cell {i+1:2d} [MD  ] {source[:75]}")
        else:
            outputs = cell.get('outputs', [])
            has_error = any(o.get('output_type') == 'error' for o in outputs)
            has_image = any('image/png' in o.get('data', {}) for o in outputs 
                          if o.get('output_type') in ['display_data', 'execute_result'])
            has_text = any(o.get('output_type') == 'stream' for o in outputs)
            executed = cell.get('execution_count') is not None
            
            parts = []
            if not executed: parts.append('NOT-RUN')
            if has_error: parts.append('ERROR')
            if has_image: parts.append('PLOT')
            if has_text: parts.append('TEXT')
            status = ', '.join(parts) if parts else 'EMPTY'
            
            text_preview = ''
            for o in outputs:
                if o.get('output_type') == 'stream':
                    text_preview = ''.join(o.get('text', []))[:120].replace('\n', ' | ')
                    break
                elif o.get('output_type') == 'error':
                    text_preview = 'ERROR: ' + o.get('ename', '') + ': ' + str(o.get('evalue', ''))[:80]
                    break
            
            print(f"  Cell {i+1:2d} [CODE] [{status:15s}] {source[:55]}")
            if text_preview:
                print(f"          >> {text_preview[:100]}")
    
    # Summary counts
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    executed_cells = [c for c in code_cells if c.get('execution_count') is not None]
    error_cells = [c for c in code_cells if any(o.get('output_type') == 'error' for o in c.get('outputs', []))]
    plot_cells = [c for c in code_cells if any('image/png' in o.get('data', {}) for o in c.get('outputs', []) if o.get('output_type') in ['display_data', 'execute_result'])]
    
    print()
    print(f"  SUMMARY: {len(code_cells)} code cells | {len(executed_cells)} executed | {len(plot_cells)} with plots | {len(error_cells)} errors")
