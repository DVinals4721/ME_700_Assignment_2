import os

def display_main_structure(startpath, max_depth=2):
    exclude = {'venv', '__pycache__', '.git', '.idea', '.vscode'}
    
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        if level > max_depth:
            continue
        
        indent = '  ' * level
        print(f'{indent}{os.path.basename(root)}/')
        
        if level < max_depth:
            subindent = '  ' * (level + 1)
            for f in files:
                if not f.startswith('.') and not f.endswith('.pyc'):
                    print(f'{subindent}{f}')
        
        dirs[:] = [d for d in dirs if d not in exclude]

# Use it like this:
display_main_structure('.')
