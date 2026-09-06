from loader import load_text_files
from splitter import split_by_structure

docs = load_text_files('D:/X/0315/data/documents')

for d in docs:
    if '说明书' in d['filename']:
        print(f'文件: {d["filename"]}')
        chunks = split_by_structure(d['content'])
        print(f'共切出 {len(chunks)} 块:')
        for i, c in enumerate(chunks):
            preview = c[:80].replace('\n', ' ')
            print(f'  {i+1}: {preview}...')