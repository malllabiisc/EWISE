import pickle
import sys


inp_file_path=sys.argv[1]
out_file_path=sys.argv[2]

with open(inp_file_path) as f:
    lines = f.readlines()

lines = [line.split() for line in lines]
d = {(line[0].strip(), line[1].strip()):[item.strip() for item in line[2:]]for line in lines}

with open(out_file_path, 'wb') as fout:
    pickle.dump(d, fout)
