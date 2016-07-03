import sys

path, module = sys.argv[1], sys.argv[2]
sys.path.append(path)
__import__(module)