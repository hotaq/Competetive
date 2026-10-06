import ast
src = open("/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/farm_ctb/train.py").read()
ast.parse(src)
print("SYNTAX OK", len(src))
