import tabm, inspect
print(inspect.signature(tabm.TabM.make))
print(inspect.getsource(tabm.TabM.make)[:2000])
