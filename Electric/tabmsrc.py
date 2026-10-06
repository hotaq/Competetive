import tabm, inspect
src = inspect.getsource(tabm.TabM.__init__)
print(src[:3000])
print("=====KWARGS=====")
print(inspect.getsource(tabm.MLPBackboneKwargs)[:1500])
