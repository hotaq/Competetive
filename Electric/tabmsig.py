import tabm, inspect
print(inspect.signature(tabm.TabM))
print(tabm.TabM.__doc__[:1500] if tabm.TabM.__doc__ else "no doc")
