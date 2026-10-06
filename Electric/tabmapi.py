import tabm, inspect
print([x for x in dir(tabm) if "ab" in x.lower() or "lass" in str(type(getattr(tabm, x)))][:20])
print([x for x in dir(tabm) if not x.startswith("_")])
from tabm import TabMClassifier
print(inspect.signature(TabMClassifier.__init__))
print(inspect.signature(TabMClassifier.fit))
