import warnings

# Suppress noisy DeprecationWarnings coming from third-party SWIG bindings used in
# the pipeline dependencies. They are external to our code and do not affect behavior.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"builtin type SwigPy(?:Packed|Object) has no __module__ attribute",
)
