import sys
print(f"Python version: {sys.version}")

try:
    import numpy
    print(f"NumPy version: {numpy.__version__}")
except ImportError as e:
    print(f"NumPy failed: {e}")

try:
    import pandas
    print(f"Pandas version: {pandas.__version__}")
except ImportError as e:
    print(f"Pandas failed: {e}")

try:
    import yfinance
    print("yfinance imported successfully")
except ImportError as e:
    print(f"yfinance failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from typing import TypedDict
    print("from typing import TypedDict success")
except ImportError:
    print("from typing import TypedDict failed")

try:
    from typing_extensions import TypedDict
    print("from typing_extensions import TypedDict success")
except ImportError:
    print("from typing_extensions import TypedDict failed")
