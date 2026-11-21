import os
import pandas as pd
def save_history(history_list, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pd.DataFrame(history_list).to_csv(filepath, index=False)