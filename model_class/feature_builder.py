import ast
import numpy as np
import pandas as pd

class FeatureBuilder:
    def __init__(self):
        self.col_freq_maps = {}  # {col: DataFrame with columns [hostName, col, col_freq]}


    def _stack_features(self, stack) -> tuple:
        """Return (length, jump_std, unique_ratio) for a stack address list in one pass."""
        n = len(stack)
        unique_ratio = len(set(stack)) / max(n, 1)
        jump_std = float(np.std(np.abs(np.diff(stack)))) if n >= 2 else 0.0
        return n, jump_std, unique_ratio

    def _generate_parent_process_table(self, df) -> pd.DataFrame:
        parent_process_lookup = df[["hostName", "processId", "processName", "userId","timestamp"]].drop_duplicates()
        parent_process_lookup = parent_process_lookup.rename(
            columns={
                    "processId": "parentProcessId",
                    "processName": "parentProcessName", 
                    "userId": "parentUserId",
                    "timestamp": "parent_timestamp",
                    }).sort_values(["parent_timestamp","hostName", "parentProcessId"]) 
        df = df.sort_values(["timestamp", "hostName", "parentProcessId"])
        df = pd.merge_asof(
            df,
            parent_process_lookup,
            left_on="timestamp",
            right_on="parent_timestamp",
            by=["hostName","parentProcessId"],
            direction="backward",        # parent time <= child time
            allow_exact_matches=False    # enforce strictly earlier
        )
        return df
        # df = df.dropna(subset=['parentProcessName', 'parentUserId'], how='all')
        # self.parent_process_table = df[["processName","parentProcessId", "parentProcessName", "parentUserId"]].drop_duplicates()
    
    def _compute_frequency_encoding(self, df: pd.DataFrame) -> None:
        host_idx = df["hostName"]
        for col in ['processId','threadId','parentProcessId','userId','mountNamespace','eventId']:
            key = pd.MultiIndex.from_arrays([host_idx, df[col]])
            freq = key.value_counts()
            df[f"{col}_freq"] = key.map(freq).astype(int)
            df[f"{col}_freq"] = np.log1p(df[f"{col}_freq"])
            self.col_freq_maps[col] = df[["hostName", col, f"{col}_freq"]].drop_duplicates()
    
    # ===========================
    # FIT
    # ===========================
    def fit(self, X: pd.DataFrame) -> "FeatureBuilder":
        self._compute_frequency_encoding(X.copy())
        return self
    
    # ===========================
    # TRANSFORM
    # ===========================
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        print(len(df), "rows before feature engineering")

        # Frequency encoding
        for col, freq_map in self.col_freq_maps.items():
            df = df.merge(freq_map, on=['hostName', col], how='left')
            df[f"{col}_freq"] = df[f"{col}_freq"].fillna(0).astype(int)
        print(len(df), "rows after frequency encoding")

        # Parent process info
        df = self._generate_parent_process_table(df)
        print(len(df), "rows after merging parent process info")

        # System process flags (vectorized)
        df['is_parent_system_process'] = df['parentProcessId'].isin([0, 1, 2]).astype(int)
        df['is_system_process'] = df['processId'].isin([0, 1, 2]).astype(int)

        # Parent relationship features (vectorized)
        df["parent_missing"] = df["parentUserId"].isna().astype(int)
        df["same_user_as_parent"] = np.where(
            df["parentUserId"].notnull(), (df["userId"] == df["parentUserId"]).astype(int), None
        )
        df["userId_binary"] = (df["userId"] < 1000).astype(int)
        df["parentUserId_binary"] = (df["parentUserId"] < 1000).astype(int)
        df["same_process_name_as_parent"] = np.where(
            df["parentProcessName"].notnull(), (df["processName"] == df["parentProcessName"]).astype(int), None
        )

        # Stack features — parse once, compute all three in a single pass
        df['stackAddresses'] = df['stackAddresses'].apply(
            lambda x: [int(i) for i in x.strip('[]').split(',') if i.strip()]
        )
        lengths, jump_stds, unique_ratios = zip(*df['stackAddresses'].apply(self._stack_features))
        df['stackAddresses_len'] = lengths
        df['stackAddresses_jump_std'] = jump_stds
        df['stackAddresses_unique_ratio'] = unique_ratios
        print(len(df), "rows after processing stackAddresses info")

        # Return value
        df["returnValue_is_error"] = (df["returnValue"] == -1).astype(int)

        # Args — parse and check path presence in one pass
        df['args_has_path'] = df['args'].apply(
            lambda s: int(any('pathname' in d['name'] for d in ast.literal_eval(s)))
        )
        print(len(df), "rows after processing arg info")

        # Mount namespace (vectorized)
        df['mountNamespace_binary'] = (df['mountNamespace'] == 4026531840).astype(int)

        return df[[
            'eventId',
            'processId_freq',
            'threadId_freq',
            'parentProcessId_freq',
            'userId_freq',
            'mountNamespace_freq',
            'eventId_freq',
            'is_system_process',
            'is_parent_system_process',
            'userId_binary',
            'parentUserId_binary',
            'same_user_as_parent',
            'same_process_name_as_parent',
            'stackAddresses_len',
            'stackAddresses_jump_std',
            'stackAddresses_unique_ratio',
            'returnValue',
            'returnValue_is_error',
            'argsNum',
            'args_has_path',
            'mountNamespace_binary',
        ]]