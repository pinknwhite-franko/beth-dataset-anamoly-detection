import ast
import logging
import numpy as np
import pandas as pd
import numpy as np
import hashlib
import os

class FeatureBuilder:
    def __init__(self):
        self.hash_feature_lookup = {}
        self.col_freq_maps = {}  # {(col): DataFrame with columns [hostName, col, col_freq]}

    def _stack_features(self, stack) -> tuple:
        """Return (length, jump_std, unique_ratio) for a stack address list in one pass."""
        n = len(stack)
        unique_ratio = len(set(stack)) / max(n, 1)
        jump_std = float(np.std(np.abs(np.diff(stack)))) if n >= 2 else 0.0
        return n, jump_std, unique_ratio

    def _stack_diversity(self, stack) -> float:
        '''
        stack diversity calculate the percentage of stack addresses that are different in a given stack trace
        
        :param stack: a trace of stack addresses
        '''
        return len(set(stack)) / max(len(stack), 1)

    def _argument_parse(self, args_str) -> list:
        '''
        check if a string representation of a list can be safely evaluated to a Python object.
        
        :param args_str: args string
        '''
        return ast.literal_eval(args_str)
    
    def _mount_ns_binary(self, ns) -> int:
        '''
        mount_ns_binary check if the mount namespace is the default one.
        
        :param ns: Description
        '''
        return int(ns == 4026531840)

    def _generate_parent_process_table(self, df) -> pd.DataFrame:
        parent_process_lookup = df[["hostName", "processId", "processName", "userId","timestamp"]].drop_duplicates()
        parent_process_lookup = parent_process_lookup.rename(
            columns={
                    "processId": "parentProcessId",
                    "processName": "parentProcessName", 
                    "userId": "parentUserId",
                    "timestamp": "parent_timestamp",
                    }).sort_values(["parent_timestamp","hostName", "parentProcessId"]) # the sorting order is important for the merge_asof to work correctly
        df = df.sort_values(["timestamp", "hostName", "parentProcessId"]) # the sorting order is important for the merge_asof to work correctly
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
    
    def _hash(self, feature_name) -> int:
        # Generate hash, convert to int, then modulo to range (e.g., 1000)
        numeric_hash = int(hashlib.sha256(feature_name.encode("utf-8")).hexdigest(), 16) % 2_000_000_000  # much larger space
        return numeric_hash
    
    def _hash_features(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        for col in feature_cols:
            df[f"{col}_hash"] = df[col].apply(lambda x: self._hash(str(x)))
            self.hash_feature_lookup[col] = df[[col, f"{col}_hash"]].drop_duplicates().to_dict(orient='list')
        return df
    
    def _compute_frequency_encoding_time_based(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["hostName", "timestamp"]).reset_index(drop=True).copy()

        host_prior_total = df.groupby("hostName").cumcount()

        # Calculate the event_id account given processid per host)
        host_prior_count = df.groupby(["hostName", "processId", "eventId"]).cumcount()
        df[f"processId_eventId_past_freq"] = np.where(
            host_prior_total > 0,
            host_prior_count / host_prior_total,
            0.0
        )

        # Calculate the frequency for each column per host)
        for col in ["processId", "threadId", "userId", "mountNamespace","eventId","parentProcessId"]:
            host_prior_count = df.groupby(["hostName", col]).cumcount()

            df[f"{col}_past_freq"] = np.where(
                host_prior_total > 0,
                host_prior_count / host_prior_total,
                0.0
            )
        
        # print(df.columns)

        return df
    
    def _computer_child_process_spawn_rate_per_parent(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["hostName", "timestamp"]).reset_index(drop=True).copy()

        host_prior_total = df.groupby("hostName").cumcount()

        # Calculate the child process spawn rate (given parentProcessId per host)
        df["is_child_process_new"] = ~df.duplicated(subset=["hostName", "parentProcessId", "processId"]).astype(int)
        df["child_process_spawn_count_so_far"] = df.groupby(["hostName","parentProcessId"])["is_child_process_new"].cumsum() - 1 # subtract 1 to exclude the current process itself
        df.drop(columns=["is_child_process_new"], inplace=True)

        df["child_process_spawn_rate_so_far"] = np.where(
            host_prior_total > 0,
            df["child_process_spawn_count_so_far"] / host_prior_total,
            0.0
        )
        return df
    
    # ===========================
    # FIT
    # ===========================
    def fit(self, X: pd.DataFrame) -> FeatureBuilder:
        return self
    
    # ===========================
    # TRANSFORM
    # ===========================
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        df = X.copy()

        print(len(df), "rows before feature engineering")

        # What: identify parent process info based on hostName and parentProcessId
        # Why: get parent process information to expand information on child parent process relationship
        # Generate parent process table for merging in transform
        df = self._generate_parent_process_table(df)

        print(len(df), "rows after merging parent process info")

        # What: Time-based frequency encoding for multiple columns
        # Why: capture commonality within each host
        df = self._compute_frequency_encoding_time_based(df)

        print(len(df), "rows after frequency encoding")

        # What: parentProcessId and as processId mapping to a binary variable should suffice.
        # suggested by research paper
        df['is_parent_system_process'] = df['parentProcessId'].isin([0, 1, 2]).astype(int)
        df['is_system_process'] = df['processId'].isin([0, 1, 2]).astype(int)

        # What: get parent process info and append it to the dataframe
        # Why: get parent process information to expand information on child parent process relationship
        df["parent_missing"] = df["parentUserId"].isna().astype(int)

        # What: Did the child process run under the same user as its parent?
        # Why: Malicious processes may run under different user accounts than their parent processes.
        df["same_user_as_parent"] = np.where(df["parentUserId"].notnull(),(df["userId"] == df["parentUserId"]).astype(int), None)

        # What: Did the parent fork and re-exec itself or spawn a different binary?
        # why: Malicious activity may involve a process spawning a different binary than itself.
        df["same_process_name_as_parent"] = np.where(df["parentProcessName"].notnull(),(df["processName"] == df["parentProcessName"]).astype(int), None)

        # What: Binary encoding of userId based on whether it is below 1000 or not.
        # Why: Distinguish between system/OS users and regular users, as system activities often
        df["userId_binary"]  = df["userId"].apply(lambda x: 1 if x < 1000 else 0)
        df["parentUserId_binary"]  = df["parentUserId"].apply(lambda x: 1 if x < 1000 else 0)

        # What: mount namespace binary encoding
        # Why: Distinguish between default and custom mount namespaces. all logs with userId ≥1000
        # had a mountNamespace of 4026531840, while some OS
        # userId traffic used different mountNamespace values.
        df['mountNamespace_binary'] = df['mountNamespace'].apply(self._mount_ns_binary)


        # What: calculate the child process spawn rate given parentProcessId per host
        # Why: A parent process spawning an unusually high number of child processes may indicate malicious behavior
        df = self._computer_child_process_spawn_rate_per_parent(df)
        print(len(df), "rows after processing parent and child process info and relationship")


        # What: Stack features
        # Why: get information on the stack trace
        df['stackAddresses'] = df['stackAddresses'].apply(
            lambda x: [int(i) for i in x.strip('[]').split(',') if i.strip()]
        )
        lengths, jump_stds, unique_ratios = zip(*df['stackAddresses'].apply(self._stack_features))
        df['stackAddresses_len'] = lengths
        df['stackAddresses_jump_std'] = jump_stds
        df['stackAddresses_unique_ratio'] = unique_ratios
        print(len(df), "rows after processing stackAddresses info")

        # What: check if the program exit with an error
        # Why: error return value might indicates the function call could have been altered by a attacker.
        df["returnValue_is_error"] = (df["returnValue"] == -1).astype(int)

        # What: check if an argument contains a file path
        # Why: arguments containing file paths might indicate file access or manipulation activities.
        df['args'] = df['args'].apply(self._argument_parse)
        df['args_has_path'] = df['args'].apply(lambda x: int(any('pathname' in d['name'] for d in x)))

        print(len(df), "rows after processing arg info")

        return df[[
            "processId_eventId_past_freq",
            "processId_past_freq",
            "threadId_past_freq",
            "eventId_past_freq",
            "userId_past_freq",
            "parentProcessId_past_freq",
            "mountNamespace_past_freq",
            "is_parent_system_process",
            "is_system_process",
            "parent_missing",
            "same_user_as_parent",
            "same_process_name_as_parent",
            "userId_binary",
            "parentUserId_binary",
            "mountNamespace_binary",
            "child_process_spawn_rate_so_far",
            "stackAddresses_unique_ratio",
            "stackAddresses_len",
            "stackAddresses_jump_std",
            "returnValue",
            "returnValue_is_error",
            "args_has_path",
            "argsNum",
        ]]
    

    

    
        
