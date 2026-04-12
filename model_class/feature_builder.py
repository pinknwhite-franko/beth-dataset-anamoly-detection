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


    def _stack_diversity(self, stack) -> float:
        '''
        stack diversity calculate the percentage of stack addresses that are different in a given stack trace
        
        :param stack: a trace of stack addresses
        '''
        return len(set(stack)) / max(len(stack), 1)

    
    def _stack_jump_std(self, stack) -> float:
        '''
        calculate the standard deviation of the differences between consecutive stack addresses
        to measure the continuity of the stack trace. 
        
        :param stack: a trace of stack addresses
        '''
        if len(stack) < 2:
            return 0
        diffs = [abs(stack[i] - stack[i+1]) for i in range(len(stack)-1)]
        return np.std(diffs)
    
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
    
    def _hash(self, feature_name) -> int:
        # Generate hash, convert to int, then modulo to range (e.g., 1000)
        numeric_hash = int(hashlib.sha256(feature_name.encode("utf-8")).hexdigest(), 16) % 2_000_000_000  # much larger space
        return numeric_hash
    
    def _hash_features(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        for col in feature_cols:
            df[f"{col}_hash"] = df[col].apply(lambda x: self._hash(str(x)))
            self.hash_feature_lookup[col] = df[[col, f"{col}_hash"]].drop_duplicates().to_dict(orient='list')
        return df
    
    # def _compute_frequency_encoding(self, df: pd.DataFrame) -> None:
    #     host_idx = df["hostName"]
    #     for col in ['processId','threadId','parentProcessId','userId','mountNamespace','eventId']:
    #         key = pd.MultiIndex.from_arrays([host_idx, df[col]])
    #         freq = key.value_counts()
    #         df[f"{col}_freq"] = key.map(freq).astype(int)
    #         df[f"{col}_freq"] = np.log1p(df[f"{col}_freq"])
    #         self.col_freq_maps[col] = df[["hostName", col, f"{col}_freq"]].drop_duplicates()

    def _compute_frequency_encoding_time_based(self, df: pd.DataFrame) -> None:
        df = df.sort_values("timestamp").reset_index(drop=True)
        # number of prior rows overall
        prior_total = np.arange(len(df))
        for col in ['processId','threadId','parentProcessId','userId','mountNamespace','eventId']:
            # number of prior occurrences of this category
            prior_count = df.groupby(col).cumcount()

            # expanding past-only frequency
            df[f"{col}_past_freq"] = np.where(
                prior_total > 0,
                prior_count / prior_total,
                0.0
            )
        
        return df
    
    # ===========================
    # FIT
    # ===========================
    def fit(self, X: pd.DataFrame) -> FeatureBuilder:
        df = X.copy()

        # Hash high cardinality categorical features
        # df = self._hash_features(df, ['processName'])
        
        # Compute frequency encoding maps on training data
        # self._compute_frequency_encoding_time_based(df)
        # print(self.col_freq_maps)

        return self
    
    # ===========================
    # TRANSFORM
    # ===========================
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        df = X.copy()

        print(len(df), "rows before feature engineering")
        # df.to_csv("./staging_dataframe/observe0.csv") # delete

        # What: Frequency encoding for multiple columns
        # Why: capture commonality within each host
        # for col, freq_map in self.col_freq_maps.items():
        #     df = df.merge(freq_map, on=['hostName', col], how='left')
        #     df[f"{col}_freq"] = df[f"{col}_freq"].fillna(0).astype(int)
        df = self._compute_frequency_encoding_time_based(df)

        print(len(df), "rows after frequency encoding")
        df.to_csv("./staging_dataframe/observe_after_freq_encoding.csv") # DELETE

        # What: identify parent process info based on hostName and parentProcessId
        # Why: get parent process information to expand information on child parent process relationship
        # Generate parent process table for merging in transform
        df = self._generate_parent_process_table(df)
        df.to_csv("./staging_dataframe/observe_after_merging_parent_info.csv") # DELETE
        # df = df.merge(self.parent_process_table, on=['parentProcessId'], how='left').drop_duplicates()

        print(len(df), "rows after merging parent process info")

        # TODO: Find Build the lineage trace for each process to get more comprehensive parent-child relationship information.
        # the lineage trace can be built based on parentProcessId and processId mapping, and the event sequence can be reconstructed based on timestamp.
        # we then have to normalize the trace replacing specific arguments(noise) with placeholders to get a more general trace pattern(signal).
        '''
        Then a full trace document becomes:
        access|pathname=/etc/ld.so.cache
        openat|pathname=/etc/ld.so.cache
        stat|pathname=/usr/bin/run-parts
        clone
        -> 
        execve|pathname=/usr/bin/run-parts|argv=run-parts_--report_/etc/cron.hourly
        openat|pathname=/etc/cron.hourly
        read
        '''

        # TODO: then we feed the trace into TF-IDF or a transformer model to get a trace embedding, which can be used as a feature for anomaly detection.
        '''
        TF-IDF has to be built based on the training data to get the term frequency and inverse document frequency, and then we can transform the trace into a TF-IDF vector.
        '''

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

        # What: Binary encoding of userId based on whether it is below 1000 or not.
        # Why: Distinguish between system/OS users and regular users, as system activities often
        df["userId_binary"]  = df["userId"].apply(lambda x: 1 if x < 1000 else 0)
        df["parentUserId_binary"]  = df["parentUserId"].apply(lambda x: 1 if x < 1000 else 0)
        # df.to_csv("./staging_dataframe/observe.csv") # DELETE

        # What: Did the parent fork and re-exec itself or spawn a different binary?
        # why: Malicious activity may involve a process spawning a different binary than itself.
        df["same_process_name_as_parent"] = np.where(df["parentProcessName"].notnull(),(df["processName"] == df["parentProcessName"]).astype(int), None)

        print(len(df), "rows after processing parent and child process info and relationship")

        # What: calculate the length of a stackAddresses
        # Why: get information on how much memory does a call use. 
        df['stackAddresses'] = df['stackAddresses'].apply(lambda x: [int(i) for i in x.strip('[]').strip(' ').split(',') if i])
        df['stackAddresses_len'] = df['stackAddresses'].apply(len)

        # What: calculate the standard deviation of the differences between consecutive stack addresses
        # Why: normal stacks are often continuous, while abnormal stacks may have large jumps.
        df['stackAddresses_jump_std'] = df['stackAddresses'].apply(self._stack_jump_std)

        # What: Calculate the percentage of stack addresses that are different in a given stack trace
        # Why: For normal function call, the stackAddresses should be highly diverse (ASLR). malicious or abnormal behavior, it often manipulates and uses stack addresses.
        df['stackAddresses_unique_ratio'] = df['stackAddresses'].apply(lambda x: self._stack_diversity(x))

        df.to_csv("./staging_dataframe/after_stackAddresses_data.csv")

        print(len(df), "rows after processing stackAddresses info")

        # What: check if the program exit with an error
        # Why: error return value might indicates the function call could have been altered by a attacker.
        df["returnValue_is_error"] = (df["returnValue"] == -1).astype(int)

        # What: check if an argument contains a file path
        # Why: arguments containing file paths might indicate file access or manipulation activities.
        df['args'] = df['args'].apply(self._argument_parse)
        df['args_has_path'] = df['args'].apply(lambda x: int(any('pathname' in d['name'] for d in x)))
        # df[['args_has_path', 'args']].head(10)

        print(len(df), "rows after processing arg info")

        # What: mount namespace binary encoding
        # Why: Distinguish between default and custom mount namespaces. all logs with userId ≥1000
        # had a mountNamespace of 4026531840, while some OS
        # userId traffic used different mountNamespace values.
        df['mountNamespace_binary'] = df['mountNamespace'].apply(self._mount_ns_binary)

        # OPTIONAL: based on testing, these features don't seem to help with the model performance. 
        # What hash processName, hostName, parentProcessName
        # Why: convert high cardinality categorical features into numeric features
        # hash_lookup = self.hash_feature_lookup['processName']
        # df["processName_hash"] = df["processName"].astype(str).apply(self.hash)
        # df["parentProcessName_hash"] = df["parentProcessName"].astype(str).fillna("").apply(self.hash)
        # df["hostName_hash"] = df["hostName"].astype(str).apply(self.hash)
        # hash_lookup = self.hash_feature_lookup['hostName']
        # df['hostName_hash'] = df['hostName'].map(dict(zip(hash_lookup['hostName'], hash_lookup['hostName_hash'])))

        df = df[[
            'eventId',
            'processId_past_freq', 
            'threadId_past_freq', 
            'parentProcessId_past_freq',
            'userId_past_freq', 
            'mountNamespace_past_freq', 
            'eventId_past_freq', 
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
            'mountNamespace_binary']]
        return df
    

    

    
        
