import ast
import numpy as np
import pandas as pd
from collections import Counter
import math
import numpy as np

class FeatureBuilder:
    def __init__(self):
        self.col_freq_maps = {}                 # {(col): DataFrame with columns [hostName, col, col_freq]}
        # self.parent_process_table = None        # parent table contains indexed by (hostName, processId)
        self.parent_lookup = {}


    def _stack_diversity(self, stack):
        '''
        stack diversity calculate the percentage of stack addresses that are different in a given stack trace
        
        :param stack: a trace of stack addresses
        '''
        return len(set(stack)) / max(len(stack), 1)

    
    def _stack_jump_std(self, stack):
        '''
        calculate the standard deviation of the differences between consecutive stack addresses
        to measure the continuity of the stack trace. 
        
        :param stack: a trace of stack addresses
        '''
        if len(stack) < 2:
            return 0
        diffs = [abs(stack[i] - stack[i+1]) for i in range(len(stack)-1)]
        return np.std(diffs)
    
    def _argument_parse(self, args_str):
        '''
        check if a string representation of a list can be safely evaluated to a Python object.
        
        :param args_str: args string
        '''
        return ast.literal_eval(args_str)
    
    def _mount_ns_binary(self, ns):
        '''
        mount_ns_binary check if the mount namespace is the default one.
        
        :param ns: Description
        '''
        return int(ns == 4026531840)

    def build_parent_lookup(self, df):
        parent_process_table = df[["hostName", "processId", "processName", "userId","timestamp"]].drop_duplicates()
        parent_process_table = parent_process_table.rename(
            columns={
                    "processId": "parentProcessId",
                    "processName": "parentProcessName", 
                    "userId": "parentUserId"
                    })
        # replace parent_process_table with this
        df = parent_process_table.sort_values(["hostName", "parentProcessId", "timestamp"])

        for (host, pid), g in df.groupby(["hostName", "parentProcessId"], sort=False):
            # store as numpy arrays for speed
            ts = g["timestamp"].to_numpy()
            # store whole rows as dict-like records
            records = g.to_dict("records")
            self.parent_lookup[(host, pid)] = (ts, records)
    
    def _find_parent_process(self, row):
        key = (row["hostName"], row["parentProcessId"])
        data = self.parent_lookup.get(key)

        if data is None:
            return row

        ts, records = data
        t = row["timestamp"]

        # index of rightmost parent_timestamp < t
        i = np.searchsorted(ts, t, side="left") - 1
        if i < 0:
            return row
        
        # append the data from records[i] to row
        parent_record = records[i]
        row["parentProcessName"] = parent_record["parentProcessName"]
        row["parentUserId"] = parent_record["parentUserId"]

        return row
    
    # ===========================
    # FIT
    # ===========================
    def fit(self, df_train: pd.DataFrame):
        """
        Learn frequency encodings from traininbg data
        """

        df = df_train.copy()

        # frequency encoding maps learned from train data only, save it to transform the test data
        self.col_freq_maps = {}
        for col in ['processId','threadId','parentProcessId','userId','mountNamespace','eventId']:
            counts = (
                df.groupby(["hostName", col], dropna=False)
                  .size()
                  .reset_index(name=f"{col}_freq")
            )
            self.col_freq_maps[col] = counts
        
        return self
    
    # ===========================
    # TRANSFORM
    # ===========================
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        # What: parentProcessId and as processId mapping to a binary variable should suffice.
        # suggested by research paper
        df['is_parent_system_process'] = df['parentProcessId'].isin([0, 1, 2]).astype(int)
        df['is_system_process'] = df['processId'].isin([0, 1, 2]).astype(int)

        # What: get parent process info and append it to the dataframe
        # Why: get parent process information to expand information on child parent process relationship
        # build parent lookup tablex
        self.build_parent_lookup(df)
        df = df.apply(self._find_parent_process, axis=1)
        print(df.columns)
        df["parent_missing"] = df["parentUserId"].isna().astype(int)

        # What: Did the child process run under the same user as its parent?
        # Why: Malicious processes may run under different user accounts than their parent processes.
        df["same_user_as_parent"] = np.where(df["parentUserId"].notnull(),(df["userId"] == df["parentUserId"]).astype(int), -1)

        # What: Binary encoding of userId based on whether it is below 1000 or not.
        # Why: Distinguish between system/OS users and regular users, as system activities often
        df["userId_binary"]  = df["userId"].apply(lambda x: 1 if x < 1000 else 0)

        # What: Frequency encoding for multiple columns
        # Why:  capture commonality within each host.
        for col, m in self.col_freq_maps.items():
            df = df.merge(m, on=["hostName", col], how="left")
            df[f"{col}_freq"] = df[f"{col}_freq"].fillna(0).astype(int)

        # What: Did the parent fork and re-exec itself or spawn a different binary?
        # why: Malicious activity may involve a process spawning a different binary than itself.
        df["same_process_name_as_parent"] = np.where(df["parentProcessName"].notnull(),(df["processName"] == df["parentProcessName"]).astype(int), -1)

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

        # What: check if the program exit with an error
        # Why: error return value might indicates the function call could have been altered by a attacker.
        df["returnValue_is_error"] = (df["returnValue"] == -1).astype(int)

        # What: check if an argument contains a file path
        # Why: arguments containing file paths might indicate file access or manipulation activities.
        df['args'] = df['args'].apply(self._argument_parse)
        df['args_has_path'] = df['args'].apply(lambda x: int(any('pathname' in d['name'] for d in x)))
        df[['args_has_path', 'args']].head(10)

        # What: mount namespace binary encoding
        # Why: Distinguish between default and custom mount namespaces. all logs with userId ≥1000
        # had a mountNamespace of 4026531840, while some OS
        # userId traffic used different mountNamespace values.
        df['mountNamespace_binary'] = df['mountNamespace'].apply(self._mount_ns_binary)
        return df
    

    

    
        
