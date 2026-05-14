import ast
import re
import pandas as pd


def simple_normalize_token(x):
    """Light normalization for quick NLP experiments."""
    x = str(x).lower()

    # very light normalization
    x = re.sub(r'0x[0-9a-f]+', ' <HEX> ', x)                 # hex / pointers
    x = re.sub(r'\b\d+\b', ' <NUM> ', x)                     # numbers
    x = re.sub(r'/tmp/[^ ]+', '/tmp/<TMP>', x)              # temp paths
    x = re.sub(r'/proc/\d+', '/proc/<PID>', x)              # proc pid paths
    x = re.sub(r'\s+', ' ', x).strip()

    return x


def args_to_text(args):
    """
    Convert parsed Args into one text string.
    
    Input example:
    [
      {'name': 'pathname', 'type': 'const char*', 'value': '/usr/bin/run-parts'},
      {'name': 'argv', 'type': 'const char*const*', 'value': ['run-parts', '--report', '/etc/cron.hourly']}
    ]
    """
    # if stored as string in dataframe, parse first
    if isinstance(args, str):
        try:
            args = ast.literal_eval(args)
        except Exception:
            return ""

    if not isinstance(args, list):
        return ""

    parts = []

    for item in args:
        if not isinstance(item, dict):
            continue

        name = item.get("name", "")
        value = item.get("value", "")

        if isinstance(value, list):
            value = " ".join(map(str, value))

        parts.append(f"{name} {value}")

    text = " ".join(parts)
    return simple_normalize_token(text)


def preprocess_args_column(df, col="Args", output_col="ArgsText"):
    """
    Add a simple NLP-ready text column from parsed Args.
    """
    df = df.copy()
    df[output_col] = df[col].apply(args_to_text)
    return df