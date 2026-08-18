from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".xlsx", ".csv"}


def load_table(file_path: str | Path) -> pd.DataFrame:
    """
    Загружает XLSX или CSV и возвращает pandas DataFrame.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if extension == ".xlsx":
        dataframe = pd.read_excel(path)

    else:
        dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            "Input file contains no data."
        )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe