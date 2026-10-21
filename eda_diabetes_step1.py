"""Первичная проверка Diabetes Prediction Dataset.
Запуск: python eda_diabetes_step1.py 'путь/к/diabetes_prediction_dataset.csv (1).zip'
Также поддерживается непосредственный путь к CSV.
"""
from pathlib import Path
import sys
import zipfile
import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            candidates = [name for name in archive.namelist() if name.lower().endswith('.csv')]
            if len(candidates) != 1:
                raise ValueError('В архиве должен быть ровно один CSV-файл')
            with archive.open(candidates[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)


def main() -> None:
    DATA_PATH = Path(__file__).resolve().parent / "diabetes_prediction_dataset.csv"
    df = load_dataset(DATA_PATH)
    print('Размер данных:', df.shape)
    print('\nТипы данных:\n', df.dtypes.to_string())
    print('\nПервые 5 строк:\n', df.head().to_string(index=False))
    print('\nПропуски:\n', df.isna().sum().to_string())
    print('\nПолные дубликаты:', df.duplicated().sum())
    if 'diabetes' in df.columns:
        print('\nРаспределение классов:\n', df['diabetes'].value_counts().to_string())
        print('\nПроценты классов:\n', df['diabetes'].value_counts(normalize=True).mul(100).round(2).to_string())
    print('\nЧисловые признаки:\n', df.describe().round(2).to_string())
    for column in df.select_dtypes(include='object'):
        print(f'\nКатегории {column}:\n', df[column].value_counts(dropna=False).to_string())


if __name__ == '__main__':
    main()