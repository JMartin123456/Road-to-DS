# 03 - Data Cleaning

This project focuses on cleaning an employee dataset using Python and Pandas.

The goal was to prepare raw data for further analysis by handling missing values, incorrect data types and duplicate records.

## Cleaning Steps

### Missing Values

Missing values were identified and replaced using appropriate methods depending on the column type.

### Data Types

Incorrect data formats were converted into usable formats.

Example:

```python
pd.to_numeric(errors="coerce")