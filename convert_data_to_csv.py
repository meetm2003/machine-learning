import pandas as pd
df = pd.read_table("emp_salary.data")
df.to_csv('emp_salary.csv', index=False)