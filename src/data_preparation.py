from datasets import load_dataset

dataset = load_dataset(
    "itisashokkumar/visit-with-us-dataset"
)

df = dataset["train"].to_pandas()

print(df.shape)
print(df.head())
print(df.info())

print(df.isnull().sum())

print(df.duplicated().sum())

print(df.describe())

print(df.dtypes)