"""Generates DP synthetic data for the gas dataset with dpsynth.

Loads the gas dataset (a pickled pandas DataFrame) and the attribute bounds,
builds a DatasetDescriptor for it and runs `dpsynth.data_generation.generate`
with the local (in-process) backend.

Example:
  pip install dp-accounting absl-py pipeline-dp pyyaml mbi  more-itertools tqdm etils cattrs apache-beam
  python3 gas/generate_gas.py --epsilon=5.0 --delta=1e-6 --mechanism=aim \
      --output_path=gas/gas_synthetic.csv
"""

import argparse
import json
import os
import sys
from typing import Any
import jax
jax.config.update("jax_enable_x64", True)

import pandas as pd


# dpsynth is checked out next to this directory and is not installed.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dpsynth")
)

from dpsynth import data_generation  # pylint: disable=g-import-not-at-top
from dpsynth import domain
from dpsynth.dataset_descriptors import csv_descriptor
from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline_transformations import types
from dpsynth.pipeline_transformations import aim
import pipeline_dp


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_THIS_DIR, "gas.pkl.gz")
BOUNDS_PATH = os.path.join(_THIS_DIR, "bounds.json")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
  """Loads the gas dataset from a gzipped pickle of a pandas DataFrame."""
  df = pd.read_pickle(path, compression="gzip")

  # `DataFormat.CSV` records are `df.iterrows()` items, i.e. (index, Series)
  # pairs. Categorical columns are not supported by the CSV descriptor, and
  # `iterrows` upcasts every row to a single dtype anyway, so categorical
  # columns are cast to the dtype of their categories. For this dataset that
  # makes the whole frame float64, which the descriptor maps to DataType.FLOAT
  # and which matches the values `iterrows` yields.
  for column in df.columns:
    if isinstance(df.dtypes[column], pd.CategoricalDtype):
      df[column] = df[column].astype(df.dtypes[column].categories.dtype)
  numeric_columns = [
      c for c in df.columns if pd.api.types.is_numeric_dtype(df.dtypes[c])
  ]
  df[numeric_columns] = df[numeric_columns].astype(float)
  return df


def load_domain_spec(path: str = BOUNDS_PATH) -> dict[str, Any]:
  """Loads min/max (and categorical) bounds into a dpsynth domain spec.

  Args:
    path: Path to a JSON file mapping a column name either to
      {"lower": ..., "upper": ...} for numerical columns or to
      {"categories": [...]} for categorical ones.

  Returns:
    A mapping from column name to a dpsynth domain attribute, suitable for
    `DatasetDescriptor.update_from_domain_specification`.
  """
  with open(path, "r") as f:
    bounds = json.load(f)

  domain_spec = {}
  for name, spec in bounds.items():
    if "categories" in spec:
      domain_spec[name] = domain.CategoricalAttribute(
          possible_values=spec["categories"]
      )
    elif "lower" in spec and "upper" in spec:
      domain_spec[name] = domain.NumericalAttribute(
          min_value=spec["lower"],
          max_value=spec["upper"],
          clip_to_range=True,
          dtype="float",
      )
    else:
      raise ValueError(f"Unsupported bounds specification for '{name}': {spec}")
  return domain_spec


def create_descriptor(
    df: pd.DataFrame, domain_spec: dict[str, Any]
) -> dataset_descriptor.DatasetDescriptor:
  """Creates a DatasetDescriptor for `df` with domains taken from bounds."""
  descriptor = csv_descriptor.get_dataset_descriptor_for_csv(df)
  descriptor.update_from_domain_specification(domain_spec)

  missing = [
      attr.name for attr in descriptor.attributes if not attr.is_initialized
  ]
  if missing:
    raise ValueError(f"No bounds found for attributes: {missing}")
  return descriptor


def generate(
    df: pd.DataFrame,
    descriptor: dataset_descriptor.DatasetDescriptor,
    epsilon: float,
    delta: float,
    mechanism: data_generation.Mechanism,
    num_out_records: int | None = None,
) -> pd.DataFrame:
  """Generates synthetic data and returns it as a DataFrame."""
  config = data_generation.DataGenerationConfig(
      epsilon=epsilon,
      delta=delta,
      mechanism=data_generation.Mechanism.AIM,
      dataset_descriptor=descriptor,
      data_format=types.DataFormat.CSV,
      num_out_records=num_out_records,
      aim_parameters=aim.AIMParameters(rounds=10)
  )

  synthetic_data = data_generation.generate(
      input_data=df.iterrows(),
      config=config,
      backend=pipeline_dp.LocalBackend(),
  )

  # The local backend is lazy, materialize the generated records.
  synthetic_data = list(synthetic_data)
  return pd.DataFrame(
      list(synthetic_data), columns=list(descriptor.attribute_names)
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", default=DATA_PATH, help="Path to gas.pkl.gz")
  parser.add_argument("--bounds", default=BOUNDS_PATH, help="Path to bounds.json")
  parser.add_argument("--epsilon", type=float, default=5.0)
  parser.add_argument("--delta", type=float, default=1e-6)
  parser.add_argument(
      "--mechanism",
      type=data_generation.Mechanism,
      choices=list(data_generation.Mechanism),
      default=data_generation.Mechanism.AIM,
  )
  parser.add_argument(
      "--num_out_records",
      type=int,
      default=None,
      help="Number of synthetic records. Defaults to the (DP) input size.",
  )
  parser.add_argument(
      "--output_path",
      default=None,
      help="Where to write the synthetic data as CSV. If unset, nothing is "
      "written.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  df = load_data(args.dataset)
  print(f"Loaded {len(df)} records with columns {list(df.columns)}")

  descriptor = create_descriptor(df, load_domain_spec(args.bounds))

  synthetic_df = generate(
      df,
      descriptor,
      epsilon=args.epsilon,
      delta=args.delta,
      mechanism=data_generation.Mechanism.AIM,
      num_out_records=args.num_out_records,
  )
  print(f"Generated {len(synthetic_df)} synthetic records:")
  print(synthetic_df.head())

  if args.output_path:
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    synthetic_df.to_csv(args.output_path, index=False)
    print(f"Saved synthetic data to {args.output_path}")


if __name__ == "__main__":
  main()
