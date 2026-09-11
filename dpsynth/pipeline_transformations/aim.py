# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of the Adaptive+Iterative Mechanism (AIM)."""

import copy
import dataclasses
import itertools
from typing import Any, TypeAlias, TypeVar

from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.discrete_mechanisms import common
from dpsynth.pipeline_transformations import diagnostic_info
from dpsynth.pipeline_transformations import independent_mechanism
from dpsynth.pipeline_transformations import marginals_computations
from dpsynth.pipeline_transformations import types
import jax
import jax.numpy as jnp
import mbi
import numpy as np
import pipeline_dp


Clique: TypeAlias = tuple[int, ...]
MarginalQuery: TypeAlias = tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class AIMParameters:
  """Parameters for AIM."""

  # attributes.
  rounds: int = 100
  pgm_iters: int = 1000
  max_model_size: int = 500

def fit_model(
    backend: pipeline_dp.PipelineBackend,
    budget_accountant: pipeline_dp.BudgetAccountant,
    data: types.Collection[tuple[int, ...]],
    descriptor: types.Collection[dataset_descriptor.DatasetDescriptor],
    parameters: AIMParameters,
    workload: list[MarginalQuery] | None = None,
    additional_output: Any | None = None,
) -> types.Collection[mbi.MarkovRandomField]:
  """Fits the model."""

  # Generate workload.
  domain = backend.map(descriptor, lambda x: x.compressed_domain, 'Get domain')
  if workload is None:
    workload = backend.map(
        domain,
        _generate_workload,
        'Generate workload',
    )
  else:
    workload = backend.to_collection([workload], data, 'Create Workload')
  # workload: singleton collection of list of marginals.

  marginals = marginals_computations.compute_exact_marginals(
      backend, data, workload, domain
  )
  marginals = backend.to_list(marginals, "ToList")
  data_list = backend.to_list(data, "ToList")

  measurements = backend.map(
      descriptor,
      lambda x: list(x.compressed_measurements()),
      'Extract Measurements',
  )
  # measurements: singleton of list[LinearMeasurements]

  independent_model = independent_mechanism.fit_model(backend, descriptor)
  # model: singleton (mbi.MarkovRandomField,)

  gaussian_spec = budget_accountant.request_budget(pipeline_dp.budget_accounting.MechanismType.GAUSSIAN, weight=5)

  def aim_gdp_fn(data: list[tuple[int, ...]],
                   domain: mbi.Domain,
                   workload: list[MarginalQuery],
                   independent_model: mbi.MarkovRandomField,
                   marignals,
                   measurements
                   ) -> mbi.MarkovRandomField:
      sigma = gaussian_spec.noise_standard_deviation
      gdp_mu = 1/sigma
      return aim_gdp(data, domain, workload, gdp_mu, independent_model, marignals, measurements)

  model = backend.map_with_side_inputs(data_list, aim_gdp_fn, [domain, workload,independent_model, marginals, measurements])

  return model


def _generate_workload(domain: mbi.Domain) -> list[MarginalQuery]:
  def tuple_to_int(t: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(int(x) for x in t)

  return [
      tuple_to_int(cl)  # pyrefly: ignore[bad-argument-type]
      for cl in itertools.combinations(domain, 3)
      if domain.size(cl) <= 1e6
  ]

# T = TypeVar('T')


def _create_new_model(
    model: mbi.MarkovRandomField,
    measurements: list[mbi.LinearMeasurement],
    pgm_iters: int,
) -> mbi.Model:
  """Adds measurements to the model and running mirror descent."""
  return mbi.estimation.MirrorDescent().estimate(
      model.domain,
      copy.copy(measurements),
      warm_start=model,
      iters=pgm_iters,
  )


def aim_gdp(data: list[tuple[int, ...]],
                 domain: mbi.Domain,
                 workload: list[MarginalQuery],
                 gdp_mu: float,
                 independent_model: mbi.MarkovRandomField,
                 marginals: list[tuple[tuple[int, ...], np.ndarray]],
                 measurements: list[mbi.LinearMeasurement],
                 ) -> mbi.MarkovRandomField:
    """Implements AIM GDP. TODO

    Args:
        data: preprocessed data
        domain: Domain corresponds to preprocessed data
        workload: workload to use in aim
        gdp_mu: GDP mu
        independent_model: model fitted to data, taking columns as independent
        marginals: non-dp marginals computed for workload: list of (columns: tuple[int, ...], n-d array for marginal)
        measurements: 1-d DP measurements, measurements[i] corresponds to i-th column
    """
    # Explore arguments
    print(f"Data: {len(data)=} {data[0]=}")
    print(f"{domain=}")
    print(f"workload={len(workload)=} {workload[0]=}")
    print(f"{gdp_mu=}")
    print(f"marginals: {len(marginals)=} {marginals[0]=}")
    print(f"measurements: {len(measurements)=} {measurements[0]=}")

    # TODO: implement GDP AIM

