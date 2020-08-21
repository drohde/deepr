"""Build RecoGym dataset as TFRecords."""

import recogym, gym
from recogym import env_1_args
import pdb
import logging
import random
from typing import List, Dict, Tuple, Callable
from functools import partial
from dataclasses import dataclass
import numpy as np
import tensorflow as tf

import deepr as dpr
from deepr.examples.movielens.utils import fields



LOGGER = logging.getLogger(__name__)


FIELDS = [fields.UID, fields.INPUT_POSITIVES, fields.TARGET_POSITIVES, fields.TARGET_NEGATIVES]


mapping = {}
for ii in range(env_1_args['num_products']):
    mapping[ii] = ii

@dataclass
class BuildRecords(dpr.jobs.Job):
    """Build MovieLens dataset as TFRecords."""

    path_train: str
    path_eval: str
    path_test: str
    min_rating: int = 4
    min_length: int = 5
    num_negatives: int = 8
    target_ratio: float = 0.2
    size_test: int = 10_000
    size_eval: int = 10_000
    shuffle_timelines: bool = True
    seed: int = 2020

    def run(self):
        env = gym.make('reco-gym-v1')
        env.init_gym(env_1_args)
        logs = env.generate_logs(10000)

        timelines_test = logs[logs['u'] < 10]
        timelines_eval = logs[logs['u'] < 10]
        timelines_train = logs[logs['u'] > 80]

        # Write datasets
        for timelines, path in zip(
            [timelines_train, timelines_test, timelines_eval], [self.path_train, self.path_test, self.path_eval]
        ):
            dpr.io.Path(path).parent.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"Writing {len(timelines)} timelines to {path}")
            LOGGER.info(f"shuffle_timelines = {self.shuffle_timelines}, num_negatives = {self.num_negatives}")

            write_records(
                partial(
                    records_generator_organic,
                    timelines=timelines,
                    target_ratio=self.target_ratio,
                    num_negatives=self.num_negatives,
                    shuffle_timelines=self.shuffle_timelines,
                    mapping=mapping,
                ),
                path,
            )


def write_records(gen: Callable, path: str):
    """Write records to path."""
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types={field.name: field.dtype for field in FIELDS},
        output_shapes={field.name: field.shape for field in FIELDS},
    )
    to_example = dpr.prepros.ToExample(fields=FIELDS)
    writer = dpr.writers.TFRecordWriter(path=path)
    writer.write(to_example(dataset))


def records_generator_organic(
    timelines: List[Tuple[str, List[int]]],
    target_ratio: float,
    num_negatives: int,
    shuffle_timelines: bool,
    mapping: Dict[int, int],
):
    for u in set(timelines['u']):
        tl = timelines[timelines['u']==u]
        v = tl[tl['z'] == 'organic'].v.tolist()

        target_negatives = [random.sample(range(len(mapping)), num_negatives) for _ in range(5)]
        yield {
            fields.UID.name: str(u),
            fields.INPUT_POSITIVES.name: v[:-1],
            fields.TARGET_POSITIVES.name: [v[-1]],
            fields.TARGET_NEGATIVES.name: target_negatives,
        }


