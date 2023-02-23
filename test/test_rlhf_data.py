#! /usr/bin/env python3
# coding=utf-8

# Ruibo Liu @Dartmouth College
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import absl.flags
from absl.testing import absltest, parameterized

from lm_classifier.data import RLHFDataModule


class RLHFDataTest(parameterized.TestCase):
    """Test cases for RLHF data."""

    def test_loading_rlhf_helpful_base(self):
        """Test the load samples function of data model."""
        root_path = "/home/ruibo/Research/lm_classifier/assets"

        dm = RLHFDataModule(task_name="helpful-base",
                            root_path=root_path,
                            model_name_or_path='roberta-large')
        dm.setup("fit")
        print(next(iter(dm.train_dataloader())))


if __name__ == '__main__':
    absltest.main()
