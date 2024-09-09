#!/usr/bin/env python

# Copyright 2018-2020 John T. Foster
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import nbconvert

with open("assignment6.ipynb") as f:
    exporter = nbconvert.PythonExporter()
    python_file, _ = exporter.from_file(f)


with open("assignment6.py", "w") as f:
    f.write(python_file)


from assignment6 import *

class TestSolution(unittest.TestCase):

    def test_kc_model(self):
        kc = KozenyCarmen('poro_perm.csv')
        np.testing.assert_allclose(kc.df.loc[0:2, 'KC model'], np.array([0.00145, 0.00145, 0.001782]), atol=0.001)

    def test_least_squares(self):
        kc = KozenyCarmen('poro_perm.csv')

        A = np.array([[1, 2],[1, 4],[1, 6],[1,10]])
        b = np.array([6, 10, 13, 22])

        np.testing.assert_allclose(kc.least_squares(A,b), np.array([1.82857143,  1.98571429]), atol=0.001)

        A = np.array([[1, 3],[1, 9],[1, 14],[1,10]])
        b = np.array([6, 10, 11, 29])

    def test_fit(self):
        kc = KozenyCarmen('poro_perm.csv')
        kc.fit()
        np.testing.assert_allclose(kc.fit(), np.array([  1.05933127e+01,   2.35173520e+04]), atol=0.001)

    def test_fit_through_zero(self):
        kc = KozenyCarmen('poro_perm.csv')
        kc.fit_through_zero()
        np.testing.assert_allclose(kc.fit_through_zero(),26133.929742741482, atol=0.001)


if __name__ == '__main__':
    unittest.main()
