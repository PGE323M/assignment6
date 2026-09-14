"""Transparent public tests for Assignment 6.

Copyright 2018-2026 John T. Foster. Apache-2.0.
Rewritten for the module-based assignment and required mathematical method.
"""

import ast
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from assignment6 import KozenyCarmen


class TestAgentArtifacts(unittest.TestCase):
    def test_baseline_instructions(self):
        text = Path('AGENTS.md').read_text().lower().replace('`', '')
        for fragment in ('readme.md', 'test.py', 'poro_perm.csv', 'wait for approval',
                         'assignment6.py', 'python -m unittest -v', 'git diff --check',
                         'stop', 'submit assignment 6', '.github/skills/submit-assignment/skill.md'):
            self.assertIn(fragment, text)
        self.assertNotIn('git add --', text, 'Keep submission steps in the supplied skill')

    def test_provided_skill_contract(self):
        text = Path('.github/skills/submit-assignment/SKILL.md').read_text()
        self.assertTrue(text.startswith('---\nname: submit-assignment\n'))
        self.assertIn('\ndescription:', text.split('---', 2)[1])
        plain = text.lower().replace('`', '')
        for fragment in ('submit assignment 6', 'do not edit', 'stop', 'git status --short',
                         'python -m unittest -v', 'git diff --check',
                         'git add -- agents.md assignment6.py', 'git diff --cached',
                         'git commit', 'git push origin head', 'github actions'):
            self.assertIn(fragment, plain)
        stages = re.findall(r'`git add ([^`]+)`', text)
        self.assertEqual(stages, ['-- AGENTS.md assignment6.py', '.'])
        self.assertIn('never use `git add .`', text)


class TestImplementationConstraints(unittest.TestCase):
    def test_vectorized_method_and_import_contract(self):
        tree = ast.parse(Path('assignment6.py').read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'KozenyCarmen')
        methods = {n.name: n for n in klass.body if isinstance(n, ast.FunctionDef)}
        names = ('__init__', 'kc_model', 'least_squares', 'fit', 'fit_through_zero')
        for name in names:
            self.assertIn(name, methods)
            for node in ast.walk(methods[name]):
                self.assertNotIsInstance(node, (ast.For, ast.While, ast.If, ast.IfExp,
                                                ast.ListComp, ast.DictComp, ast.SetComp,
                                                ast.GeneratorExp), 'Use vectorized methods')
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split('.')[0], ('numpy', 'pandas'))
            if isinstance(node, ast.ImportFrom):
                self.assertIn((node.module or '').split('.')[0], ('numpy', 'pandas'))
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, ('inv', 'pinv', 'lstsq', 'polyfit', 'curve_fit', 'apply', 'vectorize'))
            if isinstance(node, ast.Name):
                self.assertNotIn(node.id, ('inv', 'pinv', 'lstsq', 'polyfit', 'curve_fit', 'eval', 'exec', '__import__'))


class TestAssignment6Public(unittest.TestCase):
    def make_model(self, phi=(0.1, 0.2, 0.3), permeability=(2.0, 5.0, 9.0)):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / 'observations.csv'
        pd.DataFrame({'porosity': phi, 'permeability': permeability}).to_csv(path, index=False)
        return KozenyCarmen(path)

    def test_csv_and_transform(self):
        model = self.make_model()
        self.assertEqual(list(model.df.columns), ['porosity', 'permeability', 'KC model'])
        np.testing.assert_allclose(model.df['KC model'], [0.001/0.81, 0.008/0.64, 0.027/0.49])
        self.assertIsNone(model.kc_model())
        np.testing.assert_array_equal(model.df['permeability'], [2, 5, 9])

    def test_normal_equation_assembly_and_solution(self):
        model = self.make_model()
        A = np.array([[1., 2.], [1., 4.], [1., 6.], [1., 10.]])
        b = np.array([6., 10., 13., 22.])
        with patch('numpy.linalg.solve', wraps=np.linalg.solve) as solve:
            actual = model.least_squares(A, b)
        self.assertEqual(solve.call_count, 1, 'Use numpy.linalg.solve on the normal equations')
        args = solve.call_args.args
        np.testing.assert_allclose(args[0], A.T @ A)
        np.testing.assert_allclose(args[1], A.T @ b)
        np.testing.assert_allclose(actual, [1.8285714285714285, 1.9857142857142858])
        self.assertEqual(np.shape(actual), (2,))

    def test_free_intercept_fit(self):
        phi = np.array([0.08, 0.16, 0.27, 0.34])
        z = phi**3/(1-phi)**2
        model = self.make_model(phi, 4.0 + 1200.0*z)
        original = model.df.copy(deep=True)
        with patch.object(model, 'least_squares', wraps=model.least_squares) as fit:
            actual = model.fit()
        self.assertEqual(fit.call_count, 1)
        A, b = fit.call_args.args
        self.assertEqual(A.shape, (4, 2))
        np.testing.assert_allclose(A[:, 0], 1)
        np.testing.assert_allclose(A[:, 1], z)
        np.testing.assert_allclose(b, 4 + 1200*z)
        np.testing.assert_allclose(actual, [4, 1200], rtol=1e-10)
        pd.testing.assert_frame_equal(model.df, original)

    def test_mirrored_through_origin_fit(self):
        model = self.make_model()
        original = model.df.copy(deep=True)
        z = model.df['KC model'].to_numpy()
        k = model.df['permeability'].to_numpy()
        with patch.object(model, 'least_squares', wraps=model.least_squares) as fit:
            actual = model.fit_through_zero()
        self.assertEqual(fit.call_count, 1)
        A, b = fit.call_args.args
        self.assertEqual(A.shape, (6, 2))
        self.assertEqual(np.shape(b), (6,))
        np.testing.assert_array_equal(A[:, 0], np.ones(6))
        expected = sorted(zip(np.r_[z, -z], np.r_[k, -k]))
        np.testing.assert_allclose(sorted(zip(A[:, 1], b)), expected)
        self.assertTrue(np.isscalar(actual))
        self.assertAlmostEqual(actual, float(z @ k / (z @ z)), places=8)
        pd.testing.assert_frame_equal(model.df, original)

    def test_course_data_fit(self):
        model = KozenyCarmen('poro_perm.csv')
        np.testing.assert_allclose(model.fit(), [10.5933127, 23517.3520], rtol=1e-7)
        self.assertAlmostEqual(model.fit_through_zero(), 26133.929742741482, places=6)


if __name__ == '__main__':
    unittest.main()
