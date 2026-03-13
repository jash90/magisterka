"""
Testy jednostkowe dla modulu XAI (Explainable AI).

Testuje wszystkie 5 modulow XAI:
- SHAPExplainer
- LIMEExplainer
- DALEXWrapper
- EBMExplainer
- XAIComparison
"""

import pytest
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for tests

from src.xai.shap_explainer import SHAPExplainer
from src.xai.lime_explainer import LIMEExplainer
from src.xai.dalex_wrapper import DALEXWrapper
from src.xai.ebm_explainer import EBMExplainer
from src.xai.comparison import XAIComparison


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def feature_names():
    """10 feature names used across all tests."""
    return [f'feature_{i}' for i in range(10)]


@pytest.fixture(scope="module")
def sample_data():
    """50 samples x 10 features, binary target, deterministic."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 10)
    # Make target slightly correlated with first two features
    logits = 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.randn(50) * 0.5
    y = (logits > 0).astype(int)
    return X, y


@pytest.fixture(scope="module")
def trained_rf(sample_data):
    """Trained RandomForestClassifier on sample data."""
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def trained_lr(sample_data):
    """Trained LogisticRegression on sample data."""
    X, y = sample_data
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def single_instance(sample_data):
    """A single sample (1-D array) for local explanations."""
    X, _ = sample_data
    return X[0]


@pytest.fixture
def single_instance_2d(sample_data):
    """A single sample reshaped to 2-D."""
    X, _ = sample_data
    return X[0].reshape(1, -1)


# ===========================================================================
# SHAP Explainer
# ===========================================================================

class TestSHAPExplainer:
    """Tests for SHAPExplainer."""

    @pytest.fixture
    def shap_explainer(self, trained_rf, sample_data, feature_names):
        X, _ = sample_data
        return SHAPExplainer(
            model=trained_rf,
            X_background=X,
            feature_names=feature_names,
            explainer_type='auto'
        )

    # --- construction ---

    def test_init_auto_detects_tree(self, shap_explainer):
        """Auto detection should pick 'tree' for RandomForest."""
        assert shap_explainer.explainer_type == 'tree'

    def test_init_stores_feature_names(self, shap_explainer, feature_names):
        assert shap_explainer.feature_names == feature_names

    def test_init_default_feature_names(self, trained_rf, sample_data):
        """When feature_names is None, defaults are generated."""
        X, _ = sample_data
        exp = SHAPExplainer(model=trained_rf, X_background=X, feature_names=None)
        assert len(exp.feature_names) == 10
        assert exp.feature_names[0] == 'feature_0'

    def test_init_kernel_fallback(self, trained_lr, sample_data, feature_names):
        """Explicitly request kernel explainer."""
        X, _ = sample_data
        exp = SHAPExplainer(
            model=trained_lr,
            X_background=X[:20],
            feature_names=feature_names,
            explainer_type='kernel'
        )
        assert exp.explainer_type == 'kernel'

    def test_init_linear_for_logistic(self, trained_lr, sample_data, feature_names):
        """Auto-detect should pick 'linear' for LogisticRegression."""
        X, _ = sample_data
        exp = SHAPExplainer(
            model=trained_lr,
            X_background=X,
            feature_names=feature_names,
            explainer_type='auto'
        )
        assert exp.explainer_type == 'linear'

    # --- explain_instance ---

    def test_explain_instance_keys(self, shap_explainer, single_instance):
        result = shap_explainer.explain_instance(single_instance)
        expected_keys = {
            'shap_values', 'base_value', 'feature_values', 'feature_names',
            'prediction', 'probability_positive', 'feature_impacts',
            'explainer_type', 'risk_factors', 'protective_factors'
        }
        assert expected_keys.issubset(result.keys())

    def test_explain_instance_shap_values_length(self, shap_explainer, single_instance, feature_names):
        result = shap_explainer.explain_instance(single_instance)
        assert len(result['shap_values']) == len(feature_names)

    def test_explain_instance_prediction_range(self, shap_explainer, single_instance):
        result = shap_explainer.explain_instance(single_instance)
        assert result['prediction'] in (0, 1)
        assert 0.0 <= result['probability_positive'] <= 1.0

    def test_explain_instance_2d_input(self, shap_explainer, single_instance_2d):
        """2-D input should be handled without error."""
        result = shap_explainer.explain_instance(single_instance_2d)
        assert 'shap_values' in result

    def test_explain_instance_feature_impacts_sorted(self, shap_explainer, single_instance):
        result = shap_explainer.explain_instance(single_instance)
        impacts = result['feature_impacts']
        abs_vals = [abs(fi['shap_value']) for fi in impacts]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_explain_instance_direction_values(self, shap_explainer, single_instance):
        result = shap_explainer.explain_instance(single_instance)
        for fi in result['feature_impacts']:
            assert fi['direction'] in ('increases_risk', 'decreases_risk')

    def test_risk_and_protective_partition(self, shap_explainer, single_instance):
        result = shap_explainer.explain_instance(single_instance)
        total = len(result['risk_factors']) + len(result['protective_factors'])
        # features with shap_value == 0 appear in neither
        assert total <= len(result['feature_impacts'])
        for rf in result['risk_factors']:
            assert rf['shap_value'] > 0
        for pf in result['protective_factors']:
            assert pf['shap_value'] < 0

    # --- get_global_importance ---

    def test_global_importance_keys(self, shap_explainer, sample_data, feature_names):
        X, _ = sample_data
        importance = shap_explainer.get_global_importance(X[:20])
        assert set(importance.keys()) == set(feature_names)

    def test_global_importance_sorted_desc(self, shap_explainer, sample_data):
        X, _ = sample_data
        importance = shap_explainer.get_global_importance(X[:20])
        vals = list(importance.values())
        assert vals == sorted(vals, key=abs, reverse=True)

    def test_global_importance_unknown_method(self, shap_explainer, sample_data):
        X, _ = sample_data
        with pytest.raises(ValueError, match="Nieznana metoda"):
            shap_explainer.get_global_importance(X[:10], method='bogus')

    def test_global_importance_mean_method(self, shap_explainer, sample_data):
        """The 'mean' aggregation method should work without error."""
        X, _ = sample_data
        importance = shap_explainer.get_global_importance(X[:20], method='mean')
        assert len(importance) == 10

    def test_global_importance_max_method(self, shap_explainer, sample_data):
        """The 'max' aggregation method should work without error."""
        X, _ = sample_data
        importance = shap_explainer.get_global_importance(X[:20], method='max')
        assert len(importance) == 10

    # --- get_feature_ranking ---

    def test_feature_ranking_returns_list(self, shap_explainer, single_instance):
        exp = shap_explainer.explain_instance(single_instance)
        ranking = shap_explainer.get_feature_ranking(exp)
        assert isinstance(ranking, list)
        assert len(ranking) == 10

    # --- to_json ---

    def test_to_json_valid(self, shap_explainer, single_instance):
        exp = shap_explainer.explain_instance(single_instance)
        json_str = shap_explainer.to_json(exp)
        parsed = json.loads(json_str)
        assert 'base_risk' in parsed
        assert 'predicted_risk' in parsed
        assert 'top_factors' in parsed
        assert 'summary' in parsed

    def test_to_json_top_n(self, shap_explainer, single_instance):
        exp = shap_explainer.explain_instance(single_instance)
        json_str = shap_explainer.to_json(exp, top_n=3)
        parsed = json.loads(json_str)
        assert len(parsed['top_factors']) == 3

    def test_to_json_summary_direction(self, shap_explainer, single_instance):
        exp = shap_explainer.explain_instance(single_instance)
        json_str = shap_explainer.to_json(exp)
        parsed = json.loads(json_str)
        assert parsed['summary']['dominant_direction'] in ('risk', 'protective')

    # --- to_patient_friendly ---

    def test_patient_friendly_structure(self, shap_explainer, single_instance):
        exp = shap_explainer.explain_instance(single_instance)
        friendly = shap_explainer.to_patient_friendly(exp)
        assert 'risk_level' in friendly
        assert friendly['risk_level'] in ('niski', 'umiarkowany', 'podwyższony')
        assert 'main_concerns' in friendly
        assert 'positive_factors' in friendly
        assert 'note' in friendly

    # --- explain_dataset ---

    def test_explain_dataset_returns_keys(self, shap_explainer, sample_data):
        X, _ = sample_data
        result = shap_explainer.explain_dataset(X[:10])
        assert 'shap_values' in result
        assert 'base_value' in result
        assert result['shap_values'].shape == (10, 10)

    def test_explain_dataset_max_samples(self, shap_explainer, sample_data):
        X, _ = sample_data
        result = shap_explainer.explain_dataset(X, max_samples=5)
        assert result['shap_values'].shape[0] == 5


# ===========================================================================
# LIME Explainer
# ===========================================================================

class TestLIMEExplainer:
    """Tests for LIMEExplainer."""

    @pytest.fixture
    def lime_explainer(self, trained_rf, sample_data, feature_names):
        X, _ = sample_data
        return LIMEExplainer(
            model=trained_rf,
            X_train=X,
            feature_names=feature_names,
            random_state=42
        )

    # --- construction ---

    def test_init_stores_attributes(self, lime_explainer, feature_names):
        assert lime_explainer.feature_names == feature_names
        assert lime_explainer.class_names == ['Przeżycie', 'Zgon']
        assert lime_explainer.mode == 'classification'

    def test_init_custom_class_names(self, trained_rf, sample_data, feature_names):
        X, _ = sample_data
        explainer = LIMEExplainer(
            model=trained_rf,
            X_train=X,
            feature_names=feature_names,
            class_names=['Alive', 'Dead']
        )
        assert explainer.class_names == ['Alive', 'Dead']

    def test_init_predict_fn_set(self, lime_explainer):
        assert lime_explainer.predict_fn is not None

    def test_init_random_state(self, lime_explainer):
        assert lime_explainer.random_state == 42

    # --- explain_instance ---

    def test_explain_instance_keys(self, lime_explainer, single_instance):
        result = lime_explainer.explain_instance(single_instance, num_samples=500)
        expected_keys = {
            'prediction', 'prediction_label', 'probability',
            'probability_positive', 'feature_weights', 'intercept',
            'local_prediction', 'instance_values',
            'risk_factors', 'protective_factors'
        }
        assert expected_keys.issubset(result.keys())

    def test_explain_instance_prediction_range(self, lime_explainer, single_instance):
        result = lime_explainer.explain_instance(single_instance, num_samples=500)
        assert result['prediction'] in (0, 1)
        assert 0.0 <= result['probability_positive'] <= 1.0

    def test_explain_instance_2d_input(self, lime_explainer, single_instance_2d):
        """2-D input is flattened internally."""
        result = lime_explainer.explain_instance(single_instance_2d, num_samples=500)
        assert 'feature_weights' in result

    def test_explain_instance_feature_weights_are_tuples(self, lime_explainer, single_instance):
        result = lime_explainer.explain_instance(single_instance, num_samples=500)
        for fw in result['feature_weights']:
            assert len(fw) == 2  # (description, weight)
            assert isinstance(fw[1], float)

    def test_risk_protective_split(self, lime_explainer, single_instance):
        result = lime_explainer.explain_instance(single_instance, num_samples=500)
        for feat, weight in result['risk_factors']:
            assert weight > 0
        for feat, weight in result['protective_factors']:
            assert weight < 0

    def test_explain_instance_probability_keys(self, lime_explainer, single_instance):
        result = lime_explainer.explain_instance(single_instance, num_samples=500)
        assert 'Przeżycie' in result['probability']
        assert 'Zgon' in result['probability']
        total = result['probability']['Przeżycie'] + result['probability']['Zgon']
        assert abs(total - 1.0) < 1e-6

    def test_explain_instance_values_count(self, lime_explainer, single_instance, feature_names):
        result = lime_explainer.explain_instance(single_instance, num_samples=500)
        assert len(result['instance_values']) == len(feature_names)

    # --- get_feature_importance ---

    def test_get_feature_importance_returns_dict(self, lime_explainer, single_instance):
        exp = lime_explainer.explain_instance(single_instance, num_samples=500)
        importance = lime_explainer.get_feature_importance(exp)
        assert isinstance(importance, dict)
        assert len(importance) > 0

    # --- get_feature_ranking ---

    def test_feature_ranking_list(self, lime_explainer, single_instance):
        exp = lime_explainer.explain_instance(single_instance, num_samples=500)
        ranking = lime_explainer.get_feature_ranking(exp)
        assert isinstance(ranking, list)
        assert len(ranking) > 0

    def test_feature_ranking_non_absolute(self, lime_explainer, single_instance):
        exp = lime_explainer.explain_instance(single_instance, num_samples=500)
        ranking_abs = lime_explainer.get_feature_ranking(exp, absolute=True)
        ranking_raw = lime_explainer.get_feature_ranking(exp, absolute=False)
        # Both should be lists of the same features but possibly different order
        assert set(ranking_abs) == set(ranking_raw)

    # --- to_json ---

    def test_to_json_valid(self, lime_explainer, single_instance):
        exp = lime_explainer.explain_instance(single_instance, num_samples=500)
        json_str = lime_explainer.to_json(exp)
        parsed = json.loads(json_str)
        assert 'prediction' in parsed
        assert 'feature_contributions' in parsed
        for fc in parsed['feature_contributions']:
            assert fc['direction'] in ('increases_risk', 'decreases_risk')

    def test_to_json_has_intercept(self, lime_explainer, single_instance):
        exp = lime_explainer.explain_instance(single_instance, num_samples=500)
        json_str = lime_explainer.to_json(exp)
        parsed = json.loads(json_str)
        assert 'intercept' in parsed

    # --- to_patient_friendly ---

    def test_patient_friendly(self, lime_explainer, single_instance):
        exp = lime_explainer.explain_instance(single_instance, num_samples=500)
        friendly = lime_explainer.to_patient_friendly(exp)
        assert friendly['risk_level'] in ('niski', 'umiarkowany', 'podwyższony')
        assert 'recommendation' in friendly
        assert 'risk_description' in friendly

    # --- explain_batch ---

    def test_explain_batch(self, lime_explainer, sample_data):
        X, _ = sample_data
        results = lime_explainer.explain_batch(X[:3], num_features=5, num_samples=500)
        assert len(results) == 3
        for r in results:
            assert 'prediction' in r

    # --- _calculate_ranking_consistency ---

    def test_ranking_consistency_empty(self, lime_explainer):
        assert lime_explainer._calculate_ranking_consistency([]) == 0.0

    def test_ranking_consistency_perfect(self, lime_explainer):
        rankings = [['a', 'b'], ['a', 'c'], ['a', 'd']]
        assert lime_explainer._calculate_ranking_consistency(rankings) == 1.0

    def test_ranking_consistency_none_filtered(self, lime_explainer):
        rankings = [[], ['a', 'b']]
        # first entry has no top feature -> filtered out
        result = lime_explainer._calculate_ranking_consistency(rankings)
        assert 0.0 <= result <= 1.0


# ===========================================================================
# DALEX Wrapper
# ===========================================================================

class TestDALEXWrapper:
    """Tests for DALEXWrapper."""

    @pytest.fixture
    def dalex_wrapper(self, trained_rf, sample_data, feature_names):
        X, y = sample_data
        return DALEXWrapper(
            model=trained_rf,
            X=X,
            y=y,
            feature_names=feature_names,
            label='TestRF'
        )

    # --- construction ---

    def test_init_creates_explainer(self, dalex_wrapper):
        assert dalex_wrapper.explainer is not None
        assert dalex_wrapper.label == 'TestRF'

    def test_init_converts_y_dtype(self, dalex_wrapper):
        assert dalex_wrapper.y.dtype == np.float64

    def test_init_stores_feature_names(self, dalex_wrapper, feature_names):
        assert dalex_wrapper.feature_names == feature_names

    def test_init_X_is_dataframe(self, dalex_wrapper):
        import pandas as pd
        assert isinstance(dalex_wrapper.X, pd.DataFrame)

    def test_init_custom_predict_function(self, trained_rf, sample_data, feature_names):
        X, y = sample_data
        custom_fn = lambda m, d: m.predict(d).astype(float)
        wrapper = DALEXWrapper(
            model=trained_rf, X=X, y=y,
            feature_names=feature_names,
            label='CustomPred',
            predict_function=custom_fn
        )
        assert wrapper.explainer is not None

    # --- explain_instance_break_down ---

    def test_break_down_keys(self, dalex_wrapper, single_instance):
        result = dalex_wrapper.explain_instance_break_down(single_instance)
        expected_keys = {
            'intercept', 'prediction', 'contributions', 'type',
            'risk_factors', 'protective_factors'
        }
        assert expected_keys.issubset(result.keys())

    def test_break_down_type(self, dalex_wrapper, single_instance):
        result = dalex_wrapper.explain_instance_break_down(single_instance, interaction=False)
        assert result['type'] == 'break_down'

    def test_break_down_interaction_type(self, dalex_wrapper, single_instance):
        result = dalex_wrapper.explain_instance_break_down(single_instance, interaction=True)
        assert result['type'] == 'break_down_interactions'

    def test_break_down_2d_input(self, dalex_wrapper, single_instance_2d):
        result = dalex_wrapper.explain_instance_break_down(single_instance_2d)
        assert 'contributions' in result

    def test_break_down_contributions_structure(self, dalex_wrapper, single_instance):
        result = dalex_wrapper.explain_instance_break_down(single_instance)
        for c in result['contributions']:
            assert 'variable' in c
            assert 'contribution' in c
            assert 'cumulative' in c

    def test_break_down_risk_protective(self, dalex_wrapper, single_instance):
        result = dalex_wrapper.explain_instance_break_down(single_instance)
        for rf in result['risk_factors']:
            assert rf['contribution'] > 0
        for pf in result['protective_factors']:
            assert pf['contribution'] < 0

    # --- explain_instance_shap ---

    def test_dalex_shap_keys(self, dalex_wrapper, single_instance):
        result = dalex_wrapper.explain_instance_shap(single_instance, B=5)
        assert 'shap_values' in result
        assert 'sorted_features' in result
        assert 'intercept' in result

    def test_dalex_shap_sorted_features(self, dalex_wrapper, single_instance):
        result = dalex_wrapper.explain_instance_shap(single_instance, B=5)
        sorted_feats = result['sorted_features']
        abs_vals = [abs(sf['shap_value']) for sf in sorted_feats]
        assert abs_vals == sorted(abs_vals, reverse=True)

    # --- get_variable_importance ---

    def test_variable_importance_returns_dict(self, dalex_wrapper):
        vi = dalex_wrapper.get_variable_importance(B=3)
        assert isinstance(vi, dict)
        assert len(vi) > 0

    def test_variable_importance_sorted(self, dalex_wrapper):
        vi = dalex_wrapper.get_variable_importance(B=3)
        vals = list(vi.values())
        assert vals == sorted(vals, reverse=True)

    def test_variable_importance_excludes_baselines(self, dalex_wrapper):
        vi = dalex_wrapper.get_variable_importance(B=3)
        assert '_baseline_' not in vi
        assert '_full_model_' not in vi

    # --- to_json ---

    def test_to_json_valid(self, dalex_wrapper, single_instance):
        exp = dalex_wrapper.explain_instance_break_down(single_instance)
        json_str = dalex_wrapper.to_json(exp)
        parsed = json.loads(json_str)
        assert 'contributions' in parsed

    # --- get_partial_dependence ---

    def test_partial_dependence_keys(self, dalex_wrapper, feature_names):
        pdp = dalex_wrapper.get_partial_dependence(feature_names[0])
        assert 'feature' in pdp
        assert 'x_values' in pdp
        assert 'y_values' in pdp
        assert pdp['feature'] == feature_names[0]

    def test_partial_dependence_lists_not_empty(self, dalex_wrapper, feature_names):
        pdp = dalex_wrapper.get_partial_dependence(feature_names[0])
        assert len(pdp['x_values']) > 0
        assert len(pdp['y_values']) > 0

    # --- get_model_performance ---

    def test_model_performance_returns_dict(self, dalex_wrapper):
        perf = dalex_wrapper.get_model_performance()
        assert isinstance(perf, dict)


# ===========================================================================
# EBM Explainer
# ===========================================================================

class TestEBMExplainer:
    """Tests for EBMExplainer."""

    @pytest.fixture(scope="class")
    def fitted_ebm(self):
        """EBM fitted on small dataset (class-scoped to avoid re-training)."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        logits = 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.randn(50) * 0.5
        y = (logits > 0).astype(int)
        names = [f'feature_{i}' for i in range(10)]

        ebm = EBMExplainer(
            feature_names=names,
            max_rounds=50,
            interactions=2,
            outer_bags=2,
            random_state=42
        )
        ebm.fit(X, y)
        return ebm, X, y, names

    @pytest.fixture
    def unfitted_ebm(self, feature_names):
        return EBMExplainer(feature_names=feature_names, max_rounds=10, random_state=42)

    # --- construction ---

    def test_init_not_fitted(self, unfitted_ebm):
        assert unfitted_ebm._is_fitted is False

    def test_init_default_class_names(self, unfitted_ebm):
        assert unfitted_ebm.class_names == ['Przeżycie', 'Zgon']

    def test_init_custom_class_names(self):
        ebm = EBMExplainer(class_names=['No', 'Yes'])
        assert ebm.class_names == ['No', 'Yes']

    def test_init_stores_feature_names(self, feature_names):
        ebm = EBMExplainer(feature_names=feature_names)
        assert ebm.feature_names == feature_names

    # --- fit ---

    def test_fit_sets_flag(self, fitted_ebm):
        ebm, _, _, _ = fitted_ebm
        assert ebm._is_fitted is True

    def test_fit_returns_self(self):
        rng = np.random.RandomState(99)
        X = rng.randn(30, 5)
        y = rng.randint(0, 2, 30)
        ebm = EBMExplainer(max_rounds=10, outer_bags=1, interactions=0, random_state=99)
        result = ebm.fit(X, y)
        assert result is ebm

    def test_fit_auto_feature_names(self):
        rng = np.random.RandomState(99)
        X = rng.randn(30, 5)
        y = rng.randint(0, 2, 30)
        ebm = EBMExplainer(feature_names=None, max_rounds=10, outer_bags=1, interactions=0, random_state=99)
        ebm.fit(X, y)
        assert ebm.feature_names == [f'feature_{i}' for i in range(5)]

    def test_fit_override_feature_names(self):
        rng = np.random.RandomState(99)
        X = rng.randn(30, 3)
        y = rng.randint(0, 2, 30)
        ebm = EBMExplainer(feature_names=['a', 'b', 'c'], max_rounds=10, outer_bags=1, interactions=0, random_state=99)
        ebm.fit(X, y, feature_names=['x', 'y', 'z'])
        assert ebm.feature_names == ['x', 'y', 'z']

    # --- error handling for unfitted model ---

    def test_predict_unfitted_raises(self, unfitted_ebm, sample_data):
        X, _ = sample_data
        with pytest.raises(RuntimeError, match="nie jest wytrenowany"):
            unfitted_ebm.predict(X[:1])

    def test_predict_proba_unfitted_raises(self, unfitted_ebm, sample_data):
        X, _ = sample_data
        with pytest.raises(RuntimeError, match="nie jest wytrenowany"):
            unfitted_ebm.predict_proba(X[:1])

    def test_explain_global_unfitted_raises(self, unfitted_ebm):
        with pytest.raises(RuntimeError, match="nie jest wytrenowany"):
            unfitted_ebm.explain_global()

    def test_explain_local_unfitted_raises(self, unfitted_ebm, single_instance):
        with pytest.raises(RuntimeError, match="nie jest wytrenowany"):
            unfitted_ebm.explain_local(single_instance)

    def test_get_feature_function_unfitted_raises(self, unfitted_ebm):
        with pytest.raises(RuntimeError, match="nie jest wytrenowany"):
            unfitted_ebm.get_feature_function('feature_0')

    # --- predict ---

    def test_predict_output_shape(self, fitted_ebm):
        ebm, X, _, _ = fitted_ebm
        preds = ebm.predict(X[:5])
        assert preds.shape == (5,)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_output_shape(self, fitted_ebm):
        ebm, X, _, _ = fitted_ebm
        proba = ebm.predict_proba(X[:5])
        assert proba.shape == (5, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    # --- explain_global ---

    def test_explain_global_keys(self, fitted_ebm):
        ebm, _, _, _ = fitted_ebm
        result = ebm.explain_global()
        expected_keys = {
            'feature_importance', 'feature_names', 'feature_scores',
            'interactions_detected', 'n_features', 'n_interactions'
        }
        assert expected_keys.issubset(result.keys())

    def test_explain_global_importance_sorted(self, fitted_ebm):
        ebm, _, _, _ = fitted_ebm
        result = ebm.explain_global()
        scores = result['feature_scores']
        assert scores == sorted(scores, key=abs, reverse=True)

    def test_explain_global_interactions_list(self, fitted_ebm):
        ebm, _, _, _ = fitted_ebm
        result = ebm.explain_global()
        for interaction in result['interactions_detected']:
            assert ' x ' in interaction

    # --- explain_local ---

    def test_explain_local_keys(self, fitted_ebm):
        ebm, X, _, _ = fitted_ebm
        result = ebm.explain_local(X[0])
        expected_keys = {
            'prediction', 'prediction_label', 'probability',
            'probability_positive', 'intercept', 'contributions',
            'risk_factors', 'protective_factors'
        }
        assert expected_keys.issubset(result.keys())

    def test_explain_local_2d_input(self, fitted_ebm):
        ebm, X, _, _ = fitted_ebm
        result = ebm.explain_local(X[0].reshape(1, -1))
        assert 'contributions' in result

    def test_explain_local_contributions_direction(self, fitted_ebm):
        ebm, X, _, _ = fitted_ebm
        result = ebm.explain_local(X[0])
        for c in result['contributions']:
            assert c['direction'] in ('increases_risk', 'decreases_risk')

    def test_explain_local_contributions_sorted(self, fitted_ebm):
        ebm, X, _, _ = fitted_ebm
        result = ebm.explain_local(X[0])
        abs_scores = [abs(c['score']) for c in result['contributions']]
        assert abs_scores == sorted(abs_scores, reverse=True)

    def test_explain_local_prediction_range(self, fitted_ebm):
        ebm, X, _, _ = fitted_ebm
        result = ebm.explain_local(X[0])
        assert result['prediction'] in (0, 1)
        assert 0.0 <= result['probability_positive'] <= 1.0

    def test_explain_local_risk_protective(self, fitted_ebm):
        ebm, X, _, _ = fitted_ebm
        result = ebm.explain_local(X[0])
        for c in result['risk_factors']:
            assert c['score'] > 0
        for c in result['protective_factors']:
            assert c['score'] < 0

    # --- get_feature_importance ---

    def test_get_feature_importance_dict(self, fitted_ebm):
        ebm, _, _, _ = fitted_ebm
        importance = ebm.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0

    # --- get_feature_function ---

    def test_get_feature_function_valid(self, fitted_ebm):
        ebm, _, _, _ = fitted_ebm
        # EBM may zero-pad feature names internally; use actual global names
        global_exp = ebm.model.explain_global()
        actual_name = global_exp.data()['names'][0]
        func = ebm.get_feature_function(actual_name)
        assert 'feature' in func
        assert func['feature'] == actual_name
        assert 'names' in func
        assert 'scores' in func

    def test_get_feature_function_missing_raises(self, fitted_ebm):
        ebm, _, _, _ = fitted_ebm
        with pytest.raises(ValueError, match="nie istnieje"):
            ebm.get_feature_function('nonexistent_feature_xyz')

    # --- to_json ---

    def test_to_json_local(self, fitted_ebm):
        ebm, X, _, _ = fitted_ebm
        exp = ebm.explain_local(X[0])
        json_str = ebm.to_json(exp)
        parsed = json.loads(json_str)
        assert 'contributions' in parsed

    def test_to_json_global(self, fitted_ebm):
        ebm, _, _, _ = fitted_ebm
        exp = ebm.explain_global()
        json_str = ebm.to_json(exp)
        parsed = json.loads(json_str)
        assert 'feature_importance' in parsed

    # --- save / load ---

    def test_save_load_round_trip(self, fitted_ebm, tmp_path):
        ebm, X, _, _ = fitted_ebm
        path = str(tmp_path / "ebm_model.joblib")
        ebm.save_model(path)

        new_ebm = EBMExplainer()
        new_ebm.load_model(path)
        assert new_ebm._is_fitted is True

        proba_orig = ebm.predict_proba(X[:3])
        proba_loaded = new_ebm.predict_proba(X[:3])
        np.testing.assert_allclose(proba_orig, proba_loaded)

    def test_save_unfitted_raises(self, unfitted_ebm, tmp_path):
        path = str(tmp_path / "bad.joblib")
        with pytest.raises(RuntimeError, match="nie jest wytrenowany"):
            unfitted_ebm.save_model(path)

    # --- get_model_summary ---

    def test_model_summary_keys(self, fitted_ebm):
        ebm, _, _, _ = fitted_ebm
        summary = ebm.get_model_summary()
        assert 'model_type' in summary
        assert summary['model_type'] == 'ExplainableBoostingClassifier'
        assert summary['is_fitted'] is True
        assert 'n_features' in summary
        assert 'top_5_features' in summary
        assert len(summary['top_5_features']) <= 5


# ===========================================================================
# XAI Comparison
# ===========================================================================

class TestXAIComparison:
    """Tests for XAIComparison."""

    @pytest.fixture
    def comparison(self, feature_names):
        return XAIComparison(feature_names=feature_names)

    @pytest.fixture
    def shap_explanation(self, feature_names):
        """Synthetic SHAP-format explanation."""
        return {
            'feature_impacts': [
                {
                    'feature': feature_names[i],
                    'shap_value': float(10 - i) * (1 if i % 2 == 0 else -1),
                    'feature_value': float(i),
                    'direction': 'increases_risk' if i % 2 == 0 else 'decreases_risk'
                }
                for i in range(10)
            ],
            'risk_factors': [],
            'protective_factors': [],
        }

    @pytest.fixture
    def lime_explanation(self, feature_names):
        """Synthetic LIME-format explanation."""
        return {
            'feature_weights': [
                (feature_names[i], float(10 - i) * (1 if i % 3 != 0 else -1))
                for i in range(10)
            ],
            'risk_factors': [],
            'protective_factors': [],
        }

    @pytest.fixture
    def ebm_explanation(self, feature_names):
        """Synthetic EBM-format explanation."""
        return {
            'contributions': [
                {
                    'feature': feature_names[i],
                    'score': float(10 - i) * (1 if i < 5 else -1),
                    'value': float(i),
                    'direction': 'increases_risk' if i < 5 else 'decreases_risk'
                }
                for i in range(10)
            ],
        }

    @pytest.fixture
    def mixed_explanations(self, shap_explanation, lime_explanation, ebm_explanation):
        return {
            'SHAP': shap_explanation,
            'LIME': lime_explanation,
            'EBM': ebm_explanation
        }

    # --- construction ---

    def test_init_stores_names(self, comparison, feature_names):
        assert comparison.feature_names == feature_names
        assert comparison.comparison_results == {}

    def test_init_default_class_names(self, comparison):
        assert comparison.class_names == ['Przeżycie', 'Zgon']

    def test_init_custom_class_names(self, feature_names):
        comp = XAIComparison(feature_names=feature_names, class_names=['A', 'B'])
        assert comp.class_names == ['A', 'B']

    # --- compare_feature_rankings ---

    def test_compare_rankings_keys(self, comparison, mixed_explanations):
        result = comparison.compare_feature_rankings(mixed_explanations)
        expected_keys = {
            'rankings', 'importance_scores', 'agreement_matrix',
            'common_top_features', 'spearman_correlations', 'top_n'
        }
        assert expected_keys.issubset(result.keys())

    def test_compare_rankings_methods_present(self, comparison, mixed_explanations):
        result = comparison.compare_feature_rankings(mixed_explanations)
        assert set(result['rankings'].keys()) == {'SHAP', 'LIME', 'EBM'}

    def test_agreement_matrix_shape(self, comparison, mixed_explanations):
        result = comparison.compare_feature_rankings(mixed_explanations)
        matrix = result['agreement_matrix']
        assert matrix.shape == (3, 3)
        # diagonal is 1
        np.testing.assert_array_equal(np.diag(matrix.values), [1.0, 1.0, 1.0])

    def test_agreement_matrix_symmetry(self, comparison, mixed_explanations):
        result = comparison.compare_feature_rankings(mixed_explanations)
        matrix = result['agreement_matrix'].values
        np.testing.assert_allclose(matrix, matrix.T)

    def test_spearman_correlations_shape(self, comparison, mixed_explanations):
        result = comparison.compare_feature_rankings(mixed_explanations)
        corr = result['spearman_correlations']
        assert corr.shape == (3, 3)
        np.testing.assert_array_equal(np.diag(corr.values), [1.0, 1.0, 1.0])

    def test_common_top_features_not_empty(self, comparison, mixed_explanations):
        result = comparison.compare_feature_rankings(mixed_explanations, top_n=10)
        # All three methods include all 10 features, so all should be common
        assert len(result['common_top_features']) > 0

    def test_top_n_limits_ranking(self, comparison, mixed_explanations):
        result = comparison.compare_feature_rankings(mixed_explanations, top_n=3)
        for ranking in result['rankings'].values():
            assert len(ranking) <= 3

    def test_compare_rankings_saves_to_results(self, comparison, mixed_explanations):
        comparison.compare_feature_rankings(mixed_explanations)
        assert 'rankings' in comparison.comparison_results

    # --- calculate_agreement ---

    def test_calculate_agreement_keys(self, comparison, mixed_explanations):
        result = comparison.calculate_agreement(mixed_explanations)
        expected_keys = {
            'mean_ranking_agreement', 'direction_agreement',
            'spearman_mean', 'common_features', 'n_methods'
        }
        assert expected_keys.issubset(result.keys())

    def test_calculate_agreement_n_methods(self, comparison, mixed_explanations):
        result = comparison.calculate_agreement(mixed_explanations)
        assert result['n_methods'] == 3

    def test_calculate_agreement_range(self, comparison, mixed_explanations):
        result = comparison.calculate_agreement(mixed_explanations)
        assert 0.0 <= result['mean_ranking_agreement'] <= 1.0
        assert 0.0 <= result['direction_agreement'] <= 1.0

    def test_calculate_agreement_single_method(self, comparison, shap_explanation):
        result = comparison.calculate_agreement({'SHAP': shap_explanation})
        assert result['mean_ranking_agreement'] == 1.0

    def test_calculate_agreement_saves_to_results(self, comparison, mixed_explanations):
        comparison.calculate_agreement(mixed_explanations)
        assert 'agreement' in comparison.comparison_results

    # --- to_json ---

    def test_to_json_valid(self, comparison, mixed_explanations):
        comparison.compare_feature_rankings(mixed_explanations)
        json_str = comparison.to_json()
        parsed = json.loads(json_str)
        assert 'rankings' in parsed

    def test_to_json_empty_results(self, comparison):
        """JSON on empty results should still produce valid JSON."""
        json_str = comparison.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    # --- edge cases ---

    def test_empty_explanations(self, comparison):
        result = comparison.compare_feature_rankings({})
        assert result['rankings'] == {}

    def test_unknown_explanation_format(self, comparison):
        """Explanation dict with unrecognized keys should produce empty ranking."""
        result = comparison.compare_feature_rankings({'unknown': {'data': [1, 2, 3]}})
        assert result['rankings']['unknown'] == []

    def test_global_importance_format(self, comparison, feature_names):
        """Test that 'feature_importance' dict format is recognized."""
        exp = {
            'feature_importance': {fn: float(10 - i) for i, fn in enumerate(feature_names)}
        }
        result = comparison.compare_feature_rankings({'global': exp})
        assert len(result['rankings']['global']) == 10

    # --- direction agreement ---

    def test_direction_agreement_perfect(self, comparison, feature_names):
        """Two identical SHAP explanations should have direction_agreement == 1.0."""
        impacts = [
            {'feature': feature_names[i], 'shap_value': 1.0, 'feature_value': 0.0,
             'direction': 'increases_risk'}
            for i in range(10)
        ]
        exp = {'feature_impacts': impacts}
        result = comparison.calculate_agreement({'A': exp, 'B': exp})
        assert result['direction_agreement'] == 1.0

    def test_direction_agreement_opposite(self, comparison, feature_names):
        """Two explanations with opposite directions should have low agreement."""
        impacts_a = [
            {'feature': feature_names[i], 'shap_value': 1.0, 'feature_value': 0.0,
             'direction': 'increases_risk'}
            for i in range(10)
        ]
        impacts_b = [
            {'feature': feature_names[i], 'shap_value': -1.0, 'feature_value': 0.0,
             'direction': 'decreases_risk'}
            for i in range(10)
        ]
        exp_a = {'feature_impacts': impacts_a}
        exp_b = {'feature_impacts': impacts_b}
        result = comparison.calculate_agreement({'A': exp_a, 'B': exp_b})
        assert result['direction_agreement'] == 0.0

    # --- _extract_feature_ranking for various formats ---

    def test_extract_ranking_shap_format(self, comparison, feature_names):
        exp = {
            'feature_impacts': [
                {'feature': feature_names[0], 'shap_value': 0.5},
                {'feature': feature_names[1], 'shap_value': 0.3},
            ]
        }
        ranking = comparison._extract_feature_ranking(exp, 'SHAP')
        assert ranking == [feature_names[0], feature_names[1]]

    def test_extract_ranking_lime_format(self, comparison, feature_names):
        exp = {
            'feature_weights': [
                (feature_names[0] + ' > 5', 0.5),
                (feature_names[1] + ' <= 3', 0.3),
            ]
        }
        ranking = comparison._extract_feature_ranking(exp, 'LIME')
        assert feature_names[0] in ranking
        assert feature_names[1] in ranking

    def test_extract_ranking_contributions_format(self, comparison, feature_names):
        exp = {
            'contributions': [
                {'feature': feature_names[2], 'score': 0.5},
                {'variable': feature_names[3], 'contribution': 0.3},
            ]
        }
        ranking = comparison._extract_feature_ranking(exp, 'EBM')
        assert len(ranking) == 2

    # --- generate_comparison_report ---

    def test_generate_report_string(self, comparison, mixed_explanations):
        report = comparison.generate_comparison_report(mixed_explanations)
        assert isinstance(report, str)
        assert 'RAPORT POROWNANIA METOD XAI' in report or 'RAPORT' in report
        assert len(report) > 100


# ===========================================================================
# Integration / cross-module tests
# ===========================================================================

class TestXAIIntegration:
    """Integration tests that use multiple XAI modules together."""

    @pytest.fixture(scope="class")
    def full_setup(self):
        """Setup all explainers on a shared dataset."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        logits = 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.randn(50) * 0.5
        y = (logits > 0).astype(int)
        names = [f'feature_{i}' for i in range(10)]

        rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
        rf.fit(X, y)

        shap_exp = SHAPExplainer(model=rf, X_background=X, feature_names=names)
        lime_exp = LIMEExplainer(model=rf, X_train=X, feature_names=names, random_state=42)

        return X, y, names, rf, shap_exp, lime_exp

    def test_shap_lime_same_prediction(self, full_setup):
        """Both SHAP and LIME should agree on the predicted class for the same instance."""
        X, _, _, _, shap_exp, lime_exp = full_setup
        instance = X[0]

        shap_result = shap_exp.explain_instance(instance)
        lime_result = lime_exp.explain_instance(instance, num_samples=500)

        assert shap_result['prediction'] == lime_result['prediction']

    def test_comparison_with_real_explanations(self, full_setup):
        """XAIComparison with actual SHAP and LIME explanations."""
        X, _, names, _, shap_exp, lime_exp = full_setup
        instance = X[0]

        shap_result = shap_exp.explain_instance(instance)
        lime_result = lime_exp.explain_instance(instance, num_samples=500)

        comp = XAIComparison(feature_names=names)
        agreement = comp.calculate_agreement({
            'SHAP': shap_result,
            'LIME': lime_result
        })

        assert agreement['n_methods'] == 2
        assert 0.0 <= agreement['mean_ranking_agreement'] <= 1.0

    def test_comparison_report_with_real(self, full_setup):
        """Generate a full comparison report from real SHAP + LIME explanations."""
        X, _, names, _, shap_exp, lime_exp = full_setup
        instance = X[0]

        shap_result = shap_exp.explain_instance(instance)
        lime_result = lime_exp.explain_instance(instance, num_samples=500)

        comp = XAIComparison(feature_names=names)
        report = comp.generate_comparison_report({
            'SHAP': shap_result,
            'LIME': lime_result
        })
        assert isinstance(report, str)
        assert 'SHAP' in report
        assert 'LIME' in report

    @pytest.mark.slow
    def test_full_pipeline_with_ebm(self, full_setup):
        """Full pipeline: train EBM, explain, and compare with SHAP/LIME."""
        X, y, names, _, shap_exp, lime_exp = full_setup
        instance = X[0]

        # Train EBM
        ebm = EBMExplainer(
            feature_names=names, max_rounds=30, interactions=0,
            outer_bags=1, random_state=42
        )
        ebm.fit(X, y)

        shap_result = shap_exp.explain_instance(instance)
        lime_result = lime_exp.explain_instance(instance, num_samples=500)
        ebm_result = ebm.explain_local(instance)

        comp = XAIComparison(feature_names=names)
        result = comp.compare_feature_rankings({
            'SHAP': shap_result,
            'LIME': lime_result,
            'EBM': ebm_result
        })

        assert result['agreement_matrix'].shape == (3, 3)
        assert len(result['common_top_features']) >= 0

    @pytest.mark.slow
    def test_full_pipeline_with_dalex(self, full_setup):
        """Full pipeline with DALEX included."""
        X, y, names, rf, shap_exp, _ = full_setup
        instance = X[0]

        dalex = DALEXWrapper(model=rf, X=X, y=y, feature_names=names, label='RF')
        shap_result = shap_exp.explain_instance(instance)
        dalex_result = dalex.explain_instance_break_down(instance)

        comp = XAIComparison(feature_names=names)
        agreement = comp.calculate_agreement({
            'SHAP': shap_result,
            'DALEX': dalex_result
        })

        assert agreement['n_methods'] == 2
        assert 0.0 <= agreement['mean_ranking_agreement'] <= 1.0

    @pytest.mark.slow
    def test_full_pipeline_all_four(self, full_setup):
        """Full pipeline with all four XAI methods."""
        X, y, names, rf, shap_exp, lime_exp = full_setup
        instance = X[0]

        # DALEX
        dalex = DALEXWrapper(model=rf, X=X, y=y, feature_names=names, label='RF')

        # EBM
        ebm = EBMExplainer(
            feature_names=names, max_rounds=30, interactions=0,
            outer_bags=1, random_state=42
        )
        ebm.fit(X, y)

        shap_result = shap_exp.explain_instance(instance)
        lime_result = lime_exp.explain_instance(instance, num_samples=500)
        dalex_result = dalex.explain_instance_break_down(instance)
        ebm_result = ebm.explain_local(instance)

        comp = XAIComparison(feature_names=names)
        result = comp.compare_feature_rankings({
            'SHAP': shap_result,
            'LIME': lime_result,
            'DALEX': dalex_result,
            'EBM': ebm_result
        })

        assert result['agreement_matrix'].shape == (4, 4)
        # JSON serialization of full comparison should work
        json_str = comp.to_json()
        parsed = json.loads(json_str)
        assert 'rankings' in parsed


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
