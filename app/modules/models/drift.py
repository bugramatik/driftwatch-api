# encoding: utf-8
"""
Model drift reporting
---------------------

Compares the feature values recorded for a model's recent predictions against
the baseline captured when the model was registered, and reports a population
stability score per feature.
"""
import logging
import math

from flask import current_app
from flask_marshmallow import base_fields

from app.extensions.api import Namespace
from app.extensions.api import abort
from flask_restplus._http import HTTPStatus

from . import schemas
from .models import Model, Prediction


log = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DriftReportSchema(schemas.BaseModelSchema):
    """
    Response schema of a drift report.
    """
    window_size = base_fields.Integer(dump_only=True)
    threshold = base_fields.Float(dump_only=True)
    alert = base_fields.Boolean(dump_only=True)
    max_score = base_fields.Float(dump_only=True)
    features = base_fields.Raw(dump_only=True)


class FeatureDriftSchema(schemas.BaseModelSchema):
    """
    Per-feature part of a drift report.
    """
    feature = base_fields.String(dump_only=True)
    score = base_fields.Float(dump_only=True)
    baseline_mean = base_fields.Float(dump_only=True)
    window_mean = base_fields.Float(dump_only=True)
    samples = base_fields.Integer(dump_only=True)
    alert = base_fields.Boolean(dump_only=True)


BIN_COUNT = 10


def _numeric_values(rows, feature):
    """
    Pull the numeric values of one feature out of a list of input dicts.
    """
    values = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get(feature)
        if isinstance(value, bool) or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(number) or math.isinf(number):
            continue
        values.append(number)
    return values


def _bin_edges(baseline_values):
    """
    Build fixed bin edges from the baseline so both samples share the same
    buckets. The lowest and highest edges are widened to catch out-of-range
    values from the live window.
    """
    low = min(baseline_values)
    high = max(baseline_values)
    if low == high:
        low -= 0.5
        high += 0.5
    step = (high - low) / BIN_COUNT
    edges = [low + step * index for index in range(BIN_COUNT + 1)]
    edges[0] = float('-inf')
    edges[-1] = float('inf')
    return edges


def _histogram(values, edges):
    """
    Count values per bin, returning proportions that sum to one.
    """
    counts = [0] * (len(edges) - 1)
    for value in values:
        for index in range(len(counts)):
            if edges[index] <= value < edges[index + 1]:
                counts[index] += 1
                break
    total = sum(counts)
    if total == 0:
        return [0.0] * len(counts)
    return [count / total for count in counts]


def population_stability_score(baseline_values, window_values):
    """
    Population stability index between two samples of one feature.

    Returns ``None`` when either sample is empty, since a score without data
    would be misleading.
    """
    if not baseline_values or not window_values:
        return None
    edges = _bin_edges(baseline_values)
    expected = _histogram(baseline_values, edges)
    actual = _histogram(window_values, edges)
    score = 0.0
    for expected_share, actual_share in zip(expected, actual):
        # A tiny floor keeps empty bins from blowing the score up to infinity.
        expected_share = max(expected_share, 1e-4)
        actual_share = max(actual_share, 1e-4)
        score += (actual_share - expected_share) * math.log(actual_share / expected_share)
    return round(score, 4)


def build_drift_report(model, window_size, threshold):
    """
    Compute a drift report for one model.

    The baseline is the ``baseline_features`` dict stored on the model at
    registration time; the window is the ``input_features`` of the most recent
    predictions, newest first.
    """
    baseline = model.baseline_features or {}
    if not baseline:
        abort(
            code=HTTPStatus.CONFLICT,
            message="Model %d has no baseline features to compare against" % model.id
        )

    predictions = (
        Prediction.query
        .filter(Prediction.model_id == model.id)
        .order_by(Prediction.created.desc())
        .limit(window_size)
        .all()
    )
    window_rows = [prediction.input_features for prediction in predictions]

    features = []
    for feature in sorted(baseline.keys()):
        baseline_values = _numeric_values([baseline], feature)
        window_values = _numeric_values(window_rows, feature)
        score = population_stability_score(baseline_values, window_values)
        features.append({
            'feature': feature,
            'score': score if score is not None else 0.0,
            'baseline_mean': round(sum(baseline_values) / len(baseline_values), 4)
            if baseline_values else 0.0,
            'window_mean': round(sum(window_values) / len(window_values), 4)
            if window_values else 0.0,
            'samples': len(window_values),
            'alert': score is not None and score >= threshold,
        })

    max_score = max((item['score'] for item in features), default=0.0)
    return {
        'id': model.id,
        'window_size': window_size,
        'threshold': threshold,
        'alert': any(item['alert'] for item in features),
        'max_score': max_score,
        'features': features,
    }


def default_threshold():
    """
    The configured alert threshold, read at request time so tests and
    deployments can change it without rebuilding the app.
    """
    return float(current_app.config.get('DRIFT_ALERT_THRESHOLD', 0.31))
