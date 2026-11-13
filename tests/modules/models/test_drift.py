# encoding: utf-8
# pylint: disable=missing-docstring,redefined-outer-name
import pytest

from app.modules.models import drift
from app.modules.models.models import Model, Prediction


BASELINE = {
    'temperature': 21.0,
    'humidity': 45.0,
    'occupancy': 12.0,
}


def _make_model(db, name='drift-test-model'):
    model = Model(
        name=name,
        version='1.0.0',
        baseline_features=BASELINE,
    )
    with db.session.begin():
        db.session.add(model)
    return model


def _add_predictions(db, model, rows):
    with db.session.begin():
        for row in rows:
            db.session.add(Prediction(
                model_id=model.id,
                input_features=row,
                output_value=0.5,
            ))


def test_population_stability_score_is_zero_for_identical_samples():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert drift.population_stability_score(values, values) == 0.0


def test_population_stability_score_grows_with_shift():
    baseline = [float(value) for value in range(1, 21)]
    shifted = [float(value) for value in range(15, 35)]
    small = drift.population_stability_score(baseline, baseline[:10] + baseline[10:])
    large = drift.population_stability_score(baseline, shifted)
    assert small == 0.0
    assert large > 0.3


def test_population_stability_score_without_data_is_none():
    assert drift.population_stability_score([], [1.0, 2.0]) is None
    assert drift.population_stability_score([1.0, 2.0], []) is None


def test_numeric_values_skips_bad_entries():
    rows = [
        {'temperature': 20.0},
        {'temperature': 'not-a-number'},
        {'temperature': None},
        {'temperature': True},
        {'humidity': 30.0},
        'not-a-dict',
    ]
    assert drift._numeric_values(rows, 'temperature') == [20.0]


def test_build_drift_report_flags_shifted_feature(flask_app, db):
    model = _make_model(db)
    _add_predictions(db, model, [
        {'temperature': 40.0, 'humidity': 45.0, 'occupancy': 12.0}
        for _ in range(20)
    ])

    report = drift.build_drift_report(model, window_size=20, threshold=0.31)

    assert report['window_size'] == 20
    assert report['alert'] is True
    by_name = {item['feature']: item for item in report['features']}
    assert by_name['temperature']['alert'] is True
    assert by_name['humidity']['alert'] is False
    assert by_name['temperature']['samples'] == 20
    assert by_name['temperature']['window_mean'] == 40.0


def test_build_drift_report_without_baseline_conflicts(flask_app, db):
    model = Model(name='no-baseline-model', version='0.1.0', baseline_features={})
    with db.session.begin():
        db.session.add(model)

    with pytest.raises(Exception) as error:
        drift.build_drift_report(model, window_size=10, threshold=0.31)
    assert 'baseline' in str(error.value)


def test_drift_endpoint_returns_report(flask_app_client, db, regular_user):
    model = _make_model(db, name='endpoint-model')
    _add_predictions(db, model, [
        {'temperature': 21.0, 'humidity': 45.0, 'occupancy': 12.0}
        for _ in range(10)
    ])

    with flask_app_client.login(regular_user, auth_scopes=('models:read',)):
        response = flask_app_client.get(
            '/api/v1/models/%d/drift/?window_size=10' % model.id
        )

    assert response.status_code == 200
    assert response.content_type == 'application/json'
    assert response.json['alert'] is False
    assert response.json['threshold'] == 0.31
    assert len(response.json['features']) == 3
