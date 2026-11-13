# encoding: utf-8
"""
Drift report parameters
-----------------------
"""
from flask_marshmallow import base_fields

from app.extensions.api.parameters import Parameters


class DriftReportParameters(Parameters):
    """
    Query parameters of the drift report endpoint.
    """
    window_size = base_fields.Integer(
        missing=500,
        validate=lambda value: 1 <= value <= 10000,
        metadata={'description': "How many recent predictions to compare"},
    )
    threshold = base_fields.Float(
        missing=None,
        validate=lambda value: 0.0 <= value <= 10.0,
        metadata={'description': "Score above which a feature is flagged"},
    )
