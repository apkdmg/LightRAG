"""Backend of the admin Operations console.

This package must stay cheap to import: core modules (``lightrag.pipeline``,
``lightrag.operate``, ``lightrag.utils``) import :mod:`lightrag.api.ops.redact`
and :mod:`lightrag.api.ops.context`, so neither may pull in FastAPI, a storage
backend or anything else heavier than the standard library.
"""
