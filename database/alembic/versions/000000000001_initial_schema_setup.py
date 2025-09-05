"""Initial schema setup

Revision ID: 000000000001
Revises: None
Create Date: 2025-09-05 07:22:00

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa  # noqa: F401

# Import models and Base so metadata includes all tables
from database.models.base import Base
from database.models import (  # noqa: F401
    raw_data as _raw_data,
    feature_stores as _feature_stores,
    operational as _operational,
)

# revision identifiers, used by Alembic.
revision = "000000000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:  # noqa: D401
    """Apply initial schema by creating all tables defined in SQLAlchemy models."""
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:  # noqa: D401
    """Revert initial schema by dropping all tables defined in SQLAlchemy models."""
    bind = op.get_bind()
    # Drop in reverse dependency order handled by metadata.drop_all
    Base.metadata.drop_all(bind=bind)
