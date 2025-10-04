"""add raw_ohlcv_7d table

Revision ID: 7f1a2c9a3b1a
Revises: ee23c3b5d5df
Create Date: 2025-09-11 14:58:00.000000

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "7f1a2c9a3b1a"
down_revision = "ee23c3b5d5df"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "raw_ohlcv_7d",
        sa.Column("timestamp", sa.BigInteger(), nullable=False),
        sa.Column("token", sa.String(), nullable=False),
        sa.Column("open", sa.Float(precision=53), nullable=False),
        sa.Column("high", sa.Float(precision=53), nullable=False),
        sa.Column("low", sa.Float(precision=53), nullable=False),
        sa.Column("close", sa.Float(precision=53), nullable=False),
        sa.Column("volume", sa.Float(precision=53), nullable=False),
        sa.PrimaryKeyConstraint("timestamp", "token"),
        sa.UniqueConstraint("timestamp", "token", name="_timestamp_token_uc_7d"),
    )


def downgrade() -> None:
    op.drop_table("raw_ohlcv_7d")
