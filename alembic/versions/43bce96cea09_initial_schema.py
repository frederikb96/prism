"""initial_schema

Revision ID: 43bce96cea09
Revises:
Create Date: 2026-01-31 17:07:27.207338

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "43bce96cea09"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "search_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.String(length=100), nullable=False),
        sa.Column("claude_session_id", sa.String(length=100), nullable=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_sessions_user_created",
        "search_sessions",
        ["user_id", "created_at"],
    )
    op.create_index(
        "idx_sessions_claude_id",
        "search_sessions",
        ["claude_session_id"],
    )
    op.create_index("idx_sessions_status", "search_sessions", ["status"])

    op.create_table(
        "search_tasks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("worker_type", sa.String(length=50), nullable=False),
        sa.Column("worker_prompt", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["search_sessions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_tasks_session", "search_tasks", ["session_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_tasks_session", table_name="search_tasks")
    op.drop_table("search_tasks")
    op.drop_index("idx_sessions_status", table_name="search_sessions")
    op.drop_index("idx_sessions_claude_id", table_name="search_sessions")
    op.drop_index("idx_sessions_user_created", table_name="search_sessions")
    op.drop_table("search_sessions")
