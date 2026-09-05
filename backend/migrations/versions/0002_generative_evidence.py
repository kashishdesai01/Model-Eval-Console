"""Generative output evidence and benchmark sampling provenance."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("datasets", sa.Column("provenance", JSONB(), nullable=False, server_default="{}"))
    op.add_column("predictions", sa.Column("raw_output", sa.Text(), nullable=True))
    op.alter_column("predictions", "prob_pos", existing_type=sa.Float(), nullable=True)
    op.drop_constraint("predictions_pred_check", "predictions", type_="check")
    op.create_check_constraint("ck_prediction_label", "predictions", "pred IN (-1,0,1)")


def downgrade():
    # Refuse destructive downgrade if invalid generation evidence exists.
    op.execute(
        "DO $$ BEGIN IF EXISTS (SELECT 1 FROM predictions "
        "WHERE pred=-1 OR prob_pos IS NULL) THEN "
        "RAISE EXCEPTION 'Generative predictions prevent downgrade'; END IF; END $$"
    )
    op.drop_constraint("ck_prediction_label", "predictions", type_="check")
    op.create_check_constraint("predictions_pred_check", "predictions", "pred IN (0,1)")
    op.alter_column("predictions", "prob_pos", existing_type=sa.Float(), nullable=False)
    op.drop_column("predictions", "raw_output")
    op.drop_column("datasets", "provenance")
