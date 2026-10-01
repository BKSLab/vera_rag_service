"""Журнал индексации. Только новая таблица; существующие данные не меняются."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '20260921_1200'
down_revision = '20260907_1300'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'ingestion_runs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('document_id', sa.Text(), nullable=False),
        sa.Column('source_title', sa.Text(), nullable=False),
        sa.Column('kind', sa.String(20), nullable=False),
        sa.Column('source', sa.String(20), nullable=False),
        sa.Column('version', sa.Text(), nullable=False),
        sa.Column('revision_date', sa.Date(), nullable=False),
        sa.Column('section_number', sa.Text()),
        sa.Column('collection_name', sa.Text(), nullable=False),
        sa.Column('request_id', sa.Text()),
        sa.Column('change_id', sa.Integer()),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('current_stage', sa.String(40)),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True)),
        sa.Column('stages', postgresql.JSONB(), nullable=False),
        sa.Column('result', postgresql.JSONB(), nullable=False),
        sa.Column('error', sa.Text()),
    )
    op.create_index('ix_ingestion_runs_document_started', 'ingestion_runs', ['document_id', 'started_at'])
    op.create_index('ix_ingestion_runs_status_started', 'ingestion_runs', ['status', 'started_at'])
    op.create_index('ix_ingestion_runs_request_id', 'ingestion_runs', ['request_id'])
    op.create_index('ix_ingestion_runs_change_id', 'ingestion_runs', ['change_id'])


def downgrade():
    op.drop_table('ingestion_runs')
