import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base


@pytest.fixture(scope="session")
def factory():
    url = os.environ.get("MEC_TEST_DATABASE_URL")
    container = None
    if not url:
        from testcontainers.postgres import PostgresContainer

        container = PostgresContainer("postgres:16-alpine", driver="psycopg")
        container.start()
        url = container.get_connection_url()
    engine = create_engine(url)
    # Isolated test schema; the demo database is never truncated.
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA IF EXISTS mec_test CASCADE"))
        connection.execute(text("CREATE SCHEMA mec_test"))
    engine.dispose()
    engine = create_engine(url, connect_args={"options": "-csearch_path=mec_test"})
    config = Config()
    config.set_main_option(
        "script_location", str(Path(__file__).resolve().parents[1] / "migrations")
    )
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    yield sessionmaker(engine, expire_on_commit=False)
    engine.dispose()
    if container:
        container.stop()


@pytest.fixture
def db(factory) -> Iterator[Session]:
    with factory.begin() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
    with factory() as session:
        yield session
