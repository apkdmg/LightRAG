"""Tests for first-start Milvus authentication setup (lightrag.kg.milvus_auth).

A fake MilvusClient models a Milvus server with authentication on or off, so
every branch runs offline.
"""

import pytest

import lightrag.kg.milvus_auth as milvus_auth

pytestmark = pytest.mark.offline

URI = "http://milvus:19530"


class FakeServer:
    def __init__(self, auth_enabled=True, root_password="Milvus"):
        self.auth_enabled = auth_enabled
        self.passwords = {"root": root_password}
        self.roles = {"admin": set(), "public": set()}
        self.user_roles = {"root": {"admin"}}
        self.databases = {"default"}

    def check(self, user, password):
        if not self.auth_enabled:
            return True
        return user in self.passwords and self.passwords[user] == password


class FakeClient:
    server: FakeServer = None

    def __init__(self, uri, user=None, password=None, token=None, db_name=None):
        if token:
            user, password = token.split(":", 1)
        self.user = user
        if not self.server.check(user, password):
            raise PermissionError("auth check failure")

    def list_collections(self):
        return []

    def update_password(self, user, old, new):
        assert self.server.passwords[user] == old
        self.server.passwords[user] = new

    def list_databases(self):
        return sorted(self.server.databases)

    def create_database(self, name):
        self.server.databases.add(name)

    def list_roles(self):
        return list(self.server.roles)

    def create_role(self, name):
        self.server.roles[name] = set()

    def grant_privilege_v2(self, role_name, privilege, collection_name, db_name):
        self.server.roles[role_name].add((privilege, db_name, collection_name))

    def list_users(self):
        return list(self.server.passwords)

    def create_user(self, name, password):
        self.server.passwords[name] = password

    def grant_role(self, user, role):
        self.server.user_roles.setdefault(user, set()).add(role)


@pytest.fixture
def server(monkeypatch):
    def make(**kwargs):
        srv = FakeServer(**kwargs)
        FakeClient.server = srv
        monkeypatch.setattr(milvus_auth, "MilvusClient", FakeClient)
        milvus_auth._done.clear()
        return srv

    yield make
    milvus_auth._done.clear()


def setup(db_name="lightrag", root="root-secret", app_password="app-secret"):
    milvus_auth.ensure_milvus_app_user(
        uri=URI,
        app_user="lightrag",
        app_password=app_password,
        root_password=root,
        db_name=db_name,
    )


def test_fresh_install_rotates_root_and_creates_least_privilege_user(server):
    srv = server()
    setup()
    assert srv.passwords["root"] == "root-secret"
    assert srv.passwords["lightrag"] == "app-secret"
    assert "lightrag" in srv.databases
    assert srv.user_roles["lightrag"] == {"lightrag_app"}
    assert srv.roles["lightrag_app"] == {
        ("DatabaseAdmin", "lightrag", "*"),
        ("CollectionAdmin", "lightrag", "*"),
    }


def test_default_database_is_not_created(server):
    srv = server()
    setup(db_name=None)
    assert srv.databases == {"default"}
    assert ("CollectionAdmin", "default", "*") in srv.roles["lightrag_app"]


def test_second_start_is_idempotent(server):
    srv = server()
    setup()
    milvus_auth._done.clear()  # new process
    setup()
    assert srv.passwords["root"] == "root-secret"
    assert list(srv.passwords).count("lightrag") == 1


def test_no_root_password_does_nothing(server):
    srv = server()
    setup(root=None)
    assert srv.passwords == {"root": "Milvus"}


def test_wrong_root_password_is_a_clear_error(server):
    server(root_password="someone-elses-password")
    with pytest.raises(milvus_auth.MilvusAuthSetupError, match="MILVUS_ROOT_PASSWORD"):
        setup()


def test_existing_user_with_different_password_is_reported(server):
    srv = server(root_password="root-secret")
    srv.passwords["lightrag"] = "older-password"
    with pytest.raises(milvus_auth.MilvusAuthSetupError, match="MILVUS_PASSWORD"):
        setup()


def test_auth_disabled_skips_with_warning(server):
    srv = server(auth_enabled=False)
    setup()
    assert srv.passwords == {"root": "Milvus"}
    assert "lightrag_app" not in srv.roles


def test_milvus_lite_path_is_skipped(server):
    srv = server()
    milvus_auth.ensure_milvus_app_user(
        uri="/data/milvus_lite.db",
        app_user="lightrag",
        app_password="app-secret",
        root_password="root-secret",
    )
    assert srv.passwords == {"root": "Milvus"}
