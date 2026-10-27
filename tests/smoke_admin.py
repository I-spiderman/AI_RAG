"""管理员功能冒烟：登录/上传/列表/热门工单/删除/非管理员403

运行：.\.venv\Scripts\python.exe tests\smoke_admin.py [base_url]
默认 base_url = http://127.0.0.1:8000
"""

import sys
from pathlib import Path
import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
TEST_FILE = Path(__file__).resolve().parent / "data" / "test.txt"


def login(c: httpx.Client, username: str, password: str) -> dict:
    r = c.post(f"{BASE}/api/auth/login", json={"username": username, "password": password})
    r.raise_for_status()
    return r.json()


def main():
    ok = True
    with httpx.Client(timeout=60) as c:
        # 1. admin 登录，断言 role
        data = login(c, "admin", "admin123")
        assert data["role"] == "admin", f"role 期望 admin，实际 {data.get('role')}"
        print(f"[OK] admin 登录 role=admin（user_id={data['user_id']}）")
        h = {"Authorization": f"Bearer {data['token']}"}

        # 2. 上传文档
        with TEST_FILE.open("rb") as f:
            r = c.post(f"{BASE}/api/admin/documents", headers=h,
                       files={"file": (TEST_FILE.name, f, "text/plain")})
        r.raise_for_status()
        up = r.json()
        assert up["doc_id"] == "test" and up["chunk_count"] > 0
        print(f"[OK] 上传文档 doc_id={up['doc_id']} chunk_count={up['chunk_count']}")

        # 3. 文档列表（含刚上传的）
        r = c.get(f"{BASE}/api/admin/documents", headers=h)
        r.raise_for_status()
        docs = r.json()
        found = [d for d in docs if d["doc_id"] == "test"]
        assert found, "列表未包含刚上传的 test 文档"
        d = found[0]
        print(f"[OK] 文档列表 {len(docs)} 条：{d['doc_name']} chunks={d['chunk_count']} "
              f"cats={','.join(d['categories'])} time={d['created_at']}")

        # 4. 热门工单
        r = c.get(f"{BASE}/api/admin/hot-tickets", headers=h)
        r.raise_for_status()
        hot = r.json()
        assert len(hot) <= 5
        print(f"[OK] 热门工单 Top{len(hot)}：")
        for t in hot:
            print(f"     [{t['occurrences']}x] {t['question']} | {t['category']} | {t['last_occurred_at']}")

        # 5. 删除文档
        r = c.delete(f"{BASE}/api/admin/documents/test", headers=h)
        r.raise_for_status()
        assert r.json()["deleted"] == "test"
        r = c.get(f"{BASE}/api/admin/documents", headers=h)
        after = r.json()
        assert not [x for x in after if x["doc_id"] == "test"], "删除后列表仍包含 test"
        print(f"[OK] 删除文档 test，删除后剩余 {len(after)} 条")

        # 6. 非管理员 403
        try:
            u = login(c, "smoke01", "123456")
            r = c.get(f"{BASE}/api/admin/documents",
                      headers={"Authorization": f"Bearer {u['token']}"})
            assert r.status_code == 403, f"期望 403，实际 {r.status_code}"
            print("[OK] 非管理员访问 admin 接口 → 403")
        except AssertionError:
            # smoke01 可能已不存在，换 yhy
            u = login(c, "yhy", "123456")
            r = c.get(f"{BASE}/api/admin/documents",
                      headers={"Authorization": f"Bearer {u['token']}"})
            assert r.status_code == 403, f"期望 403，实际 {r.status_code}"
            print("[OK] 非管理员访问 admin 接口 → 403")

    print("\n全部管理员冒烟通过")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)
