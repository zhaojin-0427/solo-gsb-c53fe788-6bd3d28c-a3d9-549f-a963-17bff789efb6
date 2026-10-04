"""端到端 API 测试：导入→规则→标注者→标注→共识→覆盖→报告→不可变性。"""
import os
import tempfile

import pytest

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ["ADJ_DB_PATH"] = _tmp.name

from fastapi.testclient import TestClient  # noqa: E402

from app import database  # noqa: E402
from app.main import app  # noqa: E402

database.init_db()
client = TestClient(app)


@pytest.fixture(scope="module")
def setup():
    p = client.post("/api/projects", json={"name": "e2e"}).json()
    pid = p["id"]
    content = "张三和李四在上海开会。"
    # 词元：张三/和/李四/在/上海/开会/。
    words = ["张三", "和", "李四", "在", "上海", "开会", "。"]
    bounds, at = [], 0
    for w in words:
        bounds.append([at, at + len(w)])
        at += len(w)

    doc = client.post(f"/api/projects/{pid}/documents", json={
        "doc_id": "d1", "content": content, "token_bounds": bounds,
    }).json()
    assert doc["revision"] == 1

    person = client.post(f"/api/projects/{pid}/labels",
                         json={"name": "PERSON", "color": "#2563eb"}).json()["id"]
    city = client.post(f"/api/projects/{pid}/labels",
                       json={"name": "CITY", "color": "#0891b2"}).json()["id"]
    org = client.post(f"/api/projects/{pid}/labels",
                      json={"name": "ORG", "color": "#db2777"}).json()["id"]
    assert person < city
    r = client.post(f"/api/projects/{pid}/label-rules", json={
        "kind": "mutex", "label_a_id": person, "label_b_id": org})
    assert r.status_code == 201

    a = client.post(f"/api/projects/{pid}/annotators",
                    json={"name": "甲", "weight": 9}).json()["id"]
    b = client.post(f"/api/projects/{pid}/annotators",
                    json={"name": "乙", "weight": 4}).json()["id"]
    c = client.post(f"/api/projects/{pid}/annotators",
                    json={"name": "丙", "weight": 2}).json()["id"]
    return {
        "pid": pid, "content": content, "bounds": bounds,
        "person": person, "city": city, "org": org,
        "a": a, "b": b, "c": c,
    }


def test_weight_bounds(setup):
    r = client.post(f"/api/projects/{setup['pid']}/annotators",
                    json={"name": "超范围", "weight": 11})
    assert r.status_code == 422


def test_reject_out_of_bounds_and_token_cutting(setup):
    pid, doc = setup["pid"], "d1"
    # 越界
    r = client.post(f"/api/projects/{pid}/documents/{doc}/annotations", json={
        "annotator_id": setup["a"], "label_id": setup["person"],
        "start": 0, "end": 99})
    assert r.status_code == 400 and "越界" in r.json()["detail"]
    # 切开词元「张三」[0,2)：端点 1 不是词元边界
    r = client.post(f"/api/projects/{pid}/documents/{doc}/annotations", json={
        "annotator_id": setup["a"], "label_id": setup["person"],
        "start": 0, "end": 1})
    assert r.status_code == 400 and "词元" in r.json()["detail"]


def post_ann(pid, who, label, s, e):
    r = client.post(f"/api/projects/{pid}/documents/d1/annotations", json={
        "annotator_id": who, "label_id": label, "start": s, "end": e})
    assert r.status_code == 201, r.text


def test_consensus_flow_and_override(setup):
    pid = setup["pid"]
    # 张三[0,2)：甲乙 PERSON vs 丙 ORG（互斥）
    post_ann(pid, setup["a"], setup["person"], 0, 2)
    post_ann(pid, setup["b"], setup["person"], 0, 2)
    post_ann(pid, setup["c"], setup["org"], 0, 2)
    # 上海[6,8)：三人一致 CITY
    post_ann(pid, setup["a"], setup["city"], 6, 8)
    post_ann(pid, setup["b"], setup["city"], 6, 8)
    post_ann(pid, setup["c"], setup["city"], 6, 8)

    con = client.post(f"/api/projects/{pid}/documents/d1/consensus",
                      json={}).json()
    adopted = {(i["start"], i["end"], i["label_name"]) for i in con["items"] if i["adopted"]}
    rejected = [(i["start"], i["end"], i["label_name"]) for i in con["items"] if not i["adopted"]]
    assert (0, 2, "PERSON") in adopted
    assert (0, 2, "ORG") in rejected
    assert (6, 8, "CITY") in adopted
    # 拒绝项有支持者明细
    org_item = next(i for i in con["items"] if (i["start"], i["end"], i["label_name"]) == (0, 2, "ORG"))
    assert org_item["supporters"][0]["annotator_name"] == "丙"
    cid1 = con["id"]

    # 裁决者强制采纳 ORG（低权重少数派），必须给理由
    r = client.post(f"/api/projects/{pid}/consensuses/{cid1}/override", json={
        "action": "select", "start": 0, "end": 2,
        "label_id": setup["org"], "reason": ""})
    assert r.status_code == 422  # 理由必填

    con2 = client.post(f"/api/projects/{pid}/consensuses/{cid1}/override", json={
        "action": "select", "start": 0, "end": 2,
        "label_id": setup["org"],
        "reason": "外部知识库确认此处为机构简称"}).json()
    adopted2 = {(i["start"], i["end"], i["label_name"]) for i in con2["items"] if i["adopted"]}
    assert (0, 2, "ORG") in adopted2
    assert (0, 2, "PERSON") not in adopted2
    assert con2["id"] != cid1  # 新版本，旧共识保留

    old = client.get(f"/api/projects/{pid}/consensuses/{cid1}").json()
    assert any(i["adopted"] and (i["start"], i["end"], i["label_name"]) == (0, 2, "PERSON")
               for i in old["items"])
    globals()["_OVERRIDDEN_CONSENSUS"] = con2["id"]


def test_report_immutable(setup):
    pid = setup["pid"]
    cid = globals()["_OVERRIDDEN_CONSENSUS"]
    rep = client.post(f"/api/projects/{pid}/consensuses/{cid}/report", json={}).json()
    version = rep["report_version"]
    assert rep["algorithm"]["version"]
    assert rep["input_snapshot"]["annotations"]
    assert rep["parameters"]["score_scale"] == 10

    # 重新导入同名文档 => revision 2；旧报告仍可打开且内容指向 revision 1
    r = client.post(f"/api/projects/{pid}/documents", json={
        "doc_id": "d1", "content": setup["content"] + "追加",
        "token_bounds": setup["bounds"] + [[11, 13]]})
    assert r.json()["revision"] == 2

    again = client.get(f"/api/projects/{pid}/reports/{version}").json()
    assert again["input_snapshot"]["revision"] == 1
    assert again["report_version"] == version

    # 数据库层禁止 UPDATE/DELETE
    import sqlite3
    conn = sqlite3.connect(os.environ["ADJ_DB_PATH"])
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE reports SET payload_json='x' WHERE version=?", (version,))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM reports WHERE version=?", (version,))
    conn.close()


def test_partial_overlap_scoring_end_to_end():
    """同标签、边界不同但相交的标注互相贡献分数；共识页与报告数据一致。"""
    pid = client.post("/api/projects", json={"name": "partial-overlap"}).json()["id"]
    client.post(f"/api/projects/{pid}/documents",
                json={"doc_id": "d1", "content": "ABCDEFGHIJ"})
    lab = client.post(f"/api/projects/{pid}/labels",
                      json={"name": "ENTITY"}).json()["id"]
    a1 = client.post(f"/api/projects/{pid}/annotators",
                     json={"name": "甲", "weight": 8}).json()["id"]
    a2 = client.post(f"/api/projects/{pid}/annotators",
                     json={"name": "乙", "weight": 5}).json()["id"]
    # 同标签、边界不同但相交的合法区间（按单字符词元，端点均对齐）
    for who, s, e in [(a1, 0, 4), (a2, 0, 6)]:
        r = client.post(f"/api/projects/{pid}/documents/d1/annotations",
                        json={"annotator_id": who, "label_id": lab,
                              "start": s, "end": e})
        assert r.status_code == 201, r.text

    con = client.post(f"/api/projects/{pid}/documents/d1/consensus", json={}).json()
    items = {(i["start"], i["end"]): i for i in con["items"]}
    short, long_ = items[(0, 4)], items[(0, 6)]
    # dice([0,4),[0,6)) = 0.8 -> 量化 8/10；双方都获得部分重叠支持
    assert short["score"] == 8 * 10 + 5 * 8
    assert long_["score"] == 5 * 10 + 8 * 8
    assert {s["annotator_id"] for s in short["supporters"]} == {a1, a2}
    assert {s["annotator_id"] for s in long_["supporters"]} == {a1, a2}
    # 明细贡献之和 = 候选得分（页面/报告展示一致）
    for it in (short, long_):
        assert sum(s["contribution"] for s in it["supporters"]) == \
            pytest.approx(it["raw_score"])
    # 同标签相交互斥：总分高者被采纳
    assert short["adopted"] is True and long_["adopted"] is False
    assert con["total_score"] == short["score"]
    assert con["params"]["score_scale"] == 10
    assert con["algorithm_version"]

    # 报告与共识页数据一致，且带实际算法参数与版本
    rep = client.post(
        f"/api/projects/{pid}/consensuses/{con['id']}/report", json={}).json()
    assert rep["solver"]["total_score"] == con["total_score"]
    assert rep["algorithm"]["version"] == con["algorithm_version"]
    assert rep["parameters"] == con["params"]
    rep_short = next(i for i in rep["consensus"]["all_items"]
                     if (i["start"], i["end"]) == (0, 4))
    assert rep_short["score"] == short["score"]
    assert rep_short["supporters"] == short["supporters"]
