"""求解器单元测试：评分、约束、平分确定性、覆盖可行性。"""
import pytest

from app.solver import (
    InputAnnotation,
    SolveParams,
    dice_overlap,
    quantize_overlap,
    solve_consensus,
)


def ann(aid, start, end, lid, weight=5, name=None):
    return InputAnnotation(
        annotator_id=aid, annotator_name=name or f"a{aid}",
        weight=weight, start=start, end=end,
        label_id=lid, label_name=f"L{lid}",
    )


def test_dice_and_quantize():
    assert dice_overlap(0, 10, 0, 10) == 1.0
    assert dice_overlap(0, 4, 2, 6) == 0.5
    assert quantize_overlap(1.0) == 10
    assert quantize_overlap(0.5) == 5


def test_weighted_winner_and_rejection():
    # PERSON=1, ORG=2 互斥；区间相同，高权重 PERSON(8+5) 胜 ORG(3)
    annotations = [
        ann(1, 0, 2, 1, 8), ann(2, 0, 2, 1, 5), ann(3, 0, 2, 2, 3),
        # 让三名标注者都"参与"文档的另一处不冲突标注
        ann(1, 4, 6, 1, 8), ann(2, 4, 6, 1, 5), ann(3, 4, 6, 1, 3),
    ]
    res = solve_consensus(annotations, set(), {(1, 2)})
    chosen = {
        (c.start, c.end, c.label_id): res.chosen[c.index]
        for c in res.candidates
    }
    assert chosen[(0, 2, 1)] is True
    assert chosen[(0, 2, 2)] is False
    assert chosen[(4, 6, 1)] is True
    # 拒绝候选仍保留支持者明细
    rejected = res.candidates[[
        (c.start, c.end, c.label_id) for c in res.candidates
    ].index((0, 2, 2))]
    assert rejected.supporters[0]["weight"] == 3


def test_nesting_allowed_crossing_forbidden():
    # CLAIM=3 可嵌套 PERSON=1；ORG=2 与 PERSON 交叉且无嵌套 => 不能同时取
    annotations = [
        ann(1, 0, 10, 3, 5), ann(2, 0, 10, 3, 5),  # 外层 CLAIM
        ann(1, 2, 4, 1, 5), ann(2, 2, 4, 1, 5),    # 内层 PERSON（允许）
        ann(3, 8, 12, 2, 10),                      # 与 CLAIM 交叉
        ann(1, 20, 22, 1, 1), ann(2, 20, 22, 1, 1),
        ann(3, 20, 22, 1, 1),
    ]
    res = solve_consensus(annotations, {(3, 1)}, {(1, 2)})
    by = {(c.start, c.end, c.label_id): res.chosen[c.index] for c in res.candidates}
    assert by[(0, 10, 3)] is True
    assert by[(2, 4, 1)] is True       # 嵌套共存
    assert by[(8, 12, 2)] is False     # 交叉被排除


def test_tie_break_is_deterministic_and_lexicographic():
    # 两个等长同标签不重叠、得分相同 => 都取；构造互斥重叠的等分场景：
    # 区间 [0,2) 与 [2,4) 不冲突。改为 [0,4) vs [2,6) 互斥同分时取起点小的
    annotations = [
        ann(1, 0, 4, 1, 5), ann(2, 0, 4, 1, 5),
        ann(1, 2, 6, 1, 5), ann(2, 2, 6, 1, 5),
    ]
    r1 = solve_consensus(annotations, set(), set())
    r2 = solve_consensus(annotations, set(), set())
    assert [tuple(res.chosen) for res in (r1,)]
    assert r1.chosen == r2.chosen
    starts = [c.start for c, ok in zip(r1.candidates, r1.chosen) if ok]
    assert starts == [0]  # 起点小者胜


def test_override_forces_low_score_candidate():
    annotations = [
        ann(1, 0, 2, 1, 8), ann(2, 0, 2, 1, 5), ann(3, 0, 2, 2, 3),
        ann(1, 4, 6, 1, 8), ann(2, 4, 6, 1, 5), ann(3, 4, 6, 1, 3),
    ]
    res = solve_consensus(
        annotations, set(), {(1, 2)}, SolveParams(),
        force_selected_keys={(0, 2, 2)},
    )
    # 强制采纳 ORG => 同区间 PERSON 因互斥必须落选
    by = {(c.start, c.end, c.label_id): res.chosen[c.index] for c in res.candidates}
    assert by[(0, 2, 2)] is True
    assert by[(0, 2, 1)] is False


def test_conflicting_overrides_infeasible():
    annotations = [
        ann(1, 0, 2, 1, 8), ann(2, 0, 2, 1, 5),
    ]
    res = solve_consensus(
        annotations, set(), set(), SolveParams(),
        force_selected_keys={(0, 2, 1)},
        force_deselected_keys={(0, 2, 1)},
    )
    assert res.status not in ("OPTIMAL", "FEASIBLE")


def test_partial_overlap_same_label_contributes():
    # 同标签、边界不同但相交的标注互相为对方候选贡献分数
    annotations = [
        ann(1, 0, 4, 1, 8),
        ann(2, 0, 6, 1, 5),
        ann(3, 10, 12, 1, 3),   # 零重叠：不得计入
    ]
    res = solve_consensus(annotations, set(), set())
    by = {(c.start, c.end): c for c in res.candidates}
    short, long_ = by[(0, 4)], by[(0, 6)]
    # dice([0,4),[0,6)) = 2*4/(4+6) = 0.8 -> 量化 8/10
    assert short.score == 8 * 10 + 5 * 8      # 自身 1.0 + 部分重叠 0.8
    assert long_.score == 5 * 10 + 8 * 8
    assert short.raw_score == 12.0
    # 同一标注可支持多个候选
    assert {s["annotator_id"] for s in short.supporters} == {1, 2}
    assert {s["annotator_id"] for s in long_.supporters} == {1, 2}
    # 明细贡献之和与候选得分严格一致（整数映射）
    for c in (short, long_):
        assert sum(s["contribution"] for s in c.supporters) == \
            pytest.approx(c.raw_score)
    # 同标签相交 => 至多取一，总分高者胜
    assert res.chosen[short.index] is True
    assert res.chosen[long_.index] is False


def test_cross_label_and_zero_overlap_never_score():
    annotations = [
        ann(1, 0, 4, 1, 8),
        ann(2, 0, 6, 2, 10),   # 异标签相交：不计分
        ann(3, 4, 8, 1, 10),   # 同标签但端点相接（零重叠）：不计分
    ]
    res = solve_consensus(annotations, set(), set())
    by = {(c.start, c.end, c.label_id): c for c in res.candidates}
    c = by[(0, 4, 1)]
    assert [s["annotator_id"] for s in c.supporters] == [1]
    assert c.score == 8 * 10
    assert by[(0, 6, 2)].score == 10 * 10
    assert by[(4, 8, 1)].score == 10 * 10


def test_one_annotation_supports_multiple_candidates():
    # 一条长区间标注同时支持多个同标签相交候选
    annotations = [
        ann(1, 0, 10, 1, 6),
        ann(2, 2, 4, 1, 5),
        ann(3, 6, 8, 1, 5),
    ]
    res = solve_consensus(annotations, set(), set())
    by = {(c.start, c.end): c for c in res.candidates}
    outer, inner_a, inner_b = by[(0, 10)], by[(2, 4)], by[(6, 8)]
    assert {s["annotator_id"] for s in outer.supporters} == {1, 2, 3}
    assert 1 in {s["annotator_id"] for s in inner_a.supporters}
    assert 1 in {s["annotator_id"] for s in inner_b.supporters}
    # dice([0,10),[2,4)) = 4/12 -> 量化 3/10
    assert outer.score == 6 * 10 + 5 * 3 + 5 * 3
    assert inner_a.score == 5 * 10 + 6 * 3
    # 两个内层互不重叠可共存（68+68=136），优于独占的外层（90）
    assert res.chosen[outer.index] is False
    assert res.chosen[inner_a.index] is True
    assert res.chosen[inner_b.index] is True
    assert res.total_score == inner_a.score + inner_b.score


def test_integer_mapping_quantizes_each_supporter():
    # dice=0.25 -> round-half-up 量化 0.3，贡献 = 权重 × 3（而非四舍五入总分）
    annotations = [ann(1, 0, 4, 1, 5), ann(2, 3, 7, 1, 6)]
    res = solve_consensus(annotations, set(), set())
    by = {(c.start, c.end): c for c in res.candidates}
    # dice([0,4),[3,7)) = 2*1/(4+4) = 0.25 -> 3/10
    assert by[(0, 4)].score == 5 * 10 + 6 * 3
    assert by[(3, 7)].score == 6 * 10 + 5 * 3
    assert by[(0, 4)].supporters[1]["contribution"] == 1.8
    assert by[(0, 4)].supporters[1]["overlap_ratio"] == 0.25
