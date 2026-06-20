from app.search.fusion import rrf_fusion


def test_rrf_fusion_ranks_item_present_in_both_lists_highest():
    dense = ['a', 'b', 'c']
    sparse = ['b', 'a', 'd']

    result = rrf_fusion([dense, sparse])
    ranked_ids = [item_id for item_id, _ in result]

    assert ranked_ids[0] in ('a', 'b')
    assert set(ranked_ids) == {'a', 'b', 'c', 'd'}


def test_rrf_fusion_gives_higher_score_to_item_in_both_lists_than_one_list():
    dense = ['a', 'b']
    sparse = ['a', 'c']

    result = dict(rrf_fusion([dense, sparse]))

    assert result['a'] > result['b']
    assert result['a'] > result['c']


def test_rrf_fusion_preserves_rank_order_within_single_list():
    dense = ['a', 'b', 'c']

    result = rrf_fusion([dense])
    ranked_ids = [item_id for item_id, _ in result]

    assert ranked_ids == ['a', 'b', 'c']


def test_rrf_fusion_returns_empty_list_for_no_lists():
    assert rrf_fusion([]) == []


def test_rrf_fusion_handles_empty_individual_lists():
    result = rrf_fusion([[], ['x', 'y']])
    ranked_ids = [item_id for item_id, _ in result]

    assert ranked_ids == ['x', 'y']


def test_rrf_fusion_smoothing_constant_affects_score_magnitude_not_order():
    dense = ['a', 'b']

    low_k = dict(rrf_fusion([dense], k=1))
    high_k = dict(rrf_fusion([dense], k=1000))

    assert low_k['a'] > high_k['a']
    assert low_k['a'] > low_k['b']
    assert high_k['a'] > high_k['b']
