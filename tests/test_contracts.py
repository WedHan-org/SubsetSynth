from subsetsynth.masks import nonempty_proper_subsets, primitive_mappings_for_avail
from subsetsynth.locks import load_support_lock
from subsetsynth.objective import HierarchicalImportanceObjective
from subsetsynth.config import load_config


def test_mapping_counts():
    assert len(nonempty_proper_subsets((1,1,1))) == 6
    assert len(primitive_mappings_for_avail((1,1,1))) == 9
    assert len(primitive_mappings_for_avail((1,1,0))) == 2
    assert len(primitive_mappings_for_avail((1,0,1))) == 2


def test_importance_identity():
    root=__import__('pathlib').Path(__file__).resolve().parents[1]
    cfg=load_config(root/'configs/base.yaml')
    support=load_support_lock(cfg,allow_unlocked=True)
    obj=HierarchicalImportanceObjective.from_config(cfg,support)
    assert abs(sum(c.q*c.weight for c in obj.cells.values())-1.0) < 1e-12
    masses=obj.mapping_expected_target_mass()
    assert max(masses.values())/min(masses.values()) < 1.0000000001
