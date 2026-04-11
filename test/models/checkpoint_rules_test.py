from pipeline.base.configs import CheckpointConfig, CheckpointRules


def test_resolve_testing_rule_uses_default_dirs(tmp_path):
    checkpoint_rules = CheckpointRules(
        testing=CheckpointConfig(epoch=5, suffix=".keras")
    )

    result = checkpoint_rules.resolve_testing_rule(default_dirs=[tmp_path])

    assert result == {
        "dirs": [tmp_path],
        "path": None,
        "epoch": 5,
        "suffix": ".keras"
    }
