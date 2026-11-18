from trustaudit.verifier import Claim, Verifier

FLIP = {"increases": "decreases", "decreases": "increases"}


def test_true_top_claim_is_supported(verifier, row, attrs):
    top = attrs[0]
    v = verifier.verify(Claim(top["feature"], top["direction"], "strong"), row, attrs)
    assert v.supported, v.reasons
    assert v.rank == 1


def test_flipped_direction_is_rejected(verifier, row, attrs):
    top = attrs[0]
    v = verifier.verify(Claim(top["feature"], FLIP[top["direction"]], "strong"), row, attrs)
    assert not v.supported
    assert any("SHAP says" in r for r in v.reasons)


def test_exaggerated_strength_is_rejected(verifier, row, attrs):
    low = attrs[-1]
    v = verifier.verify(Claim(low["feature"], low["direction"], "strong"), row, attrs)
    assert not v.supported
    assert any("only rank" in r for r in v.reasons)


def test_unknown_feature_and_bad_values(verifier, row, attrs):
    assert not verifier.verify(Claim("made_up", "increases"), row, attrs).supported
    feat = attrs[0]["feature"]
    assert not verifier.verify(Claim(feat, "sideways"), row, attrs).supported
    assert not verifier.verify(Claim(feat, attrs[0]["direction"], "huge"), row, attrs).supported


def test_deletion_sign_matches_shap_for_top_feature(verifier, row, attrs):
    top = attrs[0]
    delta, _ = verifier.deletion_delta(row, top["feature"])
    assert (delta > 0) == (top["shap"] > 0)


def test_checks_can_be_disabled(ds, model, row, attrs):
    low = attrs[-1]
    claim = Claim(low["feature"], low["direction"], "strong")
    shap_off = Verifier(model, ds.X_train, use_shap=False, use_deletion=False)
    assert shap_off.verify(claim, row, attrs).supported   # nothing left to fail


def test_claim_from_dict_normalises():
    c = Claim.from_dict({"feature": " age ", "direction": "Increases", "strength": "STRONG"})
    assert (c.feature, c.direction, c.strength) == ("age", "increases", "strong")
