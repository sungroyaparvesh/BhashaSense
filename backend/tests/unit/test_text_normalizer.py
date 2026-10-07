from app.ml.preprocessors.text_normalizer import TextNormalizer

def test_clean_text_removes_noise_tokens():
    raw = "Hello [applause] maine kal ticket <unk> raise kiya tha   "
    cleaned = TextNormalizer.clean_text(raw)
    assert "[applause]" not in cleaned
    assert "<unk>" not in cleaned
    assert "  " not in cleaned
    assert "ticket raise kiya tha" in cleaned

def test_split_into_clauses():
    text = "Account activate nahi hua. Please escalate quickly!"
    clauses = TextNormalizer.split_into_clauses(text)
    assert len(clauses) == 2
    assert clauses[0] == "Account activate nahi hua."
    assert clauses[1] == "Please escalate quickly!"
