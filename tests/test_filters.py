from app.utils.filters import contains_offensive_language

def test_offensive():
    assert contains_offensive_language("qué mierda", "locals/ban_kwds.json")
    assert not contains_offensive_language("qué bonito", "locals/ban_kwds.json")