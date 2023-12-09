from ditto import parse


def test_parse():
    text = """
    @skhaz I need food.
    @modelonulo I need food too.
    """

    messages = parse(text)

    assert len(messages) == 2
    assert messages[0].author == "skhaz"
    assert messages[0].content == "I need food."
    assert messages[1].author == "modelonulo"
    assert messages[1].content == "I need food too."


def test_parse_mention():
    text = """
    @skhaz I love you @modelonulo
    """

    messages = parse(text)

    assert len(messages) == 1
    assert messages[0].author == "skhaz"
    assert messages[0].content == "I love you @modelonulo"
