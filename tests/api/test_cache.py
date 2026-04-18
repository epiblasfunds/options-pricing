from src.api.services.cache import ApiModelCache


def test_api_model_cache_evicts_oldest_entry():
    cache = ApiModelCache(max_entries=1)

    first = cache.get_or_load("first", lambda: {"name": "first"})
    second = cache.get_or_load("second", lambda: {"name": "second"})
    first_again = cache.get_or_load("first", lambda: {"name": "first-again"})

    assert first == {"name": "first"}
    assert second == {"name": "second"}
    assert first_again == {"name": "first-again"}
    assert list(cache.entries) == ["first"]
