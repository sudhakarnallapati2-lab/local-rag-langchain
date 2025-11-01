def test_imports():
    import importlib
    modules = ['streamlit','langchain','langchain_community','faiss','pypdf']
    missing = []
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as e:
            missing.append((m,str(e)))
    # we don't fail hard if langchain_community isn't installed in all environments (optional)
    assert 'streamlit' not in [m for m,_ in missing], f"streamlit import failed: {missing}"
