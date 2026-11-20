"""Config: the shipped TOML loads, typos stop, --fake isolates the store."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    pathlib                         import  Path

import  pytest
from    helpers                         import  ROOT

from    factdesk.config                 import  load_config, with_backend


#==========================================================================
# The shipped config loads
#==========================================================================
def test_app_toml_loads_with_paths_from_the_root() -> None:
    """
    Pin that configs/app.toml loads and every path resolves under the root.
    """
    config = load_config(ROOT / "configs" / "app.toml", root=ROOT)
    assert config.model.backend == "local"
    assert config.knowledge.chunk_overlap < config.knowledge.chunk_size
    assert config.paths.docs_dir == ROOT / "data" / "docs"
    assert config.paths.root == ROOT
    assert config.servers.model_url == "http://127.0.0.1:8000/v1"
    assert list(config.tracing.tags) == ["factdesk"]


#==========================================================================
# Typos stop the run
#==========================================================================
def test_unknown_section_and_key_stop_with_the_allowed_names(tmp_path: Path) -> None:
    """
    Pin that a misspelled section or key is refused with the allowed names.

    Input:
        . tmp_path      : temporary folder for the TOML files
    """
    section = tmp_path / "section.toml"
    section.write_text('[modle]\nbackend = "fake"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown section"):
        load_config(section, root=tmp_path)
    key = tmp_path / "key.toml"
    key.write_text('[model]\nchat_modle = "x"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="chat_modle"):
        load_config(key, root=tmp_path)


#==========================================================================
# Bad values stop the run
#==========================================================================
def test_bad_backend_and_chunk_window_are_refused(tmp_path: Path) -> None:
    """
    Pin the validation of the backend name and of overlap < size.

    Input:
        . tmp_path      : temporary folder for the TOML files
    """
    backend = tmp_path / "backend.toml"
    backend.write_text('[model]\nbackend = "cloud"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="model.backend"):
        load_config(backend, root=tmp_path)
    window = tmp_path / "window.toml"
    window.write_text("[knowledge]\nchunk_size = 10\nchunk_overlap = 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_overlap"):
        load_config(window, root=tmp_path)


#==========================================================================
# --fake isolates the vectors
#==========================================================================
def test_fake_backend_uses_its_own_embedded_collection() -> None:
    """
    Pin that the fake backend never writes into the real collection or needs a server.
    """
    config = load_config(ROOT / "configs" / "app.toml", root=ROOT)
    fake   = with_backend(config, "fake")
    assert fake.model.backend == "fake"
    assert fake.knowledge.collection == f"{config.knowledge.collection}-fake"
    assert fake.knowledge.store == "embedded"
    assert with_backend(config, "server").knowledge == config.knowledge
