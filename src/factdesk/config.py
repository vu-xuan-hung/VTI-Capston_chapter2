"""Load the FactDesk configuration from TOML into frozen config classes.

main.py creates one Config at startup and passes it down; nothing else reads
the TOML file. A project adds its own section with one more frozen class and
one more field in Config: load_config finds it by the field's type.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  tomllib
from    dataclasses                     import  dataclass, field, fields, replace
from    pathlib                         import  Path
from    typing                          import  Any, get_type_hints


#==========================================================================
# Model settings
#==========================================================================
@dataclass(frozen=True)
class ModelConfig:
    """
    Purpose: choose the chat backend and the models. "local" runs Qwen in this
    process, "server" calls the OpenAI-compatible model server, "fake" answers
    by rules with no download.
    """

    #----------------------------------------------------------------------
    # Backend and models
    #----------------------------------------------------------------------
    backend: str                        = "local"
    chat_model: str                     = "Qwen/Qwen2.5-1.5B-Instruct"
    small_chat_model: str               = "Qwen/Qwen2.5-0.5B-Instruct"
    embedding_model: str                = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_new_tokens: int                 = 300

    #----------------------------------------------------------------------
    # Reject an unknown backend at startup
    #----------------------------------------------------------------------
    def __post_init__(self) -> None:
        """
        Stop when no chat source can create the requested backend.
        """
        if self.backend not in ("local", "fake", "server"):
            raise ValueError(
                f"model.backend must be 'local', 'fake' or 'server', got {self.backend!r}"
            )


#==========================================================================
# Knowledge base settings
#==========================================================================
@dataclass(frozen=True)
class KnowledgeConfig:
    """
    Purpose: the Chroma collection and how documents are cut and searched.
    "embedded" opens the folder in this process; "server" calls `chroma run`,
    the collection Langflow reads too.
    """

    #----------------------------------------------------------------------
    # Collection
    #----------------------------------------------------------------------
    collection: str                     = "factdesk-kb"
    store: str                          = "embedded"

    #----------------------------------------------------------------------
    # Chunking (words) and search
    #----------------------------------------------------------------------
    chunk_size: int                     = 60
    chunk_overlap: int                  = 15
    top_k: int                          = 3
    threshold: float                    = 0.35      # cosine similarity, 1 - distance

    #----------------------------------------------------------------------
    # Validate the store kind and the chunk window
    #----------------------------------------------------------------------
    def __post_init__(self) -> None:
        """
        Stop on a store kind nobody opens or a window that cannot advance.
        """
        if self.store not in ("embedded", "server"):
            raise ValueError(f"knowledge.store must be 'embedded' or 'server', got {self.store!r}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                f"knowledge.chunk_overlap must be in [0, chunk_size): "
                f"{self.chunk_overlap} vs {self.chunk_size}"
            )


#==========================================================================
# File locations
#==========================================================================
@dataclass(frozen=True)
class PathConfig:
    """
    Purpose: every folder the project reads or writes. load_config resolves
    them from the project root, so commands work from any directory.
    """

    root: Path                          = Path(".")
    data_dir: Path                      = Path("data")
    docs_dir: Path                      = Path("data/docs")
    runtime_dir: Path                   = Path("runtime")     # traces, checkpoints; safe to delete
    results_dir: Path                   = Path("results")     # experiments, submitted
    kb_dir: Path                        = Path("kb_store")    # Chroma folder, embedded or `chroma run --path`

    #----------------------------------------------------------------------
    # Data, cases, flows, the learner's notes
    #----------------------------------------------------------------------
    raw_dir: Path                       = Path("data/raw")    # the fetched snapshot
    facts_path: Path                    = Path("data/facts.csv")
    countries_path: Path                = Path("data/countries.json")
    indicators_path: Path               = Path("data/indicators.json")
    notes_path: Path                    = Path("data/notes.json")
    cases_path: Path                    = Path("data/dev_cases.json")
    holdout_path: Path                  = Path("holdout/holdout_cases.json")   # solution only
    flows_dir: Path                     = Path("flows")
    my_work_dir: Path                   = Path("my_work")


#==========================================================================
# Local server addresses
#==========================================================================
@dataclass(frozen=True)
class ServerConfig:
    """
    Purpose: where the model server, the Chroma server and Langflow listen,
    and which flow the code calls.
    """

    #----------------------------------------------------------------------
    # Model server and Chroma server
    #----------------------------------------------------------------------
    model_host: str                     = "127.0.0.1"
    model_port: int                     = 8000
    chroma_host: str                    = "127.0.0.1"
    chroma_port: int                    = 8001
    request_timeout: float              = 300.0     # seconds; a CPU answer can take a minute

    #----------------------------------------------------------------------
    # Langflow
    #----------------------------------------------------------------------
    langflow_url: str                   = "http://127.0.0.1:7860"
    flow_name: str                      = "factdesk-rag"

    #----------------------------------------------------------------------
    # Base URL of the OpenAI-compatible API
    #----------------------------------------------------------------------
    @property
    def model_url(self) -> str:
        """
        Build the base URL that ServerChat, ServerEmbeddings and Langflow use.

        Output:
            . url           : e.g. http://127.0.0.1:8000/v1
        """
        return f"http://{self.model_host}:{self.model_port}/v1"


#==========================================================================
# Tracing and LangSmith
#==========================================================================
@dataclass(frozen=True)
class TracingConfig:
    """
    Purpose: how runs are labelled, locally and on LangSmith, and the name of
    the LangSmith dataset the cloud experiment creates once.
    """

    project: str                        = "factdesk"
    tags: tuple[str, ...]               = ("factdesk",)
    dataset: str                        = "factdesk-dataset"


#==========================================================================
# The data and the name table
#==========================================================================
@dataclass(frozen=True)
class FactsConfig:
    """
    Purpose: which raw files feed facts.csv, which years are kept, which
    World Bank codes are aggregates (dropped) and the everyday country names
    that the World Bank names do not contain.
    """

    #----------------------------------------------------------------------
    # Raw files (under paths.raw_dir) and years
    #----------------------------------------------------------------------
    population_csv: str                 = "population/data/population.csv"
    gdp_csv: str                        = "gdp/data/gdp.csv"
    min_year: int                       = 1990
    keep_aggregates: list[str]          = field(default_factory=list)

    #----------------------------------------------------------------------
    # Hand-checked lists
    #----------------------------------------------------------------------
    aggregate_codes: list[str]          = field(default_factory=list)
    aliases: dict[str, list[str]]       = field(default_factory=dict)


#==========================================================================
# Number formatting and the verification rules
#==========================================================================
@dataclass(frozen=True)
class FormatConfig:
    """
    Purpose: how numbers are written for the reader and how strictly the
    check reads them back: decimals per kind, the scale words the model may
    use, and which small integers and years the check ignores.
    """

    #----------------------------------------------------------------------
    # Decimals of each kind of number
    #----------------------------------------------------------------------
    money_decimals: int                 = 2         # decimals in the unit used: 4.26 trillion
    per_person_decimals: int            = 2         # US$ per person: 39,986.93
    ratio_decimals: int                 = 2
    percent_decimals: int               = 2

    #----------------------------------------------------------------------
    # What the check reads
    #----------------------------------------------------------------------
    scales: dict[str, float]            = field(default_factory=lambda: {
        "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12,
    })
    small_int_max: int                  = 12        # whole numbers up to this are counting words, not facts
    year_min: int                       = 1900
    year_max: int                       = 2100
    slack: float                        = 1.0       # 1.0 = a number may differ by half a unit of its last digit


#==========================================================================
# Linking, the graph and the model prompt budget
#==========================================================================
@dataclass(frozen=True)
class GraphConfig:
    """
    Purpose: the numbers that steer the graph: how many chunks the linker
    reads, when two candidates are "too close to tell", the minimum score of
    a vector match, and how many times the answer may be re-written.
    """

    link_k: int                         = 3
    ambiguity_margin: float             = 0.05      # top-2 country scores closer than this = ambiguous
    country_min_score: float            = 0.45      # a vector match below this is "unknown country"
    country_confident_score: float      = 0.70      # a vector match below this is only a guess: the user confirms it
    indicator_min_score: float          = 0.50      # below this the indicator is "not in the data"
    max_tries: int                      = 2         # explain attempts before the template answer
    outbox_file: str                    = "outbox.jsonl"    # in runtime_dir: one line per clarification request
    agent_max_steps: int                = 12        # LangGraph recursion limit of the free agent (v1)


#==========================================================================
# The RAG slice that Langflow also builds
#==========================================================================
@dataclass(frozen=True)
class SliceConfig:
    """
    Purpose: every value of the "what does this number mean" chain, so the
    code and the Langflow flow are built from the same text. The chain searches
    the whole collection: Langflow's Chroma DB component has no metadata
    filter, so the code does not use one either. flows/FLOW_SPEC.json repeats
    these values; a test pins that they agree.
    """

    k: int                              = 3
    max_tokens: int                     = 200
    system: str                         = ""
    template: str                       = ""
    parser_pattern: str                 = "[{chunk_id}] {text}"


#==========================================================================
# Evaluation
#==========================================================================
@dataclass(frozen=True)
class EvalConfig:
    """
    Purpose: how an expected number is compared and where eval puts its files.
    """

    result_rel_tol: float               = 0.001     # relative tolerance of result_exact
    trace_label_sep: str                = "-"       # traces are saved as <version>-<case id>


#==========================================================================
# All sections
#==========================================================================
@dataclass(frozen=True)
class Config:
    """
    Purpose: group every section; created once in main.py and passed down.
    A project adds its own section class as one more field here.
    """

    model: ModelConfig                  = ModelConfig()
    knowledge: KnowledgeConfig          = KnowledgeConfig()
    paths: PathConfig                   = PathConfig()
    servers: ServerConfig               = ServerConfig()
    tracing: TracingConfig              = TracingConfig()
    facts: FactsConfig                  = FactsConfig()
    numbers: FormatConfig               = FormatConfig()
    graph: GraphConfig                  = GraphConfig()
    rag: SliceConfig                    = SliceConfig()
    evaluation: EvalConfig              = EvalConfig()


#==========================================================================
# Load one TOML file
#==========================================================================
def load_config(path: Path, *, root: Path) -> Config:
    """
    Read a TOML file into Config; a typo in a section or key stops the run.

    Input:
        . path          : the TOML file, e.g. configs/app.toml
        . root          : project root that relative paths start from
    Output:
        . config        : every section, each validated by its own class
    """

    #----------------------------------------------------------------------
    # Read the file and refuse unknown sections
    #----------------------------------------------------------------------
    with path.open("rb") as file:
        raw = tomllib.load(file)
    hints   = get_type_hints(Config)
    known   = [item.name for item in fields(Config)]
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ValueError(f"{path.name}: unknown section(s) {unknown}; known: {known}")

    #----------------------------------------------------------------------
    # Build each section from its class
    #----------------------------------------------------------------------
    sections: dict[str, Any] = {}
    for name in known:
        sections[name] = _section(hints[name], raw.get(name, {}), name=name)
    paths = sections["paths"]
    sections["paths"] = replace(
        paths, **{item.name: root / getattr(paths, item.name) for item in fields(paths)}
    )
    return Config(**sections)


#==========================================================================
# Build one section
#==========================================================================
def _section(section_class: type, values: dict[str, Any], *, name: str) -> Any:
    """
    Create one config class from its TOML table, refusing unknown keys.

    Input:
        . section_class : the frozen dataclass of this section
        . values        : the TOML table of this section
        . name          : the section name, for the error message
    Output:
        . section       : an instance of section_class
    """
    allowed = {item.name for item in fields(section_class)}
    extra   = sorted(set(values) - allowed)
    if extra:
        raise ValueError(f"[{name}]: unknown key(s) {extra}; allowed: {sorted(allowed)}")
    return section_class(**values)


#==========================================================================
# Switch the backend (the --fake flag)
#==========================================================================
def with_backend(config: Config, backend: str) -> Config:
    """
    Return the same config with another model backend. "fake" also uses an
    embedded collection of its own, so fake vectors never mix with real ones
    and no server is needed.

    Input:
        . config        : the loaded configuration
        . backend       : "local", "fake" or "server"
    Output:
        . config        : a new Config; the old one is unchanged
    """
    model = replace(config.model, backend=backend)
    if backend != "fake":
        return replace(config, model=model)
    knowledge = replace(
        config.knowledge,
        collection  = f"{config.knowledge.collection}-fake",
        store       = "embedded",
    )
    return replace(config, model=model, knowledge=knowledge)
