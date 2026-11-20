"""The desk: the facts table, the entity collection, the model and the two versions, wired from one Config.

v1 is the free agent, v2 is the graph. Both return the same contract.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  sqlite3
from    pathlib                         import  Path
from    typing                          import  Any

from    langchain_core.runnables        import  RunnableConfig
from    langgraph.checkpoint.memory     import  InMemorySaver
from    langgraph.checkpoint.sqlite     import  SqliteSaver
from    langgraph.types                 import  Command

from    factdesk.agents.free_agent       import  run_agent
from    factdesk.compute.table           import  FactsTable, load_facts
from    factdesk.config                  import  Config
from    factdesk.contract                import  Answer, answer_from_state
from    factdesk.graph.build             import  GraphDeps, build_graph
from    factdesk.knowledge.linker        import  Link, link_country
from    factdesk.knowledge.names         import  load_country_index, load_indicator_index
from    factdesk.knowledge.vector_store  import  open_vector_store
from    factdesk.llm.chat_source         import  ChatSource


#==========================================================================
# The desk
#==========================================================================
class Desk:
    """
    Purpose: open everything once (table, indexes, vector store, lazy chat
    model, compiled graph) and answer questions with v1 (agent) or v2 (graph).
    Nothing heavy loads until a question needs it.
    """

    #----------------------------------------------------------------------
    # Wire the parts
    #----------------------------------------------------------------------
    def __init__(self, config: Config, *, chat: ChatSource | None = None, checkpointer: Any = None) -> None:
        """
        Read the facts table and the alias indexes; the store, the model and
        the graph are created on first use.

        Input:
            . config        : the whole configuration
            . chat          : a chat source; None = ChatSource(config.model, config.servers)
            . checkpointer  : a LangGraph checkpointer; None = in memory (one process)
        """

        #------------------------------------------------------------------
        # Load the facts and wire the dependencies
        #------------------------------------------------------------------
        paths              = config.paths
        self.config        = config
        self.table         = load_facts(paths.facts_path, paths.countries_path, paths.indicators_path)
        self.store         = open_vector_store(config)
        self.chat          = chat or ChatSource(config.model, config.servers)
        self._checkpointer = checkpointer or InMemorySaver()
        self._deps         = GraphDeps(
            table           = self.table,
            store           = self.store,
            chat            = self.chat,
            country_index   = load_country_index(paths.countries_path),
            indicator_index = load_indicator_index(paths.indicators_path),
            numbers         = config.numbers,
            graph           = config.graph,
            outbox          = paths.runtime_dir / config.graph.outbox_file,
        )
        self._app: Any     = None

    #----------------------------------------------------------------------
    # v2: ask the graph
    #----------------------------------------------------------------------
    def ask(self, question: str, *, thread_id: str, run_config: RunnableConfig | None = None) -> Answer:
        """
        Run the graph on a question. A question with an ambiguous country name
        stops at the interrupt and comes back as status "needs_clarification";
        continue it with resume() under the same thread_id.

        Input:
            . question      : the user's question
            . thread_id     : the checkpoint thread; resume needs the same one
            . run_config    : tags, metadata, tracer
        Output:
            . answer        : the contract
        """
        start = {"question": question, "thread_id": thread_id, "max_tries": self.config.graph.max_tries}
        return self._finish(self._graph().invoke(start, self._config(thread_id, run_config)), question)

    #----------------------------------------------------------------------
    # v2: continue after the interrupt
    #----------------------------------------------------------------------
    def resume(self, *, thread_id: str, choice: str, run_config: RunnableConfig | None = None) -> Answer:
        """
        Give the interrupted graph the user's choice (a country code) and run on.
        Works in a NEW process when the checkpointer is the SQLite file.

        Input:
            . thread_id     : the thread of the interrupted ask
            . choice        : the code the user picked, e.g. KOR
            . run_config    : tags, metadata, tracer
        Output:
            . answer        : the contract
        """
        app    = self._graph()
        config = self._config(thread_id, run_config)
        if not app.get_state(config).next:
            raise ValueError(f"thread {thread_id!r} has nothing to resume (not asked yet, or already finished)")
        out      = app.invoke(Command(resume=choice), config)
        question = app.get_state(config).values["question"]
        return self._finish(out, question)

    #----------------------------------------------------------------------
    # v1: ask the free agent
    #----------------------------------------------------------------------
    def ask_agent(self, question: str, *, run_config: RunnableConfig | None = None) -> Answer:
        """
        Run the free agent (v1) on a question.

        Input:
            . question      : the user's question
            . run_config    : tags, metadata, tracer
        Output:
            . answer        : the contract
        """
        return run_agent(
            question,
            chat        = self.chat.get(),
            table       = self.table,
            numbers     = self.config.numbers,
            graph       = self.config.graph,
            run_config  = run_config,
        )

    #----------------------------------------------------------------------
    # Link one country name
    #----------------------------------------------------------------------
    def link(self, text: str) -> Link:
        """
        Link one country name the way the graph does (the `link` command).

        Input:
            . text          : a name as a question may write it
        Output:
            . link          : status, code and candidates
        """
        return link_country(
            text,
            index   = self._deps.country_index,
            names   = self.table.names,
            store   = self.store,
            graph   = self.config.graph,
        )

    #----------------------------------------------------------------------
    # The compiled graph, built once
    #----------------------------------------------------------------------
    def _graph(self) -> Any:
        """
        Compile the graph with the desk's checkpointer the first time.

        Output:
            . app           : the compiled LangGraph graph
        """
        if self._app is None:
            self._app = build_graph(self._deps, self._checkpointer)
        return self._app

    #----------------------------------------------------------------------
    # Merge the thread into the run config
    #----------------------------------------------------------------------
    def _config(self, thread_id: str, run_config: RunnableConfig | None) -> RunnableConfig:
        """
        Add the checkpoint thread to the caller's tags, metadata and callbacks.

        Input:
            . thread_id     : the checkpoint thread
            . run_config    : the caller's config or None
        Output:
            . config        : the config for invoke
        """
        return {**(run_config or {}), "configurable": {"thread_id": thread_id}}

    #----------------------------------------------------------------------
    # From graph output to contract
    #----------------------------------------------------------------------
    def _finish(self, out: dict[str, Any], question: str) -> Answer:
        """
        Turn the graph output into the contract. An interrupted graph becomes
        status "needs_clarification" with the question for the user.

        Input:
            . out           : what invoke returned (state values, and "__interrupt__" when stopped)
            . question      : the user's question
        Output:
            . answer        : the contract
        """
        if "__interrupt__" in out:
            ask     = out["__interrupt__"][0].value
            options = ", ".join(f"{item['code']} ({item['name']})" for item in ask["options"])
            text    = f"Which one do you mean by {ask['text']!r}? Options: {options}. Resume with the code."
            out     = {**out, "status": "needs_clarification", "reason": text, "answer": text, "verified": False}
        return answer_from_state({**out, "question": question}, units=self.table.units)


#==========================================================================
# The SQLite checkpointer of the command line
#==========================================================================
def open_checkpointer(runtime_dir: Path) -> SqliteSaver:
    """
    Open runtime/checkpoints.sqlite: the file that lets a NEW process resume a thread.

    Input:
        . runtime_dir   : the project's runtime folder
    Output:
        . saver         : a SqliteSaver
    """
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(runtime_dir / "checkpoints.sqlite", check_same_thread=False))
