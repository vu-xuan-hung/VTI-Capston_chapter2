"""Wire the nodes and edges of the FactDesk graph (v2).

    START -> parse -> link --decline--> decline -> END
                       |--clarify (interrupt) -> link
                       '--compute --decline--> decline -> END
                                  '--explain -> verify --END
                                       ^          |--explain (retry)
                                       |          '--template -> END
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    dataclasses                     import  dataclass
from    functools                       import  partial
from    pathlib                         import  Path
from    typing                          import  Any

from    langgraph.graph                 import  END, START, StateGraph

from    factdesk.compute.table           import  FactsTable
from    factdesk.config                  import  FormatConfig, GraphConfig
from    factdesk.graph.nodes.clarify     import  clarify_node
from    factdesk.graph.nodes.compute     import  compute_node
from    factdesk.graph.nodes.decline     import  decline_node
from    factdesk.graph.nodes.explain     import  explain_node
from    factdesk.graph.nodes.link        import  link_node
from    factdesk.graph.nodes.parse       import  parse_node
from    factdesk.graph.nodes.verify      import  template_node, verify_node
from    factdesk.graph.routes            import  after_compute, after_link, after_verify
from    factdesk.graph.state             import  FactState
from    factdesk.knowledge.vector_store  import  VectorStore
from    factdesk.llm.chat_source         import  ChatSource


#==========================================================================
# Everything the nodes need
#==========================================================================
@dataclass(frozen=True)
class GraphDeps:
    """
    Purpose: the parts the nodes share (data, indexes, store, model, settings),
    created once by the desk and handed to build_graph.
    """

    table: FactsTable
    store: VectorStore
    chat: ChatSource
    country_index: dict[str, list[str]]
    indicator_index: dict[str, list[str]]
    numbers: FormatConfig
    graph: GraphConfig
    outbox: Path


#==========================================================================
# Build the graph
#==========================================================================
def build_graph(deps: GraphDeps, checkpointer: Any) -> Any:
    """
    Add the nodes (each with the parts it needs), the fixed edges and the three
    routers, and compile with the checkpointer (needed by interrupt).

    Input:
        . deps          : the shared parts
        . checkpointer  : a LangGraph checkpointer (SqliteSaver or InMemorySaver)
    Output:
        . app           : the compiled graph; invoke with {"question", "thread_id", "max_tries"}
    """

    #----------------------------------------------------------------------
    # Nodes
    #----------------------------------------------------------------------
    graph = StateGraph(FactState)
    graph.add_node("parse", partial(parse_node, chat=deps.chat))
    graph.add_node("link", partial(
        link_node,
        store           = deps.store,
        names           = deps.table.names,
        country_index   = deps.country_index,
        indicator_index = deps.indicator_index,
        graph           = deps.graph,
    ))
    graph.add_node("clarify", partial(clarify_node, outbox=deps.outbox))
    graph.add_node("compute", partial(compute_node, table=deps.table, numbers=deps.numbers))
    graph.add_node("explain", partial(explain_node, chat=deps.chat, table=deps.table, numbers=deps.numbers))
    graph.add_node("verify", partial(verify_node, numbers=deps.numbers))
    graph.add_node("template", partial(template_node, table=deps.table, numbers=deps.numbers))
    graph.add_node("decline", partial(decline_node, table=deps.table))

    #----------------------------------------------------------------------
    # Edges: fixed ones and the three routers
    #----------------------------------------------------------------------
    graph.add_edge(START, "parse")
    graph.add_edge("parse", "link")
    graph.add_conditional_edges("link", after_link, ["decline", "clarify", "compute"])
    graph.add_edge("clarify", "link")
    graph.add_conditional_edges("compute", after_compute, ["explain", "decline"])
    graph.add_edge("explain", "verify")
    graph.add_conditional_edges("verify", after_verify, [END, "explain", "template"])
    graph.add_edge("template", END)
    graph.add_edge("decline", END)
    return graph.compile(checkpointer=checkpointer)
