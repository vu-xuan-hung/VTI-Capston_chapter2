"""Run the project from the terminal: python main.py [--fake] COMMAND ...

Kit commands are declared here and implemented in factdesk/runtime/cli.py;
project commands are declared and implemented in factdesk/runtime/project_commands.py.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  argparse
import  sys
from    dataclasses                     import  dataclass
from    pathlib                         import  Path

from    factdesk.ai_logger                 import  set_up_ai_logger
from    factdesk.config                    import  load_config, with_backend
from    factdesk.runtime.cli               import  kit_commands
from    factdesk.runtime.project_commands  import  add_project_parsers, project_commands


#==========================================================================
# Where main.py finds its files
#==========================================================================
@dataclass(frozen=True)
class CliConfig:
    """
    Purpose: the project root (the folder of main.py) and the default config.
    """

    root: Path                          = Path(__file__).resolve().parent

    #----------------------------------------------------------------------
    # Default config file
    #----------------------------------------------------------------------
    @property
    def config_path(self) -> Path:
        """
        Locate configs/app.toml next to main.py.

        Output:
            . path          : the default --config value
        """
        return self.root / "configs" / "app.toml"


#==========================================================================
# Declare the commands
#==========================================================================
def build_parser(cli: CliConfig) -> argparse.ArgumentParser:
    """
    Declare the global flags, the kit commands and the project commands.

    Input:
        . cli           : root and default config path
    Output:
        . parser        : the argument parser
    """

    #----------------------------------------------------------------------
    # Global flags
    #----------------------------------------------------------------------
    top = argparse.ArgumentParser(description="Capstone project: kit commands + project commands")
    top.add_argument("--config", type=Path, default=cli.config_path)
    top.add_argument("--fake", action="store_true", help="rule-based model, hashing vectors: no download")
    top.add_argument("--log-level", default="WARNING", help="DEBUG, INFO, WARNING, ERROR")
    commands = top.add_subparsers(dest="command", required=True)

    #----------------------------------------------------------------------
    # Servers
    #----------------------------------------------------------------------
    serve = commands.add_parser("serve-model", help="run the OpenAI-compatible model server")
    serve.add_argument("--small", action="store_true", help="the small chat model (weak laptop)")
    serve.add_argument("--fake", dest="serve_fake", action="store_true", help="the fake backend: no download")
    serve.add_argument("--port", type=int, help="default: servers.model_port")
    commands.add_parser("servers", help="which local servers answer: model, Chroma, Langflow")

    #----------------------------------------------------------------------
    # Knowledge base
    #----------------------------------------------------------------------
    commands.add_parser("index", help="build or refresh the collection from the docs folder")
    search = commands.add_parser("search", help="the nearest chunks of a query, with scores")
    search.add_argument("query")
    search.add_argument("--k", type=int, help="default: knowledge.top_k")
    search.add_argument("--where", action="append", help="metadata filter key=value (repeatable)")
    search.add_argument("--trace", action="store_true", help="also save a trace in runtime/traces")

    #----------------------------------------------------------------------
    # Traces, keys, flows, experiments
    #----------------------------------------------------------------------
    trace = commands.add_parser("trace-show", help="print a saved trace as a tree")
    trace.add_argument("path", type=Path)
    commands.add_parser("scan-keys", help="look for API keys in the files you will submit")
    flow_show = commands.add_parser("flow-show", help="read an exported flow file; no server needed")
    flow_show.add_argument("path", type=Path)
    flow_ask = commands.add_parser("flow-ask", help="ask the Langflow flow one question over its API")
    flow_ask.add_argument("question")
    flow_ask.add_argument("--k", type=int, help="the flow's search k, for this run only")
    flow_ask.add_argument("--session", help="a session id: keeps conversations apart")
    compare = commands.add_parser("compare", help="compare two saved experiments case by case")
    compare.add_argument("before")
    compare.add_argument("after")

    #----------------------------------------------------------------------
    # PROJECT COMMANDS: declared in factdesk/runtime/project_commands.py
    #----------------------------------------------------------------------
    add_project_parsers(commands)
    return top


#==========================================================================
# Entry point
#==========================================================================
def main(argv: list[str] | None = None) -> int:
    """
    Parse the arguments, load the config once, run one command.

    Input:
        . argv          : arguments without the program name; None = sys.argv
    Output:
        . code          : 0 ok, 1 a check said [NO], 2 cannot do this, 130 stopped
    """

    #----------------------------------------------------------------------
    # Parse the arguments and merge the command tables
    #----------------------------------------------------------------------
    cli      = CliConfig()
    args     = build_parser(cli).parse_args(argv)
    kit      = kit_commands()
    project  = project_commands()
    repeated = sorted(set(kit) & set(project))
    if repeated:
        raise SystemExit(f"command name(s) used by the kit and the project: {repeated}")
    #----------------------------------------------------------------------
    # Load the config once and run the command
    #----------------------------------------------------------------------
    set_up_ai_logger(args.log_level)
    config = load_config(args.config, root=cli.root)
    config = with_backend(config, "fake") if args.fake else config
    try:
        return {**kit, **project}[args.command](config, args)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError) as error:
        print(f"Cannot do this: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
