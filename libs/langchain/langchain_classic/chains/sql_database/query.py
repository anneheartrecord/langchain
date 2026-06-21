from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, TypedDict
from warnings import warn

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from langchain_classic.chains.sql_database.prompt import PROMPT, SQL_PROMPTS

if TYPE_CHECKING:
    from langchain_community.utilities.sql_database import SQLDatabase


def _strip(text: str) -> str:
    return text.strip()


# ---------------------------------------------------------------------------
# Security helpers: input sanitization + output validation
# ---------------------------------------------------------------------------

_SUSPICIOUS_DB_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(the\s+)?above\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+", re.IGNORECASE),
    re.compile(r"disregard\s+(all|previous)", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
]

# Matches the sample-row comment block returned by SQLDatabase.get_table_info.
_SAMPLE_ROWS_BLOCK_RE: re.Pattern[str] = re.compile(r"/\*[\s\S]*?\*/", re.DOTALL)

# Strips SQL comments before structural analysis so DML keywords hidden in
# comments (e.g. inside CTE bodies) cannot bypass the validator.
_STRIP_COMMENTS_RE: re.Pattern[str] = re.compile(
    r"--[^\n]*|/\*.*?\*/",
    re.DOTALL,
)

# Strips SQL string literals so structural checks avoid false positives from
# semicolons or DML keywords inside quoted values.  Handles '' and \' escapes.
_STRIP_STRINGS_RE: re.Pattern[str] = re.compile(
    r"'(?:[^'\\]|''|\\.)*'|\"(?:[^\"\\]|\"\"|\\.)*\"",
    re.DOTALL,
)

# Token scanner: matches parentheses, semicolons, and SQL identifiers/verbs.
_TOKEN_RE: re.Pattern[str] = re.compile(r"[();]|\b[A-Z_]\w*\b", re.IGNORECASE)

# DML/DDL verbs that are never permitted in a read-only query at any depth.
_FORBIDDEN_VERBS: frozenset[str] = frozenset(
    {
        "DELETE",
        "INSERT",
        "UPDATE",
        "MERGE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "GRANT",
        "REVOKE",
        "REPLACE",
        "VACUUM",
        "PRAGMA",
        "ATTACH",
        "DETACH",
        "EXEC",
        "EXECUTE",
    }
)


def sanitize_user_question(question: str, max_len: int = 2000) -> str:
    """Cap length and strip role-tag mimics and 'SQLQuery:' from user input."""
    q = question[:max_len]
    q = re.sub(r"</?\s*(system|assistant|user)\s*>", "", q, flags=re.IGNORECASE)
    return re.sub(r"\bSQLQuery\s*:", "", q, flags=re.IGNORECASE)


def sanitize_table_info(table_info: str) -> str:
    """Warn and redact sample rows when prompt-injection patterns are detected.

    If any suspicious pattern is found, the ``/* … */`` sample-row comment
    blocks are replaced with a ``[redacted]`` placeholder so the injected
    instruction never reaches the LLM prompt.
    """
    for pat in _SUSPICIOUS_DB_PATTERNS:
        if pat.search(table_info):
            msg = (
                "create_sql_query_chain: suspicious prompt-injection pattern "
                f"({pat.pattern!r}) detected in DB sample rows. "
                "Sample rows have been redacted."
            )
            warn(msg, stacklevel=2)
            return _SAMPLE_ROWS_BLOCK_RE.sub(
                "/* [sample rows redacted due to suspicious content] */",
                table_info,
            )
    return table_info


def _preprocess_sql(sql: str) -> str:
    """Strip comments then string literals for safe structural analysis."""
    no_comments = _STRIP_COMMENTS_RE.sub(" ", sql)
    return _STRIP_STRINGS_RE.sub("''", no_comments)


def validate_sql_output(sql: str) -> str:
    """Reject anything that is not a single read-only SELECT (or WITH … SELECT).

    Uses a single-pass token scanner to check for forbidden DML/DDL verbs,
    multi-statement SQL, and ``SELECT … INTO`` side-effects.  Both SQL comments
    and string literals are stripped before scanning so keywords or semicolons
    that appear inside quoted or commented text do not cause false rejections or
    false passes.

    Handles:
    - Multi-statement SQL (any ``;`` at parenthesis depth 0)
    - DDL / DML as the main statement or inside data-modifying CTEs
    - ``SELECT … INTO`` table/file writes (MySQL, PostgreSQL, SQL Server)
    - CTE column-alias lists (e.g. ``WITH c(id) AS (SELECT id FROM t) …``)
    """
    s = sql.strip().rstrip(";").strip()
    if not s:
        msg = "create_sql_query_chain: LLM returned empty SQL"
        raise ValueError(msg)

    s_clean = _preprocess_sql(s)

    # One pass over all tokens, tracking paren depth.
    depth = 0
    first_token: str | None = None
    prev_was_select = False

    for m in _TOKEN_RE.finditer(s_clean):
        tok = m.group()
        upper = tok.upper()

        if tok == "(":
            depth += 1
            prev_was_select = False
            continue
        if tok == ")":
            depth -= 1
            prev_was_select = False
            continue
        if tok == ";":
            if depth == 0:
                msg = (
                    "create_sql_query_chain: refusing multi-statement SQL "
                    "(multi-statement SQL is blocked)."
                )
                raise ValueError(msg)
            prev_was_select = False
            continue

        # Identifier / keyword token.
        if first_token is None:
            first_token = upper

        if upper in _FORBIDDEN_VERBS:
            msg = (
                f"create_sql_query_chain: forbidden verb {upper!r} detected "
                "in generated SQL."
            )
            raise ValueError(msg)

        # Block SELECT … INTO in all forms (table creation, OUTFILE, @var).
        if upper == "INTO" and prev_was_select and depth == 0:
            msg = "create_sql_query_chain: SELECT INTO writes are not permitted."
            raise ValueError(msg)

        prev_was_select = upper == "SELECT" and depth == 0

    if first_token not in ("SELECT", "WITH"):
        msg = (
            "create_sql_query_chain: statement must start with SELECT or "
            f"WITH; got: {first_token!r}"
        )
        raise ValueError(msg)
    return s + ";"


class SQLInput(TypedDict):
    """Input for a SQL Chain."""

    question: str


class SQLInputWithTables(TypedDict):
    """Input for a SQL Chain."""

    question: str
    table_names_to_use: list[str]


def create_sql_query_chain(
    llm: BaseLanguageModel,
    db: SQLDatabase,
    prompt: BasePromptTemplate | None = None,
    k: int = 5,
    *,
    get_col_comments: bool | None = None,
) -> Runnable[SQLInput | SQLInputWithTables | dict[str, Any], str]:
    r"""Create a chain that generates SQL queries.

    *Security Note*: This chain generates SQL queries for the given database.

        The SQLDatabase class provides a get_table_info method that can be used
        to get column information as well as sample data from the table.

        To mitigate risk of leaking sensitive data, limit permissions
        to read and scope to the tables that are needed.

        Optionally, use the SQLInputWithTables input type to specify which tables
        are allowed to be accessed.

        Control access to who can submit requests to this chain.

        See https://docs.langchain.com/oss/python/security-policy for more information.

    Args:
        llm: The language model to use.
        db: The SQLDatabase to generate the query for.
        prompt: The prompt to use. If none is provided, will choose one
            based on dialect.  See Prompt section below for more.
        k: The number of results per select statement to return.
        get_col_comments: Whether to retrieve column comments along with table info.

    Returns:
        A chain that takes in a question and generates a SQL query that answers
        that question.

    Example:
        ```python
        # pip install -U langchain langchain-community langchain-openai
        from langchain_openai import ChatOpenAI
        from langchain_classic.chains import create_sql_query_chain
        from langchain_community.utilities import SQLDatabase

        db = SQLDatabase.from_uri("sqlite:///Chinook.db")
        model = ChatOpenAI(model="gpt-5.5", temperature=0)
        chain = create_sql_query_chain(model, db)
        response = chain.invoke({"question": "How many employees are there"})
        ```

    Prompt:
        If no prompt is provided, a default prompt is selected based on the SQLDatabase
        dialect. If one is provided, it must support input variables:

            * input: The user question plus suffix "\\nSQLQuery: " is passed here.
            * top_k: The number of results per select statement (the `k` argument to
                this function) is passed in here.
            * table_info: Table definitions and sample rows are passed in here. If the
                user specifies "table_names_to_use" when invoking chain, only those
                will be included. Otherwise, all tables are included.
            * dialect (optional): If dialect input variable is in prompt, the db
                dialect will be passed in here.

        Here's an example prompt:

        ```python
        from langchain_core.prompts import PromptTemplate

        template = '''Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Use the following format:

        Question: "Question here"
        SQLQuery: "SQL Query to run"
        SQLResult: "Result of the SQLQuery"
        Answer: "Final answer here"

        Only use the following tables:

        {table_info}.

        Question: {input}'''
        prompt = PromptTemplate.from_template(template)
        ```
    """  # noqa: E501
    if prompt is not None:
        prompt_to_use = prompt
    elif db.dialect in SQL_PROMPTS:
        prompt_to_use = SQL_PROMPTS[db.dialect]
    else:
        prompt_to_use = PROMPT
    if {"input", "top_k", "table_info"}.difference(
        prompt_to_use.input_variables + list(prompt_to_use.partial_variables),
    ):
        msg = (
            f"Prompt must have input variables: 'input', 'top_k', "
            f"'table_info'. Received prompt with input variables: "
            f"{prompt_to_use.input_variables}. Full prompt:\n\n{prompt_to_use}"
        )
        raise ValueError(msg)
    if "dialect" in prompt_to_use.input_variables:
        prompt_to_use = prompt_to_use.partial(dialect=db.dialect)

    table_info_kwargs = {}
    if get_col_comments:
        if db.dialect not in ("postgresql", "mysql", "oracle"):
            msg = (
                f"get_col_comments=True is only supported for dialects "
                f"'postgresql', 'mysql', and 'oracle'. Received dialect: "
                f"{db.dialect}"
            )
            raise ValueError(msg)
        table_info_kwargs["get_col_comments"] = True

    inputs = {
        "input": lambda x: sanitize_user_question(x["question"]) + "\nSQLQuery: ",
        "table_info": lambda x: sanitize_table_info(
            db.get_table_info(
                table_names=x.get("table_names_to_use"),
                **table_info_kwargs,
            )
        ),
    }
    return (
        RunnablePassthrough.assign(**inputs)  # type: ignore[return-value]
        | (
            lambda x: {
                k: v
                for k, v in x.items()
                if k not in ("question", "table_names_to_use")
            }
        )
        | prompt_to_use.partial(top_k=str(k))
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
        | validate_sql_output
    )
