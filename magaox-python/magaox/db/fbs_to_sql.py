from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Any

from lark import Lark, Transformer, v_args

FBS_GRAMMAR = r"""
?start: schema

schema: namespace_decl? decl*

namespace_decl: "namespace" dotted_ident ";"
dotted_ident: IDENT ("." IDENT)*

decl: table_decl
    | root_decl

table_decl: "table" IDENT "{" field_decl* "}"

field_decl: IDENT ":" ftype default_value? metadata? ";"
default_value: "=" scalar

metadata: "(" meta_entry_list? ")"
meta_entry_list: meta_entry ("," meta_entry)*
meta_entry: IDENT (":" scalar)?

?ftype: array_type
     | scalar_type
     | IDENT               -> user_type

// arrays you mentioned: [string], [float], [long], etc.
array_type: "[" scalar_type "]"

scalar_type: "bool"        -> t_bool
           | "byte"        -> t_byte
           | "ubyte"       -> t_ubyte
           | "short"       -> t_short
           | "ushort"      -> t_ushort
           | "int"         -> t_int
           | "uint"        -> t_uint
           | "long"        -> t_long
           | "ulong"       -> t_ulong
           | "float"       -> t_float
           | "double"      -> t_double
           | "int8"        -> t_int8
           | "uint8"       -> t_uint8
           | "int16"       -> t_int16
           | "uint16"      -> t_uint16
           | "int32"       -> t_int32
           | "uint32"      -> t_uint32
           | "int64"       -> t_int64
           | "uint64"      -> t_uint64
           | "float32"     -> t_float32
           | "float64"     -> t_float64
           | "string"      -> t_string

?scalar: BOOLEAN
       | FLOAT
       | INTEGER
       | STRING

root_decl: "root_type" IDENT ";"

BOOLEAN: "true" | "false"
FLOAT.2: /[-+]?(?:(?:(?:\.[0-9]+)|(?:[0-9]+\.[0-9]*)|(?:[0-9]+))(?:[eE][-+]?[0-9]+)?|(?:nan|inf|infinity))/i
INTEGER.1: /[-+]?[0-9]+/
STRING: /"(?:\\.|[^"\\])*"/
IDENT: /[a-zA-Z_][a-zA-Z0-9_]*/

%import common.WS
%ignore WS
LINE_COMMENT: /\/\/[^\n]*/
BLOCK_COMMENT: /\/\*[\s\S]*?\*\//
%ignore LINE_COMMENT
%ignore BLOCK_COMMENT
"""

# ftype kinds:
# ("scalar", "int"), ("user", "TempCtrl"), ("array", "string")
FType = Tuple[str, str]

@dataclass(frozen=True)
class Field:
    name: str
    ftype: FType
    default: Optional[str] = None  # SQL literal
    deprecated: bool = False

@dataclass
class TableDecl:
    name: str
    fields: List[Field]

@dataclass
class Schema:
    namespace: Optional[str]
    tables: Dict[str, TableDecl]
    root_type: Optional[str]

@v_args(inline=True)
class BuildSchema(Transformer):
    def __init__(self):
        super().__init__()
        self._namespace: Optional[str] = None
        self._tables: Dict[str, TableDecl] = {}
        self._root: Optional[str] = None

    def schema(self, *items):
        return Schema(self._namespace, self._tables, self._root)

    def dotted_ident(self, *parts):
        return ".".join(str(p) for p in parts)

    def namespace_decl(self, dotted):
        self._namespace = str(dotted)

    def table_decl(self, name, *fields):
        # Filter out anything that's not a Field (shouldn't happen, but safe)
        real_fields = [f for f in fields if isinstance(f, Field)]
        self._tables[str(name)] = TableDecl(str(name), real_fields)

    # --- metadata -> list[(key, value_sql_or_None)]
    def metadata(self, entries=None):
        # entries is either list[...] or None
        return entries or []

    def meta_entry_list(self, *entries):
        # Return a list, not a tuple
        return list(entries)

    def meta_entry(self, key, value=None):
        # value is already a SQL literal string (or None)
        return (str(key), value)

    def default_value(self, scalar):
        return scalar

    def field_decl(self, fname, ftype, *rest):
        """
        With inline=True and optional default/metadata:
          rest can be:
            ()
            (default_sql,)
            (metadata_list,)
            (default_sql, metadata_list)
        We must disambiguate the single-item case.
        """
        default_sql: Optional[str] = None
        metadata: List[Tuple[str, Optional[str]]] = []

        if len(rest) == 1:
            (only,) = rest
            if isinstance(only, list):
                metadata = only
            else:
                default_sql = only
        elif len(rest) == 2:
            default_sql, metadata = rest  # type: ignore[misc]
        elif len(rest) > 2:
            raise ValueError(f"Unexpected field_decl tail: {rest!r}")

        deprecated = False
        for k, v in metadata:
            # support (deprecated) and (deprecated:true)
            if k == "deprecated" and (v is None or v == "TRUE"):
                deprecated = True

        return Field(
            name=str(fname),
            ftype=ftype,
            default=default_sql,
            deprecated=deprecated,
        )

    def user_type(self, ident):
        return ("user", str(ident))

    def array_type(self, scalar_type):
        kind, ty = scalar_type  # ("scalar", "float") etc
        if kind != "scalar":
            raise ValueError("Only arrays of scalar types supported here.")
        return ("array", ty)

    # scalar types -> ("scalar", name)
    def t_bool(self):    return ("scalar", "bool")
    def t_byte(self):    return ("scalar", "byte")
    def t_ubyte(self):   return ("scalar", "ubyte")
    def t_short(self):   return ("scalar", "short")
    def t_ushort(self):  return ("scalar", "ushort")
    def t_int(self):     return ("scalar", "int")
    def t_uint(self):    return ("scalar", "uint")
    def t_long(self):    return ("scalar", "long")
    def t_ulong(self):   return ("scalar", "ulong")
    def t_float(self):   return ("scalar", "float")
    def t_double(self):  return ("scalar", "double")
    def t_int8(self):    return ("scalar", "int8")
    def t_uint8(self):   return ("scalar", "uint8")
    def t_int16(self):   return ("scalar", "int16")
    def t_uint16(self):  return ("scalar", "uint16")
    def t_int32(self):   return ("scalar", "int32")
    def t_uint32(self):  return ("scalar", "uint32")
    def t_int64(self):   return ("scalar", "int64")
    def t_uint64(self):  return ("scalar", "uint64")
    def t_float32(self): return ("scalar", "float32")
    def t_float64(self): return ("scalar", "float64")
    def t_string(self):  return ("scalar", "string")

    # scalars -> SQL literals
    def BOOLEAN(self, tok):
        return "TRUE" if tok.value == "true" else "FALSE"

    def INTEGER(self, tok):
        return tok.value

    def FLOAT(self, tok):
        v = tok.value.lower()
        if v in ("nan", "+nan", "-nan"):
            return "'NaN'"
        if v in ("inf", "+inf", "infinity", "+infinity"):
            return "'Infinity'"
        if v in ("-inf", "-infinity"):
            return "'-Infinity'"
        return tok.value

    def STRING(self, tok):
        inner = tok.value[1:-1].replace("'", "''")
        return f"'{inner}'"

    def root_decl(self, ident):
        self._root = str(ident)

# ---- SQL mapping (Postgres-flavored) ----
def sql_type_for_scalar(t: str) -> str:
    mapping = {
        "bool": "BOOLEAN",
        "string": "TEXT",
        "float": "REAL",
        "float32": "REAL",
        "double": "DOUBLE PRECISION",
        "float64": "DOUBLE PRECISION",

        "int": "INTEGER",
        "int32": "INTEGER",

        "uint": "BIGINT",
        "uint32": "BIGINT",

        "long": "BIGINT",
        "int64": "BIGINT",

        "ulong": "NUMERIC(20)",
        "uint64": "NUMERIC(20)",

        "int8": "SMALLINT",
        "uint8": "SMALLINT",
        "byte": "SMALLINT",
        "ubyte": "SMALLINT",

        "short": "SMALLINT",
        "int16": "SMALLINT",
        "ushort": "INTEGER",
        "uint16": "INTEGER",
    }
    if t not in mapping:
        raise ValueError(f"Unknown scalar type: {t}")
    return mapping[t]

def sql_type_for_array_scalar(t: str, array_mode: str = "array") -> str:
    if array_mode == "jsonb":
        return "JSONB"
    return f"{sql_type_for_scalar(t)}[]"

def colname(parts: List[str], joiner: str = "__") -> str:
    return joiner.join(parts)

def flatten_table(
    schema: Schema,
    table: str,
    prefix: Optional[List[str]] = None,
    joiner: str = "__",
    array_mode: str = "array",  # "array" or "jsonb"
    seen: Optional[Set[str]] = None,
):
    prefix = prefix or []
    seen = seen or set()
    if table in seen:
        raise ValueError(f"Cycle detected: {' -> '.join(list(seen) + [table])}")
    seen.add(table)

    if table not in schema.tables:
        raise KeyError(f"Unknown table referenced: {table}")

    t = schema.tables[table]
    out = []
    for f in t.fields:
        kind, ty = f.ftype
        name = colname(prefix + [f.name], joiner=joiner)

        if kind == "scalar":
            out.append((name, sql_type_for_scalar(ty), f.default, f.deprecated))
        elif kind == "array":
            out.append((name, sql_type_for_array_scalar(ty, array_mode=array_mode), f.default, f.deprecated))
        elif kind == "user":
            out.extend(flatten_table(schema, ty, prefix + [f.name], joiner, array_mode, seen.copy()))
        else:
            raise ValueError(f"Unknown field kind: {kind}")
    return out

def create_table_sql(table_name: str, schema: Schema, joiner: str = "__", array_mode: str = "array") -> str:
    if not schema.root_type:
        raise ValueError("No root_type declared.")
    cols = [
        ('ts', 'TIMESTAMPTZ', None, False),
        ('device', 'VARCHAR(50)', None, False),
    ]
    cols.extend(flatten_table(schema, schema.root_type, joiner=joiner, array_mode=array_mode))

    def qident(s: str) -> str:
        # columns like tempCtrl__setpt are safe, but quoting avoids reserved word issues
        return '"' + s.replace('"', '""') + '"'

    lines = [f'CREATE TABLE IF NOT EXISTS "{table_name}" (']
    col_lines = []
    for name, ty, default, deprecated in cols:
        pieces = [f"  {qident(name)} {ty}"]
        if default is not None:
            pieces.append(f"DEFAULT {default}")
        # Policy: non-deprecated fields are guaranteed present
        if not deprecated:
            pieces.append("NOT NULL")
        col_lines.append(" ".join(pieces))
    col_lines.append("  PRIMARY KEY (ts, device)")
    lines.append(",\n".join(col_lines))
    lines.append(");")
    lines.append(f"CREATE INDEX IF NOT EXISTS {table_name}_device_ts ON {table_name} (device, ts);")
    return "\n".join(lines)

def fbs_to_sql(table_name: str, text: str, joiner: str = "__", array_mode: str = "array") -> str:
    """
    array_mode:
      - "array" => Postgres typed arrays (TEXT[], REAL[], BIGINT[], ...)
      - "jsonb" => JSONB for any vector field
    """
    parser = Lark(FBS_GRAMMAR, parser="lalr", lexer="contextual", maybe_placeholders=False)
    tree = parser.parse(text)
    schema = BuildSchema().transform(tree)
    return create_table_sql(table_name, schema, joiner=joiner, array_mode=array_mode)
