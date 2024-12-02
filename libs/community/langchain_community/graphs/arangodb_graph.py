import itertools
import json
import os
from collections import defaultdict
from math import ceil
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_community.graphs.graph_document import Document, GraphDocument, Node
from langchain_community.graphs.neo4j_graph import value_sanitize
from langchain_community.graphs.graph_store import GraphStore

try:
    from arango.database import Database
    from arango.graph import Graph

    ARANGO_INSTALLED = True
except ImportError:
    print("ArangoDB not installed, please install with `pip install python-arango`.")
    ARANGO_INSTALLED = False


class ArangoGraph(GraphStore):
    """ArangoDB wrapper for graph operations.

    Parameters:
    db (arango.database.Database): ArangoDB database instance.
    sanitize (bool): A flag to indicate whether to remove lists with
            more than 128 elements from results. Useful for removing
            embedding-like properties from database responses. Default is False.
    include_examples (bool): A flag whether to scan the database for
            example values and use them in the graph schema. Default is True.
    graph_name (str): The name of the graph to use to generate the schema. If
            None, the entire database will be used.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(
        self,
        db: Database,
        include_examples: bool = True,
        graph_name: Optional[str] = None,
    ) -> None:
        if not ARANGO_INSTALLED:
            m = "ArangoDB not installed, please install with `pip install python-arango`."
            raise ImportError(m)

        self.__db: Database = db
        self.__schema = self.generate_schema(include_examples=include_examples, graph_name=graph_name)

    @property
    def db(self) -> "Database":
        return self.__db

    @property
    def schema(self) -> Dict[str, Any]:
        """Returns the schema of the Graph Database as a structured object"""
        return self.__schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the schema of the Graph Database as a structured object"""
        return self.__schema

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Graph Database as a string"""
        return json.dumps(self.__schema)

    def set_schema(self, schema: Dict[str, Any]) -> None:
        """Sets a custom schema for the ArangoDB Database."""
        self.__schema = schema

    def refresh_schema(
        self,
        sample_ratio: float = 0,
        graph_name: Optional[str] = None,
        include_examples: bool = True,
    ) -> None:
        """Refresh the graph schema information."""
        self.__schema = self.generate_schema(sample_ratio, graph_name, include_examples)

    def generate_schema(
        self,
        sample_ratio: float = 0,
        graph_name: Optional[str] = None,
        include_examples: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generates the schema of the ArangoDB Database and returns it

        Parameters:
        sample_ratio (float): A ratio (0 to 1) to determine the
        ratio of documents/edges used (in relation to the Collection size)
        to render each Collection Schema.
        graph_name (str): The name of the graph to use to generate the schema. If
            None, the entire database will be used.
        include_examples (bool): A flag whether to scan the database for
            example values and use them in the graph schema. Default is True.
        """
        if not 0 <= sample_ratio <= 1:
            raise ValueError("**sample_ratio** value must be in between 0 to 1")

        if graph_name:
            # Fetch a single graph
            graph: Graph = self.db.graph(graph_name)
            edge_definitions = graph.edge_definitions()

            graph_schema: List[Dict[str, Any]] = [{"name": graph_name, "edge_definitions": edge_definitions}]

            # Fetch graph-specific collections
            collection_names = set(graph.vertex_collections())
            for edge_definition in edge_definitions:
                collection_names.add(edge_definition["edge_collection"])

        else:
            # Fetch all graphs
            graph_schema: List[Dict[str, Any]] = [
                {"graph_name": g["name"], "edge_definitions": g["edge_definitions"]} for g in self.db.graphs()
            ]

            # Fetch all collections
            collection_names = {collection["name"] for collection in self.db.collections()}

        # Stores the schema of every ArangoDB Document/Edge collection
        collection_schema: List[Dict[str, Any]] = []
        for collection in self.db.collections():
            if collection["system"]:
                continue

            if collection["name"] not in collection_names:
                continue

            # Extract collection name, type, and size
            col_name: str = collection["name"]
            col_type: str = collection["type"]
            col_size: int = self.db.collection(col_name).count()

            # Skip collection if empty
            if col_size == 0:
                continue

            # Set number of ArangoDB documents/edges to retrieve
            limit_amount = ceil(sample_ratio * col_size) or 1

            aql = f"""
                FOR doc in @@col_name
                    LIMIT {limit_amount}
                    RETURN doc
            """

            doc: Dict[str, Any]
            properties: List[Dict[str, str]] = []
            for doc in self.db.aql.execute(aql, bind_vars={"@col_name": col_name}):
                for key, value in doc.items():
                    properties.append({"name": key, "type": type(value).__name__})

            collection_schema_entry = {
                "name": col_name,
                "type": col_type,
                f"properties": properties,
            }

            if include_examples:
                collection_schema_entry[f"example"] = value_sanitize(doc)

            collection_schema.append(collection_schema_entry)

        return {"graph_schema": graph_schema, "collection_schema": collection_schema}

    def query(self, query: str, top_k: Optional[int] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        """Query the ArangoDB database."""
        cursor = self.__db.aql.execute(query, **kwargs)
        return [value_sanitize(doc) for doc in itertools.islice(cursor, top_k)]

    def explain(self, query: str, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """Explain an AQL query without executing it."""
        return self.__db.aql.explain(query)

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
        batch_size: int = 1000,
        graph_name: Optional[str] = None,
    ) -> None:
        """
        This method constructs nodes and relationships in the graph based on the
        provided GraphDocument objects.

        Parameters:
        - graph_documents (List[GraphDocument]): A list of GraphDocument objects
        that contain the nodes and relationships to be added to the graph. Each
        GraphDocument should encapsulate the structure of part of the graph,
        including nodes, relationships, and the source document information.
        - include_source (bool, optional): If True, stores the source document
        and links it to nodes in the graph using the MENTIONS relationship.
        This is useful for tracing back the origin of data. Merges source
        documents based on the `id` property from the source document metadata
        if available; otherwise it calculates the MD5 hash of `page_content`
        for merging process. Defaults to False.
        - graph_name (str): The name of the ArangoDB General Graph to create. If None,
            no graph will be created.
        """
        if not graph_documents:
            return

        nodes = defaultdict(list)
        edges = defaultdict(list)
        edge_definitions_dict = defaultdict(lambda: defaultdict(set))

        if include_source:
            if not self.db.has_collection("MENTIONS"):
                self.db.create_collection("MENTIONS", edge=True)

            if not self.db.has_collection("GraphDocumentSource"):
                self.db.create_collection("GraphDocumentSource")

            edge_definitions_dict["MENTIONS"] = {
                "edge_collection": "MENTIONS",
                "from_vertex_collections": {"GraphDocumentSource"},
                "to_vertex_collections": set(),
            }

        for document in graph_documents:
            for i, node in enumerate(document.nodes, 1):
                node_id = str(node.id).replace(' ', '_')
                node_type = node.type.replace(' ', '_')
                node_data = {"_key": node_id, **node.properties}
                nodes[node_type].append(node_data)

                if i % batch_size == 0:
                    self.__import_data(nodes, is_edge=False)

            self.__import_data(nodes, is_edge=False)

            # Insert relationships
            for i, rel in enumerate(document.relationships, 1):
                source: Node = rel.source
                target: Node = rel.target

                rel_type = rel.type.replace(' ', '_')
                source_type = source.type.replace(' ', '_')
                target_type = target.type.replace(' ', '_')

                source_id = str(source.id).replace(' ', '_')
                target_id = str(target.id).replace(' ', '_')

                edge_definitions_dict[rel_type]["edge_collection"].add(rel_type)
                edge_definitions_dict[rel_type]["from_vertex_collections"].add(source_type)
                edge_definitions_dict[rel_type]["to_vertex_collections"].add(target_type)

                rel_data = {
                    "_from": f"{source_type}/{source_id}",
                    "_to": f"{target_type}/{target_id}",
                    **rel.properties,
                }

                edges[rel_type].append(rel_data)

                if i % batch_size == 0:
                    self.__import_data(edges, is_edge=True)

            self.__import_data(edges, is_edge=True)

            # Insert source document if required
            if include_source:
                doc_source: Document = document.source

                _key = str(doc_source.metadata.get("id", uuid4())).replace(' ', '_')
                source_data = {
                    "_key": _key,
                    "text": doc_source.page_content,
                    "metadata": doc_source.metadata,
                }

                self.db.collection("GraphDocumentSource").insert(source_data, overwrite=True)

                mentions = []
                mentions_col = self.db.collection("MENTIONS")
                for i, node in enumerate(document.nodes, 1):
                    node_id = str(node.id).replace(' ', '_')
                    node_type = node.type.replace(' ', '_')
                    edge_definitions_dict["MENTIONS"]["to_vertex_collections"].add(node_type)

                    mentions.append(
                        {
                            "_from": f"GraphDocumentSource/{_key}",
                            "_to": f"{node_type}/{node_id}",
                        }
                    )

                    if i % batch_size == 0:
                        mentions_col.import_bulk(mentions, on_duplicate="update")
                        mentions.clear()

                mentions_col.import_bulk(mentions, on_duplicate="update")

        if graph_name:
            edge_definitions = []
            for k, v in edge_definitions_dict.items():
                edge_definitions.append(
                    {
                        "edge_collection": k,
                        "from_vertex_collections": list(v["from_vertex_collections"]),
                        "to_vertex_collections": list(v["to_vertex_collections"]),
                    }
                )

            if not self.db.has_graph(graph_name):
                self.db.create_graph(graph_name, edge_definitions)
            else:
                graph = self.db.graph(graph_name)
                for e_d in edge_definitions:
                    if not graph.has_edge_definition(e_d["edge_collection"]):
                        graph.create_edge_definition(*e_d.values())
                    else:
                        graph.replace_edge_definition(*e_d.values())

        # Refresh schema after insertions
        self.refresh_schema()

    @classmethod
    def from_db_credentials(
        cls,
        url: Optional[str] = None,
        dbname: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Any:
        """Convenience constructor that builds Arango DB from credentials.

        Args:
            url: Arango DB url. Can be passed in as named arg or set as environment
                var ``ARANGODB_URL``. Defaults to "http://localhost:8529".
            dbname: Arango DB name. Can be passed in as named arg or set as
                environment var ``ARANGODB_DBNAME``. Defaults to "_system".
            username: Can be passed in as named arg or set as environment var
                ``ARANGODB_USERNAME``. Defaults to "root".
            password: Can be passed ni as named arg or set as environment var
                ``ARANGODB_PASSWORD``. Defaults to "".

        Returns:
            An arango.database.StandardDatabase.
        """
        db = get_arangodb_client(url=url, dbname=dbname, username=username, password=password)
        return cls(db)

    def __import_data(self, data: Dict[str, List[Dict[str, Any]]], is_edge: bool) -> None:
        for collection, batch in data.items():
            if not self.db.has_collection(collection):
                self.db.create_collection(collection, edge=is_edge)

            self.db.collection(collection).import_bulk(batch, on_duplicate="update")

        data.clear()


def get_arangodb_client(
    url: Optional[str] = None,
    dbname: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Any:
    """Get the Arango DB client from credentials.

    Args:
        url: Arango DB url. Can be passed in as named arg or set as environment
            var ``ARANGODB_URL``. Defaults to "http://localhost:8529".
        dbname: Arango DB name. Can be passed in as named arg or set as
            environment var ``ARANGODB_DBNAME``. Defaults to "_system".
        username: Can be passed in as named arg or set as environment var
            ``ARANGODB_USERNAME``. Defaults to "root".
        password: Can be passed ni as named arg or set as environment var
            ``ARANGODB_PASSWORD``. Defaults to "".

    Returns:
        An arango.database.StandardDatabase.
    """
    try:
        from arango import ArangoClient
    except ImportError as e:
        m = "Unable to import arango, please install with `pip install python-arango`."
        raise ImportError(m) from e

    _url: str = url or os.environ.get("ARANGODB_URL", "http://localhost:8529")  # type: ignore[assignment]
    _dbname: str = dbname or os.environ.get("ARANGODB_DBNAME", "_system")  # type: ignore[assignment]
    _username: str = username or os.environ.get("ARANGODB_USERNAME", "root")  # type: ignore[assignment]
    _password: str = password or os.environ.get("ARANGODB_PASSWORD", "")  # type: ignore[assignment]

    return ArangoClient(_url).db(_dbname, _username, _password, verify=True)
