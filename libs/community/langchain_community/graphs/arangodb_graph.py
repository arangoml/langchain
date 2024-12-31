import itertools
import json
import os
import re
from collections import defaultdict
from math import ceil
from typing import Any, DefaultDict, Dict, List, Optional

from langchain_community.graphs.graph_document import (
    Document,
    GraphDocument,
    Node,
    Relationship,
)
from langchain_community.graphs.graph_store import GraphStore

try:
    from arango.database import StandardDatabase
    from arango.graph import Graph

    ARANGO_INSTALLED = True
except ImportError:
    print("ArangoDB not installed, please install with `pip install python-arango`.")
    ARANGO_INSTALLED = False

SOURCE_VERTEX_COLLECTION = "SOURCE"
SOURCE_EDGE_COLLECTION = "HAS_SOURCE"
ENTITY_VERTEX_COLLECTION = "ENTITY"
ENTITY_EDGE_COLLECTION = "LINKS_TO"


# Taken from langchain_community/graphs/neo4j_graph.py
def value_sanitize(d: Any) -> Any:
    """Sanitize the input dictionary or list.

    Sanitizes the input by removing embedding-like values,
    lists with more than 128 elements, that are mostly irrelevant for
    generating answers in a LLM context. These properties, if left in
    results, can occupy significant context space and detract from
    the LLM's performance by introducing unnecessary noise and cost.

    Args:
        d (Any): The input dictionary or list to sanitize.

    Returns:
        Any: The sanitized dictionary or list.
    """
    LIST_LIMIT = 128

    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                sanitized_value = value_sanitize(value)
                if (
                    sanitized_value is not None
                ):  # Check if the sanitized value is not None
                    new_dict[key] = sanitized_value
            elif isinstance(value, list):
                if len(value) < LIST_LIMIT:
                    sanitized_value = value_sanitize(value)
                    if (
                        sanitized_value is not None
                    ):  # Check if the sanitized value is not None
                        new_dict[key] = sanitized_value
                # Do not include the key if the list is oversized
            else:
                new_dict[key] = value
        return new_dict
    elif isinstance(d, list):
        if len(d) < LIST_LIMIT:
            return [
                value_sanitize(item) for item in d if value_sanitize(item) is not None
            ]
        else:
            return None
    else:
        return d


class ArangoGraph(GraphStore):
    """ArangoDB wrapper for graph operations.

    Parameters:
    - db (arango.database.StandardDatabase): ArangoDB database instance.
    - include_examples (bool): A flag whether to scan the database for
        example values and use them in the graph schema. Default is True.
    - graph_name (str): The name of the graph to use to generate the schema. If
        None, the entire database will be used.
    - generate_schema_on_init (bool): A flag whether to generate the schema
        on initialization. Default is True.

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
        db: StandardDatabase,
        include_examples: bool = True,
        graph_name: Optional[str] = None,
        generate_schema_on_init: bool = True,
    ) -> None:
        if not ARANGO_INSTALLED:
            m = "ArangoDB not installed, please install with `pip install python-arango`."
            raise ImportError(m)

        self.__db = db
        self.__async_db = db.begin_async_execution()

        self.__schema = {}
        if generate_schema_on_init:
            self.__schema = self.generate_schema(
                include_examples=include_examples, graph_name=graph_name
            )

    @property
    def db(self) -> "StandardDatabase":
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
        """
        Refresh the graph schema information.

        Parameters:
        - sample_ratio (float): A ratio (0 to 1) to determine the
        ratio of documents/edges used (in relation to the Collection size) to render
        each Collection Schema.
        - graph_name (str): The name of the graph to use to generate the schema. If
            None, the entire database will be used.
        - include_examples (bool): A flag whether to scan the database for
            example values and use them in the graph schema. Default is True.
        """
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
        - sample_ratio (float): A ratio (0 to 1) to determine the
        ratio of documents/edges used (in relation to the Collection size)
        to render each Collection Schema.
        - graph_name (str): The name of the graph to use to generate the schema. If
            None, the entire database will be used.
        - include_examples (bool): A flag whether to scan the database for
            example values and use them in the graph schema. Default is True.
        """
        if not 0 <= sample_ratio <= 1:
            raise ValueError("**sample_ratio** value must be in between 0 to 1")

        if graph_name:
            # Fetch a single graph
            graph: Graph = self.db.graph(graph_name)
            edge_definitions = graph.edge_definitions()

            graph_schema: List[Dict[str, Any]] = [
                {"name": graph_name, "edge_definitions": edge_definitions}
            ]

            # Fetch graph-specific collections
            collection_names = set(graph.vertex_collections())
            for edge_definition in edge_definitions:
                collection_names.add(edge_definition["edge_collection"])

        else:
            # Fetch all graphs
            graph_schema: List[Dict[str, Any]] = [
                {"graph_name": g["name"], "edge_definitions": g["edge_definitions"]}
                for g in self.db.graphs()
            ]

            # Fetch all collections
            collection_names = {
                collection["name"] for collection in self.db.collections()
            }

        # Stores the schema of every ArangoDB Document/Edge collection
        collection_schema: List[Dict[str, Any]] = []
        for collection in self.db.collections():
            if collection["system"] or collection["name"] not in collection_names:
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

    def query(
        self, query: str, top_k: Optional[int] = None, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Execute an AQL query and return the results.

        Parameters:
        - query (str): The AQL query to execute.
        - top_k (int): The number of results to return. If None, all results are returned.
        - kwargs: Additional keyword arguments to pass to the AQL query.

        Returns:
        - A list of dictionaries containing the query results.
        """
        cursor = self.__db.aql.execute(query, **kwargs)
        return [value_sanitize(doc) for doc in itertools.islice(cursor, top_k)]

    def explain(self, query: str, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Explain an AQL query without executing it.

        Parameters:
        - query (str): The AQL query to explain.
        - args: Additional positional arguments to pass to the AQL query.
        - kwargs: Additional keyword arguments to pass to the AQL query.

        Returns:
        - A list of dictionaries containing the query explanation.
        """
        return self.__db.aql.explain(query)

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
        graph_name: Optional[str] = None,
        update_graph_definition_if_exists: bool = False,
        batch_size: int = 1000,
        use_one_entity_collection: bool = True,
        insert_async: bool = False,
        source_collection_name: str | None = None,
        source_edge_collection_name: str | None = None,
        entity_collection_name: str | None = None,
        entity_edge_collection_name: str | None = None,
        source_metadata_fields_to_extract_to_top_level: set[str] = {},
    ) -> None:
        """
        Constructs nodes & relationships in the graph based on the
        provided GraphDocument objects.

        Parameters:
        - graph_documents (List[GraphDocument]): A list of GraphDocument objects
        that contain the nodes and relationships to be added to the graph. Each
        GraphDocument should encapsulate the structure of part of the graph,
        including nodes, relationships, and the source document information.
        - include_source (bool, optional): If True, stores the source document
        and links it to nodes in the graph using the HAS_SOURCE relationship.
        This is useful for tracing back the origin of data. Merges source
        documents based on the `id` property from the source document if available,
        otherwise it calculates the Farmhash hash of `page_content`
        for merging process. Defaults to False.
        - graph_name (str): The name of the ArangoDB General Graph to create. If None,
            no graph will be created.
        - update_graph_definition_if_exists (bool): If True, updates the graph Edge Definitions
        if it already exists. Defaults to False. Not used if `graph_name` is None. It is
        recommended to set this to True if `use_one_entity_collection` is set to False.
        - batch_size (int): The number of nodes/edges to insert in a single batch.
        - use_one_entity_collection (bool): If True, all nodes are stored in a single
        entity collection. If False, nodes are stored in separate collections based
        on their type. Defaults to True.
        - insert_async (bool): If True, inserts data asynchronously. Defaults to False.
        - source_collection_name (str): The name of the collection to store the source
        documents. Defaults to "SOURCE".
        - source_edge_collection_name (str): The name of the edge collection to store
        the relationships between source documents and nodes. Defaults to "HAS_SOURCE".
        - entity_collection_name (str): The name of the collection to store the nodes.
        Defaults to "ENTITY". Only used if `use_one_entity_collection` is True.
        - entity_edge_collection_name (str): The name of the edge collection to store
        the relationships between nodes. Defaults to "LINKS_TO". Only used if
        `use_one_entity_collection` is True.
        - source_metadata_fields_to_extract_to_top_level (set[str]): A set of field names
        to extract from the source document metadata and add to the top level of the source
        document. This is useful for making metadata fields more accessible for querying.
        Defaults to an empty set. Only used if `include_source` is True.
        """
        if not graph_documents:
            print("No Graph Documents to insert.")
            return

        try:
            import farmhash
        except ImportError:
            m = "Farmhash not installed, please install with `pip install cityhash`."
            raise ImportError(m)

        #########
        # Setup #
        #########

        suffix = f"{graph_name}_" if graph_name else ""
        if not source_collection_name:
            source_collection_name = f"{suffix}{SOURCE_VERTEX_COLLECTION}"
        if not source_edge_collection_name:
            source_edge_collection_name = f"{suffix}{SOURCE_EDGE_COLLECTION}"
        if not entity_collection_name:
            entity_collection_name = f"{suffix}{ENTITY_VERTEX_COLLECTION}"
        if not entity_edge_collection_name:
            entity_edge_collection_name = f"{suffix}{ENTITY_EDGE_COLLECTION}"

        insertion_db = self.__async_db if insert_async else self.__db
        nodes: DefaultDict[str, list[dict[str, Any]]] = defaultdict(list)
        edges: DefaultDict[str, list[dict[str, Any]]] = defaultdict(list)
        edge_definitions_dict: DefaultDict[str, DefaultDict[str, set[str]]] = (
            defaultdict(lambda: defaultdict(set))
        )

        if include_source:
            self.__create_collection(source_collection_name)
            self.__create_collection(source_edge_collection_name, is_edge=True)

            from_cols = {entity_collection_name} if use_one_entity_collection else set()
            edge_definitions_dict[source_edge_collection_name] = {
                "from_vertex_collections": from_cols,
                "to_vertex_collections": {source_collection_name},
            }

        if use_one_entity_collection:
            self.__create_collection(entity_collection_name)
            self.__create_collection(entity_edge_collection_name, is_edge=True)

            edge_definitions_dict[entity_edge_collection_name] = {
                "from_vertex_collections": {entity_collection_name},
                "to_vertex_collections": {entity_collection_name},
            }

        process_node_fn = (
            self.__process_node_as_entity
            if use_one_entity_collection
            else self.__process_node_as_type
        )

        process_edge_fn = (
            self.__process_edge_as_entity
            if use_one_entity_collection
            else self.__process_edge_as_type
        )

        source_id_hash = None

        #######################
        # Document Processing #
        #######################

        for document in graph_documents:
            # 1. Process Source Document
            if include_source:
                source_id_hash = self.__process_source(
                    document.source,
                    source_collection_name,
                    source_metadata_fields_to_extract_to_top_level,
                )

            # 2. Process Nodes
            node_key_map = {}
            for i, node in enumerate(document.nodes, 1):
                node_key = self.__hash(node.id)
                node_key_map[node.id] = node_key
                node_type = process_node_fn(
                    node_key, node, nodes, entity_collection_name
                )

                # 2.1 Link Source Document to Node
                if include_source:
                    edges[source_edge_collection_name].append(
                        {
                            "_from": f"{node_type}/{node_key}",
                            "_to": f"{source_collection_name}/{source_id_hash}",
                        }
                    )

                    if not use_one_entity_collection:
                        edge_definitions_dict[source_edge_collection_name][
                            "from_vertex_collections"
                        ].add(node_type)

                # 2.2 Batch Insert
                if i % batch_size == 0:
                    self.__import_data(insertion_db, nodes, is_edge=False)
                    self.__import_data(insertion_db, edges, is_edge=True)

            self.__import_data(insertion_db, nodes, is_edge=False)
            self.__import_data(insertion_db, edges, is_edge=True)

            # 3. Process Edges
            for i, edge in enumerate(document.relationships, 1):
                source, target = edge.source, edge.target

                source_key = self.__get_node_key(
                    source,
                    nodes,
                    node_key_map,
                    entity_collection_name,
                    process_node_fn,
                )

                target_key = self.__get_node_key(
                    target,
                    nodes,
                    node_key_map,
                    entity_collection_name,
                    process_node_fn,
                )

                edge_key = self.__hash(f"{source.id}{edge.type}{target.id}")
                process_edge_fn(
                    edge,
                    edge_key,
                    source_key,
                    target_key,
                    edges,
                    entity_collection_name,
                    entity_edge_collection_name,
                    edge_definitions_dict,
                )

                # 3.1 Batch Insert
                if i % batch_size == 0:
                    self.__import_data(insertion_db, edges, is_edge=True)
                    self.__import_data(insertion_db, nodes, is_edge=False)

            self.__import_data(insertion_db, edges, is_edge=True)
            self.__import_data(insertion_db, nodes, is_edge=False)

        ##################
        # Graph Creation #
        ##################

        if graph_name:
            edge_definitions = [
                {
                    "edge_collection": k,
                    "from_vertex_collections": list(v["from_vertex_collections"]),
                    "to_vertex_collections": list(v["to_vertex_collections"]),
                }
                for k, v in edge_definitions_dict.items()
            ]

            if not self.db.has_graph(graph_name):
                self.db.create_graph(graph_name, edge_definitions)

            elif update_graph_definition_if_exists:
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
        db = get_arangodb_client(
            url=url, dbname=dbname, username=username, password=password
        )
        return cls(db)

    def __import_data(
        self,
        db: "StandardDatabase",
        data: Dict[str, List[Dict[str, Any]]],
        is_edge: bool,
    ) -> None:
        """Imports data into the ArangoDB database in bulk."""
        for collection, batch in data.items():
            self.__create_collection(collection, is_edge)
            db.collection(collection).import_bulk(batch, on_duplicate="update")

        data.clear()

    def __create_collection(
        self, collection_name: str, is_edge: bool = False, **kwargs
    ) -> None:
        """Creates a collection in the ArangoDB database if it does not exist."""
        if not self.db.has_collection(collection_name):
            self.db.create_collection(collection_name, edge=is_edge, **kwargs)

    def __process_node_as_entity(
        self,
        node_key: str,
        node: Node,
        nodes: DefaultDict[str, list],
        entity_collection_name: str,
    ) -> tuple[str, str]:
        """Processes a Graph Document Node into ArangoDB as a unanimous Entity."""
        nodes[entity_collection_name].append(
            {
                "_key": node_key,
                "name": node.id,
                "type": node.type,
                **node.properties,
            }
        )
        return entity_collection_name

    def __process_node_as_type(
        self, node_key: str, node: Node, nodes: DefaultDict[str, list], _: str
    ) -> str:
        """Processes a Graph Document Node into ArangoDB based on its Node Type."""
        node_type = self.__sanitize_collection_name(node.type)
        nodes[node_type].append({"_key": node_key, "name": node.id, **node.properties})
        return node_type

    def __process_edge_as_entity(
        self,
        edge: Relationship,
        edge_key: str,
        source_key: str,
        target_key: str,
        edges: DefaultDict[str, list],
        entity_collection_name: str,
        entity_edge_collection_name: str,
        _: DefaultDict[str, DefaultDict[str, set[str]]],
    ) -> None:
        """Processes a Graph Document Edge into ArangoDB as a unanimous Entity."""
        edges[entity_edge_collection_name].append(
            {
                "_key": edge_key,
                "_from": f"{entity_collection_name}/{source_key}",
                "_to": f"{entity_collection_name}/{target_key}",
                "type": edge.type,
                **edge.properties,
            }
        )

    def __process_edge_as_type(
        self,
        edge: Relationship,
        edge_key: str,
        source_key: str,
        target_key: str,
        edges: DefaultDict[str, list],
        _1: str,
        _2: str,
        edge_definitions_dict: DefaultDict[str, DefaultDict[str, set[str]]],
    ) -> None:
        """Processes a Graph Document Edge into ArangoDB based on its Edge Type."""
        source: Node = edge.source
        target: Node = edge.target

        edge_type = self.__sanitize_collection_name(edge.type)
        source_type = self.__sanitize_collection_name(source.type)
        target_type = self.__sanitize_collection_name(target.type)

        edge_definitions_dict[edge_type]["from_vertex_collections"].add(source_type)
        edge_definitions_dict[edge_type]["to_vertex_collections"].add(target_type)

        edges[edge_type].append(
            {
                "_key": edge_key,
                "_from": f"{source_type}/{source_key}",
                "_to": f"{target_type}/{target_key}",
                **edge.properties,
            }
        )

    def __get_node_key(
        self,
        node: Node,
        nodes: DefaultDict[str, list],
        node_key_map: Dict[str, str],
        entity_collection_name: str,
        process_node_fn: Any,
    ) -> str:
        """Gets the key of a node and processes it if it doesn't exist."""
        if node.id in node_key_map:
            return node_key_map[node.id]

        node_key = self.__hash(node.id)
        node_key_map[node.id] = node_key
        process_node_fn(node_key, node, nodes, entity_collection_name)

        return node_key

    def __process_source(
        self,
        source: Document,
        source_collection_name: str,
        source_metadata_fields_to_extract_to_top_level: set[str],
    ) -> str:
        """Processes a Graph Document Source into ArangoDB."""
        source_id = self.__hash(
            source.id if source.id else source.page_content.encode("utf-8")
        )

        top_level_fields = {}
        metadata = source.metadata
        for field in source_metadata_fields_to_extract_to_top_level:
            if field in metadata:
                top_level_fields[field] = metadata.pop(field)

        doc = {
            "_key": source_id,
            "text": source.page_content,
            "type": source.type,
            "metadata": metadata,
            **top_level_fields,
        }

        self.db.collection(source_collection_name).insert(doc, overwrite=True)

        return source_id

    def __hash(self, value: Any) -> str:
        """Applies the Farmhash hash function to a value."""
        import farmhash

        try:
            value_str = str(value)
        except Exception:
            raise ValueError("Value must be a string or have a string representation.")

        return str(farmhash.Fingerprint64(value_str))

    def __sanitize_collection_name(self, name: str) -> str:
        """
        Modifies a string to adhere to ArangoDB collection name rules.

        - Trims the name to 256 characters if it's too long.
        - Replaces invalid characters with underscores (_).
        - Ensures the name starts with a letter (prepends 'a' if needed).
        """
        if not name:
            raise ValueError("Collection name cannot be empty.")

        name = name[:256]

        # Replace invalid characters with underscores
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

        # Ensure the name starts with a letter; prepend 'a' if not
        if not re.match(r"^[a-zA-Z]", name):
            name = f"Collection_{name}"

        return name


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
