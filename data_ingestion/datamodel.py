from pydantic import Field
from typing import Annotated
from pydantic import BaseModel
from semantic_kernel.data import (
    vectorstoremodel,
    VectorStoreRecordDataField,
    VectorStoreRecordKeyField,
    VectorStoreRecordVectorField,
)


@vectorstoremodel
class SKDataModel(BaseModel):
    id: Annotated[str, VectorStoreRecordKeyField]
    chunk: Annotated[
        str,
        VectorStoreRecordDataField(
            is_full_text_searchable=True,
            has_embedding=True,
            embedding_property_name="embedding",
        ),
    ]
    embedding: Annotated[
        list[float] | None,
        VectorStoreRecordVectorField(local_embedding=True, dimensions=1536),
    ] = None
    metadata: Annotated[
        str | None, VectorStoreRecordDataField(is_full_text_searchable=False)
    ] = None
    doc_id: Annotated[str | None, VectorStoreRecordDataField(is_filterable=True)] = None
    topic: Annotated[str | None, VectorStoreRecordDataField(is_filterable=True)] = None
    subtopic: Annotated[str | None, VectorStoreRecordDataField(is_filterable=True)] = (
        None
    )
    connector: Annotated[str | None, VectorStoreRecordDataField(is_filterable=True)] = (
        None
    )


@vectorstoremodel
class SKQdrantDataModel(BaseModel):
    id: Annotated[str, VectorStoreRecordKeyField]
    node_content: Annotated[
        str,
        VectorStoreRecordDataField(
            is_full_text_searchable=True,
            has_embedding=True,
            embedding_property_name="embedding",
        ),
        Field(alias="_node_content"),
    ]
    embedding: Annotated[
        list[float] | None,
        VectorStoreRecordVectorField(local_embedding=True, dimensions=1536),
    ] = None
    doc_id: Annotated[str | None, VectorStoreRecordDataField(is_filterable=True)] = None
    topic: Annotated[str | None, VectorStoreRecordDataField(is_filterable=True)] = None
    subtopic: Annotated[str | None, VectorStoreRecordDataField(is_filterable=True)] = (
        None
    )
    connector: Annotated[str | None, VectorStoreRecordDataField(is_filterable=True)] = (
        None
    )
    file_path: Annotated[str | None, VectorStoreRecordDataField(is_filterable=True)] = (
        None
    )
    file_name: Annotated[str | None, VectorStoreRecordDataField(is_filterable=True)] = (
        None
    )
    document_id: Annotated[
        str | None, VectorStoreRecordDataField(is_filterable=True)
    ] = None
    ref_doc_id: Annotated[
        str | None, VectorStoreRecordDataField(is_filterable=True)
    ] = None
