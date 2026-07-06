class QdrantCollectionSchemaError(Exception):
    """Существующая коллекция Qdrant несовместима с ожидаемой схемой сервиса."""

    def __init__(self, collection_name: str, problems: list[str]):
        self.collection_name = collection_name
        self.problems = problems
        super().__init__(collection_name, problems)

    def __str__(self) -> str:
        return (
            f'Коллекция Qdrant {self.collection_name!r} несовместима с ожидаемой схемой: '
            + '; '.join(self.problems)
        )
