class LegalSyncClientError(Exception):
    """Ошибка обращения к Legal Sync Service.

    Постановка документа на контроль — не часть индексации: к этому моменту
    документ уже проиндексирован в Qdrant. Поэтому такой отказ не отменяет
    ingestion, а логируется и требует ручной постановки на контроль
    (FASTAPI_PATTERNS.md, раздел 9 — деградация при частичном отказе).
    """

    def __init__(self, error_details: str):
        self.error_details = error_details
        super().__init__(self.error_details)

    def __str__(self) -> str:
        return f'Ошибка обращения к Legal Sync Service. Подробности: {self.error_details}'


class LegalSyncRequisitesError(LegalSyncClientError):
    """У документа нет реквизитов, без которых мониторинг невозможен.

    Мониторинг изменений ищет акт в банке правовых актов по номеру и дате
    подписания. Без номера документ поставить на контроль нельзя, и угадывать
    его здесь недопустимо: ошибка привела бы к отслеживанию чужого акта.
    """

    def __init__(self, document_id: str):
        self.document_id = document_id
        super().__init__(document_id)

    def __str__(self) -> str:
        return (
            f'У документа {self.document_id!r} нет номера акта, '
            'постановка на контроль изменений невозможна.'
        )
