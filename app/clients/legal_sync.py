from datetime import date

import httpx

from app.core.config_logger import logger
from app.core.settings import LegalSyncSettings
from app.exceptions.legal_sync import LegalSyncClientError, LegalSyncRequisitesError
from app.models.metadata import Category


class LegalSyncClient:
    """Клиент Legal Sync Service — сервиса мониторинга изменений НПА.

    Через него база знаний ставит документ на контроль: с этого момента
    Legal Sync Service ежедневно проверяет, не вышли ли акты, меняющие этот
    документ, и в дату вступления изменений в силу возвращает новую редакцию
    изменённых статей через `PUT /document/{id}/sections/{number}`.

    Реквизиты акта передаются структурно. Выводить их из наименования нельзя:
    ошибка привела бы к контролю за чужим актом, а заметить такую подмену
    практически невозможно.
    """

    TRACKED_DOCUMENTS_PATH = '/api/v1/tracked-documents'

    def __init__(self, httpx_client: httpx.AsyncClient, settings: LegalSyncSettings):
        self.httpx_client = httpx_client
        self.settings = settings

    async def register_tracked_document(
        self,
        document_id: str,
        category: Category,
        act_type: str | None,
        act_number: str | None,
        act_date: date,
        act_title: str,
        source_title: str,
        audience: str,
        topics: list[str],
        ingestion_id: str | None = None,
    ) -> dict:
        """Ставит документ на контроль изменений.

        Args:
            document_id: Идентификатор документа в базе знаний.
            category: Категория источника.
            act_type: Вид акта.
            act_number: Номер акта.
            act_date: Дата подписания акта.
            act_title: Наименование акта без реквизитов.
            source_title: Полное наименование документа для ссылки на источник.
            audience: Целевая аудитория.
            topics: Темы документа.

        Returns:
            Созданная запись реестра Legal Sync Service.

        Raises:
            LegalSyncRequisitesError: У документа нет реквизитов, нужных для мониторинга.
            LegalSyncClientError: Legal Sync Service недоступен или отклонил запрос.
        """
        if self.settings.legal_sync_api_key is None:
            raise LegalSyncClientError('LEGAL_SYNC_API_KEY не задан.')
        if not act_number:
            raise LegalSyncRequisitesError(document_id)

        payload = {
            'document_id': document_id,
            'short_name': act_type or act_title,
            'full_title': act_title,
            'category': category,
            'audience': audience,
            'topics': topics,
            'source_title': source_title,
            'publication_block': self.settings.publication_block_for(category),
            'document_number': act_number,
            'adoption_date': act_date.isoformat(),
        }
        if ingestion_id:
            payload['rag_ingestion_id'] = ingestion_id
        url = f'{self.settings.legal_sync_base_url}{self.TRACKED_DOCUMENTS_PATH}'
        headers = {'X-API-Key': self.settings.legal_sync_api_key.get_secret_value()}
        if ingestion_id:
            headers['X-Request-ID'] = ingestion_id

        logger.info('🔍 Постановка документа на контроль изменений. document_id=%s.', document_id)
        try:
            response = await self.httpx_client.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.settings.legal_sync_timeout_seconds,
            )
        except (httpx.TimeoutException, httpx.TransportError) as error:
            raise LegalSyncClientError(
                f'Legal Sync Service недоступен: {type(error).__name__}: {error}'
            ) from error

        if response.status_code == httpx.codes.CONFLICT:
            # Документ уже на контроле — повторный ingest не должен выглядеть
            # как отказ: контроль уже обеспечен, делать нечего.
            logger.info('✅ Документ уже стоит на контроле. document_id=%s.', document_id)
            return {'document_id': document_id, 'already_tracked': True}
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise LegalSyncClientError(
                f'Legal Sync Service отклонил постановку на контроль: '
                f'HTTP {response.status_code}. {response.text[:500]}'
            )

        logger.info('✅ Документ поставлен на контроль изменений. document_id=%s.', document_id)
        return response.json()
