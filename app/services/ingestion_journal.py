import asyncio
import hashlib
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from copy import deepcopy
from datetime import UTC, datetime
from time import monotonic
from uuid import uuid4

from app.core.config_logger import logger
from app.core.request_context import get_request_id
from app.exceptions.ingestion import (
    RawTextTooLargeError,
    StaleRevisionError,
    TooManyChunksError,
    TopicsNotAllowedForCategoryError,
)
from app.repositories.ingestion_run import IngestionRunRepository

STAGE_LABELS = {
    'validation': 'Проверка входных данных',
    'lock': 'Ожидание очереди документа',
    'revision': 'Проверка редакции и текущих чанков',
    'preprocess': 'Разбор текста документа',
    'chunking': 'Разбиение и проверка чанков',
    'enrichment': 'Обогащение чанков',
    'embedding': 'Расчёт векторов',
    'upsert': 'Запись новой редакции в Qdrant',
    'integrity': 'Проверка количества и идентификаторов в Qdrant',
    'registry': 'Сохранение реестра документов',
    'retire': 'Снятие старой редакции с актуальности',
    'change_log': 'Сохранение итога обновления статьи',
    'monitoring': 'Постановка на контроль в Legal Sync',
    'cleanup': 'Удаление новых чанков после ошибки проверки',
}
STATUS_LABELS = {
    'running': 'В работе', 'succeeded': 'Завершено', 'warning': 'Требует внимания',
    'failed': 'Ошибка', 'rejected': 'Отклонено', 'interrupted': 'Прервано',
    'stalled': 'Нет сигнала', 'pending': 'Ожидает', 'not_run': 'Не выполнялось',
    'skipped': 'Пропущено',
}
_trace: ContextVar['IngestionTrace | None'] = ContextVar('ingestion_trace', default=None)


def current_trace() -> 'IngestionTrace | None':
    return _trace.get()


class Stage:
    def __init__(self, key: str):
        self.data = {'key': key, 'status': 'pending', 'details': {}}

    def update(self, **values) -> None:
        self.data['details'].update(values)

    def warning(self, message: str) -> None:
        self.data['status'] = 'warning'
        self.data.setdefault('warnings', []).append(message[:2000])

    def skip(self, reason: str) -> None:
        self.data.update(status='skipped', reason=reason)


class IngestionTrace:
    """Журнал не меняет результат записи в Qdrant при отказе своей БД.

    Heartbeat сохраняет в том числе счётчики долгих этапов. Потерянный
    heartbeat означает неизвестный исход, а не доказанное падение процесса.
    """

    def __init__(self, repository: IngestionRunRepository | None, *, raw_text: str, **metadata):
        self.repository = repository
        keys = ['validation', 'lock', 'revision'] if metadata['kind'] == 'document' else ['lock', 'validation', 'revision']
        if metadata['kind'] == 'document':
            keys += ['preprocess']
        keys += ['chunking', 'enrichment', 'embedding', 'upsert', 'integrity']
        keys += ['registry', 'retire', 'monitoring'] if metadata['kind'] == 'document' else ['retire', 'change_log']
        self.steps = {key: Stage(key) for key in keys}
        now = datetime.now(UTC)
        self.values = dict(
            id=str(uuid4()), status='running', current_stage=None, started_at=now,
            updated_at=now, finished_at=None, error=None, request_id=get_request_id(),
            result={'input_characters': len(raw_text), 'input_sha256': hashlib.sha256(raw_text.encode()).hexdigest()},
            **metadata,
        )
        self.lock = asyncio.Lock()
        self.journal_failed = False

    async def persist(self):
        if self.repository is None:
            return
        async with self.lock:
            values = deepcopy(self.values)
            values.update(updated_at=datetime.now(UTC), stages=[deepcopy(step.data) for step in self.steps.values()])
            try:
                async with asyncio.timeout(5):
                    await self.repository.save(values)
            except Exception:
                self.journal_failed = True
                if self.values['status'] == 'succeeded':
                    self.values['status'] = 'warning'
                logger.exception('Не удалось сохранить журнал ingestion %s.', self.values['id'])

    async def heartbeat(self):
        while True:
            await asyncio.sleep(5)
            await self.persist()

    @asynccontextmanager
    async def run(self):
        token = _trace.set(self)
        heartbeat = None
        try:
            await self.persist()
            if self.repository is not None:
                heartbeat = asyncio.create_task(self.heartbeat())
            try:
                yield self
            except BaseException as error:
                if isinstance(error, asyncio.CancelledError):
                    self.values['status'] = 'interrupted'
                elif isinstance(error, ValueError | StaleRevisionError | RawTextTooLargeError
                                | TooManyChunksError | TopicsNotAllowedForCategoryError):
                    self.values['status'] = 'rejected'
                else:
                    self.values['status'] = 'failed'
                self.values['error'] = f'{type(error).__name__}: {error}'[:4000]
                self.values['current_stage'] = next(
                    (key for key, step in self.steps.items() if step.data['status'] in ('failed', 'interrupted')),
                    self.values['current_stage'],
                )
                raise
            else:
                self.values['status'] = 'warning' if self.warnings else 'succeeded'
            finally:
                if heartbeat:
                    heartbeat.cancel()
                    with suppress(asyncio.CancelledError):
                        await heartbeat
                self.values['finished_at'] = datetime.now(UTC)
                for step in self.steps.values():
                    if step.data['status'] == 'pending':
                        step.data['status'] = 'not_run'
                if self.journal_failed:
                    self.values['result']['journal_warning'] = 'Во время обработки журнал был временно недоступен.'
                await self.persist()
        finally:
            _trace.reset(token)

    @property
    def warnings(self) -> list[str]:
        messages = [message for step in self.steps.values() for message in step.data.get('warnings', [])]
        if self.journal_failed:
            messages.append('Во время обработки журнал был временно недоступен.')
        return messages


@asynccontextmanager
async def ingestion_stage(key: str, **details):
    trace = current_trace()
    stage = trace.steps.setdefault(key, Stage(key)) if trace else Stage(key)
    stage.update(**details)
    started = monotonic()
    stage.data.update(status='running', started_at=datetime.now(UTC).isoformat())
    if trace:
        trace.values['current_stage'] = key
        await trace.persist()
    try:
        yield stage
    except BaseException as error:
        stage.data.update(
            status='interrupted' if isinstance(error, asyncio.CancelledError) else 'failed',
            error=f'{type(error).__name__}: {error}'[:4000],
        )
        raise
    finally:
        if stage.data['status'] == 'running':
            stage.data['status'] = 'succeeded'
        stage.data.update(finished_at=datetime.now(UTC).isoformat(), duration_seconds=round(monotonic() - started, 3))
        if trace:
            trace.values['current_stage'] = key
            await trace.persist()


def stage_warning(key: str, message: str):
    trace = current_trace()
    if trace:
        trace.steps.setdefault(key, Stage(key)).warning(message)


def chunk_completed(key: str):
    trace = current_trace()
    if trace and key in trace.steps:
        details = trace.steps[key].data['details']
        details['completed'] = details.get('completed', 0) + 1
