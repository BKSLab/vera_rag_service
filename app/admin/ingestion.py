from datetime import UTC, datetime
from urllib.parse import urlencode, urlsplit

from fastapi.encoders import jsonable_encoder
from sqladmin import BaseView, expose
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.settings import get_settings
from app.db.models.ingestion_run import IngestionRun
from app.services.ingestion_history import STALE_AFTER, run_view
from app.services.ingestion_journal import STAGE_LABELS, STATUS_LABELS


def external_admin_url(base: str | None, path: str, **query) -> str | None:
    if not base or urlsplit(str(base)).scheme not in ('http', 'https'):
        return None
    return f'{str(base).rstrip("/")}{path}?{urlencode(query)}'


class IngestionLogView(BaseView):
    name = 'Обновления коллекции'
    icon = 'fa-solid fa-route'

    @expose('/ingestion-log', methods=['GET'])
    async def ingestion_log(self, request: Request):
        try:
            page = int(request.query_params.get('page', '1'))
            if not 1 <= page <= 1_000_000:
                raise ValueError
        except ValueError as error:
            raise HTTPException(400, 'Некорректный номер страницы.') from error
        filters = {key: request.query_params.get(key, '').strip() for key in ('document_id', 'status', 'kind', 'request_id', 'change_id')}
        conditions = []
        for key in ('document_id', 'request_id'):
            if filters[key]:
                conditions.append(getattr(IngestionRun, key) == filters[key])
        if filters['change_id']:
            try:
                change_id = int(filters['change_id'])
                if not 0 < change_id < 2**31:
                    raise ValueError
            except ValueError as error:
                raise HTTPException(400, 'Некорректный номер изменения.') from error
            conditions.append(IngestionRun.change_id == change_id)
        if filters['kind']:
            if filters['kind'] not in ('document', 'section'):
                raise HTTPException(400, 'Неизвестный тип операции.')
            conditions.append(IngestionRun.kind == filters['kind'])
        if filters['status']:
            if filters['status'] not in ('running', 'succeeded', 'warning', 'failed', 'rejected', 'interrupted', 'stalled'):
                raise HTTPException(400, 'Неизвестный статус.')
            if filters['status'] in ('running', 'stalled'):
                conditions += [IngestionRun.status == 'running']
                cutoff = datetime.now(UTC) - STALE_AFTER
                conditions.append(IngestionRun.updated_at < cutoff if filters['status'] == 'stalled' else IngestionRun.updated_at >= cutoff)
            else:
                conditions.append(IngestionRun.status == filters['status'])
        error = None
        try:
            async with self._admin_ref.session_maker() as session:
                total = await session.scalar(select(func.count()).select_from(IngestionRun).where(*conditions))
                runs = (await session.scalars(select(IngestionRun).where(*conditions).order_by(
                    IngestionRun.started_at.desc(), IngestionRun.id.desc(),
                ).offset((page - 1) * 25).limit(25))).all()
                runs = [run_view(run) for run in runs]
        except (SQLAlchemyError, OSError):
            error, runs, total = 'Журнал временно недоступен.', [], 0
        return await self.templates.TemplateResponse(request, 'ingestion_log.html', {
            'title': self.name, 'runs': runs, 'total': total, 'page': page, 'filters': filters,
            'status_labels': STATUS_LABELS, 'stage_labels': STAGE_LABELS, 'error': error,
            'previous_url': str(request.url.include_query_params(page=page - 1)),
            'next_url': str(request.url.include_query_params(page=page + 1)),
        }, status_code=503 if error else 200)


class IngestionRunView(BaseView):
    name = 'Этапы обновления'

    def is_visible(self, request: Request) -> bool:
        return False

    @expose('/ingestion-run/{run_id}', methods=['GET'])
    async def ingestion_run(self, request: Request):
        async with self._admin_ref.session_maker() as session:
            run = await session.get(IngestionRun, request.path_params['run_id'])
            if run is None:
                raise HTTPException(404, 'Операция не найдена.')
            values = run_view(run)
        if request.query_params.get('format') == 'json':
            return JSONResponse(jsonable_encoder(values), headers={'Cache-Control': 'no-store'})
        response = await self.templates.TemplateResponse(request, 'ingestion_run.html', {
            'title': self.name, 'run': values, 'status_labels': STATUS_LABELS, 'stage_labels': STAGE_LABELS,
            'sync_url': external_admin_url(get_settings().legal_sync.legal_sync_admin_base_url,
                                          '/admin/document-trace', document_id=run.document_id),
            'history_url': '/admin/ingestion-log?' + urlencode({'document_id': run.document_id}),
            'chunks_url': '/admin/document-chunks?' + urlencode({'document_id': run.document_id, 'version': run.version}),
        })
        response.headers['Cache-Control'] = 'no-store'
        return response
