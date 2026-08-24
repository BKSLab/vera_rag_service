from sqlalchemy import Text

from app.db.models.document import Document


def test_document_source_title_has_no_length_limit():
    """`source_title` — полное официальное наименование документа (вид акта,
    дата, номер, название), а не короткое обозначение: у федеральных законов
    оно длиннее 255 символов, поэтому колонка обязана оставаться `Text`
    (миграция `20260825_1200`)."""
    column_type = Document.__table__.c.source_title.type

    assert isinstance(column_type, Text)
    assert getattr(column_type, 'length', None) is None
