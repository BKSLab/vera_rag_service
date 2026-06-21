from types import SimpleNamespace

from app.admin.views import _fmt_json


def test_fmt_json_escapes_html_in_value():
    """Регрессия на stored XSS (ADM-1/SEC-3) — содержимое из реальных
    документов/LLM-вывода может содержать `<script>` и т.п., и не должно
    попадать в HTML-страницу админки неэкранированным."""
    model = SimpleNamespace(final_response=[{'text': '<script>alert(1)</script>'}])

    rendered = _fmt_json(model, 'final_response')

    assert '<script>' not in str(rendered)
    assert '&lt;script&gt;' in str(rendered)
