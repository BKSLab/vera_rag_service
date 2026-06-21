from starlette.requests import Request

from app.admin.csrf import get_or_create_csrf_token, verify_csrf_token


def make_request(session: dict | None = None) -> Request:
    scope = {'type': 'http', 'method': 'GET', 'path': '/admin/document-upload', 'headers': [], 'session': session or {}}
    return Request(scope)


def test_get_or_create_csrf_token_creates_and_stores_token_in_session():
    request = make_request()

    token = get_or_create_csrf_token(request)

    assert token
    assert request.session['csrf_token'] == token


def test_get_or_create_csrf_token_reuses_existing_session_token():
    request = make_request(session={'csrf_token': 'existing-token'})

    token = get_or_create_csrf_token(request)

    assert token == 'existing-token'


def test_verify_csrf_token_succeeds_for_matching_token():
    request = make_request(session={'csrf_token': 'matching-token'})

    assert verify_csrf_token(request, 'matching-token') is True


def test_verify_csrf_token_fails_for_mismatched_token():
    request = make_request(session={'csrf_token': 'expected-token'})

    assert verify_csrf_token(request, 'wrong-token') is False


def test_verify_csrf_token_fails_when_no_token_in_session():
    request = make_request()

    assert verify_csrf_token(request, 'anything') is False


def test_verify_csrf_token_fails_when_submitted_token_is_none():
    request = make_request(session={'csrf_token': 'expected-token'})

    assert verify_csrf_token(request, None) is False
