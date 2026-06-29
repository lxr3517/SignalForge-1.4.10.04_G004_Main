from fastapi.testclient import TestClient
from app.main import app


def test_home_page_loads():
    client = TestClient(app)
    response = client.get('/')
    assert response.status_code == 200
    assert 'SignalForge' in response.text


def test_health_endpoints_load():
    client = TestClient(app)
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] in {'ok', 'degraded'}

    response = client.get('/health/deep')
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] in {'ok', 'degraded'}
    assert payload['checks']['database']['status'] == 'ok'
