import pytest
from unittest.mock import patch
from honeypot import app, db

@pytest.fixture
def client(tmp_path):
    """Provides a Flask test client with a temporary SQLite database."""
    with patch("honeypot.db.get_record_count", return_value=100), \
         patch("honeypot.db.get_unique_ips", return_value=10), \
         patch("honeypot.db.get_blocked_count", return_value=5), \
         patch("honeypot.db.log_request"):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

def test_homepage(client):
    """Test the homepage returns a 200 OK status."""
    response = client.get('/')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'ok'
    assert 'Victor Honeypot' in data['message']

def test_secret_data_trap(client):
    """Test the honeypot trap endpoint returns 200 (unless blocked)."""
    # Since we are using a test client without the ML model loaded, 
    # it won't block the request, returning 200 OK.
    response = client.get('/secret-data')
    assert response.status_code == 200
    data = response.get_json()
    assert data['page'] == 'secret-data'
    assert 'warning' in data

def test_api_status(client):
    """Test the server status API endpoint."""
    response = client.get('/api/status')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'running'
    assert 'total_requests' in data
