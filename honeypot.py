"""
Victor Honeypot Server
Simulates a web application and logs all traffic to SQLite for analysis.
Run with: python honeypot.py
"""

from flask import Flask, request, jsonify
from datetime import datetime
import uuid
import os
import atexit
from database import TrafficDatabase

app = Flask(__name__)
db = TrafficDatabase()
atexit.register(db.close)  # ensure SQLite connection is released on exit

# Session tracking
CURRENT_SESSION = str(uuid.uuid4())

@app.before_request
def log_request():
    """Log every incoming request to SQLite"""
    timestamp = datetime.utcnow().isoformat()
    ip = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    referer = request.headers.get('Referer', 'none')
    accept_lang = request.headers.get('Accept-Language', 'none')
    path = request.path
    method = request.method
    
    # Store request data for after_request
    request.log_data = {
        'timestamp': timestamp,
        'ip': ip,
        'user_agent': user_agent,
        'referer': referer,
        'accept_lang': accept_lang,
        'path': path,
        'method': method,
        'session_id': CURRENT_SESSION
    }


@app.after_request
def save_request(response):
    """Save request to database after response"""
    log_data = getattr(request, 'log_data', {})
    if log_data:
        db.log_request(
            timestamp=log_data['timestamp'],
            ip=log_data['ip'],
            user_agent=log_data['user_agent'],
            referer=log_data['referer'],
            accept_lang=log_data['accept_lang'],
            path=log_data['path'],
            method=log_data['method'],
            status_code=response.status_code,
            session_id=log_data['session_id']
        )
    
    return response


@app.route('/', methods=['GET'])
def home():
    """Homepage"""
    return jsonify({
        'status': 'ok',
        'message': 'Victor Honeypot Server',
        'timestamp': datetime.utcnow().isoformat()
    }), 200


@app.route('/articles', methods=['GET'])
def articles():
    """Articles page"""
    return jsonify({
        'status': 'ok',
        'page': 'articles',
        'articles': [
            {'id': 1, 'title': 'Article 1'},
            {'id': 2, 'title': 'Article 2'}
        ],
        'timestamp': datetime.utcnow().isoformat()
    }), 200


@app.route('/about', methods=['GET'])
def about():
    """About page"""
    return jsonify({
        'status': 'ok',
        'page': 'about',
        'content': 'This is the about page',
        'timestamp': datetime.utcnow().isoformat()
    }), 200


@app.route('/secret-data', methods=['GET'])
def secret():
    """
    Honeypot endpoint - legit users won't know about this.
    Bots will find it through aggressive scanning.
    """
    return jsonify({
        'status': 'ok',
        'page': 'secret-data',
        'warning': 'This endpoint should not be public',
        'timestamp': datetime.utcnow().isoformat()
    }), 200


@app.route('/api/status', methods=['GET'])
def api_status():
    """API endpoint for checking server status"""
    total = db.get_record_count()
    unique_ips = db.get_unique_ips()
    
    return jsonify({
        'status': 'running',
        'total_requests': total,
        'unique_ips': unique_ips,
        'current_session': CURRENT_SESSION,
        'timestamp': datetime.utcnow().isoformat()
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Not found',
        'path': request.path
    }), 404


@app.errorhandler(500)
def server_error(error):
    """Handle server errors"""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  VICTOR HONEYPOT SERVER")
    print("=" * 60)
    print(f"Starting honeypot server on http://127.0.0.1:5000")
    print(f"Session ID: {CURRENT_SESSION}")
    print()
    print("Available endpoints:")
    print("  GET /               - Homepage")
    print("  GET /articles       - Articles page")
    print("  GET /about          - About page")
    print("  GET /secret-data    - Honeypot endpoint")
    print("  GET /api/status     - Server status")
    print()
    print("Traffic is being logged to: data/victor_traffic.db")
    print()
    print("To start traffic simulation:")
    print("  1. Keep this server running (in background or separate terminal)")
    print("  2. Run: python simulate_traffic.py")
    print("=" * 60)
    
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)
