from flask import Flask, render_template, Response
from pyngrok import ngrok
import sqlite3
import os
from anpr import generate_frames, init_database

os.system("taskkill /f /im ngrok.exe >nul 2>&1")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, 'plates.db')
template_folder = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=template_folder)

def initialize_app_database():
    conn, cursor = init_database(db_path)
    conn.close()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT plate_text, timestamp FROM plates ORDER BY id DESC')
        records = cursor.fetchall()
        conn.close()
        return render_template('index.html', records=records)
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
        initialize_app_database()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT plate_text, timestamp FROM plates ORDER BY id DESC')
        records = cursor.fetchall()
        conn.close()
        return render_template('index.html', records=records)

if __name__ == '__main__':
    
    public_url = ngrok.connect(5000)
    print(f' * Ngrok tunnel: {public_url}')
    app.run(debug=True, use_reloader=False)

