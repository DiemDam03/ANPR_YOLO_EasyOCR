from flask import Flask, render_template
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, 'plates.db')
template_folder = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=template_folder)

@app.route('/')
def index():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT plate_text, timestamp FROM plates ORDER BY id DESC')
    records = cursor.fetchall()
    conn.close()
    return render_template('index.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)