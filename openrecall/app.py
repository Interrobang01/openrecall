import glob
import os
import subprocess
import sys
from threading import Thread

import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from jinja2 import BaseLoader

from openrecall.config import appdata_folder, screenshots_path
from openrecall.database import (
    create_db,
    get_all_entries,
    get_timestamps,
    get_timeline_entries,
)
from openrecall.nlp import cosine_similarity, get_embedding
from openrecall.screenshot import capture_state, record_screenshots_thread
from openrecall.utils import human_readable_time, timestamp_to_human_readable

app = Flask(__name__)

app.jinja_env.filters["human_readable_time"] = human_readable_time
app.jinja_env.filters["timestamp_to_human_readable"] = timestamp_to_human_readable
app.jinja_env.globals["screenshots_path"] = screenshots_path

base_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OpenRecall</title>
  <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.3.0/font/bootstrap-icons.css">
  <style>
    .slider-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
    }
    .slider { width: 80%; }
    .slider-value { margin-top: 10px; font-size: 1.2em; }
    .image-container { margin-top: 20px; text-align: center; }
    .image-container img { max-width: 100%; height: auto; }

    /* Capture indicator dot */
    #capture-dot {
      display: inline-block;
      width: 10px; height: 10px;
      border-radius: 50%;
      background: #6c757d;
      margin-right: 6px;
      transition: background 0.3s;
      flex-shrink: 0;
    }
    #capture-dot.active {
      background: #28a745;
      animation: blink 0.6s ease-out;
    }
    @keyframes blink {
      0%   { box-shadow: 0 0 0 0 rgba(40,167,69,0.7); }
      70%  { box-shadow: 0 0 0 8px rgba(40,167,69,0); }
      100% { box-shadow: 0 0 0 0 rgba(40,167,69,0); }
    }

    /* Search card text snippet */
    .ocr-snippet {
      font-size: 0.72rem;
      color: #6c757d;
      max-height: 3.5em;
      overflow: hidden;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
    }
    .similarity-badge {
      font-size: 0.75rem;
    }
    .card-meta { font-size: 0.78rem; color: #555; }

    /* Metrics table */
    .metrics-table td, .metrics-table th { font-size: 0.82rem; }
  </style>
</head>
<body>
<nav class="navbar navbar-light bg-light">
  <div class="container-fluid d-flex align-items-center flex-wrap" style="gap: 8px;">

    <!-- Capture indicator -->
    <span id="capture-dot" title="Last capture time"></span>
    <span id="capture-info" class="text-muted mr-2" style="font-size:0.78rem; white-space:nowrap;">idle</span>

    <!-- Search bar -->
    <form class="form-inline flex-grow-1 d-flex" action="/search" method="get" style="min-width:200px;">
      <input class="form-control flex-grow-1 mr-1" type="search" name="q" placeholder="Search" aria-label="Search"
             value="{{ request.args.get('q', '') }}">
      <button class="btn btn-outline-secondary" type="submit"><i class="bi bi-search"></i></button>
    </form>

    <!-- Storage stats -->
    <span id="storage-badge" class="badge badge-light border text-muted" style="font-size:0.75rem; white-space:nowrap;" title="Storage usage"></span>

    <!-- Open folder button -->
    <button class="btn btn-sm btn-outline-secondary" id="open-folder-btn"
            title="{{ screenshots_path }}">
      <i class="bi bi-folder2-open"></i>
    </button>

    <!-- Metrics link -->
    <a href="/metrics" class="btn btn-sm btn-outline-secondary" title="Performance metrics">
      <i class="bi bi-speedometer2"></i>
    </a>

  </div>
</nav>

{% block content %}{% endblock %}

<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.3/dist/umd/popper.min.js"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>

<script>
// ---- Capture status polling ----
let lastCaptureTs = 0;
function pollStatus() {
  fetch('/api/status').then(r => r.json()).then(data => {
    const dot = document.getElementById('capture-dot');
    const info = document.getElementById('capture-info');
    if (data.last_capture_ts && data.last_capture_ts !== lastCaptureTs) {
      lastCaptureTs = data.last_capture_ts;
      dot.classList.add('active');
      setTimeout(() => dot.classList.remove('active'), 800);
    }
    if (data.last_capture_ts) {
      const ago = Math.round((Date.now()/1000) - data.last_capture_ts);
      info.textContent = ago < 5 ? 'just now' : ago + 's ago';
      dot.title = 'Captures this session: ' + data.captures_this_session +
                  ' | MSSIM: ' + (data.last_mssim !== null ? data.last_mssim : '—');
    }
  }).catch(() => {});
}
setInterval(pollStatus, 3000);
pollStatus();

// ---- Storage stats ----
function loadStats() {
  fetch('/api/stats').then(r => r.json()).then(data => {
    function fmt(b) {
      if (b >= 1073741824) return (b/1073741824).toFixed(1) + ' GB';
      if (b >= 1048576) return (b/1048576).toFixed(1) + ' MB';
      return (b/1024).toFixed(0) + ' KB';
    }
    const badge = document.getElementById('storage-badge');
    badge.textContent = fmt(data.screenshots_size_bytes + data.db_size_bytes) +
                        ' · ' + data.entry_count + ' entries';
    badge.title = 'Screenshots: ' + fmt(data.screenshots_size_bytes) +
                  ' | DB: ' + fmt(data.db_size_bytes);
    if (data.screenshots_size_bytes + data.db_size_bytes > 5368709120) {
      badge.classList.remove('badge-light');
      badge.classList.add('badge-warning');
    }
  }).catch(() => {});
}
loadStats();

// ---- Open folder ----
document.getElementById('open-folder-btn').addEventListener('click', function() {
  fetch('/open-folder', {method: 'POST'}).then(r => r.json()).then(d => {
    if (!d.ok) alert('Could not open folder: ' + d.error);
  }).catch(() => {});
});
</script>
</body>
</html>
"""


class StringLoader(BaseLoader):
    def get_source(self, environment, template):
        if template == "base_template":
            return base_template, None, lambda: True
        return None, None, None


app.jinja_env.loader = StringLoader()


@app.route("/")
def timeline():
    timeline_entries = [
        {
            "timestamp": entry.timestamp,
            "monitor_id": entry.monitor_id,
            "image_filename": entry.image_filename,
        }
        for entry in get_timeline_entries()
    ]
    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
{% if timeline_entries|length > 0 %}
<div class="container mt-3">

  <!-- Date range filter -->
  <div class="card mb-3">
    <div class="card-body py-2">
      <div class="form-row align-items-center">
        <div class="col-auto">
          <label class="col-form-label col-form-label-sm font-weight-bold">From</label>
        </div>
        <div class="col-auto">
          <input type="datetime-local" id="dateFrom" class="form-control form-control-sm">
        </div>
        <div class="col-auto">
          <label class="col-form-label col-form-label-sm font-weight-bold">To</label>
        </div>
        <div class="col-auto">
          <input type="datetime-local" id="dateTo" class="form-control form-control-sm">
        </div>
        <div class="col-auto">
          <button class="btn btn-sm btn-outline-secondary" id="resetRange">Reset</button>
        </div>
        <div class="col-auto text-muted small" id="rangeInfo"></div>
      </div>
    </div>
  </div>

  <div class="slider-container">
    <input type="range" class="slider custom-range" id="discreteSlider" min="0" max="0" step="1" value="0">
    <div class="slider-value" id="sliderValue"></div>
  </div>
  <div class="image-container">
    <img id="timestampImage" src="" alt="Screenshot" style="display:none;">
  </div>
</div>

<script>
const allEntries = {{ timeline_entries|tojson }};  // descending order by timestamp, then monitor
let filtered = allEntries.slice();

function toLocalDatetimeInput(ts) {
  const d = new Date(ts * 1000);
  // format as YYYY-MM-DDTHH:MM for datetime-local input
  const pad = n => String(n).padStart(2, '0');
  return d.getFullYear() + '-' + pad(d.getMonth()+1) + '-' + pad(d.getDate()) +
         'T' + pad(d.getHours()) + ':' + pad(d.getMinutes());
}

// Set initial date inputs to span of data
const dateFrom = document.getElementById('dateFrom');
const dateTo   = document.getElementById('dateTo');
dateFrom.value = toLocalDatetimeInput(allEntries[allEntries.length - 1].timestamp);
dateTo.value   = toLocalDatetimeInput(allEntries[0].timestamp);

const slider       = document.getElementById('discreteSlider');
const sliderValue  = document.getElementById('sliderValue');
const img          = document.getElementById('timestampImage');
const rangeInfo    = document.getElementById('rangeInfo');

function applyFilter() {
  const from = dateFrom.value ? new Date(dateFrom.value).getTime() / 1000 : 0;
  const to   = dateTo.value   ? new Date(dateTo.value).getTime()   / 1000 : Infinity;
  filtered = allEntries.filter(item => item.timestamp >= from && item.timestamp <= to);
  slider.max = Math.max(0, filtered.length - 1);
  slider.value = slider.max;
  rangeInfo.textContent = filtered.length + ' of ' + allEntries.length + ' screenshots';
  updateDisplay();
}

function updateDisplay() {
  if (filtered.length === 0) {
    sliderValue.textContent = 'No screenshots in range';
    img.style.display = 'none';
    return;
  }
  // slider goes 0 (oldest) → max (newest); filtered is desc, so index = max - slider.value
  const idx = filtered.length - 1 - parseInt(slider.value);
  const item = filtered[idx];
  sliderValue.textContent = new Date(item.timestamp * 1000).toLocaleString() + ' · monitor ' + item.monitor_id;
  img.src = '/static/' + item.image_filename;
  img.style.display = '';
}

slider.addEventListener('input', updateDisplay);
dateFrom.addEventListener('change', applyFilter);
dateTo.addEventListener('change', applyFilter);
document.getElementById('resetRange').addEventListener('click', function() {
  dateFrom.value = toLocalDatetimeInput(allEntries[allEntries.length - 1].timestamp);
  dateTo.value   = toLocalDatetimeInput(allEntries[0].timestamp);
  applyFilter();
});

applyFilter();
</script>
{% else %}
<div class="container mt-3">
  <div class="alert alert-info">Nothing recorded yet, wait a few seconds.</div>
</div>
{% endif %}
{% endblock %}
""",
        timeline_entries=timeline_entries,
    )


@app.route("/search")
def search():
    q = (request.args.get("q") or "").strip()
    entries = get_all_entries()
    embeddings = [np.asarray(entry.embedding, dtype=np.float32) for entry in entries]
    query_embedding = get_embedding(q)
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    indices = np.argsort(similarities)[::-1] if similarities else []
    results = [
        {
            "timestamp": entries[i].timestamp,
        "monitor_id": entries[i].monitor_id,
        "image_filename": entries[i].image_filename or f"{entries[i].timestamp}.webp",
            "app": entries[i].app or "",
            "title": entries[i].title or "",
            "text": entries[i].text or "",
            "similarity": round(similarities[i] * 100, 1),
        }
        for i in indices
    ]

    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
<div class="container mt-3">
  <div class="mb-3">
    <a href="/" class="btn btn-sm btn-outline-secondary"><i class="bi bi-house"></i> Timeline</a>
  </div>
  {% if not results %}
    <div class="alert alert-info">No results.</div>
  {% else %}
  <p class="text-muted small mb-2">{{ results|length }} results</p>
  <div class="row">
    {% for r in results %}
    <div class="col-md-3 mb-4">
      <div class="card h-100 shadow-sm">
        <a href="#" data-toggle="modal" data-target="#modal-{{ loop.index0 }}" class="d-block overflow-hidden"
           style="max-height:160px;">
          <img src="/static/{{ r.image_filename }}" alt="screenshot" class="card-img-top"
               style="object-fit:cover; height:160px;">
        </a>
        <div class="card-body p-2">
          <div class="d-flex justify-content-between align-items-start mb-1">
            <span class="badge badge-primary similarity-badge">{{ r.similarity }}%</span>
            <span class="card-meta">{{ r.timestamp | timestamp_to_human_readable }}</span>
          </div>
          <div class="card-meta">Monitor {{ r.monitor_id }}</div>
          {% if r.app %}
          <div class="card-meta text-truncate" title="{{ r.title }}">
            <i class="bi bi-app-indicator"></i> {{ r.app }}{% if r.title %} — {{ r.title }}{% endif %}
          </div>
          {% endif %}
          {% if r.text %}
          <div class="mt-1" style="max-height:6rem; overflow-y:auto; font-size:0.72rem; color:#6c757d; white-space:pre-wrap; border-top:1px solid #eee; padding-top:4px;">{{ r.text }}</div>
          {% endif %}
        </div>
      </div>
    </div>

    <div class="modal fade" id="modal-{{ loop.index0 }}" tabindex="-1" role="dialog" aria-hidden="true">
      <div class="modal-dialog" role="document" style="max-width:95vw; margin:1vh auto;">
        <div class="modal-content">
          <div class="modal-header py-2">
            <div>
              <span class="badge badge-primary">{{ r.similarity }}% match</span>
              <span class="ml-2 text-muted small">{{ r.timestamp | timestamp_to_human_readable }}</span>
              {% if r.app %}<span class="ml-2 text-muted small">{{ r.app }}{% if r.title %} — {{ r.title }}{% endif %}</span>{% endif %}
            </div>
            <button type="button" class="close" data-dismiss="modal"><span>&times;</span></button>
          </div>
          <div class="modal-body p-0" style="background:#000;">
            <img src="/static/{{ r.image_filename }}" alt="screenshot"
                 style="width:100%; max-height:80vh; object-fit:contain; display:block;">
          </div>
          {% if r.text %}
          <div class="modal-footer py-2 d-block">
            <p class="mb-1 small text-muted font-weight-bold">OCR text:</p>
            <pre class="small mb-0" style="max-height:10rem; overflow-y:auto; white-space:pre-wrap; font-size:0.75rem;">{{ r.text }}</pre>
          </div>
          {% endif %}
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
  {% endif %}
</div>
{% endblock %}
""",
        results=results,
    )


@app.route("/static/<filename>")
def serve_image(filename):
    return send_from_directory(screenshots_path, filename)


@app.route("/api/stats")
def api_stats():
    """Returns storage and database statistics as JSON."""
    db_path = os.path.join(appdata_folder, "recall.db")
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    webp_files = glob.glob(os.path.join(screenshots_path, "*.webp"))
    screenshots_size = sum(os.path.getsize(f) for f in webp_files)
    timestamps = get_timestamps()
    return jsonify({
        "db_size_bytes": db_size,
        "screenshots_size_bytes": screenshots_size,
        "entry_count": len(timestamps),
        "screenshot_count": len(webp_files),
        "oldest_timestamp": timestamps[-1] if timestamps else None,
        "newest_timestamp": timestamps[0] if timestamps else None,
    })


@app.route("/api/status")
def api_status():
    """Returns live capture loop state as JSON."""
    return jsonify({
        "last_capture_ts": capture_state["last_capture_ts"],
        "captures_this_session": capture_state["captures_this_session"],
        "last_mssim": capture_state["last_mssim"],
        "recent_timings": list(capture_state["recent_timings"]),
    })


@app.route("/open-folder", methods=["POST"])
def open_folder():
    """Opens the screenshots folder in the system file manager."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", screenshots_path])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", screenshots_path])
        else:
            subprocess.Popen(["xdg-open", screenshots_path])
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)})


@app.route("/metrics")
def metrics():
    """Shows a live performance metrics dashboard."""
    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
<div class="container mt-3">
  <h5>Performance Metrics</h5>
  <p class="text-muted small">
    Timing for each capture that resulted in a saved screenshot.
    Updates automatically every 5 seconds.
  </p>

  <div class="row mb-3">
    <div class="col-md-4">
      <div class="card text-center">
        <div class="card-body py-2">
          <div class="h4 mb-0" id="m-captures">—</div>
          <div class="small text-muted">captures this session</div>
        </div>
      </div>
    </div>
    <div class="col-md-4">
      <div class="card text-center">
        <div class="card-body py-2">
          <div class="h4 mb-0" id="m-mssim">—</div>
          <div class="small text-muted">last MSSIM score</div>
        </div>
      </div>
    </div>
    <div class="col-md-4">
      <div class="card text-center">
        <div class="card-body py-2">
          <div class="h4 mb-0" id="m-last-ts">—</div>
          <div class="small text-muted">last capture</div>
        </div>
      </div>
    </div>
  </div>

  <h6 class="mt-3">Recent capture timings</h6>
  <div class="table-responsive">
    <table class="table table-sm table-hover metrics-table" id="timings-table">
      <thead class="thead-light">
        <tr>
          <th>Time</th>
          <th>MSSIM check</th>
          <th>OCR</th>
          <th>Embedding</th>
          <th>DB insert</th>
          <th>Total</th>
          <th>Text?</th>
        </tr>
      </thead>
      <tbody id="timings-body">
        <tr><td colspan="7" class="text-muted">Waiting for captures…</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
function loadMetrics() {
  fetch('/api/status').then(r => r.json()).then(data => {
    document.getElementById('m-captures').textContent = data.captures_this_session;
    document.getElementById('m-mssim').textContent    = data.last_mssim !== null ? data.last_mssim : '—';
    if (data.last_capture_ts) {
      document.getElementById('m-last-ts').textContent = new Date(data.last_capture_ts * 1000).toLocaleString();
    }

    const tbody = document.getElementById('timings-body');
    const timings = data.recent_timings || [];
    if (timings.length === 0) return;
    tbody.innerHTML = '';
    // most recent first
    [...timings].reverse().forEach(t => {
      const tr = document.createElement('tr');
      tr.innerHTML =
        '<td>' + new Date(t.timestamp * 1000).toLocaleTimeString() + '</td>' +
        '<td>' + t.mssim_ms + ' ms</td>' +
        '<td>' + t.ocr_ms + ' ms</td>' +
        '<td>' + t.embedding_ms + ' ms</td>' +
        '<td>' + t.db_ms + ' ms</td>' +
        '<td><strong>' + t.total_ms + ' ms</strong></td>' +
        '<td>' + (t.had_text ? '✓' : '—') + '</td>';
      tbody.appendChild(tr);
    });
  }).catch(() => {});
}
loadMetrics();
setInterval(loadMetrics, 5000);
</script>
{% endblock %}
""",
    )


if __name__ == "__main__":
    create_db()

    print(f"Appdata folder: {appdata_folder}")

    # Start the thread to record screenshots
    t = Thread(target=record_screenshots_thread)
    t.start()

    app.run(port=8082)
