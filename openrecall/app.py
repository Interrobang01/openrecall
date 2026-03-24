import glob
import datetime
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from threading import Thread
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from jinja2 import BaseLoader
from PIL import Image

from openrecall.config import (
  OPENRECALL_AV1_CRF,
  OPENRECALL_AV1_PLAYBACK_FPS,
  OPENRECALL_AV1_PRESET,
  OPENRECALL_AV1_SVTAV1_PARAMS,
  OPENRECALL_AV1_THREADS,
    OPENRECALL_FFMPEG_BIN,
    OPENRECALL_STORAGE_BACKEND,
    RUNTIME_CONFIG_KEYS,
  args,
    appdata_folder,
    check_ffmpeg_av1_capabilities,
    config_file_path,
    get_runtime_config_values,
    media_path,
    pending_frames_path,
    segments_path,
    thumbnails_path,
    write_runtime_config_file,
)
from openrecall.database import (
    create_db,
    delete_entries_by_segment_filenames,
    get_all_entries,
    get_media_entries_for_segments,
    get_pending_segment_recovery_entries,
  get_segment_frame_index,
    get_timestamps,
    get_timeline_entries,
)
from openrecall.nlp import (
  EMBEDDING_DIM,
  cosine_similarity,
  dot_product,
  euclidean_distance,
  get_embedding,
  manhattan_distance,
)
from openrecall.screenshot import (
  capture_state,
  clear_capture_pause,
  is_capture_paused,
  record_screenshots_thread,
  request_capture_stop,
  set_capture_pause_for_seconds,
  set_capture_pause_forever,
)
from openrecall.hotkeys import start_hotkey_listener
from openrecall.tray import start_linux_tray
from openrecall.utils import human_readable_time, timestamp_to_human_readable

app = Flask(__name__)

app.jinja_env.filters["human_readable_time"] = human_readable_time
app.jinja_env.filters["timestamp_to_human_readable"] = timestamp_to_human_readable
app.jinja_env.globals["storage_media_path"] = media_path

frame_cache_path = os.path.join(media_path, "frame_cache")
os.makedirs(frame_cache_path, exist_ok=True)
quarantine_segments_path = os.path.join(media_path, "quarantine_segments")

STARTUP_SEGMENT_RECOVERY_LIMIT = 8
startup_recovery_state: Dict[str, object] = {
  "ran": False,
  "checked_segments": 0,
  "quarantined_segments": [],
  "entries_removed": 0,
  "thumbs_removed": 0,
  "cache_removed": 0,
  "pending_recovered_segments": 0,
  "pending_recovered_frames": 0,
  "pending_recovery_failed_segments": [],
  "pending_orphan_frames": 0,
  "pending_orphan_frames_purged": 0,
}

SEARCH_METRICS = {
  "cosine": "Cosine",
  "dot": "Dot product",
  "euclidean": "Euclidean distance",
  "manhattan": "Manhattan distance",
}

PROXIMITY_MIN_SECONDS = 1
PROXIMITY_MAX_SECONDS_FALLBACK = 31536000
PROXIMITY_SLIDER_STEPS = 1000


def _json_safe(value):
  if isinstance(value, dict):
    return {key: _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, np.ndarray):
    return value.tolist()
  return value


def _embedding_magnitude(embedding: np.ndarray) -> float:
  """Returns L2 norm of an embedding, or 0 for empty/invalid vectors."""
  vector = np.asarray(embedding, dtype=np.float32)
  if vector.size == 0:
    return 0.0
  return float(np.linalg.norm(vector))


def _safe_media_name(filename: str, allowed_extensions: Tuple[str, ...]) -> str:
  """Validates media filename and allowed extension, returning safe basename."""
  candidate = os.path.basename((filename or "").strip())
  if not candidate or candidate != (filename or "").strip():
    return ""
  if not candidate.endswith(allowed_extensions):
    return ""
  return candidate


def _open_file_in_system_manager(filepath: str) -> Optional[str]:
  """Opens the file location in platform file manager, returns error string on failure."""
  try:
    if sys.platform == "win32":
      subprocess.Popen(["explorer", "/select,", filepath])
    elif sys.platform == "darwin":
      subprocess.Popen(["open", "-R", filepath])
    else:
      subprocess.Popen(["xdg-open", os.path.dirname(filepath)])
    return None
  except OSError as exc:
    return str(exc)

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

    .search-advanced-container {
      position: relative;
      min-width: 220px;
    }
    .search-advanced-panel {
      position: absolute;
      top: calc(100% + 6px);
      left: 0;
      z-index: 1050;
      width: min(680px, 92vw);
      display: none;
    }
    .search-advanced-panel.open {
      display: block;
    }
  </style>
</head>
<body>
<nav class="navbar navbar-light bg-light">
  <div class="container-fluid d-flex align-items-center flex-wrap" style="gap: 8px;">

    <!-- Capture indicator -->
    <span id="capture-dot" title="Last capture time"></span>
    <span id="capture-info" class="text-muted mr-2" style="font-size:0.78rem; white-space:nowrap;">idle</span>

    <div class="btn-group btn-group-sm mr-2" role="group" aria-label="Capture controls">
      <button class="btn btn-outline-secondary" id="pause-5m-btn" title="Pause capture for 5 minutes">Pause 5m</button>
      <button class="btn btn-outline-secondary" id="pause-30m-btn" title="Pause capture for 30 minutes">Pause 30m</button>
      <button class="btn btn-outline-secondary" id="pause-forever-btn" title="Pause capture indefinitely">Pause forever</button>
      <button class="btn btn-outline-secondary" id="resume-btn" title="Resume capture">Resume</button>
    </div>

    <span id="capture-action-feedback" class="text-muted mr-2" style="font-size:0.74rem; white-space:nowrap;"></span>

    <!-- Search bar + advanced filters -->
    <div class="search-advanced-container flex-grow-1">
      <form id="topSearchForm" class="d-flex" action="/search" method="get" style="min-width:200px;">
        <input id="topSearchInput" class="form-control flex-grow-1 mr-1" type="search" name="q" placeholder="Search" aria-label="Search"
               value="{{ search_q }}" autocomplete="off">
        <button id="topSearchFiltersBtn" type="button" class="btn btn-outline-secondary mr-1" title="Search filters">
          <i class="bi bi-sliders"></i>
        </button>
        <button class="btn btn-outline-secondary" type="submit"><i class="bi bi-search"></i></button>

        <div id="topSearchAdvancedPanel" class="search-advanced-panel card shadow-sm">
          <div class="card-body py-2">
            <div class="form-row">
              <div class="col-md-4 mb-2">
                <label class="small text-muted mb-1">Metric</label>
                <select class="form-control form-control-sm" name="metric" aria-label="Search metric">
                  <option value="cosine" {% if search_metric == 'cosine' %}selected{% endif %}>Cosine</option>
                  <option value="dot" {% if search_metric == 'dot' %}selected{% endif %}>Dot product</option>
                  <option value="euclidean" {% if search_metric == 'euclidean' %}selected{% endif %}>Euclidean distance</option>
                  <option value="manhattan" {% if search_metric == 'manhattan' %}selected{% endif %}>Manhattan distance</option>
                </select>
              </div>
              <div class="col-md-4 mb-2">
                <label class="small text-muted mb-1">From</label>
                <input type="datetime-local" class="form-control form-control-sm" name="date_from" value="{{ search_date_from }}">
              </div>
              <div class="col-md-4 mb-2">
                <label class="small text-muted mb-1">To</label>
                <input type="datetime-local" class="form-control form-control-sm" name="date_to" value="{{ search_date_to }}">
              </div>
              <div class="col-md-6 mb-2">
                <label class="small text-muted mb-1">Focused window contains</label>
                <input class="form-control form-control-sm" type="text" name="window_filter" value="{{ search_window_filter }}" placeholder="substring">
              </div>
              <div class="col-md-3 mb-2">
                <label class="small text-muted mb-1">Monitor</label>
                <select class="form-control form-control-sm" name="monitor_id">
                  <option value="" {% if not search_monitor_filter %}selected{% endif %}>All monitors</option>
                  {% for monitor_id in search_monitor_options %}
                  <option value="{{ monitor_id }}" {% if search_monitor_filter == (monitor_id|string) %}selected{% endif %}>Monitor {{ monitor_id }}</option>
                  {% endfor %}
                </select>
              </div>
              <div class="col-md-3 mb-2 d-flex align-items-end">
                <a class="btn btn-sm btn-outline-secondary w-100" href="/search?q={{ search_q|urlencode }}">Reset filters</a>
              </div>
              <div class="col-12 mb-1">
                <label class="small text-muted mb-1 d-flex justify-content-between">
                  <span>Proximity</span>
                  <span id="topSearchProximityValue" class="font-weight-bold">{{ search_proximity_human }}</span>
                </label>
                <input id="topSearchProximityLevel" type="range" class="custom-range"
                       min="0" max="{{ search_proximity_slider_steps }}" step="1"
                       value="{{ search_proximity_level }}" data-max-seconds="{{ search_proximity_max_seconds }}">
                <input id="topSearchProximitySeconds" type="hidden" name="proximity_seconds" value="{{ search_proximity_seconds }}">
                <input type="hidden" name="proximity_level" value="{{ search_proximity_level }}" id="topSearchProximityLevelHidden">
                <div class="small text-muted">Log scale: 1s → {{ search_proximity_max_human }}</div>
              </div>
            </div>
          </div>
        </div>
      </form>
    </div>

    <!-- Storage stats -->
    <span id="storage-badge" class="badge badge-light border text-muted" style="font-size:0.75rem; white-space:nowrap;" title="Storage usage"></span>

    {% if request.path != '/' %}
    <a href="/" class="btn btn-sm btn-outline-secondary" title="Go to timeline">
      <i class="bi bi-house"></i> Timeline
    </a>
    {% endif %}

    <span id="recovery-badge" class="badge badge-light border text-muted" style="font-size:0.75rem; white-space:nowrap;" title="Startup recovery status"></span>

    <!-- Open folder button -->
    <button class="btn btn-sm btn-outline-secondary" id="open-folder-btn"
          title="{{ storage_media_path }}">
      <i class="bi bi-folder2-open"></i>
    </button>

    <!-- Metrics link -->
    <a href="/metrics" class="btn btn-sm btn-outline-secondary" title="Performance metrics">
      <i class="bi bi-speedometer2"></i>
    </a>

    <a href="/config" class="btn btn-sm btn-outline-secondary" title="Configuration">
      <i class="bi bi-gear"></i>
    </a>

    <button class="btn btn-sm btn-outline-danger" id="hard-stop-btn" title="Stop OpenRecall process">
      <i class="bi bi-power"></i> Stop
    </button>

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
    const nowTs = Math.floor(Date.now() / 1000);
    const pausedUntil = Number(data.paused_until_ts || 0);
    const paused = !!data.is_paused;

    if (data.last_capture_ts && data.last_capture_ts !== lastCaptureTs) {
      lastCaptureTs = data.last_capture_ts;
      dot.classList.add('active');
      setTimeout(() => dot.classList.remove('active'), 800);
    }
    if (paused) {
      if (data.paused_indefinitely) {
        info.textContent = 'paused indefinitely';
      } else {
        const left = Math.max(0, pausedUntil - nowTs);
        info.textContent = 'paused ' + left + 's';
      }
      dot.title = 'Capture paused';
      dot.classList.remove('active');
    } else if (data.last_capture_ts) {
      const captureAgo = Math.round((Date.now()/1000) - data.last_capture_ts);
      const captureText = captureAgo < 5 ? 'cap just now' : 'cap ' + captureAgo + 's';
      const segmentAgo = Number(data.last_segment_ts || 0) > 0
        ? Math.max(0, Math.round((Date.now()/1000) - Number(data.last_segment_ts || 0)))
        : null;
      const segmentText = segmentAgo === null ? 'seg —' : (segmentAgo < 5 ? 'seg just now' : 'seg ' + segmentAgo + 's');
      info.textContent = captureText + ' · ' + segmentText;
      dot.title = 'Captures this session: ' + data.captures_this_session +
                  ' | MSSIM: ' + (data.last_mssim !== null ? data.last_mssim : '—');
    } else {
      info.textContent = 'idle';
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
    const totalSize = data.segment_size_bytes + data.thumbnail_size_bytes + data.pending_frame_size_bytes + data.db_size_bytes;
    badge.textContent = fmt(totalSize) +
                        ' · ' + data.entry_count + ' entries';
    badge.title = 'AV1 segments: ' + fmt(data.segment_size_bytes) +
                  ' | Thumbnails: ' + fmt(data.thumbnail_size_bytes) +
                  ' | Pending full-res: ' + fmt(data.pending_frame_size_bytes) +
                  ' | DB: ' + fmt(data.db_size_bytes);
    if (totalSize > 5368709120) {
      badge.classList.remove('badge-light');
      badge.classList.add('badge-warning');
    }
  }).catch(() => {});
}
loadStats();

// ---- Startup recovery status ----
function loadRecoveryStatus() {
  fetch('/api/recovery-status').then(r => r.json()).then(data => {
    const badge = document.getElementById('recovery-badge');
    if (!badge) {
      return;
    }

    const quarantined = (data.quarantined_segments || []).length;
    if (quarantined > 0) {
      badge.classList.remove('badge-light');
      badge.classList.add('badge-warning');
      badge.textContent = 'Tail repaired ' + quarantined + ' seg';
      badge.title = 'Startup tail check quarantined ' + quarantined + ' segment(s), removed ' +
                    (data.entries_removed || 0) + ' DB entries';
    } else {
      badge.classList.remove('badge-warning');
      badge.classList.add('badge-light');
      badge.textContent = 'Tail check clean';
      badge.title = 'Startup tail check found no unreadable recent segments';
    }
  }).catch(() => {});
}
loadRecoveryStatus();

// ---- Open folder ----
document.getElementById('open-folder-btn').addEventListener('click', function() {
  fetch('/open-folder', {method: 'POST'}).then(r => r.json()).then(d => {
    if (!d.ok) alert('Could not open folder: ' + d.error);
  }).catch(() => {});
});

function proximitySecondsFromLevel(level, maxSeconds, sliderMax) {
  const safeMaxSeconds = Math.max(1, Number(maxSeconds || 1));
  const safeSliderMax = Math.max(1, Number(sliderMax || 1000));
  const safeLevel = Math.max(0, Math.min(safeSliderMax, Number(level || 0)));
  if (safeMaxSeconds <= 1) {
    return 1;
  }
  const exponent = safeLevel / safeSliderMax;
  return Math.max(1, Math.min(safeMaxSeconds, Math.round(Math.exp(Math.log(safeMaxSeconds) * exponent))));
}

function formatProximitySeconds(seconds) {
  const value = Math.max(1, Number(seconds || 1));
  if (value < 60) {
    return value + 's';
  }
  if (value < 3600) {
    if (value % 60 === 0) {
      return (value / 60) + 'm';
    }
    return (value / 60).toFixed(1) + 'm';
  }
  if (value % 3600 === 0) {
    return (value / 3600) + 'h';
  }
  if (value < 86400) {
    return (value / 3600).toFixed(1) + 'h';
  }
  if (value % 86400 === 0) {
    return (value / 86400) + 'd';
  }
  return (value / 86400).toFixed(1) + 'd';
}

const topSearchForm = document.getElementById('topSearchForm');
const topSearchInput = document.getElementById('topSearchInput');
const topSearchFiltersBtn = document.getElementById('topSearchFiltersBtn');
const topSearchAdvancedPanel = document.getElementById('topSearchAdvancedPanel');
const topSearchProximityLevel = document.getElementById('topSearchProximityLevel');
const topSearchProximityValue = document.getElementById('topSearchProximityValue');
const topSearchProximitySeconds = document.getElementById('topSearchProximitySeconds');
const topSearchProximityLevelHidden = document.getElementById('topSearchProximityLevelHidden');

function openTopSearchAdvancedPanel() {
  if (topSearchAdvancedPanel) {
    topSearchAdvancedPanel.classList.add('open');
  }
}

function closeTopSearchAdvancedPanel() {
  if (topSearchAdvancedPanel) {
    topSearchAdvancedPanel.classList.remove('open');
  }
}

function syncTopSearchProximityFields() {
  if (!topSearchProximityLevel || !topSearchProximitySeconds || !topSearchProximityValue) {
    return;
  }
  const maxSeconds = Number(topSearchProximityLevel.dataset.maxSeconds || 1);
  const sliderMax = Number(topSearchProximityLevel.max || 1000);
  const level = Number(topSearchProximityLevel.value || 0);
  const seconds = proximitySecondsFromLevel(level, maxSeconds, sliderMax);

  topSearchProximitySeconds.value = String(seconds);
  topSearchProximityValue.textContent = formatProximitySeconds(seconds);
  if (topSearchProximityLevelHidden) {
    topSearchProximityLevelHidden.value = String(level);
  }
}

if (topSearchInput) {
  topSearchInput.addEventListener('focus', openTopSearchAdvancedPanel);
  topSearchInput.addEventListener('click', openTopSearchAdvancedPanel);
}

if (topSearchFiltersBtn) {
  topSearchFiltersBtn.addEventListener('click', function() {
    if (!topSearchAdvancedPanel) {
      return;
    }
    const isOpen = topSearchAdvancedPanel.classList.contains('open');
    if (isOpen) {
      closeTopSearchAdvancedPanel();
    } else {
      openTopSearchAdvancedPanel();
    }
  });
}

if (topSearchProximityLevel) {
  topSearchProximityLevel.addEventListener('input', syncTopSearchProximityFields);
  syncTopSearchProximityFields();
}

if (topSearchForm) {
  topSearchForm.addEventListener('submit', function() {
    syncTopSearchProximityFields();
    closeTopSearchAdvancedPanel();
  });
}

document.addEventListener('click', function(event) {
  if (!topSearchForm || !topSearchAdvancedPanel) {
    return;
  }
  if (!topSearchForm.contains(event.target)) {
    closeTopSearchAdvancedPanel();
  }
});

function postJson(url, payload) {
  return fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload || {})
  }).then(r => r.json());
}

const captureActionFeedback = document.getElementById('capture-action-feedback');
function setCaptureFeedback(message, timeoutMs) {
  if (!captureActionFeedback) {
    return;
  }
  captureActionFeedback.textContent = message;
  if (timeoutMs && timeoutMs > 0) {
    setTimeout(function() {
      if (captureActionFeedback.textContent === message) {
        captureActionFeedback.textContent = '';
      }
    }, timeoutMs);
  }
}

const pause5mButton = document.getElementById('pause-5m-btn');
if (pause5mButton) {
  pause5mButton.addEventListener('click', function() {
    postJson('/api/capture/pause', {minutes: 5}).then(() => {
      setCaptureFeedback('Capture paused for 5 minutes.', 3500);
      pollStatus();
    }).catch(() => {
      setCaptureFeedback('Failed to pause capture.', 4000);
    });
  });
}

const pause30mButton = document.getElementById('pause-30m-btn');
if (pause30mButton) {
  pause30mButton.addEventListener('click', function() {
    postJson('/api/capture/pause', {minutes: 30}).then(() => {
      setCaptureFeedback('Capture paused for 30 minutes.', 3500);
      pollStatus();
    }).catch(() => {
      setCaptureFeedback('Failed to pause capture.', 4000);
    });
  });
}

const pauseForeverButton = document.getElementById('pause-forever-btn');
if (pauseForeverButton) {
  pauseForeverButton.addEventListener('click', function() {
    postJson('/api/capture/pause-forever', {}).then(() => {
      setCaptureFeedback('Capture paused indefinitely.', 3500);
      pollStatus();
    }).catch(() => {
      setCaptureFeedback('Failed to pause capture.', 4000);
    });
  });
}

const resumeButton = document.getElementById('resume-btn');
if (resumeButton) {
  resumeButton.addEventListener('click', function() {
    postJson('/api/capture/resume', {}).then(() => {
      setCaptureFeedback('Capture resumed.', 3000);
      pollStatus();
    }).catch(() => {
      setCaptureFeedback('Failed to resume capture.', 4000);
    });
  });
}

const hardStopButton = document.getElementById('hard-stop-btn');
if (hardStopButton) {
  hardStopButton.addEventListener('click', function() {
    const confirmed = window.confirm('Stop OpenRecall process now? This will close the app.');
    if (!confirmed) {
      return;
    }

    postJson('/api/hard-stop', {}).then(() => {
      setCaptureFeedback('Stopping OpenRecall…', 2000);
      setTimeout(function() {
        location.reload();
      }, 800);
    }).catch(() => {
      setCaptureFeedback('Failed to stop OpenRecall.', 4000);
    });
  });
}
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


@app.context_processor
def inject_search_form_defaults() -> Dict[str, object]:
    """Provides default search form values for navbar advanced filters."""
    raw_metric = (request.args.get("metric") or "cosine").strip().lower()
    metric = raw_metric if raw_metric in SEARCH_METRICS else "cosine"

    raw_proximity_seconds = request.args.get("proximity_seconds", type=int)
    proximity_seconds = max(
      PROXIMITY_MIN_SECONDS,
      raw_proximity_seconds if raw_proximity_seconds is not None else PROXIMITY_MIN_SECONDS,
    )

    raw_proximity_level = request.args.get("proximity_level", type=int)
    proximity_level = raw_proximity_level if raw_proximity_level is not None else 0
    proximity_level = max(0, min(PROXIMITY_SLIDER_STEPS, proximity_level))

    return {
      "search_q": (request.args.get("q") or "").strip(),
      "search_metric": metric,
      "search_date_from": (request.args.get("date_from") or "").strip(),
      "search_date_to": (request.args.get("date_to") or "").strip(),
      "search_window_filter": (request.args.get("window_filter") or "").strip(),
      "search_monitor_filter": (request.args.get("monitor_id") or "").strip(),
      "search_monitor_options": [],
      "search_proximity_seconds": proximity_seconds,
      "search_proximity_level": proximity_level,
      "search_proximity_human": _format_proximity_human(proximity_seconds),
      "search_proximity_max_seconds": PROXIMITY_MAX_SECONDS_FALLBACK,
      "search_proximity_max_human": _format_proximity_human(PROXIMITY_MAX_SECONDS_FALLBACK),
      "search_proximity_slider_steps": PROXIMITY_SLIDER_STEPS,
    }


def _segment_recency_key(segment_filepath: str) -> int:
  """Returns sortable recency key from segment filename, falling back to mtime."""
  basename = os.path.basename(segment_filepath)
  ts_token = basename.split("_", 1)[0]
  try:
    return int(ts_token)
  except ValueError:
    return int(os.path.getmtime(segment_filepath) * 1000)


def _is_segment_decodable(segment_filepath: str) -> bool:
  """Returns whether ffmpeg can decode the segment without errors."""
  ffmpeg_command = [
    OPENRECALL_FFMPEG_BIN,
    "-hide_banner",
    "-loglevel",
    "error",
    "-v",
    "error",
    "-i",
    segment_filepath,
    "-map",
    "0:v:0",
    "-f",
    "null",
    "-",
  ]
  try:
    result = subprocess.run(
      ffmpeg_command,
      check=False,
      capture_output=True,
      text=True,
      timeout=45,
    )
    return result.returncode == 0
  except (OSError, subprocess.TimeoutExpired):
    return False


def _recover_recent_corrupt_segments() -> None:
  """Quarantines unreadable tail segments and removes related DB/media entries."""
  startup_recovery_state["ran"] = True
  startup_recovery_state["checked_segments"] = 0
  startup_recovery_state["quarantined_segments"] = []
  startup_recovery_state["entries_removed"] = 0
  startup_recovery_state["thumbs_removed"] = 0
  startup_recovery_state["cache_removed"] = 0

  segment_filepaths = glob.glob(os.path.join(segments_path, "*.mkv"))
  if not segment_filepaths:
    return

  recent_segment_paths = sorted(
    segment_filepaths,
    key=_segment_recency_key,
    reverse=True,
  )[:STARTUP_SEGMENT_RECOVERY_LIMIT]
  startup_recovery_state["checked_segments"] = len(recent_segment_paths)

  broken_segment_names: List[str] = []
  for segment_filepath in recent_segment_paths:
    if not _is_segment_decodable(segment_filepath):
      broken_segment_names.append(os.path.basename(segment_filepath))

  if not broken_segment_names:
    return

  os.makedirs(quarantine_segments_path, exist_ok=True)

  media_entries = get_media_entries_for_segments(broken_segment_names)
  thumbs_to_delete = {thumb for _, thumb in media_entries}

  for segment_name in broken_segment_names:
    source_path = os.path.join(segments_path, segment_name)
    if not os.path.exists(source_path):
      continue

    destination_path = os.path.join(quarantine_segments_path, segment_name)
    if os.path.exists(destination_path):
      root, ext = os.path.splitext(segment_name)
      destination_path = os.path.join(
        quarantine_segments_path,
        f"{root}.recovered{ext}",
      )

    try:
      shutil.move(source_path, destination_path)
    except OSError as exc:
      print(
        f"Startup recovery: failed to quarantine segment {segment_name}: {exc}",
        file=sys.stderr,
      )

  deleted_entries = delete_entries_by_segment_filenames(broken_segment_names)

  removed_thumbs = 0
  for thumb_name in thumbs_to_delete:
    thumb_path = os.path.join(thumbnails_path, thumb_name)
    if os.path.exists(thumb_path):
      try:
        os.remove(thumb_path)
        removed_thumbs += 1
      except OSError:
        pass

  for thumb_name in thumbs_to_delete:
    pending_frame_path = os.path.join(pending_frames_path, thumb_name)
    if os.path.exists(pending_frame_path):
      try:
        os.remove(pending_frame_path)
      except OSError:
        pass

  removed_cached_frames = 0
  broken_prefixes = {os.path.splitext(name)[0] for name in broken_segment_names}
  for cache_path in glob.glob(os.path.join(frame_cache_path, "*.png")):
    cache_name = os.path.basename(cache_path)
    if any(cache_name.startswith(prefix) for prefix in broken_prefixes):
      try:
        os.remove(cache_path)
        removed_cached_frames += 1
      except OSError:
        pass

  print(
    "Startup recovery: quarantined unreadable segments "
    f"count={len(broken_segment_names)} entries_removed={deleted_entries} "
    f"thumbs_removed={removed_thumbs} cache_removed={removed_cached_frames}",
    file=sys.stderr,
  )

  startup_recovery_state["quarantined_segments"] = sorted(broken_segment_names)
  startup_recovery_state["entries_removed"] = int(deleted_entries)
  startup_recovery_state["thumbs_removed"] = int(removed_thumbs)
  startup_recovery_state["cache_removed"] = int(removed_cached_frames)


def _encode_pending_frames_into_segment(
  segment_name: str,
  ordered_thumb_names: List[str],
) -> bool:
  """Encodes ordered pending full-resolution WebPs into one AV1 segment."""
  if not ordered_thumb_names:
    return False

  frame_filepaths = [
    os.path.join(pending_frames_path, thumb_name)
    for thumb_name in ordered_thumb_names
  ]
  if any(not os.path.exists(path) for path in frame_filepaths):
    return False

  first_frame = None
  width = 0
  height = 0
  try:
    with Image.open(frame_filepaths[0]) as image:
      first_frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = first_frame.shape[:2]
  except OSError:
    return False

  if width <= 0 or height <= 0:
    return False

  segment_filepath = os.path.join(segments_path, segment_name)
  temp_segment_path = f"{segment_filepath}.recovering.mkv"
  fps = OPENRECALL_AV1_PLAYBACK_FPS

  ffmpeg_command = [
    OPENRECALL_FFMPEG_BIN,
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "rgb24",
    "-video_size",
    f"{width}x{height}",
    "-framerate",
    f"{fps:.6f}",
    "-i",
    "-",
    "-an",
    "-c:v",
    "libsvtav1",
    "-crf",
    str(OPENRECALL_AV1_CRF),
    "-preset",
    OPENRECALL_AV1_PRESET,
  ]

  if OPENRECALL_AV1_THREADS > 0:
    ffmpeg_command.extend(["-threads", str(OPENRECALL_AV1_THREADS)])
  if OPENRECALL_AV1_SVTAV1_PARAMS:
    ffmpeg_command.extend(["-svtav1-params", OPENRECALL_AV1_SVTAV1_PARAMS])

  ffmpeg_command.extend(["-f", "matroska"])
  ffmpeg_command.append(temp_segment_path)

  process = None
  try:
    process = subprocess.Popen(
      ffmpeg_command,
      stdin=subprocess.PIPE,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.PIPE,
    )
    if process.stdin is None:
      return False

    for index, frame_path in enumerate(frame_filepaths):
      if index == 0 and first_frame is not None:
        frame_rgb = first_frame
      else:
        with Image.open(frame_path) as frame_image:
          frame_rgb = np.asarray(frame_image.convert("RGB"), dtype=np.uint8)

      if frame_rgb.shape[:2] != (height, width):
        raise RuntimeError("Pending frame dimensions do not match within segment batch")

      try:
        process.stdin.write(np.ascontiguousarray(frame_rgb, dtype=np.uint8).tobytes())
      except (BrokenPipeError, OSError) as exc:
        stderr_text = ""
        if process.stderr is not None:
          stderr_text = (process.stderr.read() or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(
          f"ffmpeg write failed for segment {segment_name}: {stderr_text or exc}"
        ) from exc

    process.stdin.close()
    process.stdin = None

    stderr_bytes = b""
    try:
      _, stderr_bytes = process.communicate(timeout=90)
    except ValueError:
      try:
        process.wait(timeout=90)
      except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)
      if process.stderr is not None:
        stderr_bytes = process.stderr.read() or b""
    except subprocess.TimeoutExpired:
      process.kill()
      _, stderr_bytes = process.communicate()

    if process.returncode != 0:
      stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
      raise RuntimeError(f"ffmpeg failed for segment {segment_name}: {stderr_text}")

    os.replace(temp_segment_path, segment_filepath)

    for frame_path in frame_filepaths:
      try:
        os.remove(frame_path)
      except OSError:
        pass

    return True
  except (OSError, RuntimeError) as exc:
    if os.path.exists(temp_segment_path):
      try:
        os.remove(temp_segment_path)
      except OSError:
        pass
    print(
      f"Startup pending recovery: failed encoding segment {segment_name}: {exc}",
      file=sys.stderr,
    )
    return False
  finally:
    if process is not None and process.stdin is not None:
      try:
        process.stdin.close()
      except OSError:
        pass


def _recover_pending_webp_segments() -> None:
  """Encodes startup pending full-resolution WebPs into missing AV1 segments."""
  startup_recovery_state["pending_recovered_segments"] = 0
  startup_recovery_state["pending_recovered_frames"] = 0
  startup_recovery_state["pending_recovery_failed_segments"] = []
  startup_recovery_state["pending_orphan_frames"] = 0
  startup_recovery_state["pending_orphan_frames_purged"] = 0

  pending_frame_paths = glob.glob(os.path.join(pending_frames_path, "*.webp"))
  pending_frame_paths.extend(glob.glob(os.path.join(pending_frames_path, "*.png")))
  if not pending_frame_paths:
    return

  pending_thumb_names = [os.path.basename(path) for path in pending_frame_paths]
  pending_paths_by_name = {
    os.path.basename(path): path
    for path in pending_frame_paths
  }
  pending_set = set(pending_thumb_names)

  def _purge_orphan_files(orphan_thumb_names: List[str]) -> Tuple[int, int]:
    purged_count = 0
    failed_count = 0
    for thumb_name in sorted(set(orphan_thumb_names)):
      orphan_path = pending_paths_by_name.get(thumb_name)
      if not orphan_path or not os.path.exists(orphan_path):
        continue
      try:
        os.remove(orphan_path)
        purged_count += 1
      except OSError:
        failed_count += 1
    return purged_count, failed_count

  recovery_rows = get_pending_segment_recovery_entries(pending_thumb_names)
  if not recovery_rows:
    purged, failed = _purge_orphan_files(pending_thumb_names)
    startup_recovery_state["pending_orphan_frames_purged"] = purged
    startup_recovery_state["pending_orphan_frames"] = failed
    return

  grouped_rows: Dict[str, List[object]] = defaultdict(list)
  for row in recovery_rows:
    segment_path = os.path.join(segments_path, row.segment_filename)
    if os.path.exists(segment_path):
      continue
    grouped_rows[row.segment_filename].append(row)

  recovered_segments = 0
  recovered_frames = 0
  failed_segments: List[str] = []
  attached_thumbs = {row.thumb_filename for row in recovery_rows}
  orphan_thumb_names = list(pending_set - attached_thumbs)
  purged_orphans, failed_orphan_deletes = _purge_orphan_files(orphan_thumb_names)

  for segment_name, rows in grouped_rows.items():
    expected_rows = get_media_entries_for_segments([segment_name])
    expected_thumbs = {thumb for _, thumb in expected_rows}
    if not expected_thumbs:
      failed_segments.append(segment_name)
      continue

    if not expected_thumbs.issubset(pending_set):
      failed_segments.append(segment_name)
      continue

    ordered_thumb_names: List[str] = []
    seen_thumbs = set()
    for row in rows:
      if row.thumb_filename in expected_thumbs and row.thumb_filename not in seen_thumbs:
        ordered_thumb_names.append(row.thumb_filename)
        seen_thumbs.add(row.thumb_filename)

    if len(ordered_thumb_names) != len(expected_thumbs):
      failed_segments.append(segment_name)
      continue

    if _encode_pending_frames_into_segment(segment_name, ordered_thumb_names):
      recovered_segments += 1
      recovered_frames += len(ordered_thumb_names)
    else:
      failed_segments.append(segment_name)

  startup_recovery_state["pending_recovered_segments"] = recovered_segments
  startup_recovery_state["pending_recovered_frames"] = recovered_frames
  startup_recovery_state["pending_recovery_failed_segments"] = sorted(set(failed_segments))
  startup_recovery_state["pending_orphan_frames"] = failed_orphan_deletes
  startup_recovery_state["pending_orphan_frames_purged"] = purged_orphans

  if recovered_segments or failed_segments or purged_orphans or failed_orphan_deletes:
    print(
      "Startup pending recovery: "
      f"segments_recovered={recovered_segments} "
      f"frames_recovered={recovered_frames} "
      f"segments_failed={len(set(failed_segments))} "
      f"orphan_frames_purged={purged_orphans} "
      f"orphan_frames={failed_orphan_deletes}",
      file=sys.stderr,
    )


def _parse_search_query(raw_query: str) -> Tuple[str, List[str]]:
    """Parses query into semantic text and exact OCR phrases from quoted segments."""
    exact_phrases: List[str] = []
    semantic_parts: List[str] = []
    buffer: List[str] = []
    in_quotes = False
    escape = False

    for char in raw_query:
      if escape:
        buffer.append(char)
        escape = False
        continue

      if char == "\\":
        escape = True
        continue

      if char == '"':
        chunk = "".join(buffer).strip()
        if in_quotes:
          if chunk:
            exact_phrases.append(chunk)
            semantic_parts.append(chunk)
        else:
          if chunk:
            semantic_parts.append(chunk)
        buffer = []
        in_quotes = not in_quotes
        continue

      if not in_quotes and char.isspace():
        chunk = "".join(buffer).strip()
        if chunk:
          semantic_parts.append(chunk)
        buffer = []
        continue

      buffer.append(char)

    if escape:
      buffer.append("\\")

    trailing_chunk = "".join(buffer).strip()
    if trailing_chunk:
      if in_quotes:
        exact_phrases.append(trailing_chunk)
      semantic_parts.append(trailing_chunk)

    semantic_query = " ".join(semantic_parts).strip()
    return semantic_query, exact_phrases


def _entry_matches_exact_phrases(entry_text: str, exact_phrases: List[str]) -> bool:
    """Checks whether entry OCR text contains all exact phrases (case-insensitive)."""
    if not exact_phrases:
      return True

    haystack = (entry_text or "").lower()
    return all(phrase.lower() in haystack for phrase in exact_phrases)


def _resolve_search_metric(raw_metric: str) -> str:
    """Normalizes and validates the requested search metric."""
    candidate = (raw_metric or "").strip().lower()
    if candidate in SEARCH_METRICS:
      return candidate
    return "cosine"


def _contains_unquoted_parentheses(raw_query: str) -> bool:
    """Returns whether raw query contains unquoted parentheses."""
    in_quotes = False
    escape = False
    for char in raw_query:
      if escape:
        escape = False
        continue
      if char == "\\":
        escape = True
        continue
      if char == '"':
        in_quotes = not in_quotes
        continue
      if not in_quotes and char in "()":
        return True
    return False


def _parse_embedding_expression(raw_query: str) -> Optional[List[Tuple[str, int]]]:
    """Parses expressions like '(queen) - (king) + (woman)' into weighted terms."""
    if not _contains_unquoted_parentheses(raw_query):
      return None

    expression = raw_query.strip()
    if not expression:
      return None

    terms: List[Tuple[str, int]] = []
    index = 0
    next_sign = 1

    while index < len(expression):
      while index < len(expression) and expression[index].isspace():
        index += 1

      if index >= len(expression):
        break

      if expression[index] != "(":
        return None

      index += 1
      start_index = index
      has_nested_parenthesis = False
      while index < len(expression) and expression[index] != ")":
        if expression[index] == "(":
          has_nested_parenthesis = True
        index += 1

      if index >= len(expression) or has_nested_parenthesis:
        return None

      term_text = expression[start_index:index].strip()
      if not term_text:
        return None

      terms.append((term_text, next_sign))
      index += 1

      while index < len(expression) and expression[index].isspace():
        index += 1

      if index >= len(expression):
        break

      operator = expression[index]
      if operator == "+":
        next_sign = 1
      elif operator == "-":
        next_sign = -1
      else:
        return None
      index += 1

    return terms if terms else None


def _compute_search_score(
    query_embedding: np.ndarray,
    entry_embedding: np.ndarray,
    metric: str,
) -> float:
    """Computes metric-specific score between query and entry embeddings."""
    if metric == "dot":
      return dot_product(query_embedding, entry_embedding)
    if metric == "euclidean":
      return euclidean_distance(query_embedding, entry_embedding)
    if metric == "manhattan":
      return manhattan_distance(query_embedding, entry_embedding)
    return cosine_similarity(query_embedding, entry_embedding)


def _score_sort_descending(metric: str) -> bool:
    """Returns whether higher score means better match for metric."""
    return metric in {"cosine", "dot"}


def _format_search_score(metric: str, score: float, used_expression: bool) -> str:
    """Formats metric score text for UI badges."""
    if metric == "cosine":
      return f"{round(score * 100, 1)}%"
    if metric == "dot":
      return f"{score:.4f}"
    if metric == "euclidean":
      return f"d={score:.4f}"
    if metric == "manhattan":
      return f"L1={score:.4f}"
    if used_expression:
      return f"{score:.4f}"
    return "—"


def _timestamp_to_datetime_local_input(timestamp: int) -> str:
    """Formats UNIX timestamp to YYYY-MM-DDTHH:MM in local time for input fields."""
    local_dt = datetime.datetime.fromtimestamp(int(timestamp))
    return local_dt.strftime("%Y-%m-%dT%H:%M")


def _parse_datetime_local_to_timestamp(raw_value: str) -> Optional[int]:
    """Parses datetime-local value (YYYY-MM-DDTHH:MM) to local UNIX timestamp."""
    value = (raw_value or "").strip()
    if not value:
      return None

    try:
      parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
      return None

    return int(parsed.timestamp())


def _entry_matches_window_filter(entry_app: str, entry_title: str, window_filter: str) -> bool:
    """Returns whether focused window metadata contains window_filter substring."""
    needle = (window_filter or "").strip().lower()
    if not needle:
      return True

    app_text = (entry_app or "").lower()
    title_text = (entry_title or "").lower()
    combined_text = (f"{entry_app or ''} {entry_title or ''}").lower()
    return needle in app_text or needle in title_text or needle in combined_text


def _entry_matches_monitor_filter(entry_monitor_id: int, monitor_filter: Optional[int]) -> bool:
    """Returns whether entry monitor matches requested monitor filter."""
    if monitor_filter is None:
      return True
    return int(entry_monitor_id) == int(monitor_filter)


def _entry_in_date_range(entry_timestamp: int, from_ts: Optional[int], to_ts: Optional[int]) -> bool:
    """Returns whether timestamp falls inside inclusive [from_ts, to_ts] bounds."""
    if from_ts is not None and entry_timestamp < from_ts:
      return False
    if to_ts is not None and entry_timestamp > to_ts:
      return False
    return True


def _apply_proximity_dedup(results: List[dict], proximity_seconds: int) -> List[dict]:
    """Greedy dedupe by timestamp proximity, preserving top-ranked representatives."""
    if proximity_seconds <= 0:
      return results

    kept: List[dict] = []
    kept_timestamps: List[int] = []
    for result in results:
      timestamp = int(result.get("timestamp") or 0)
      is_blocked = any(abs(timestamp - existing_ts) < proximity_seconds for existing_ts in kept_timestamps)
      if is_blocked:
        continue
      kept.append(result)
      kept_timestamps.append(timestamp)
    return kept


def _format_proximity_human(seconds: int) -> str:
    """Formats proximity seconds into compact human-readable text."""
    if seconds <= 0:
      return "Off"
    if seconds < 60:
      return f"{seconds}s"
    if seconds < 3600:
      minutes = seconds / 60.0
      if seconds % 60 == 0:
        return f"{int(minutes)}m"
      return f"{minutes:.1f}m"

    hours = seconds / 3600.0
    if seconds % 3600 == 0:
      return f"{int(hours)}h"
    return f"{hours:.1f}h"


def _resolve_proximity_max_seconds(oldest_ts: Optional[int], newest_ts: Optional[int]) -> int:
    """Returns max proximity seconds for slider end.

    Uses half of available recording span when possible, capped at one year.
    """
    if oldest_ts is None or newest_ts is None or newest_ts <= oldest_ts:
      return PROXIMITY_MAX_SECONDS_FALLBACK

    half_span = max(PROXIMITY_MIN_SECONDS, (int(newest_ts) - int(oldest_ts)) // 2)
    return min(PROXIMITY_MAX_SECONDS_FALLBACK, half_span)


def _proximity_seconds_to_level(seconds: int, max_seconds: int) -> int:
    """Maps proximity seconds to logarithmic slider level."""
    safe_max = max(PROXIMITY_MIN_SECONDS, int(max_seconds))
    safe_seconds = max(PROXIMITY_MIN_SECONDS, min(int(seconds), safe_max))
    if safe_max <= PROXIMITY_MIN_SECONDS:
      return 0

    ratio = np.log(float(safe_seconds)) / np.log(float(safe_max))
    level = int(round(ratio * PROXIMITY_SLIDER_STEPS))
    return max(0, min(PROXIMITY_SLIDER_STEPS, level))


def _proximity_level_to_seconds(level: int, max_seconds: int) -> int:
    """Maps logarithmic slider level to proximity seconds."""
    safe_max = max(PROXIMITY_MIN_SECONDS, int(max_seconds))
    safe_level = max(0, min(PROXIMITY_SLIDER_STEPS, int(level)))
    if safe_max <= PROXIMITY_MIN_SECONDS:
      return PROXIMITY_MIN_SECONDS

    exponent = float(safe_level) / float(PROXIMITY_SLIDER_STEPS)
    seconds = int(round(float(safe_max) ** exponent))
    return max(PROXIMITY_MIN_SECONDS, min(safe_max, seconds))


@app.route("/")
def timeline():
    timeline_entries = [
        {
            "timestamp": entry.timestamp,
            "monitor_id": entry.monitor_id,
            "segment_filename": entry.segment_filename,
            "segment_pts_ms": entry.segment_pts_ms,
            "thumb_filename": entry.thumb_filename,
      "app": entry.app or "",
      "title": entry.title or "",
      "text": entry.text or "",
            "embedding_magnitude": entry.embedding_magnitude,
            "embedding_is_zero": entry.embedding_is_zero,
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
        <div class="col-auto">
          <button class="btn btn-sm btn-outline-secondary" id="toggleMonitorOrder">Reverse monitor order</button>
        </div>
        <div class="col-auto text-muted small" id="rangeInfo"></div>
      </div>
    </div>
  </div>

  <div class="slider-container mb-2">
    <input type="range" class="slider custom-range" id="discreteSlider" min="0" max="0" step="1" value="0">
    <div class="slider-value text-muted" id="sliderValue"></div>
    <div class="small text-muted" id="monitorInfo"></div>
  </div>

  <div id="monitorPanels" class="row"></div>

  <div class="modal fade" id="timelineModal" tabindex="-1" role="dialog" aria-hidden="true">
    <div class="modal-dialog" role="document" style="max-width:min(1200px, 92vw); margin:2vh auto;">
      <div class="modal-content">
        <div class="modal-header py-2">
          <div id="timelineModalMeta"></div>
          <button type="button" class="close" data-dismiss="modal"><span>&times;</span></button>
        </div>
        <div class="modal-body p-0 position-relative" style="background:#000;">
          <button type="button" class="btn btn-sm btn-outline-light position-absolute" id="timelinePrevBtn"
                  title="Previous timestamp (←)" style="left:10px; top:50%; transform:translateY(-50%); z-index:3;">
            <i class="bi bi-arrow-left"></i>
          </button>
          <button type="button" class="btn btn-sm btn-outline-light position-absolute" id="timelineNextBtn"
                  title="Next timestamp (→)" style="right:10px; top:50%; transform:translateY(-50%); z-index:3;">
            <i class="bi bi-arrow-right"></i>
          </button>
          <img id="timelineModalImage" src="" alt="timeline screenshot"
               style="max-height:80vh; max-width:100%; width:auto; height:auto; object-fit:contain; display:block; margin:0 auto;">
        </div>
        <div class="modal-footer py-2 d-block">
          <div class="d-flex flex-wrap align-items-center mb-2" style="gap:6px;">
            <span class="badge badge-warning" id="timelineSourceBadge">Thumbnail</span>
            <span class="badge badge-light border text-muted" id="timelineEmbeddingBadge"></span>
            <button type="button" class="btn btn-sm btn-outline-secondary ml-auto" id="timelineOpenFileBtn">Open frame file</button>
          </div>
          <div class="small text-muted mb-2" id="timelineOpenFileFeedback"></div>
          <p class="mb-1 small text-muted font-weight-bold">OCR text:</p>
          <pre id="timelineModalText" class="small mb-0" style="max-height:10rem; overflow-y:auto; white-space:pre-wrap; font-size:0.75rem;"></pre>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const allEntries = {{ timeline_entries|tojson }};  // descending order by timestamp, then monitor
let filteredEntries = allEntries.slice();
let groupedByTimestamp = [];

const monitorIds = Array.from(
  new Set(allEntries.map(item => Number(item.monitor_id)))
).sort((a, b) => a - b);
const monitorOrderStorageKey = 'openrecall.timeline.reverseMonitorOrder';

function loadReverseMonitorOrderPreference() {
  try {
    return window.localStorage.getItem(monitorOrderStorageKey) === '1';
  } catch (_error) {
    return false;
  }
}

function saveReverseMonitorOrderPreference(enabled) {
  try {
    window.localStorage.setItem(monitorOrderStorageKey, enabled ? '1' : '0');
  } catch (_error) {
  }
}

let reverseMonitorOrder = loadReverseMonitorOrderPreference();

const monitorColumnClass =
  monitorIds.length <= 1
    ? 'col-12 mb-3'
    : monitorIds.length === 2
      ? 'col-12 col-lg-6 mb-3'
      : 'col-12 col-md-6 col-xl-4 mb-3';

function toLocalDatetimeInput(ts) {
  const d = new Date(ts * 1000);
  // format as YYYY-MM-DDTHH:MM for datetime-local input
  const pad = n => String(n).padStart(2, '0');
  return d.getFullYear() + '-' + pad(d.getMonth()+1) + '-' + pad(d.getDate()) +
         'T' + pad(d.getHours()) + ':' + pad(d.getMinutes());
}

function groupEntriesByTimestamp(entries) {
  const grouped = new Map();
  entries.forEach(item => {
    const key = String(item.timestamp);
    if (!grouped.has(key)) {
      grouped.set(key, { timestamp: item.timestamp, items: {} });
    }
    grouped.get(key).items[String(item.monitor_id)] = item;
  });

  return Array.from(grouped.values()).sort((a, b) => b.timestamp - a.timestamp);
}

// Set initial date inputs to span of data
const dateFrom = document.getElementById('dateFrom');
const dateTo   = document.getElementById('dateTo');
dateFrom.value = toLocalDatetimeInput(allEntries[allEntries.length - 1].timestamp);
dateTo.value   = toLocalDatetimeInput(allEntries[0].timestamp);

const slider       = document.getElementById('discreteSlider');
const sliderValue  = document.getElementById('sliderValue');
const rangeInfo    = document.getElementById('rangeInfo');
const monitorInfo  = document.getElementById('monitorInfo');
const monitorPanels = document.getElementById('monitorPanels');
const modalElement = document.getElementById('timelineModal');
const modalImage = document.getElementById('timelineModalImage');
const modalMeta = document.getElementById('timelineModalMeta');
const modalText = document.getElementById('timelineModalText');
const timelineSourceBadge = document.getElementById('timelineSourceBadge');
const timelineEmbeddingBadge = document.getElementById('timelineEmbeddingBadge');
const timelineOpenFileBtn = document.getElementById('timelineOpenFileBtn');
const timelineOpenFileFeedback = document.getElementById('timelineOpenFileFeedback');
const toggleMonitorOrderButton = document.getElementById('toggleMonitorOrder');

let latestDisplayToken = 0;
const fullFrameRetryCount = 8;
const fullFrameRetryDelayMs = 280;
const timelinePrevBtn = document.getElementById('timelinePrevBtn');
const timelineNextBtn = document.getElementById('timelineNextBtn');
let timelineModalObjectUrl = '';
let timelineModalLoadToken = 0;

function setTimelineFeedback(message, isError) {
  if (!timelineOpenFileFeedback) {
    return;
  }
  timelineOpenFileFeedback.textContent = message;
  timelineOpenFileFeedback.className = isError ? 'small text-danger mb-2' : 'small text-muted mb-2';
}

function setTimelineSourceBadge(source) {
  if (!timelineSourceBadge) {
    return;
  }

  timelineSourceBadge.className = 'badge';
  if (source === 'video_frame') {
    timelineSourceBadge.classList.add('badge-success');
    timelineSourceBadge.textContent = 'Video frame';
    return;
  }

  if (source === 'pending_webp') {
    timelineSourceBadge.classList.add('badge-info');
    timelineSourceBadge.textContent = 'Lossless WebP (pending video)';
    return;
  }

  timelineSourceBadge.classList.add('badge-warning');
  timelineSourceBadge.textContent = 'Thumbnail';
}

function formatEmbeddingBadge(item) {
  const magnitude = Number(item.embedding_magnitude || 0);
  if (item.embedding_is_zero || magnitude <= 1e-8) {
    return 'Embedding ‖e‖=0 (zero)';
  }
  return 'Embedding ‖e‖=' + magnitude.toFixed(4);
}

function buildFrameAttemptUrl(frameUrl, attempt) {
  return frameUrl + '&retry=' + attempt + '_' + Date.now();
}

function fetchFrameWithRetries(frameUrl, expectedToken, onSuccess, onFailure) {
  function attemptFetch(attempt) {
    if (expectedToken !== timelineModalLoadToken) {
      return;
    }

    const trialUrl = buildFrameAttemptUrl(frameUrl, attempt);
    fetch(trialUrl)
      .then(function(response) {
        if (!response.ok) {
          throw new Error('Frame request failed');
        }
        const source = response.headers.get('X-OpenRecall-Frame-Source') || 'video_frame';
        return response.blob().then(function(blob) {
          return {blob: blob, source: source};
        });
      })
      .then(function(result) {
        if (expectedToken !== timelineModalLoadToken) {
          return;
        }
        onSuccess(result.blob, result.source);
      })
      .catch(function() {
        if (attempt < fullFrameRetryCount) {
          setTimeout(function() {
            attemptFetch(attempt + 1);
          }, fullFrameRetryDelayMs);
          return;
        }
        onFailure();
      });
  }

  attemptFetch(0);
}

function findLatestForMonitorAtOrBefore(monitorId, timestamp) {
  for (let i = 0; i < groupedByTimestamp.length; i += 1) {
    const candidate = groupedByTimestamp[i];
    if (candidate.timestamp <= timestamp) {
      const entry = candidate.items[String(monitorId)];
      if (entry) {
        return entry;
      }
    }
  }
  return null;
}

function buildFrameUrl(item) {
  return '/frame?segment=' + encodeURIComponent(item.segment_filename) +
         '&pts_ms=' + encodeURIComponent(item.segment_pts_ms) +
         '&thumb=' + encodeURIComponent(item.thumb_filename);
}

function applyFilter() {
  const from = dateFrom.value ? new Date(dateFrom.value).getTime() / 1000 : 0;
  const to   = dateTo.value   ? new Date(dateTo.value).getTime()   / 1000 : Infinity;
  filteredEntries = allEntries.filter(item => item.timestamp >= from && item.timestamp <= to);
  groupedByTimestamp = groupEntriesByTimestamp(filteredEntries);

  slider.max = Math.max(0, groupedByTimestamp.length - 1);
  slider.value = slider.max;
  rangeInfo.textContent = groupedByTimestamp.length + ' timestamps · ' +
                          filteredEntries.length + ' screenshots';
  updateDisplay();
}

function upgradeImageToFull(imageElement, item, expectedToken) {
  const frameUrl = buildFrameUrl(item);

  function tryLoadFullFrame(attempt) {
    if (expectedToken !== latestDisplayToken) {
      return;
    }

    const trialUrl = frameUrl + '&retry=' + attempt + '_' + Date.now();
    const fullFrame = new Image();
    fullFrame.onload = function () {
      if (expectedToken === latestDisplayToken) {
        imageElement.src = trialUrl;
      }
    };
    fullFrame.onerror = function () {
      if (attempt < fullFrameRetryCount) {
        setTimeout(function () {
          tryLoadFullFrame(attempt + 1);
        }, fullFrameRetryDelayMs);
      }
    };
    fullFrame.src = trialUrl;
  }

  tryLoadFullFrame(0);
}

function openTimelineModal(item, groupTimestamp, isFallback) {
  timelineModalLoadToken += 1;
  if (timelineModalObjectUrl) {
    URL.revokeObjectURL(timelineModalObjectUrl);
    timelineModalObjectUrl = '';
  }
  modalImage.src = '/static/' + item.thumb_filename;
  modalImage.dataset.segment = item.segment_filename;
  modalImage.dataset.ptsMs = String(item.segment_pts_ms || 0);
  modalImage.dataset.thumb = item.thumb_filename;
  modalImage.dataset.monitorId = String(item.monitor_id);
  modalImage.dataset.groupTimestamp = String(groupTimestamp || item.timestamp);
  modalImage.dataset.fullLoaded = '0';
  modalImage.dataset.fullLoading = '0';

  const displayTimestamp = Number(groupTimestamp || item.timestamp);
  const appTitle = item.app
    ? ' <span class="ml-2 text-muted small">' + item.app + (item.title ? ' — ' + item.title : '') + '</span>'
    : '';
  const fallbackBadge = isFallback
    ? '<span class="badge badge-secondary ml-2">Fallback frame</span>'
    : '';

  modalMeta.innerHTML =
    '<span class="badge badge-primary">Monitor ' + item.monitor_id + '</span>' +
    '<span class="ml-2 text-muted small">' + new Date(displayTimestamp * 1000).toLocaleString() + '</span>' +
    fallbackBadge +
    appTitle;
  modalText.textContent = item.text || 'No OCR text.';
  if (timelineEmbeddingBadge) {
    timelineEmbeddingBadge.textContent = formatEmbeddingBadge(item);
  }
  setTimelineSourceBadge('thumbnail');
  modalImage.dataset.source = 'thumbnail';
  setTimelineFeedback('', false);

  setTimeout(function() {
    upgradeTimelineModalImage();
  }, 60);

  if (window.jQuery) {
    window.jQuery('#timelineModal').modal('show');
  }
}

function upgradeTimelineModalImage() {
  const image = document.getElementById('timelineModalImage');
  if (!image || image.dataset.fullLoaded === '1' || image.dataset.fullLoading === '1') {
    return;
  }

  image.dataset.fullLoading = '1';
  const expectedToken = timelineModalLoadToken;
  const frameUrl = '/frame?segment=' + encodeURIComponent(image.dataset.segment || '') +
    '&pts_ms=' + encodeURIComponent(image.dataset.ptsMs || '0') +
    '&thumb=' + encodeURIComponent(image.dataset.thumb || '');

  fetchFrameWithRetries(
    frameUrl,
    expectedToken,
    function(blob, source) {
      if (timelineModalObjectUrl) {
        URL.revokeObjectURL(timelineModalObjectUrl);
      }
      timelineModalObjectUrl = URL.createObjectURL(blob);
      image.src = timelineModalObjectUrl;
      image.dataset.fullLoaded = '1';
      image.dataset.fullLoading = '0';
      image.dataset.source = source;
      setTimelineSourceBadge(source);
    },
    function() {
      image.dataset.fullLoading = '0';
    }
  );
}

function openCurrentTimelineFile() {
  const payload = {
    segment_filename: modalImage.dataset.segment || '',
    thumb_filename: modalImage.dataset.thumb || '',
    source: modalImage.dataset.source || 'thumbnail',
  };

  postJson('/api/open-media-file', payload)
    .then(function(response) {
      if (!response.ok) {
        setTimelineFeedback(response.error || 'Unable to open frame file.', true);
        return;
      }
      setTimelineFeedback('Opened: ' + (response.opened_path || 'file'), false);
    })
    .catch(function() {
      setTimelineFeedback('Unable to open frame file.', true);
    });
}

function renderMonitorPanels(group) {
  monitorPanels.innerHTML = '';

  const orderedMonitorIds = reverseMonitorOrder ? monitorIds.slice().reverse() : monitorIds;

  orderedMonitorIds.forEach(monitorId => {
    const exactEntry = group.items[String(monitorId)] || null;
    const entry = exactEntry || findLatestForMonitorAtOrBefore(monitorId, group.timestamp);
    const isFallback = !exactEntry && !!entry;
    const col = document.createElement('div');
    col.className = monitorColumnClass;

    const card = document.createElement('div');
    card.className = 'card h-100 shadow-sm';

    const header = document.createElement('div');
    header.className = 'card-header py-1 px-2 d-flex justify-content-between align-items-center';
    header.innerHTML = '<strong>Monitor ' + monitorId + '</strong>' +
                       '<span class="small text-muted">' +
                       (isFallback ? 'fallback' : (entry ? 'captured' : 'no frame')) +
                       '</span>';
    card.appendChild(header);

    const body = document.createElement('div');
    body.className = 'card-body p-2';

    if (!entry) {
      body.innerHTML = '<div class="text-muted small">No screenshot for this timestamp.</div>';
    } else {
      const trigger = document.createElement('a');
      trigger.href = '#';
      trigger.className = 'd-flex justify-content-center align-items-center';
      trigger.style.background = '#000';
      trigger.style.minHeight = '28vh';
      trigger.style.maxHeight = '65vh';
      trigger.style.borderRadius = '4px';

      const image = document.createElement('img');
      image.src = '/static/' + entry.thumb_filename;
      image.alt = 'monitor screenshot';
      image.style.maxHeight = '65vh';
      image.style.maxWidth = '100%';
      image.style.width = '100%';
      image.style.height = '100%';
      image.style.objectFit = 'contain';
      image.style.display = 'block';
      image.style.margin = '0 auto';
      trigger.appendChild(image);

      trigger.addEventListener('click', function(evt) {
        evt.preventDefault();
        openTimelineModal(entry, group.timestamp, isFallback);
      });

      body.appendChild(trigger);

      if (isFallback) {
        const fallbackMeta = document.createElement('div');
        fallbackMeta.className = 'small text-muted mt-1';
        fallbackMeta.textContent = 'Showing most recent frame: ' +
          new Date(entry.timestamp * 1000).toLocaleString();
        body.appendChild(fallbackMeta);
      }

      setTimeout(function() {
        upgradeImageToFull(image, entry, latestDisplayToken);
      }, 120);
    }

    card.appendChild(body);
    col.appendChild(card);
    monitorPanels.appendChild(col);
  });
}

function updateDisplay() {
  latestDisplayToken += 1;

  if (groupedByTimestamp.length === 0) {
    sliderValue.textContent = 'No screenshots in range';
    monitorInfo.textContent = '';
    monitorPanels.innerHTML = '';
    return;
  }

  const idx = groupedByTimestamp.length - 1 - parseInt(slider.value);
  const group = groupedByTimestamp[idx];
  const capturedMonitors = Object.keys(group.items).length;
  sliderValue.textContent = new Date(group.timestamp * 1000).toLocaleString();
  monitorInfo.textContent = capturedMonitors + ' monitor(s) captured at this timestamp';
  renderMonitorPanels(group);
}

function navigateTimelineModal(delta) {
  if (!modalImage || !modalImage.dataset.groupTimestamp || !modalImage.dataset.monitorId) {
    return;
  }

  const currentGroupTimestamp = Number(modalImage.dataset.groupTimestamp);
  const monitorId = Number(modalImage.dataset.monitorId);

  const currentIndex = groupedByTimestamp.findIndex(function(item) {
    return item.timestamp === currentGroupTimestamp;
  });
  if (currentIndex < 0) {
    return;
  }

  const nextIndex = currentIndex + delta;
  if (nextIndex < 0 || nextIndex >= groupedByTimestamp.length) {
    return;
  }

  const targetTimestamp = groupedByTimestamp[nextIndex].timestamp;
  const nextEntry = findLatestForMonitorAtOrBefore(monitorId, targetTimestamp);
  if (!nextEntry) {
    return;
  }

  const targetGroup = groupedByTimestamp[nextIndex];
  const exactEntry = targetGroup.items[String(monitorId)] || null;
  const isFallback = !exactEntry && !!nextEntry;

  openTimelineModal(nextEntry, targetTimestamp, isFallback);
}

if (window.jQuery) {
  window.jQuery('#timelineModal').on('shown.bs.modal', function() {
    upgradeTimelineModalImage();
  });
}

if (timelinePrevBtn) {
  timelinePrevBtn.addEventListener('click', function() {
    navigateTimelineModal(1);
  });
}

if (timelineNextBtn) {
  timelineNextBtn.addEventListener('click', function() {
    navigateTimelineModal(-1);
  });
}

if (timelineOpenFileBtn) {
  timelineOpenFileBtn.addEventListener('click', function() {
    openCurrentTimelineFile();
  });
}

document.addEventListener('keydown', function(event) {
  const modalShown = modalElement && modalElement.classList.contains('show');
  if (!modalShown) {
    return;
  }

  if (event.key === 'ArrowLeft') {
    event.preventDefault();
    navigateTimelineModal(1);
  } else if (event.key === 'ArrowRight') {
    event.preventDefault();
    navigateTimelineModal(-1);
  }
});

slider.addEventListener('input', updateDisplay);
dateFrom.addEventListener('change', applyFilter);
dateTo.addEventListener('change', applyFilter);
document.getElementById('resetRange').addEventListener('click', function() {
  dateFrom.value = toLocalDatetimeInput(allEntries[allEntries.length - 1].timestamp);
  dateTo.value   = toLocalDatetimeInput(allEntries[0].timestamp);
  applyFilter();
});

if (toggleMonitorOrderButton) {
  toggleMonitorOrderButton.textContent = reverseMonitorOrder
    ? 'Normal monitor order'
    : 'Reverse monitor order';

  toggleMonitorOrderButton.addEventListener('click', function() {
    reverseMonitorOrder = !reverseMonitorOrder;
    saveReverseMonitorOrderPreference(reverseMonitorOrder);
    this.textContent = reverseMonitorOrder ? 'Normal monitor order' : 'Reverse monitor order';
    updateDisplay();
  });
}

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
    metric = _resolve_search_metric(request.args.get("metric") or "cosine")
    raw_monitor_filter = (request.args.get("monitor_id") or "").strip()
    window_filter = (request.args.get("window_filter") or "").strip()

    entries = get_all_entries()
    oldest_ts = int(entries[-1].timestamp) if entries else None
    newest_ts = int(entries[0].timestamp) if entries else None

    date_from_raw = (request.args.get("date_from") or "").strip()
    date_to_raw = (request.args.get("date_to") or "").strip()
    if not date_from_raw and oldest_ts is not None:
      date_from_raw = _timestamp_to_datetime_local_input(oldest_ts)
    if not date_to_raw and newest_ts is not None:
      date_to_raw = _timestamp_to_datetime_local_input(newest_ts)

    from_ts = _parse_datetime_local_to_timestamp(date_from_raw)
    to_ts = _parse_datetime_local_to_timestamp(date_to_raw)
    if from_ts is not None and to_ts is not None and from_ts > to_ts:
      from_ts, to_ts = to_ts, from_ts

    monitor_filter: Optional[int] = None
    if raw_monitor_filter:
      try:
        monitor_filter = int(raw_monitor_filter)
      except ValueError:
        monitor_filter = None

    proximity_max_seconds = _resolve_proximity_max_seconds(oldest_ts, newest_ts)
    proximity_level = request.args.get("proximity_level", type=int)
    proximity_seconds = request.args.get("proximity_seconds", type=int)
    if proximity_level is not None:
      proximity_seconds = _proximity_level_to_seconds(proximity_level, proximity_max_seconds)
    elif proximity_seconds is None:
      proximity_seconds = PROXIMITY_MIN_SECONDS
    proximity_seconds = max(PROXIMITY_MIN_SECONDS, min(proximity_max_seconds, int(proximity_seconds)))
    proximity_level = _proximity_seconds_to_level(proximity_seconds, proximity_max_seconds)

    monitor_options = sorted({int(entry.monitor_id) for entry in entries})

    semantic_query, exact_phrases = _parse_search_query(q)
    expression_terms = _parse_embedding_expression(q)

    candidate_indices = [
        index
        for index, entry in enumerate(entries)
        if _entry_matches_exact_phrases(entry.text or "", exact_phrases)
        and _entry_in_date_range(int(entry.timestamp), from_ts, to_ts)
        and _entry_matches_window_filter(entry.app or "", entry.title or "", window_filter)
        and _entry_matches_monitor_filter(int(entry.monitor_id), monitor_filter)
    ]

    results = []
    if semantic_query or expression_terms:
        if expression_terms:
            query_embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            for term_text, sign in expression_terms:
                query_embedding = query_embedding + (sign * get_embedding(term_text))
        else:
            query_embedding = get_embedding(semantic_query)

        scored_candidates = []
        for index in candidate_indices:
            emb = np.asarray(entries[index].embedding, dtype=np.float32)
            if emb.size == 0 or emb.shape != query_embedding.shape:
                continue
            score = _compute_search_score(query_embedding, emb, metric)
            scored_candidates.append((index, score))

        scored_candidates.sort(
            key=lambda item: item[1],
            reverse=_score_sort_descending(metric),
        )
        for index, score in scored_candidates:
            embedding_magnitude = _embedding_magnitude(entries[index].embedding)
            results.append(
                {
                    "timestamp": entries[index].timestamp,
                    "monitor_id": entries[index].monitor_id,
                    "segment_filename": entries[index].segment_filename,
                    "segment_pts_ms": entries[index].segment_pts_ms,
                    "thumb_filename": entries[index].thumb_filename,
                    "app": entries[index].app or "",
                    "title": entries[index].title or "",
                    "text": entries[index].text or "",
                    "score": _format_search_score(metric, score, bool(expression_terms)),
                    "embedding_magnitude": embedding_magnitude,
                    "embedding_is_zero": embedding_magnitude <= 1e-8,
                }
            )
    elif exact_phrases:
        for index in candidate_indices:
            embedding_magnitude = _embedding_magnitude(entries[index].embedding)
            results.append(
                {
                    "timestamp": entries[index].timestamp,
                    "monitor_id": entries[index].monitor_id,
                    "segment_filename": entries[index].segment_filename,
                    "segment_pts_ms": entries[index].segment_pts_ms,
                    "thumb_filename": entries[index].thumb_filename,
                    "app": entries[index].app or "",
                    "title": entries[index].title or "",
                    "text": entries[index].text or "",
                    "score": "Exact",
                    "embedding_magnitude": embedding_magnitude,
                    "embedding_is_zero": embedding_magnitude <= 1e-8,
                }
            )

        results.sort(key=lambda item: item["timestamp"], reverse=True)

    total_before_proximity = len(results)
    results = _apply_proximity_dedup(results, proximity_seconds)

    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
<div class="container mt-3">
  {% if not results %}
    <div class="alert alert-info">No results.</div>
  {% else %}
  <p class="text-muted small mb-2">{{ results|length }} results (from {{ total_before_proximity }} before proximity dedupe) · {{ metric_label }}</p>
  <div class="row">
    {% for r in results %}
    <div class="col-md-3 mb-4">
      <div class="card h-100 shadow-sm">
        <a href="#" data-result-index="{{ loop.index0 }}" class="d-block overflow-hidden search-result-open"
           style="max-height:160px;">
          <img src="/static/{{ r.thumb_filename }}" alt="thumbnail" class="card-img-top"
               style="object-fit:cover; height:160px;">
        </a>
        <div class="card-body p-2">
          <div class="d-flex justify-content-between align-items-start mb-1">
            <span class="badge badge-primary similarity-badge">{{ r.score }}</span>
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
    {% endfor %}
  </div>

  <div class="modal fade" id="searchModal" tabindex="-1" role="dialog" aria-hidden="true">
    <div class="modal-dialog" role="document" style="max-width:min(1200px, 92vw); margin:2vh auto;">
      <div class="modal-content">
        <div class="modal-header py-2">
          <div id="searchModalMeta"></div>
          <button type="button" class="close" data-dismiss="modal"><span>&times;</span></button>
        </div>
        <div class="modal-body p-0 position-relative" style="background:#000;">
          <button type="button" class="btn btn-sm btn-outline-light position-absolute" id="searchPrevBtn"
                  title="Previous result (←)" style="left:10px; top:50%; transform:translateY(-50%); z-index:3;">
            <i class="bi bi-arrow-left"></i>
          </button>
          <button type="button" class="btn btn-sm btn-outline-light position-absolute" id="searchNextBtn"
                  title="Next result (→)" style="right:10px; top:50%; transform:translateY(-50%); z-index:3;">
            <i class="bi bi-arrow-right"></i>
          </button>
          <img id="searchModalImage" src="" alt="thumbnail"
               style="max-height:80vh; max-width:100%; width:100%; height:100%; object-fit:contain; display:block; margin:0 auto;">
        </div>
        <div class="modal-footer py-2 d-block">
          <div class="d-flex flex-wrap align-items-center mb-2" style="gap:6px;">
            <span class="badge badge-warning" id="searchSourceBadge">Thumbnail</span>
            <span class="badge badge-light border text-muted" id="searchEmbeddingBadge"></span>
            <button type="button" class="btn btn-sm btn-outline-secondary ml-auto" id="searchOpenFileBtn">Open frame file</button>
          </div>
          <div class="small text-muted mb-2" id="searchOpenFileFeedback"></div>
          <p class="mb-1 small text-muted font-weight-bold">OCR text:</p>
          <pre id="searchModalText" class="small mb-0" style="max-height:10rem; overflow-y:auto; white-space:pre-wrap; font-size:0.75rem;"></pre>
        </div>
      </div>
    </div>
  </div>
  {% endif %}
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {
  const results = {{ results|tojson }};
  const fullFrameRetryCount = 8;
  const fullFrameRetryDelayMs = 280;
  const modalId = '#searchModal';
  const image = document.getElementById('searchModalImage');
  const text = document.getElementById('searchModalText');
  const meta = document.getElementById('searchModalMeta');
  const sourceBadge = document.getElementById('searchSourceBadge');
  const embeddingBadge = document.getElementById('searchEmbeddingBadge');
  const openFileBtn = document.getElementById('searchOpenFileBtn');
  const openFileFeedback = document.getElementById('searchOpenFileFeedback');
  const prevBtn = document.getElementById('searchPrevBtn');
  const nextBtn = document.getElementById('searchNextBtn');
  let currentIndex = -1;
  let searchModalObjectUrl = '';
  let searchModalLoadToken = 0;

  function setOpenFileFeedback(message, isError) {
    if (!openFileFeedback) {
      return;
    }
    openFileFeedback.textContent = message;
    openFileFeedback.className = isError ? 'small text-danger mb-2' : 'small text-muted mb-2';
  }

  function setSourceBadge(source) {
    if (!sourceBadge) {
      return;
    }
    sourceBadge.className = 'badge';
    if (source === 'video_frame') {
      sourceBadge.classList.add('badge-success');
      sourceBadge.textContent = 'Video frame';
      return;
    }
    if (source === 'pending_webp') {
      sourceBadge.classList.add('badge-info');
      sourceBadge.textContent = 'Lossless WebP (pending video)';
      return;
    }
    sourceBadge.classList.add('badge-warning');
    sourceBadge.textContent = 'Thumbnail';
  }

  function formatEmbeddingBadge(result) {
    const magnitude = Number(result.embedding_magnitude || 0);
    if (result.embedding_is_zero || magnitude <= 1e-8) {
      return 'Embedding ‖e‖=0 (zero)';
    }
    return 'Embedding ‖e‖=' + magnitude.toFixed(4);
  }

  function buildTrialUrl(frameUrl, attempt) {
    return frameUrl + '&retry=' + attempt + '_' + Date.now();
  }

  function fetchFrameWithRetries(frameUrl, expectedToken, onSuccess, onFailure) {
    function tryFetch(attempt) {
      if (expectedToken !== searchModalLoadToken) {
        return;
      }
      const trialUrl = buildTrialUrl(frameUrl, attempt);
      fetch(trialUrl)
        .then(function(response) {
          if (!response.ok) {
            throw new Error('Frame request failed');
          }
          const source = response.headers.get('X-OpenRecall-Frame-Source') || 'video_frame';
          return response.blob().then(function(blob) {
            return {blob: blob, source: source};
          });
        })
        .then(function(result) {
          if (expectedToken !== searchModalLoadToken) {
            return;
          }
          onSuccess(result.blob, result.source);
        })
        .catch(function() {
          if (attempt < fullFrameRetryCount) {
            setTimeout(function() {
              tryFetch(attempt + 1);
            }, fullFrameRetryDelayMs);
            return;
          }
          onFailure();
        });
    }

    tryFetch(0);
  }

  function buildFrameUrl(result) {
    return '/frame?segment=' + encodeURIComponent(result.segment_filename) +
      '&pts_ms=' + encodeURIComponent(result.segment_pts_ms || 0) +
      '&thumb=' + encodeURIComponent(result.thumb_filename || '');
  }

  function upgradeModalImage(result) {
    if (!image || image.dataset.fullLoaded === '1' || image.dataset.fullLoading === '1') {
      return;
    }
    image.dataset.fullLoading = '1';
    const expectedToken = searchModalLoadToken;

    const frameUrl = buildFrameUrl(result);

    fetchFrameWithRetries(
      frameUrl,
      expectedToken,
      function(blob, source) {
        if (searchModalObjectUrl) {
          URL.revokeObjectURL(searchModalObjectUrl);
        }
        searchModalObjectUrl = URL.createObjectURL(blob);
        image.src = searchModalObjectUrl;
        image.dataset.fullLoaded = '1';
        image.dataset.fullLoading = '0';
        image.dataset.source = source;
        setSourceBadge(source);
      },
      function() {
        image.dataset.fullLoading = '0';
      }
    );
  }

  function updateNavButtons() {
    if (!prevBtn || !nextBtn) {
      return;
    }
    prevBtn.disabled = currentIndex <= 0;
    nextBtn.disabled = currentIndex >= results.length - 1;
  }

  function renderResult(index) {
    if (index < 0 || index >= results.length) {
      return;
    }
    currentIndex = index;
    searchModalLoadToken += 1;
    const result = results[index];

    if (searchModalObjectUrl) {
      URL.revokeObjectURL(searchModalObjectUrl);
      searchModalObjectUrl = '';
    }
    image.src = '/static/' + result.thumb_filename;
    image.dataset.fullLoaded = '0';
    image.dataset.fullLoading = '0';
    image.dataset.source = 'thumbnail';

    const appTitle = result.app
      ? ' <span class="ml-2 text-muted small">' + result.app + (result.title ? ' — ' + result.title : '') + '</span>'
      : '';

    meta.innerHTML =
      '<span class="badge badge-primary">' + result.score + '</span>' +
      '<span class="ml-2 text-muted small">' + new Date(result.timestamp * 1000).toLocaleString() + '</span>' +
      appTitle;

    text.textContent = result.text || 'No OCR text.';
    if (embeddingBadge) {
      embeddingBadge.textContent = formatEmbeddingBadge(result);
    }
    setSourceBadge('thumbnail');
    setOpenFileFeedback('', false);
    updateNavButtons();
    setTimeout(function() {
      upgradeModalImage(result);
    }, 120);
  }

  function showResult(index) {
    renderResult(index);
    if (window.jQuery) {
      window.jQuery(modalId).modal('show');
    }
  }

  function navigate(delta) {
    if (currentIndex < 0) {
      return;
    }
    const nextIndex = currentIndex + delta;
    if (nextIndex < 0 || nextIndex >= results.length) {
      return;
    }
    renderResult(nextIndex);
  }

  if (window.jQuery) {
    window.jQuery(modalId).on('shown.bs.modal', function() {
      if (currentIndex >= 0) {
        upgradeModalImage(results[currentIndex]);
      }
    });
  }

  document.querySelectorAll('.search-result-open[data-result-index]').forEach(function(trigger) {
    trigger.addEventListener('click', function(event) {
      event.preventDefault();
      const idx = Number(trigger.getAttribute('data-result-index'));
      showResult(idx);
    });
  });

  if (prevBtn) {
    prevBtn.addEventListener('click', function() {
      navigate(-1);
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener('click', function() {
      navigate(1);
    });
  }

  if (openFileBtn) {
    openFileBtn.addEventListener('click', function() {
      if (currentIndex < 0 || currentIndex >= results.length) {
        return;
      }
      const current = results[currentIndex];
      postJson('/api/open-media-file', {
        segment_filename: current.segment_filename || '',
        thumb_filename: current.thumb_filename || '',
        source: image.dataset.source || 'thumbnail',
      })
        .then(function(response) {
          if (!response.ok) {
            setOpenFileFeedback(response.error || 'Unable to open frame file.', true);
            return;
          }
          setOpenFileFeedback('Opened: ' + (response.opened_path || 'file'), false);
        })
        .catch(function() {
          setOpenFileFeedback('Unable to open frame file.', true);
        });
    });
  }

  document.addEventListener('keydown', function(event) {
    const modalElement = document.querySelector(modalId);
    if (!modalElement || !modalElement.classList.contains('show')) {
      return;
    }

    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      navigate(-1);
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      navigate(1);
    }
  });
});
</script>
{% endblock %}
""",
        results=results,
  metric_label=SEARCH_METRICS[metric],
  total_before_proximity=total_before_proximity,
  search_q=q,
  search_metric=metric,
  search_date_from=date_from_raw,
  search_date_to=date_to_raw,
  search_window_filter=window_filter,
  search_monitor_filter=(str(monitor_filter) if monitor_filter is not None else ""),
  search_monitor_options=monitor_options,
  search_proximity_seconds=proximity_seconds,
  search_proximity_level=proximity_level,
  search_proximity_human=_format_proximity_human(proximity_seconds),
  search_proximity_max_seconds=proximity_max_seconds,
  search_proximity_max_human=_format_proximity_human(proximity_max_seconds),
  search_proximity_slider_steps=PROXIMITY_SLIDER_STEPS,
    )


@app.route("/static/<filename>")
def serve_image(filename):
    return send_from_directory(thumbnails_path, filename)


@app.route("/frame")
def serve_frame():
    """Extracts and serves a full-size frame from an AV1 segment."""
    segment_name = (request.args.get("segment") or "").strip()
    pts_ms = request.args.get("pts_ms", type=int)
    thumb_name = (request.args.get("thumb") or "").strip()

    if not segment_name or pts_ms is None:
        return jsonify({"error": "Missing required query params: segment and pts_ms"}), 400

    safe_segment_name = os.path.basename(segment_name)
    if safe_segment_name != segment_name or not safe_segment_name.endswith(".mkv"):
        return jsonify({"error": "Invalid segment filename"}), 400

    safe_pts_ms = max(0, pts_ms)
    segment_filepath = os.path.join(segments_path, safe_segment_name)

    safe_thumb_name = ""
    if thumb_name:
        candidate_thumb = os.path.basename(thumb_name)
        if candidate_thumb == thumb_name and candidate_thumb.endswith((".webp", ".jpg", ".jpeg")):
            safe_thumb_name = candidate_thumb

    pending_frame_filepath = ""
    if safe_thumb_name:
      pending_frame_filepath = os.path.join(pending_frames_path, safe_thumb_name)

    if not os.path.exists(segment_filepath):
      if pending_frame_filepath and os.path.exists(pending_frame_filepath):
        response = send_from_directory(pending_frames_path, safe_thumb_name)
        response.headers["X-OpenRecall-Frame-Source"] = "pending_webp"
        return response
      return jsonify({"error": "Segment not found"}), 404

    frame_index = None
    if safe_thumb_name:
        frame_index = get_segment_frame_index(safe_segment_name, safe_thumb_name)

    if frame_index is not None:
        frame_cache_filename = f"{os.path.splitext(safe_segment_name)[0]}_n{frame_index}.png"
    else:
        frame_cache_filename = f"{os.path.splitext(safe_segment_name)[0]}_{safe_pts_ms}.png"
    frame_cache_filepath = os.path.join(frame_cache_path, frame_cache_filename)

    if not os.path.exists(frame_cache_filepath):
        temp_filepath = (
            f"{os.path.splitext(frame_cache_filepath)[0]}"
            f".tmp-{os.getpid()}.png"
        )
        pts_seconds = safe_pts_ms / 1000.0
        ffmpeg_attempts = []
        if frame_index is not None:
            ffmpeg_attempts.append(
                [
                    OPENRECALL_FFMPEG_BIN,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    segment_filepath,
                    "-vf",
                    f"select=eq(n\\,{frame_index})",
                    "-vsync",
                    "vfr",
                    "-frames:v",
                    "1",
                    "-an",
                    "-f",
                    "image2",
                    temp_filepath,
                ]
            )

        ffmpeg_attempts.extend(
            [
                [
                    OPENRECALL_FFMPEG_BIN,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    segment_filepath,
                    "-ss",
                    f"{pts_seconds:.3f}",
                    "-frames:v",
                    "1",
                    "-an",
                    "-f",
                    "image2",
                    temp_filepath,
                ],
                [
                    OPENRECALL_FFMPEG_BIN,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{pts_seconds:.3f}",
                    "-i",
                    segment_filepath,
                    "-frames:v",
                    "1",
                    "-an",
                    "-f",
                    "image2",
                    temp_filepath,
                ],
                [
                    OPENRECALL_FFMPEG_BIN,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-sseof",
                    "-0.001",
                    "-i",
                    segment_filepath,
                    "-frames:v",
                    "1",
                    "-an",
                    "-f",
                    "image2",
                    temp_filepath,
                ],
            ]
        )

        last_error = ""
        extracted = False
        for ffmpeg_command in ffmpeg_attempts:
            try:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                subprocess.run(
                    ffmpeg_command,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if os.path.exists(temp_filepath) and os.path.getsize(temp_filepath) > 0:
                    os.replace(temp_filepath, frame_cache_filepath)
                    extracted = True
                    break
            except subprocess.CalledProcessError as exc:
                last_error = (exc.stderr or "").strip()

        if not extracted:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return jsonify({"error": f"Failed to decode frame: {last_error or 'unknown error'}"}), 500

    response = send_from_directory(frame_cache_path, frame_cache_filename)
    response.headers["X-OpenRecall-Frame-Source"] = "video_frame"
    return response


@app.route("/api/stats")
def api_stats():
    """Returns storage and database statistics as JSON."""
    db_path = os.path.join(appdata_folder, "recall.db")
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    av1_segment_files = glob.glob(os.path.join(segments_path, "*.mkv"))
    thumbnail_files = glob.glob(os.path.join(thumbnails_path, "*.webp"))
    thumbnail_files.extend(glob.glob(os.path.join(thumbnails_path, "*.jpg")))
    pending_frame_files = glob.glob(os.path.join(pending_frames_path, "*.webp"))
    pending_frame_files.extend(glob.glob(os.path.join(pending_frames_path, "*.png")))
    segment_size = sum(os.path.getsize(path) for path in av1_segment_files)
    thumbnail_size = sum(os.path.getsize(path) for path in thumbnail_files)
    pending_frame_size = sum(os.path.getsize(path) for path in pending_frame_files)
    timestamps = get_timestamps()
    return jsonify(
      _json_safe(
        {
          "db_size_bytes": db_size,
          "segment_size_bytes": segment_size,
          "thumbnail_size_bytes": thumbnail_size,
          "pending_frame_size_bytes": pending_frame_size,
          "entry_count": len(timestamps),
          "segment_count": len(av1_segment_files),
          "thumbnail_count": len(thumbnail_files),
          "pending_frame_count": len(pending_frame_files),
          "oldest_timestamp": timestamps[-1] if timestamps else None,
          "newest_timestamp": timestamps[0] if timestamps else None,
        }
      )
    )


@app.route("/api/status")
def api_status():
    """Returns live capture loop state as JSON."""
    paused_until_ts = int(capture_state.get("paused_until_ts") or 0)
    return jsonify(
        _json_safe(
            {
                "last_capture_ts": capture_state["last_capture_ts"],
              "last_segment_ts": capture_state.get("last_segment_ts", 0),
                "captures_this_session": capture_state["captures_this_session"],
                "last_mssim": capture_state["last_mssim"],
                "recent_timings": list(capture_state["recent_timings"]),
                "paused_until_ts": paused_until_ts,
                "paused_indefinitely": bool(capture_state.get("paused_indefinitely")),
                "is_paused": is_capture_paused(),
                "stop_requested": bool(capture_state.get("stop_requested")),
                "status": str(capture_state.get("status") or "unknown"),
                "status_updated_ts": int(capture_state.get("status_updated_ts") or 0),
                "last_blocked_reason": str(capture_state.get("last_blocked_reason") or ""),
                "last_blocked_terms": list(capture_state.get("last_blocked_terms") or []),
                "last_blocked_ts": int(capture_state.get("last_blocked_ts") or 0),
            }
        )
    )


@app.route("/api/capture/pause", methods=["POST"])
def api_capture_pause():
    """Pauses capture loop for requested number of minutes."""
    payload = request.get_json(silent=True) or {}
    minutes = payload.get("minutes", 15)
    try:
        minutes_int = max(1, int(minutes))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid minutes"}), 400

    paused_until_ts = set_capture_pause_for_seconds(minutes_int * 60)
    return jsonify({"ok": True, "paused_until_ts": paused_until_ts})


@app.route("/api/capture/pause-forever", methods=["POST"])
def api_capture_pause_forever():
    """Pauses capture loop indefinitely until resumed."""
    set_capture_pause_forever()
    return jsonify({"ok": True, "paused_indefinitely": True})


@app.route("/api/capture/resume", methods=["POST"])
def api_capture_resume():
    """Resumes capture loop immediately."""
    clear_capture_pause()
    return jsonify({"ok": True})


@app.route("/api/hard-stop", methods=["POST"])
def api_hard_stop():
    """Stops capture loop and terminates the process shortly after response."""
    clear_capture_pause()
    request_capture_stop()

    def delayed_exit() -> None:
        import time

        time.sleep(0.3)
        os._exit(0)

    Thread(target=delayed_exit, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/recovery-status")
def api_recovery_status():
    """Returns startup segment recovery summary."""
    return jsonify(_json_safe(startup_recovery_state))


@app.route("/open-folder", methods=["POST"])
def open_folder():
    """Opens the storage folder in the system file manager."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", media_path])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", media_path])
        else:
            subprocess.Popen(["xdg-open", media_path])
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)})


@app.route("/api/open-media-file", methods=["POST"])
def api_open_media_file():
    """Opens frame-related media file in system file manager.

    Resolves the best available file for the requested source state.
    """
    payload = request.get_json(silent=True) or {}
    source = (payload.get("source") or "").strip().lower()
    segment_name = _safe_media_name(
      payload.get("segment_filename") or "",
      (".mkv",),
    )
    thumb_name = _safe_media_name(
      payload.get("thumb_filename") or "",
      (".webp", ".jpg", ".jpeg", ".png"),
    )

    preferred_paths: List[str] = []
    if source == "thumbnail" and thumb_name:
      preferred_paths.append(os.path.join(thumbnails_path, thumb_name))
    elif source == "pending_webp" and thumb_name:
      preferred_paths.append(os.path.join(pending_frames_path, thumb_name))
    elif source == "video_frame" and segment_name:
      preferred_paths.append(os.path.join(segments_path, segment_name))

    fallback_paths: List[str] = []
    if thumb_name:
      fallback_paths.append(os.path.join(thumbnails_path, thumb_name))
      fallback_paths.append(os.path.join(pending_frames_path, thumb_name))
    if segment_name:
      fallback_paths.append(os.path.join(segments_path, segment_name))

    candidate_paths: List[str] = []
    seen_paths = set()
    for path in preferred_paths + fallback_paths:
      if path not in seen_paths:
        seen_paths.add(path)
        candidate_paths.append(path)

    target_path = next((path for path in candidate_paths if os.path.exists(path)), "")
    if not target_path:
      return jsonify(
        {
          "ok": False,
          "error": "Frame file is not available right now (drive may be locked or unmounted).",
        }
      ), 404

    open_error = _open_file_in_system_manager(target_path)
    if open_error:
      return jsonify({"ok": False, "error": open_error}), 500

    return jsonify(
      {
        "ok": True,
        "opened_path": os.path.basename(target_path),
      }
    )


def _parse_config_form_value(raw_value: str) -> object:
    """Parses posted config string into bool/int/float/string value."""
    value = (raw_value or "").strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


@app.route("/config", methods=["GET", "POST"])
def config_page():
    """Shows and persists runtime configuration values."""
    save_error = ""
    save_success = ""

    effective_values = get_runtime_config_values()
    if request.method == "POST":
        new_values = {}
        for key in RUNTIME_CONFIG_KEYS:
            if key in request.form:
                new_values[key] = _parse_config_form_value(request.form.get(key, ""))

        try:
            write_runtime_config_file(new_values)
            save_success = "Saved config file. Restart app to apply changes."
            effective_values = {**effective_values, **new_values}
        except OSError as exc:
            save_error = f"Failed saving config: {exc}"

    def _format_field_value(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    formatted_values = {
        key: _format_field_value(value)
        for key, value in effective_values.items()
    }

    field_meta: Dict[str, Dict[str, str]] = {
        "OPENRECALL_AV1_CRF": {
            "label": "AV1 CRF",
            "hint": "Lower = better quality and larger segments. Higher = smaller segments and lower visual fidelity.",
        },
        "OPENRECALL_AV1_PRESET": {
            "label": "AV1 Preset",
            "hint": "Lower = slower encode, usually better compression. Higher = faster encode, usually larger files.",
        },
        "OPENRECALL_AV1_THREADS": {
            "label": "AV1 Encode Threads",
            "hint": "0 lets ffmpeg choose automatically. Set >0 to cap CPU thread usage.",
        },
        "OPENRECALL_AV1_SVTAV1_PARAMS": {
            "label": "SVT-AV1 Raw Params",
            "hint": "Advanced ffmpeg `-svtav1-params` string, e.g. `lp=2:scd=0`.",
        },
        "OPENRECALL_AV1_PLAYBACK_FPS": {
            "label": "Segment Playback FPS",
            "hint": "Playback frame rate used when writing AV1 segments.",
        },
        "OPENRECALL_AV1_SEGMENT_FRAMES": {
            "label": "Lossless WebPs Before AV1 Encode",
            "hint": "How many full-resolution lossless WebP frames are buffered per monitor before flushing to one AV1 segment.",
        },
        "OPENRECALL_THUMB_QUALITY": {
            "label": "Thumbnail Quality",
            "hint": "Lossy WebP thumbnail quality (1-100). Lower values reduce storage and can reduce utility.",
        },
        "OPENRECALL_THUMB_MAX_DIMENSION": {
            "label": "Thumbnail Max Dimension",
            "hint": "Maximum thumbnail width/height in pixels. Smaller values save storage and memory.",
        },
        "OPENRECALL_CAPTURE_INTERVAL_SECONDS": {
            "label": "Capture Interval (seconds)",
            "hint": "Delay between capture checks. Lower values capture more often but increase CPU/storage usage.",
        },
        "OPENRECALL_SIMILARITY_FRAME_WIDTH": {
            "label": "Similarity Check Width",
            "hint": "Downscale width for frame similarity checks. 0 disables downscaling.",
        },
        "OPENRECALL_VERBOSE_CAPTURE_LOGS": {
            "label": "Verbose Capture Logs",
            "hint": "Boolean: true/false.",
        },
        "OPENRECALL_STORAGE_BACKEND": {
            "label": "Storage Backend",
            "hint": "Must remain `av1_hybrid`.",
        },
        "OPENRECALL_FFMPEG_BIN": {
            "label": "ffmpeg Binary",
            "hint": "Executable name/path used for AV1 encoding and decode operations.",
        },
        "OPENRECALL_BLACKLIST_WINDOWS": {
            "label": "Window Blacklist Terms",
            "hint": "Comma-separated terms. Matching windows are skipped from capture.",
        },
        "OPENRECALL_BLACKLIST_WORDS": {
            "label": "OCR Blacklist Terms",
            "hint": "Comma-separated terms. Captures whose OCR contains these terms are skipped.",
        },
        "OPENRECALL_HOTKEY_PAUSE_5M": {
            "label": "Hotkey: Pause 5m",
            "hint": "Global hotkey chord for pausing capture 5 minutes.",
        },
        "OPENRECALL_HOTKEY_PAUSE_30M": {
            "label": "Hotkey: Pause 30m",
            "hint": "Global hotkey chord for pausing capture 30 minutes.",
        },
        "OPENRECALL_HOTKEY_PAUSE_FOREVER": {
            "label": "Hotkey: Pause Forever",
            "hint": "Global hotkey chord for indefinite pause.",
        },
        "OPENRECALL_HOTKEY_RESUME": {
            "label": "Hotkey: Resume",
            "hint": "Global hotkey chord for resuming capture.",
        },
    }

    config_sections: List[Dict[str, object]] = [
        {
            "title": "Optimization",
            "description": (
                "Storage/performance/utility tuning. These settings directly trade off file size, memory use, "
                "decode speed, and visual usefulness."
            ),
            "keys": [
                "OPENRECALL_AV1_CRF",
                "OPENRECALL_AV1_PRESET",
                "OPENRECALL_AV1_THREADS",
                "OPENRECALL_AV1_PLAYBACK_FPS",
                "OPENRECALL_THUMB_QUALITY",
                "OPENRECALL_THUMB_MAX_DIMENSION",
                "OPENRECALL_SIMILARITY_FRAME_WIDTH",
            ],
        },
        {
            "title": "AV1 Segment Batching",
            "description": (
                "Batching now rotates by frame count, not by time. "
                "This controls how many lossless full-resolution WebPs are staged before AV1 encode."
            ),
            "keys": [
                "OPENRECALL_AV1_SEGMENT_FRAMES",
                "OPENRECALL_AV1_SVTAV1_PARAMS",
            ],
        },
        {
            "title": "Capture & Runtime",
            "description": "Core runtime behavior and diagnostics.",
            "keys": [
                "OPENRECALL_CAPTURE_INTERVAL_SECONDS",
                "OPENRECALL_VERBOSE_CAPTURE_LOGS",
                "OPENRECALL_STORAGE_BACKEND",
                "OPENRECALL_FFMPEG_BIN",
            ],
        },
        {
            "title": "Privacy Filters",
            "description": "Sensitive-window and OCR-term exclusion rules.",
            "keys": [
                "OPENRECALL_BLACKLIST_WINDOWS",
                "OPENRECALL_BLACKLIST_WORDS",
            ],
        },
        {
            "title": "Hotkeys",
            "description": "Global pause/resume shortcuts.",
            "keys": [
                "OPENRECALL_HOTKEY_PAUSE_5M",
                "OPENRECALL_HOTKEY_PAUSE_30M",
                "OPENRECALL_HOTKEY_PAUSE_FOREVER",
                "OPENRECALL_HOTKEY_RESUME",
            ],
        },
    ]

    section_keys = {
        key
        for section in config_sections
        for key in section["keys"]
    }
    ungrouped_keys = [
        key
        for key in RUNTIME_CONFIG_KEYS
        if key not in section_keys
    ]
    if ungrouped_keys:
        config_sections.append(
            {
                "title": "Other",
                "description": "Remaining runtime values.",
                "keys": ungrouped_keys,
            }
        )

    pending_frame_files = glob.glob(os.path.join(pending_frames_path, "*.webp"))
    pending_frame_files.extend(glob.glob(os.path.join(pending_frames_path, "*.png")))
    pending_frame_count = len(pending_frame_files)

    capture_interval_seconds = float(
        effective_values.get("OPENRECALL_CAPTURE_INTERVAL_SECONDS", 1.0)
    )
    segment_frame_count = int(effective_values.get("OPENRECALL_AV1_SEGMENT_FRAMES", 1))
    estimated_segment_window_seconds = max(
        1,
        int(round(capture_interval_seconds * segment_frame_count)),
    )

    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
<div class="container mt-3">
  <h5>Configuration</h5>
  <p class="text-muted small mb-2">
    Config file: {{ config_file_path }}. Environment variables still override file values.
  </p>
  <p class="text-muted small">
    Most settings are loaded at startup. Save here, then restart OpenRecall.
  </p>
  <p class="text-muted small mb-3">
    AV1 batching is frame-count based. Segment duration-in-seconds config is deprecated for this pipeline.
  </p>

  <div class="card mb-3">
    <div class="card-body py-2">
      <div class="small text-muted mb-1">Startup-only CLI settings (not persisted here):</div>
      <div class="small"><strong>storage_path:</strong> {{ cli_values.storage_path }}</div>
      <div class="small"><strong>primary_monitor_only:</strong> {{ cli_values.primary_monitor_only }}</div>
    </div>
  </div>

  <div class="card mb-3">
    <div class="card-body py-2">
      <div class="small text-muted mb-1">Pending full-resolution buffer transparency:</div>
      <div class="small">Lossless pending WebPs currently on disk: <strong>{{ pending_frame_count }}</strong></div>
      <div class="small">Estimated batch window before AV1 flush: <strong>~{{ estimated_segment_window_seconds }}s</strong> (capture interval × frame count)</div>
      <div class="small text-muted">Estimate is approximate because unchanged frames are skipped and monitor activity varies.</div>
    </div>
  </div>

  {% if save_error %}
    <div class="alert alert-danger py-2">{{ save_error }}</div>
  {% endif %}
  {% if save_success %}
    <div class="alert alert-success py-2">{{ save_success }}</div>
  {% endif %}

  <form method="post" action="/config">
    {% for section in config_sections %}
    <div class="card mb-3">
      <div class="card-header py-2">
        <strong>{{ section["title"] }}</strong>
        <div class="small text-muted">{{ section["description"] }}</div>
      </div>
      <div class="card-body pb-1">
        <div class="form-row">
          {% for key in section["keys"] %}
          <div class="form-group col-md-6">
            <label for="cfg-{{ key }}" class="small font-weight-bold mb-1">{{ field_meta.get(key, {}).get('label', key) }}</label>
            <input id="cfg-{{ key }}" name="{{ key }}" class="form-control form-control-sm" value="{{ formatted_values.get(key, '') }}">
            <small class="form-text text-muted">{{ field_meta.get(key, {}).get('hint', '') }}</small>
          </div>
          {% endfor %}
        </div>
      </div>
    </div>
    {% endfor %}

    <div class="card">
      <div class="card-footer d-flex justify-content-between align-items-center">
        <small class="text-muted">Changes are persisted to JSON config and applied on next startup.</small>
        <button type="submit" class="btn btn-sm btn-primary">Save</button>
      </div>
    </div>
  </form>
</div>
{% endblock %}
""",
        config_file_path=config_file_path,
        formatted_values=formatted_values,
        field_meta=field_meta,
        config_sections=config_sections,
        pending_frame_count=pending_frame_count,
        estimated_segment_window_seconds=estimated_segment_window_seconds,
        cli_values={
          "storage_path": args.storage_path or "(default appdata path)",
          "primary_monitor_only": bool(args.primary_monitor_only),
        },
        save_error=save_error,
        save_success=save_success,
    )


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
          <th>Encoding</th>
          <th>DB insert</th>
          <th>Total</th>
          <th>Text?</th>
        </tr>
      </thead>
      <tbody id="timings-body">
        <tr><td colspan="8" class="text-muted">Waiting for captures…</td></tr>
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
        '<td>' + (t.encode_ms !== undefined ? t.encode_ms : '—') + ' ms</td>' +
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
    try:
        if OPENRECALL_STORAGE_BACKEND != "av1_hybrid":
            raise RuntimeError(
                "Only OPENRECALL_STORAGE_BACKEND=av1_hybrid is supported. "
                f"Current value: {OPENRECALL_STORAGE_BACKEND!r}"
            )
        check_ffmpeg_av1_capabilities()
    except RuntimeError as exc:
        print(f"Startup capability check failed: {exc}", file=sys.stderr)
        sys.exit(1)

    create_db()
    _recover_recent_corrupt_segments()
    _recover_pending_webp_segments()

    print(f"Appdata folder: {appdata_folder}")

    # Start global hotkeys when available.
    hotkey_listener = start_hotkey_listener()
    if hotkey_listener is None:
      print("Global hotkeys unavailable (install/check pynput and desktop session).")

    tray_thread = start_linux_tray()
    if tray_thread is None:
      print("Tray icon unavailable (install/check python3-gi and AppIndicator support).")

    # Start the thread to record screenshots
    t = Thread(target=record_screenshots_thread)
    t.start()

    app.run(port=8082)
