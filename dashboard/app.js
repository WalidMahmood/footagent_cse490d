const API_URL = "http://localhost:8000";
let isRunning = false;
let matchData = null;
let currentFrameIdx = 0;
let totalFrames = 0;

// ── Upload ──────────────────────────────────────────────
document.getElementById('video-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    document.getElementById('upload-status').innerText = "Uploading...";

    try {
        const response = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
        const data = await response.json();
        if (data.status === 'success') {
            document.getElementById('upload-status').innerText = "Uploaded! Starting pipeline...";
            addLog("Video uploaded: " + file.name);
            startMatch(data.file_path, `match_${Date.now()}`);
        }
    } catch (err) {
        document.getElementById('upload-status').innerText = "Upload failed. Is the server running?";
        addLog(`Upload error: ${err.message}`);
    }
});

document.getElementById('btn-process').addEventListener('click', () => {
    const input = document.getElementById('video-upload');
    if (input.files.length > 0) input.dispatchEvent(new Event('change'));
    else input.click();
});

// ── Pipeline ────────────────────────────────────────────
async function startMatch(videoPath, matchId) {
    try {
        const resp = await fetch(`${API_URL}/match/start?video_path=${encodeURIComponent(videoPath)}&match_id=${matchId}`, { method: 'POST' });
        const data = await resp.json();
        if (data.status === 'success') {
            addLog(`Pipeline started: ${matchId}`);
            isRunning = true;
            setStatus('live');
            hideIdleHint();
            // Give backend thread time to initialize before first poll
            setTimeout(() => pollStatus(matchId), 1500);
        }
    } catch (err) {
        addLog(`Error: ${err.message}`);
    }
}

async function pollStatus(matchId) {
    if (!isRunning) return;
    try {
        const resp = await fetch(`${API_URL}/match/status`);
        const status = await resp.json();

        document.getElementById('frame-val').innerText = status.frame_idx || 0;
        document.getElementById('players-val').innerText = status.players_count || 0;
        document.getElementById('xg-val').innerText = (status.xg || 0).toFixed(3);
        document.getElementById('obc-val').innerText = (status.max_obc || 0).toFixed(3);
        currentFrameIdx = status.frame_idx || 0;
        totalFrames = status.total_frames || 0;

        // Update FPS and VRAM in topbar if elements exist
        const fpsEl = document.getElementById('fps-val');
        if (fpsEl) fpsEl.innerText = status.fps ? status.fps.toFixed(0) : 'N/A';

        if (status.active) {
            fetchLatestMemory(matchId || status.match_id);
            setTimeout(() => pollStatus(matchId), 600);
        } else if (status.frame_idx > 0) {
            // Only mark complete if we actually processed frames
            isRunning = false;
            setStatus('offline');
            addLog("Processing complete");
            fetchLatestMemory(matchId || status.match_id);
            fetchReport(matchId || status.match_id);
        } else {
            // Still waiting for first frame, keep polling
            setTimeout(() => pollStatus(matchId), 1000);
        }
    } catch (err) {
        setTimeout(() => pollStatus(matchId), 2000);
    }
}

async function fetchLatestMemory(matchId) {
    try {
        const resp = await fetch(`${API_URL}/match/memory/${matchId}`);
        const memory = await resp.json();
        matchData = memory;

        if (!memory.frames || memory.frames.length === 0) return;

        const lastFrame = memory.frames[memory.frames.length - 1];
        totalFrames = memory.frames.length;

        updatePlayerList(lastFrame);
        update3DScene(lastFrame);
        updateMetrics(lastFrame, memory);
        updateScrubber(memory);
    } catch (err) { /* server may not be ready */ }
}

async function fetchReport(matchId) {
    try {
        const resp = await fetch(`${API_URL}/match/report/${matchId}`);
        if (!resp.ok) return;
        const report = await resp.json();
        if (report.summary) {
            document.getElementById('report-duration').innerText = `${report.summary.duration_sec}s`;
            document.getElementById('report-moments').innerText = report.summary.key_moments;
            document.getElementById('report-avg-xg').innerText = report.summary.avg_xg.toFixed(3);
            document.getElementById('report-max-xg').innerText = report.summary.max_xg.toFixed(3);
        }
        if (report.top_obc_players && report.top_obc_players.length) {
            renderTopOBC(report.top_obc_players.slice(0,5).map(p => [String(p.player_id), p.avg_obc, p.class]));
        }
        addLog("Match report loaded");
    } catch (err) { /* report may not be ready */ }
}

// ── Render functions ────────────────────────────────────
function updatePlayerList(frame) {
    const list = document.getElementById('player-list');
    const players = (frame.players || []).filter(p => p.class_name !== 'ball');
    const scores = (frame.off_ball && frame.off_ball.player_scores) || {};

    const sorted = [...players].sort((a, b) => {
        return (scores[String(b.player_id)] || 0) - (scores[String(a.player_id)] || 0);
    });

    list.innerHTML = sorted.map(p => {
        const obc = scores[String(p.player_id)] || 0;
        const team = p.team;
        const dotColor = team === 0 ? '#ff2222' : (team === 1 ? '#2277ff' : '#666');
        const cls = p.class_name || 'player';
        const clsLabel = cls === 'goalkeeper' ? 'GK' : (cls === 'referee' ? 'REF' : 'PLY');
        const obcCls = obc > 0.65 ? 'obc-hi' : (obc > 0.3 ? 'obc-mid' : 'obc-lo');
        const barColor = obc > 0.65 ? 'var(--accent)' : (obc > 0.3 ? 'var(--warn)' : 'rgba(255,255,255,0.15)');
        const barW = (obc * 100).toFixed(0);
        return `
            <div class="player-row" onclick="focusPlayer(${p.player_id})">
                <div class="player-dot" style="background:${dotColor}"></div>
                <div class="player-id">#${p.player_id}</div>
                <div class="player-cls">${clsLabel}</div>
                <div class="player-obc ${obcCls}">${obc.toFixed(2)}</div>
            </div>
            <div class="obc-bar-row"><div class="obc-bar-fill" style="width:${barW}%;background:${barColor}"></div></div>
        `;
    }).join('');

    document.getElementById('player-count-badge').innerText = players.length;
    document.getElementById('players-val').innerText = players.length;
}

function update3DScene(frame) {
    const players = frame.players || [];
    const scores = (frame.off_ball && frame.off_ball.player_scores) || {};
    players.forEach(p => {
        const score = scores[String(p.player_id)] || 0;
        updatePlayer(p.player_id, p.team, p.pitch_x, p.pitch_y, score, p.class_name);
    });
}

function updateMetrics(frame, memory) {
    const ob = frame.off_ball || {};
    const scores = ob.player_scores || {};
    const maxObc = Object.values(scores).length ? Math.max(...Object.values(scores)) : 0;

    document.getElementById('xg-val').innerText = (ob.possession_prob || 0).toFixed(3);
    document.getElementById('obc-val').innerText = maxObc.toFixed(3);

    if (frame.vram_usage) {
        document.getElementById('vram-val').innerText = `${frame.vram_usage.toFixed(1)} GB`;
    }

    if (memory.frames) {
        document.getElementById('report-duration').innerText = `${Math.round(memory.frames.length / 24)}s`;
    }

    const incidents = memory.incidents || [];
    document.getElementById('report-moments').innerText = incidents.length;

    // Top OBC live
    const top5 = Object.entries(scores).sort((a,b) => b[1]-a[1]).slice(0,5);
    if (top5.length) renderTopOBC(top5.map(([pid, s]) => [pid, s, null]));

    // Key moments
    if (incidents.length) {
        const container = document.getElementById('moments-list');
        container.innerHTML = incidents.slice(-10).reverse().map(m => `
            <div class="moment-row">
                <div class="moment-frame">F:${m.frame}</div>
                <div class="moment-xg">xG:${(m.xg||0).toFixed(3)}</div>
                <div class="moment-ply">${m.type || 'high-xG'}</div>
            </div>
        `).join('');
    }
}

function renderTopOBC(entries) {
    const container = document.getElementById('top-obc-list');
    container.innerHTML = entries.map(([pid, obc, cls], i) => {
        const barW = (obc * 100).toFixed(0);
        const color = obc > 0.65 ? 'var(--accent)' : (obc > 0.35 ? 'var(--warn)' : 'rgba(255,255,255,0.2)');
        const clsLabel = cls === 'goalkeeper' ? 'GK' : (cls === 'referee' ? 'REF' : '');
        return `
            <div class="obc-entry">
                <div class="obc-rank">${i+1}.</div>
                <div class="obc-pid">#${pid}</div>
                ${clsLabel ? `<div class="obc-cls">${clsLabel}</div>` : ''}
                <div class="obc-score" style="color:${color}">${parseFloat(obc).toFixed(3)}</div>
            </div>
            <div class="obc-bar-wrap"><div class="obc-bar-inner" style="width:${barW}%;background:${color}"></div></div>
        `;
    }).join('');
}

function updateScrubber(memory) {
    if (!memory.frames || !memory.frames.length) return;
    const total = memory.frames.length;
    const pct = ((currentFrameIdx / Math.max(total,1)) * 100).toFixed(1);
    document.getElementById('scrubber-fill').style.width = `${pct}%`;

    const totalSec = Math.round(total / 24);
    const currentSec = Math.round(currentFrameIdx / 24);
    document.getElementById('scrub-current').innerText = fmtTime(currentSec);
    document.getElementById('scrub-total').innerText = fmtTime(totalSec);

    const markers = document.getElementById('scrubber-markers');
    markers.innerHTML = '';
    (memory.incidents || []).forEach(inc => {
        const pos = ((inc.frame / Math.max(total,1)) * 100).toFixed(1);
        const m = document.createElement('div');
        m.className = 'scrubber-marker';
        m.style.left = `${pos}%`;
        markers.appendChild(m);
    });
}

function fmtTime(sec) {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

function setStatus(mode) {
    const badge = document.getElementById('status-badge');
    if (mode === 'live') {
        badge.innerText = 'PROCESSING';
        badge.className = 'status-badge status-live';
    } else {
        badge.innerText = 'OFFLINE';
        badge.className = 'status-badge status-offline';
    }
}

function hideIdleHint() {
    const el = document.getElementById('hud-idle');
    if (el) el.classList.add('hidden');
}

function addLog(msg) {
    const log = document.getElementById('event-log');
    const time = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="log-time">[${time}]</span>${msg}`;
    log.prepend(entry);
}

function focusPlayer(pid) {
    if (typeof players !== 'undefined' && players[pid] && players[pid].mesh) {
        const pos = players[pid].mesh.position;
        camera.position.set(pos.x, 28, pos.z + 28);
        controls.target.set(pos.x, 0, pos.z);
        controls.update();
        addLog(`Focused on player #${pid}`);
    }
}
