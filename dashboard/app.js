const API_URL = "http://localhost:8000";
let isRunning = false;
let matchData = null;
let currentFrameIdx = 0;
let totalFrames = 0;

document.getElementById('video-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    document.getElementById('upload-status').innerText = "Uploading...";

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.status === 'success') {
            document.getElementById('upload-status').innerText = "Uploaded! Starting pipeline...";
            addLog("Video uploaded successfully");
            startMatch(data.file_path, `match_${Date.now()}`);
        }
    } catch (err) {
        document.getElementById('upload-status').innerText = "Upload failed. Is the server running?";
        addLog(`Upload error: ${err.message}`);
    }
});

document.getElementById('btn-process').addEventListener('click', () => {
    const input = document.getElementById('video-upload');
    if (input.files.length > 0) {
        input.dispatchEvent(new Event('change'));
    } else {
        input.click();
    }
});

async function startMatch(videoPath, matchId) {
    try {
        const response = await fetch(`${API_URL}/match/start?video_path=${encodeURIComponent(videoPath)}&match_id=${matchId}`, {
            method: 'POST'
        });
        const data = await response.json();
        if (data.status === 'success') {
            addLog(`Pipeline started: ${matchId}`);
            isRunning = true;
            setStatus('live');
            pollStatus(matchId);
        }
    } catch (err) {
        addLog(`Error: ${err.message}`);
    }
}

async function pollStatus(matchId) {
    if (!isRunning) return;

    try {
        const response = await fetch(`${API_URL}/match/status`);
        const status = await response.json();

        document.getElementById('frame-val').innerText = status.frame_idx || 0;
        currentFrameIdx = status.frame_idx || 0;

        if (status.active) {
            fetchLatestMemory(matchId || status.match_id);
            setTimeout(() => pollStatus(matchId), 300);
        } else {
            isRunning = false;
            setStatus('offline');
            addLog("Processing complete");
            fetchLatestMemory(matchId || status.match_id);
        }
    } catch (err) {
        setTimeout(() => pollStatus(matchId), 2000);
    }
}

async function fetchLatestMemory(matchId) {
    try {
        const response = await fetch(`${API_URL}/match/memory/${matchId}`);
        const memory = await response.json();
        matchData = memory;

        if (!memory.frames || memory.frames.length === 0) return;

        const lastFrame = memory.frames[memory.frames.length - 1];
        totalFrames = memory.frames.length;

        updatePlayerList(lastFrame);
        update3DScene(lastFrame);
        updateMetrics(lastFrame, memory);
        updateScrubber(memory);

    } catch (err) {
        // Server may not be ready
    }
}

function updatePlayerList(frame) {
    const list = document.getElementById('player-list');
    const players = (frame.players || []).filter(p => p.class_name !== 'ball');
    const scores = (frame.off_ball && frame.off_ball.player_scores) || {};

    const sorted = [...players].sort((a, b) => {
        const sa = scores[String(a.player_id)] || 0;
        const sb = scores[String(b.player_id)] || 0;
        return sb - sa;
    });

    list.innerHTML = sorted.map(p => {
        const obc = scores[String(p.player_id)] || 0;
        const team = p.team;
        const dotColor = team === 0 ? '#ff3333' : (team === 1 ? '#3388ff' : '#888');
        const cls = p.class_name || 'player';
        const clsLabel = cls === 'goalkeeper' ? 'GK' : (cls === 'referee' ? 'REF' : 'PLY');
        const obcClass = obc > 0.7 ? 'obc-high' : (obc > 0.3 ? 'obc-mid' : 'obc-low');

        return `
            <div class="player-row" onclick="focusPlayer(${p.player_id})">
                <div class="player-dot" style="background: ${dotColor}"></div>
                <div class="player-id">#${p.player_id}</div>
                <div class="player-class">${clsLabel}</div>
                <div class="player-obc ${obcClass}">${obc.toFixed(2)}</div>
            </div>
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
    const offBall = frame.off_ball || {};
    const scores = offBall.player_scores || {};
    const maxObc = Object.values(scores).length > 0 ? Math.max(...Object.values(scores)) : 0;

    document.getElementById('xg-val').innerText = (offBall.possession_prob || 0).toFixed(3);
    document.getElementById('obc-val').innerText = maxObc.toFixed(3);

    if (frame.vram_usage) {
        document.getElementById('vram-val').innerText = `${frame.vram_usage.toFixed(1)} GB`;
    }

    // Report
    if (memory.stats) {
        document.getElementById('report-duration').innerText =
            `${Math.round((memory.stats.total_frames || 0) / 24)}s`;
    }
    if (memory.frames) {
        document.getElementById('report-duration').innerText =
            `${Math.round(memory.frames.length / 24)}s`;
    }

    const incidents = memory.incidents || [];
    document.getElementById('report-moments').innerText = incidents.length;

    // Top OBC
    updateTopOBC(scores);
}

function updateTopOBC(scores) {
    const sorted = Object.entries(scores)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);

    const container = document.getElementById('top-obc-list');
    container.innerHTML = sorted.map(([pid, obc], i) => {
        const barWidth = (obc * 100).toFixed(0);
        const color = obc > 0.7 ? 'var(--accent)' : (obc > 0.4 ? 'var(--warn)' : 'var(--text-dim)');
        return `
            <div class="player-row">
                <div style="font-size: 11px; color: var(--text-dim); min-width: 16px;">${i + 1}.</div>
                <div class="player-id">#${pid}</div>
                <div class="player-obc" style="color: ${color}">${obc.toFixed(3)}</div>
            </div>
            <div class="obc-bar-container">
                <div class="obc-bar" style="width: ${barWidth}%; background: ${color};"></div>
            </div>
        `;
    }).join('');
}

function updateScrubber(memory) {
    if (!memory.frames || memory.frames.length === 0) return;

    const total = memory.frames.length;
    const pct = ((currentFrameIdx / Math.max(total, 1)) * 100).toFixed(1);
    document.getElementById('scrubber-fill').style.width = `${pct}%`;

    const totalSec = Math.round(total / 24);
    const currentSec = Math.round(currentFrameIdx / 24);
    document.getElementById('scrub-current').innerText = formatTime(currentSec);
    document.getElementById('scrub-total').innerText = formatTime(totalSec);

    // Mark key moments on scrubber
    const markers = document.getElementById('scrubber-markers');
    markers.innerHTML = '';
    if (memory.incidents) {
        memory.incidents.forEach(inc => {
            const pos = ((inc.frame / Math.max(total, 1)) * 100).toFixed(1);
            const marker = document.createElement('div');
            marker.className = 'scrubber-marker';
            marker.style.left = `${pos}%`;
            markers.appendChild(marker);
        });
    }
}

function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
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

function addLog(msg) {
    const log = document.getElementById('log');
    const time = new Date().toLocaleTimeString([], {
        hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="log-time">[${time}]</span> ${msg}`;
    log.prepend(entry);
}

function focusPlayer(pid) {
    if (players[pid] && players[pid].mesh) {
        const pos = players[pid].mesh.position;
        camera.position.set(pos.x, 30, pos.z + 30);
        controls.target.set(pos.x, 0, pos.z);
        controls.update();
        addLog(`Focused on player #${pid}`);
    }
}

// Load report JSON if available
async function loadReport(path) {
    try {
        const response = await fetch(path);
        const report = await response.json();

        if (report.summary) {
            document.getElementById('report-duration').innerText = `${report.summary.duration_sec}s`;
            document.getElementById('report-moments').innerText = report.summary.key_moments;
            document.getElementById('report-avg-xg').innerText = report.summary.avg_xg.toFixed(3);
            document.getElementById('report-max-xg').innerText = report.summary.max_xg.toFixed(3);
        }

        if (report.top_obc_players) {
            const container = document.getElementById('top-obc-list');
            container.innerHTML = report.top_obc_players.slice(0, 5).map((p, i) => {
                const color = p.avg_obc > 0.7 ? 'var(--accent)' : 'var(--warn)';
                return `
                    <div class="player-row">
                        <div style="font-size: 11px; color: var(--text-dim); min-width: 16px;">${i + 1}.</div>
                        <div class="player-id">#${p.player_id}</div>
                        <div class="player-class">${p.class === 'goalkeeper' ? 'GK' : 'PLY'}</div>
                        <div class="player-obc obc-high">${p.avg_obc.toFixed(3)}</div>
                    </div>
                    <div class="obc-bar-container">
                        <div class="obc-bar" style="width: ${(p.avg_obc * 100).toFixed(0)}%; background: ${color};"></div>
                    </div>
                `;
            }).join('');
        }

        if (report.key_moments) {
            const container = document.getElementById('moments-list');
            container.innerHTML = report.key_moments.slice(0, 10).map(m => `
                <div class="moment-row">
                    <div class="moment-frame">F:${m.frame}</div>
                    <div class="moment-xg">xG:${m.xg.toFixed(3)}</div>
                    <div class="moment-player">#${m.top_player} (${m.top_obc.toFixed(2)})</div>
                </div>
            `).join('');
        }

        addLog("Report loaded");
    } catch (err) {
        // No report available
    }
}
