const API_URL = "http://localhost:8000";
let isRunning = false;

// Upload Handler
document.getElementById('video-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    document.getElementById('upload-status').innerText = "Uploading video...";
    
    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            document.getElementById('upload-status').innerText = "Video uploaded! Initializing pipeline...";
            startMatch(data.file_path, `match_${Date.now()}`);
        }
    } catch (err) {
        document.getElementById('upload-status').innerText = "Upload failed.";
        console.error(err);
    }
});

async function startMatch(videoPath, matchId) {
    try {
        const response = await fetch(`${API_URL}/match/start?video_path=${encodeURIComponent(videoPath)}&match_id=${matchId}`, {
            method: 'POST'
        });
        const data = await response.json();
        if (data.status === 'success') {
            addLog(`Pipeline started for match: ${matchId}`);
            isRunning = true;
            pollStatus();
        }
    } catch (err) {
        addLog(`Error starting match: ${err.message}`);
    }
}

async function pollStatus() {
    if (!isRunning) return;

    try {
        const response = await fetch(`${API_URL}/match/status`);
        const status = await response.json();

        // Update UI
        document.getElementById('status-pill').innerText = status.active ? "PROCESSING" : "FINISHED";
        document.getElementById('status-pill').className = status.active ? "status-pill status-active" : "status-pill";
        document.getElementById('frame-val').innerText = status.frame_idx;
        
        if (status.active) {
            // Fetch latest memory to update 3D scene
            fetchLatestMemory(status.match_id);
            setTimeout(pollStatus, 200); // Poll every 200ms
        } else {
            isRunning = false;
            addLog("Processing completed.");
        }
    } catch (err) {
        console.error(err);
        setTimeout(pollStatus, 1000);
    }
}

async function fetchLatestMemory(matchId) {
    try {
        const response = await fetch(`${API_URL}/match/memory/${matchId}`);
        const memory = await response.json();

        if (memory.frames && memory.frames.length > 0) {
            const lastFrame = memory.frames[memory.frames.length - 1];
            
            // Update 3D Players
            const playerScores = (lastFrame.off_ball && lastFrame.off_ball.player_scores) || {};
            lastFrame.players.forEach(p => {
                const score = playerScores[p.player_id] || 0;
                updatePlayer(p.player_id, p.team, p.pitch_x, p.pitch_y, score);
            });

            // Update OBC Metrics
            if (lastFrame.off_ball) {
                document.getElementById('obc-val').innerText = lastFrame.off_ball.possession_prob.toFixed(4);
                document.getElementById('threat-val').innerText = lastFrame.off_ball.threat_level;
                document.getElementById('threat-val').style.color = 
                    lastFrame.off_ball.threat_level === 'high' ? '#ff3300' : (lastFrame.off_ball.threat_level === 'medium' ? '#ffcc00' : '#00ff88');
            }

            // Update System stats if available in future
            if (lastFrame.vram_usage) {
                document.getElementById('vram-val').innerText = `${lastFrame.vram_usage.toFixed(2)} GB`;
            }
        }
        
        // Log incidents
        if (memory.incidents && memory.incidents.length > 0) {
            const lastIncident = memory.incidents[memory.incidents.length - 1];
            addLog(`VAR: Potential ${lastIncident.analysis.foul_type} detected at frame ${lastIncident.frame}`);
        }
    } catch (err) {
        console.error(err);
    }
}

function addLog(msg) {
    const log = document.getElementById('log');
    const time = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="log-time">[${time}]</span> ${msg}`;
    log.prepend(entry);
}
