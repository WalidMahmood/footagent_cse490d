/* pitch3d.js — Full-screen 3D pitch. Canvas fills #pitch-canvas which is position:fixed inset:0 */
let scene, camera, renderer, controls;
let players = {};
let ballMesh = null;
let initialized = false;

function init3D() {
    if (initialized) return;
    initialized = true;

    const container = document.getElementById('pitch-canvas');
    const W = window.innerWidth;
    const H = window.innerHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x03030c);
    scene.fog = new THREE.FogExp2(0x03030c, 0.0032);

    camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 1000);
    camera.position.set(0, 75, 85);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.maxPolarAngle = Math.PI / 2.05;
    controls.minDistance = 15;
    controls.maxDistance = 220;
    controls.target.set(0, 0, 0);

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.25);
    scene.add(ambient);

    const sun = new THREE.DirectionalLight(0xfff5e0, 0.7);
    sun.position.set(40, 90, 50);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = 300;
    sun.shadow.camera.left = -80;
    sun.shadow.camera.right = 80;
    sun.shadow.camera.top = 60;
    sun.shadow.camera.bottom = -60;
    scene.add(sun);

    const fill = new THREE.DirectionalLight(0x8af0cc, 0.15);
    fill.position.set(-40, 50, -30);
    scene.add(fill);

    const accent = new THREE.SpotLight(0x00e87b, 0.3);
    accent.position.set(0, 130, 0);
    accent.angle = Math.PI / 2.5;
    accent.penumbra = 0.5;
    scene.add(accent);

    createPitch();
    createBallMesh();
    animate();
}

function createPitch() {
    // Pitch surface with stripe pattern
    const pitchGeo = new THREE.PlaneGeometry(105, 68, 21, 17);
    const pitchMat = new THREE.MeshStandardMaterial({
        color: 0x0a1a0a,
        roughness: 0.95,
        metalness: 0.0,
    });
    const pitchMesh = new THREE.Mesh(pitchGeo, pitchMat);
    pitchMesh.rotation.x = -Math.PI / 2;
    pitchMesh.receiveShadow = true;
    scene.add(pitchMesh);

    // Pitch stripes (alternating dark/light green)
    for (let i = 0; i < 10; i++) {
        const stripeGeo = new THREE.PlaneGeometry(10.5, 68);
        const stripeMat = new THREE.MeshStandardMaterial({
            color: i % 2 === 0 ? 0x0a1a0a : 0x0d2010,
            roughness: 0.95,
            transparent: true,
            opacity: 0.9,
        });
        const stripe = new THREE.Mesh(stripeGeo, stripeMat);
        stripe.rotation.x = -Math.PI / 2;
        stripe.position.set(-47.25 + i * 10.5, 0.01, 0);
        scene.add(stripe);
    }

    const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 });
    const pts = [];

    // Boundary
    pts.push(new THREE.Vector3(-52.5,0.1,-34), new THREE.Vector3(52.5,0.1,-34));
    pts.push(new THREE.Vector3(52.5,0.1,-34), new THREE.Vector3(52.5,0.1,34));
    pts.push(new THREE.Vector3(52.5,0.1,34), new THREE.Vector3(-52.5,0.1,34));
    pts.push(new THREE.Vector3(-52.5,0.1,34), new THREE.Vector3(-52.5,0.1,-34));
    // Halfway
    pts.push(new THREE.Vector3(0,0.1,-34), new THREE.Vector3(0,0.1,34));

    // Penalty areas (16.5m deep, 40.3m wide)
    const paD=16.5, paW=20.15;
    pts.push(new THREE.Vector3(-52.5,0.1,-paW), new THREE.Vector3(-52.5+paD,0.1,-paW));
    pts.push(new THREE.Vector3(-52.5+paD,0.1,-paW), new THREE.Vector3(-52.5+paD,0.1,paW));
    pts.push(new THREE.Vector3(-52.5+paD,0.1,paW), new THREE.Vector3(-52.5,0.1,paW));
    pts.push(new THREE.Vector3(52.5,0.1,-paW), new THREE.Vector3(52.5-paD,0.1,-paW));
    pts.push(new THREE.Vector3(52.5-paD,0.1,-paW), new THREE.Vector3(52.5-paD,0.1,paW));
    pts.push(new THREE.Vector3(52.5-paD,0.1,paW), new THREE.Vector3(52.5,0.1,paW));

    // Goal areas (5.5m deep, 18.3m wide)
    const gaD=5.5, gaW=9.16;
    pts.push(new THREE.Vector3(-52.5,0.1,-gaW), new THREE.Vector3(-52.5+gaD,0.1,-gaW));
    pts.push(new THREE.Vector3(-52.5+gaD,0.1,-gaW), new THREE.Vector3(-52.5+gaD,0.1,gaW));
    pts.push(new THREE.Vector3(-52.5+gaD,0.1,gaW), new THREE.Vector3(-52.5,0.1,gaW));
    pts.push(new THREE.Vector3(52.5,0.1,-gaW), new THREE.Vector3(52.5-gaD,0.1,-gaW));
    pts.push(new THREE.Vector3(52.5-gaD,0.1,-gaW), new THREE.Vector3(52.5-gaD,0.1,gaW));
    pts.push(new THREE.Vector3(52.5-gaD,0.1,gaW), new THREE.Vector3(52.5,0.1,gaW));

    const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
    scene.add(new THREE.LineSegments(lineGeo, lineMat));

    // Center circle
    const circleGeo = new THREE.RingGeometry(9.1, 9.25, 64);
    const circleMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5, side: THREE.DoubleSide });
    const circle = new THREE.Mesh(circleGeo, circleMat);
    circle.rotation.x = -Math.PI / 2;
    circle.position.y = 0.11;
    scene.add(circle);

    // Center spot
    const spotGeo = new THREE.CircleGeometry(0.3, 16);
    const spot = new THREE.Mesh(spotGeo, circleMat);
    spot.rotation.x = -Math.PI / 2;
    spot.position.y = 0.11;
    scene.add(spot);

    // Penalty spots
    for (const sx of [-52.5 + 11, 52.5 - 11]) {
        const ps = new THREE.Mesh(new THREE.CircleGeometry(0.25, 12), circleMat);
        ps.rotation.x = -Math.PI / 2;
        ps.position.set(sx, 0.11, 0);
        scene.add(ps);
    }

    // Goal nets
    const goalMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.22 });
    addGoalNet(-52.5, goalMat);
    addGoalNet(52.5, goalMat);

    // Accent glow on pitch center
    const glowGeo = new THREE.CircleGeometry(15, 32);
    const glowMat = new THREE.MeshBasicMaterial({ color: 0x00e87b, transparent: true, opacity: 0.025, side: THREE.DoubleSide });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    glow.rotation.x = -Math.PI / 2;
    glow.position.y = 0.05;
    scene.add(glow);
}

function addGoalNet(x, mat) {
    const gw = 3.66, gh = 2.44, gd = 1.8;
    const sign = x < 0 ? 1 : -1;
    const ps = [];
    ps.push(new THREE.Vector3(x,0,-gw), new THREE.Vector3(x,gh,-gw));
    ps.push(new THREE.Vector3(x,gh,-gw), new THREE.Vector3(x,gh,gw));
    ps.push(new THREE.Vector3(x,gh,gw), new THREE.Vector3(x,0,gw));
    ps.push(new THREE.Vector3(x,gh,-gw), new THREE.Vector3(x+sign*gd,gh,-gw));
    ps.push(new THREE.Vector3(x,gh,gw), new THREE.Vector3(x+sign*gd,gh,gw));
    ps.push(new THREE.Vector3(x+sign*gd,0,-gw), new THREE.Vector3(x+sign*gd,gh,-gw));
    ps.push(new THREE.Vector3(x+sign*gd,gh,-gw), new THREE.Vector3(x+sign*gd,gh,gw));
    ps.push(new THREE.Vector3(x+sign*gd,gh,gw), new THREE.Vector3(x+sign*gd,0,gw));
    ps.push(new THREE.Vector3(x,0,-gw), new THREE.Vector3(x+sign*gd,0,-gw));
    ps.push(new THREE.Vector3(x,0,gw), new THREE.Vector3(x+sign*gd,0,gw));
    const g = new THREE.BufferGeometry().setFromPoints(ps);
    scene.add(new THREE.LineSegments(g, mat));
}

function createBallMesh() {
    const geo = new THREE.SphereGeometry(0.55, 16, 16);
    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 1.2, roughness: 0.2 });
    ballMesh = new THREE.Mesh(geo, mat);
    ballMesh.visible = false;
    ballMesh.castShadow = true;
    scene.add(ballMesh);

    const glowSpr = new THREE.Sprite(new THREE.SpriteMaterial({
        map: buildGlowTex(0xffffff),
        transparent: true,
        blending: THREE.AdditiveBlending,
    }));
    glowSpr.scale.set(4, 4, 1);
    ballMesh.add(glowSpr);
}

function updateBall(px, py) {
    if (!ballMesh) return;
    ballMesh.visible = true;
    ballMesh.position.set(px - 52.5, 0.55, py - 34);
}

function updatePlayer(id, team, px, py, score, className) {
    const posX = px - 52.5;
    const posZ = py - 34;

    if (className === 'ball') {
        updateBall(px, py);
        return;
    }

    if (!players[id]) {
        let baseColor, geo, yPos;

        if (className === 'goalkeeper') {
            baseColor = team === 0 ? 0xff7700 : (team === 1 ? 0x00aaff : 0xffcc00);
            geo = new THREE.CylinderGeometry(0.7, 1.0, 2.4, 6);
            yPos = 1.2;
        } else if (className === 'referee') {
            baseColor = 0xffff00;
            geo = new THREE.OctahedronGeometry(0.9);
            yPos = 1.4;
        } else {
            baseColor = team === 0 ? 0xff2222 : (team === 1 ? 0x2277ff : 0x999999);
            geo = new THREE.SphereGeometry(0.95, 14, 14);
            yPos = 1.4;
        }

        const mat = new THREE.MeshStandardMaterial({
            color: baseColor,
            emissive: baseColor,
            emissiveIntensity: 0.35,
            roughness: 0.55,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        scene.add(mesh);

        const glowSpr = new THREE.Sprite(new THREE.SpriteMaterial({
            map: buildGlowTex(baseColor),
            transparent: true,
            blending: THREE.AdditiveBlending,
        }));
        glowSpr.scale.set(3, 3, 1);
        mesh.add(glowSpr);

        const label = makeLabel(`#${id}`, className === 'goalkeeper' ? '#ffcc00' : (className === 'referee' ? '#ffff00' : '#ffffff'));
        label.position.set(0, 2.1, 0);
        mesh.add(label);

        players[id] = { mesh, glowSpr, baseColor, yPos };
    }

    const p = players[id];
    p.mesh.position.set(posX, p.yPos, posZ);

    if (score > 0.25) {
        const intensity = Math.min(score * 5, 8);
        const threatHex = score > 0.65 ? 0x00e87b : 0xffaa00;
        p.mesh.material.emissive.setHex(threatHex);
        p.mesh.material.emissiveIntensity = intensity * 0.18 + 0.35;
        p.glowSpr.scale.set(3 + score * 5, 3 + score * 5, 1);
    } else {
        p.mesh.material.emissive.setHex(p.baseColor);
        p.mesh.material.emissiveIntensity = 0.35;
        p.glowSpr.scale.set(3, 3, 1);
    }
}

function makeLabel(text, color) {
    const canvas = document.createElement('canvas');
    canvas.width = 128; canvas.height = 52;
    const ctx = canvas.getContext('2d');
    ctx.font = 'bold 34px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color || '#ffffff';
    ctx.shadowColor = 'rgba(0,0,0,0.8)';
    ctx.shadowBlur = 6;
    ctx.fillText(text, 64, 26);
    const tex = new THREE.CanvasTexture(canvas);
    const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
    spr.scale.set(3.2, 1.3, 1);
    return spr;
}

function buildGlowTex(hexColor) {
    const canvas = document.createElement('canvas');
    canvas.width = 64; canvas.height = 64;
    const ctx = canvas.getContext('2d');
    const hex = hexColor.toString(16).padStart(6, '0');
    const r = parseInt(hex.slice(0,2),16);
    const g = parseInt(hex.slice(2,4),16);
    const b = parseInt(hex.slice(4,6),16);
    const gr = ctx.createRadialGradient(32,32,0, 32,32,32);
    gr.addColorStop(0, `rgba(${r},${g},${b},0.55)`);
    gr.addColorStop(0.4, `rgba(${r},${g},${b},0.18)`);
    gr.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = gr;
    ctx.fillRect(0,0,64,64);
    return new THREE.CanvasTexture(canvas);
}

function clearPlayers() {
    Object.values(players).forEach(p => scene.remove(p.mesh));
    players = {};
    if (ballMesh) ballMesh.visible = false;
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function resetCamera() {
    camera.position.set(0, 75, 85);
    controls.target.set(0, 0, 0);
    controls.update();
}

window.addEventListener('resize', () => {
    if (!renderer) return;
    const W = window.innerWidth;
    const H = window.innerHeight;
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    renderer.setSize(W, H);
});

// Init after full page load so layout is computed
window.addEventListener('load', init3D);
// Fallback: also try DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
    requestAnimationFrame(() => requestAnimationFrame(init3D));
});
