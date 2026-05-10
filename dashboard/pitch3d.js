let scene, camera, renderer, controls, pitch;
let players = {};
let ballMesh = null;

function init3D() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050505);
    scene.fog = new THREE.FogExp2(0x050505, 0.004);

    const container = document.getElementById('threejs-canvas');
    const w = container.clientWidth;
    const h = container.clientHeight;

    camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000);
    camera.position.set(0, 80, 90);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxPolarAngle = Math.PI / 2.1;
    controls.minDistance = 20;
    controls.maxDistance = 200;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight.position.set(30, 80, 40);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    scene.add(dirLight);

    const spotLight = new THREE.SpotLight(0x00e87b, 0.4);
    spotLight.position.set(0, 120, 0);
    spotLight.angle = Math.PI / 3;
    scene.add(spotLight);

    createPitch();
    createBall();
    animate();
}

function createPitch() {
    const geometry = new THREE.PlaneGeometry(105, 68);
    const material = new THREE.MeshStandardMaterial({
        color: 0x0d1a0d,
        side: THREE.DoubleSide,
        roughness: 0.9,
        metalness: 0.0
    });
    pitch = new THREE.Mesh(geometry, material);
    pitch.rotation.x = -Math.PI / 2;
    pitch.receiveShadow = true;
    scene.add(pitch);

    const lineMat = new THREE.LineBasicMaterial({ color: 0x00e87b, transparent: true, opacity: 0.35 });
    const pts = [];

    // Boundary
    pts.push(new THREE.Vector3(-52.5, 0.05, -34), new THREE.Vector3(52.5, 0.05, -34));
    pts.push(new THREE.Vector3(52.5, 0.05, -34), new THREE.Vector3(52.5, 0.05, 34));
    pts.push(new THREE.Vector3(52.5, 0.05, 34), new THREE.Vector3(-52.5, 0.05, 34));
    pts.push(new THREE.Vector3(-52.5, 0.05, 34), new THREE.Vector3(-52.5, 0.05, -34));

    // Halfway
    pts.push(new THREE.Vector3(0, 0.05, -34), new THREE.Vector3(0, 0.05, 34));

    // Penalty areas (16.5m from goal line, 40.32m wide)
    const paW = 20.16, paD = 16.5;
    pts.push(new THREE.Vector3(-52.5, 0.05, -paW), new THREE.Vector3(-52.5 + paD, 0.05, -paW));
    pts.push(new THREE.Vector3(-52.5 + paD, 0.05, -paW), new THREE.Vector3(-52.5 + paD, 0.05, paW));
    pts.push(new THREE.Vector3(-52.5 + paD, 0.05, paW), new THREE.Vector3(-52.5, 0.05, paW));

    pts.push(new THREE.Vector3(52.5, 0.05, -paW), new THREE.Vector3(52.5 - paD, 0.05, -paW));
    pts.push(new THREE.Vector3(52.5 - paD, 0.05, -paW), new THREE.Vector3(52.5 - paD, 0.05, paW));
    pts.push(new THREE.Vector3(52.5 - paD, 0.05, paW), new THREE.Vector3(52.5, 0.05, paW));

    // Goal areas (5.5m from goal line, 18.32m wide)
    const gaW = 9.16, gaD = 5.5;
    pts.push(new THREE.Vector3(-52.5, 0.05, -gaW), new THREE.Vector3(-52.5 + gaD, 0.05, -gaW));
    pts.push(new THREE.Vector3(-52.5 + gaD, 0.05, -gaW), new THREE.Vector3(-52.5 + gaD, 0.05, gaW));
    pts.push(new THREE.Vector3(-52.5 + gaD, 0.05, gaW), new THREE.Vector3(-52.5, 0.05, gaW));

    pts.push(new THREE.Vector3(52.5, 0.05, -gaW), new THREE.Vector3(52.5 - gaD, 0.05, -gaW));
    pts.push(new THREE.Vector3(52.5 - gaD, 0.05, -gaW), new THREE.Vector3(52.5 - gaD, 0.05, gaW));
    pts.push(new THREE.Vector3(52.5 - gaD, 0.05, gaW), new THREE.Vector3(52.5, 0.05, gaW));

    const lineGeom = new THREE.BufferGeometry().setFromPoints(pts);
    scene.add(new THREE.LineSegments(lineGeom, lineMat));

    // Center circle
    const circleGeom = new THREE.RingGeometry(9.1, 9.25, 64);
    const circleMat = new THREE.MeshBasicMaterial({ color: 0x00e87b, transparent: true, opacity: 0.35, side: THREE.DoubleSide });
    const circle = new THREE.Mesh(circleGeom, circleMat);
    circle.rotation.x = -Math.PI / 2;
    circle.position.y = 0.06;
    scene.add(circle);

    // Center spot
    const spotGeom = new THREE.CircleGeometry(0.3, 16);
    const spot = new THREE.Mesh(spotGeom, circleMat);
    spot.rotation.x = -Math.PI / 2;
    spot.position.y = 0.06;
    scene.add(spot);

    // Goal nets (simple box outlines)
    const goalMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25 });
    createGoalNet(-52.5, goalMat);
    createGoalNet(52.5, goalMat);
}

function createGoalNet(x, mat) {
    const w = 3.66, h = 2.44, d = 1.5;
    const sign = x < 0 ? -1 : 1;
    const pts = [];
    const bx = x;
    const ex = x + sign * -d;

    pts.push(new THREE.Vector3(bx, 0, -w), new THREE.Vector3(bx, h, -w));
    pts.push(new THREE.Vector3(bx, h, -w), new THREE.Vector3(bx, h, w));
    pts.push(new THREE.Vector3(bx, h, w), new THREE.Vector3(bx, 0, w));
    pts.push(new THREE.Vector3(bx, h, -w), new THREE.Vector3(ex, h, -w));
    pts.push(new THREE.Vector3(bx, h, w), new THREE.Vector3(ex, h, w));
    pts.push(new THREE.Vector3(ex, 0, -w), new THREE.Vector3(ex, h, -w));
    pts.push(new THREE.Vector3(ex, h, -w), new THREE.Vector3(ex, h, w));
    pts.push(new THREE.Vector3(ex, h, w), new THREE.Vector3(ex, 0, w));
    pts.push(new THREE.Vector3(bx, 0, -w), new THREE.Vector3(ex, 0, -w));
    pts.push(new THREE.Vector3(bx, 0, w), new THREE.Vector3(ex, 0, w));

    const geom = new THREE.BufferGeometry().setFromPoints(pts);
    scene.add(new THREE.LineSegments(geom, mat));
}

function createBall() {
    const geom = new THREE.SphereGeometry(0.5, 16, 16);
    const mat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        emissive: 0xffffff,
        emissiveIntensity: 0.8,
        roughness: 0.3
    });
    ballMesh = new THREE.Mesh(geom, mat);
    ballMesh.visible = false;
    ballMesh.castShadow = true;
    scene.add(ballMesh);

    const glowMat = new THREE.SpriteMaterial({
        map: createGlowTexture(0xffffff),
        transparent: true,
        blending: THREE.AdditiveBlending
    });
    const glowSprite = new THREE.Sprite(glowMat);
    glowSprite.scale.set(3, 3, 1);
    ballMesh.add(glowSprite);
}

function updateBall(x, y) {
    if (!ballMesh) return;
    ballMesh.visible = true;
    ballMesh.position.set(x - 52.5, 0.5, y - 34);
}

function updatePlayer(id, team, x, y, score, className) {
    const posX = x - 52.5;
    const posZ = y - 34;

    if (className === 'ball') {
        updateBall(x, y);
        return;
    }

    if (!players[id]) {
        let baseColor, geometry, yPos;

        if (className === 'goalkeeper') {
            baseColor = team === 0 ? 0xff6600 : (team === 1 ? 0x00ccff : 0xffcc00);
            geometry = new THREE.CylinderGeometry(0.8, 1.2, 2.5, 6);
            yPos = 1.25;
        } else if (className === 'referee') {
            baseColor = 0xffff00;
            geometry = new THREE.OctahedronGeometry(1.0);
            yPos = 1.5;
        } else {
            baseColor = team === 0 ? 0xff3333 : (team === 1 ? 0x3388ff : 0x888888);
            geometry = new THREE.SphereGeometry(1.0, 16, 16);
            yPos = 1.5;
        }

        const material = new THREE.MeshStandardMaterial({
            color: baseColor,
            emissive: baseColor,
            emissiveIntensity: 0.4,
            roughness: 0.5
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        scene.add(mesh);

        const glowMat = new THREE.SpriteMaterial({
            map: createGlowTexture(baseColor),
            transparent: true,
            blending: THREE.AdditiveBlending
        });
        const glow = new THREE.Sprite(glowMat);
        glow.scale.set(3.5, 3.5, 1);
        mesh.add(glow);

        // ID label
        const label = createTextSprite(`#${id}`, className === 'goalkeeper' ? '#ffcc00' : '#ffffff');
        label.position.set(0, 2.2, 0);
        mesh.add(label);

        players[id] = { mesh, glow, baseColor, yPos, className: className || 'player' };
    }

    const p = players[id];
    p.mesh.position.set(posX, p.yPos, posZ);

    if (score > 0.3) {
        const intensity = Math.min(score * 4.0, 6.0);
        const threatColor = score > 0.7 ? 0x00e87b : 0xffaa00;
        p.mesh.material.emissive.setHex(threatColor);
        p.mesh.material.emissiveIntensity = intensity;
        p.glow.scale.set(3.5 + intensity * 0.8, 3.5 + intensity * 0.8, 1);
    } else {
        p.mesh.material.emissive.setHex(p.baseColor);
        p.mesh.material.emissiveIntensity = 0.4;
        p.glow.scale.set(3.5, 3.5, 1);
    }
}

function createTextSprite(text, color) {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.font = 'bold 36px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color || '#ffffff';
    ctx.fillText(text, 64, 32);
    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(3, 1.5, 1);
    return sprite;
}

function createGlowTexture(color) {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    const hex = color.toString(16).padStart(6, '0');
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, `rgba(${r},${g},${b},0.6)`);
    gradient.addColorStop(0.4, `rgba(${r},${g},${b},0.2)`);
    gradient.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(canvas);
}

function clearPlayers() {
    Object.keys(players).forEach(id => {
        scene.remove(players[id].mesh);
    });
    players = {};
    if (ballMesh) ballMesh.visible = false;
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function resetCamera() {
    camera.position.set(0, 80, 90);
    controls.target.set(0, 0, 0);
    controls.update();
}

window.addEventListener('resize', () => {
    const container = document.getElementById('threejs-canvas');
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
});

init3D();
