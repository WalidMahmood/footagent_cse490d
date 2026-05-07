let scene, camera, renderer, controls, pitch;
let players = {};
let wireframeMode = false;

function init3D() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050505);
    
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 70, 100);
    
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.getElementById('threejs-canvas').appendChild(renderer.domElement);
    
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);
    
    const spotLight = new THREE.SpotLight(0x00ff88, 1);
    spotLight.position.set(0, 100, 0);
    spotLight.castShadow = true;
    scene.add(spotLight);

    createPitch();
    animate();
}

function createPitch() {
    // International Pitch Dimensions (scaled): 105m x 68m
    const geometry = new THREE.PlaneGeometry(105, 68);
    const material = new THREE.MeshStandardMaterial({ 
        color: 0x111111, 
        side: THREE.DoubleSide,
        roughness: 0.8
    });
    pitch = new THREE.Mesh(geometry, material);
    pitch.rotation.x = -Math.PI / 2;
    scene.add(pitch);

    // Pitch Lines (using LineSegments)
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0x00ff88, transparent: true, opacity: 0.5 });
    
    const points = [];
    // Boundary
    points.push(new THREE.Vector3(-52.5, 0.1, -34), new THREE.Vector3(52.5, 0.1, -34));
    points.push(new THREE.Vector3(52.5, 0.1, -34), new THREE.Vector3(52.5, 0.1, 34));
    points.push(new THREE.Vector3(52.5, 0.1, 34), new THREE.Vector3(-52.5, 0.1, 34));
    points.push(new THREE.Vector3(-52.5, 0.1, 34), new THREE.Vector3(-52.5, 0.1, -34));
    
    // Halfway line
    points.push(new THREE.Vector3(0, 0.1, -34), new THREE.Vector3(0, 0.1, 34));

    const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
    const pitchLines = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(pitchLines);
    
    // Center Circle
    const circleGeom = new THREE.RingGeometry(9.15, 9.25, 64);
    const circle = new THREE.Mesh(circleGeom, lineMaterial);
    circle.rotation.x = -Math.PI / 2;
    circle.position.y = 0.11;
    scene.add(circle);
}

function updatePlayer(id, team, x, y, score = 0) {
    // Map normalized pitch coords (-52.5 to 52.5, -34 to 34)
    const posX = x - 52.5;
    const posZ = y - 34;

    if (!players[id]) {
        const baseColor = team === 0 ? 0xff3300 : (team === 1 ? 0x0088ff : 0xaaaaaa);
        const geometry = new THREE.SphereGeometry(1.2, 16, 16);
        const material = new THREE.MeshStandardMaterial({ 
            color: baseColor, 
            emissive: baseColor,
            emissiveIntensity: 0.5
        });
        const sphere = new THREE.Mesh(geometry, material);
        scene.add(sphere);
        
        // Add Glow
        const spriteMaterial = new THREE.SpriteMaterial({
            map: createGlowTexture(baseColor),
            transparent: true,
            blending: THREE.AdditiveBlending
        });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(4, 4, 1);
        sphere.add(sprite);
        
        players[id] = { mesh: sphere, sprite: sprite, baseColor: baseColor };
    }

    const p = players[id];
    p.mesh.position.set(posX, 1.5, posZ);
    
    // Dynamic Threat Highlighting
    // If score is high (>0.5), make it glow orange/yellow
    if (score > 0.1) {
        const threatIntensity = Math.min(score * 5.0, 10.0);
        p.mesh.material.emissive.setHex(0xffaa00); // Threat color
        p.mesh.material.emissiveIntensity = threatIntensity;
        p.sprite.scale.set(4 + threatIntensity, 4 + threatIntensity, 1);
    } else {
        p.mesh.material.emissive.setHex(p.baseColor);
        p.mesh.material.emissiveIntensity = 0.5;
        p.sprite.scale.set(4, 4, 1);
    }
}

function createGlowTexture(color) {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, 'rgba(255,255,255,1)');
    gradient.addColorStop(0.2, `rgba(${parseInt(color.toString(16).slice(0,2),16)}, ${parseInt(color.toString(16).slice(2,4),16)}, ${parseInt(color.toString(16).slice(4,6),16)}, 0.5)`);
    gradient.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(canvas);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function resetCamera() {
    camera.position.set(0, 70, 100);
    controls.reset();
}

function toggleWireframe() {
    wireframeMode = !wireframeMode;
    scene.traverse((obj) => {
        if (obj.material) obj.material.wireframe = wireframeMode;
    });
}

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

init3D();
