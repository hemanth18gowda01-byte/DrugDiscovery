import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.172.0/build/three.module.js";

const canvas = document.querySelector("#dna-canvas");
if (canvas) {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: true,
    powerPreference: "low-power",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.7));
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
  camera.position.set(0, 0, 19);

  scene.add(new THREE.AmbientLight(0x60708c, 1.3));
  const mintLight = new THREE.PointLight(0x3d75bb, 32, 28);
  mintLight.position.set(-4, 4, 7);
  scene.add(mintLight);
  const amberLight = new THREE.PointLight(0x758095, 16, 24);
  amberLight.position.set(5, -2, -3);
  scene.add(amberLight);

  const helix = new THREE.Group();
  scene.add(helix);
  const bioObjects = new THREE.Group();
  scene.add(bioObjects);

  const wireMaterial = new THREE.MeshBasicMaterial({
    color: 0x4a88d0,
    wireframe: true,
    transparent: true,
    opacity: 0.18,
  });
  const ringMaterial = new THREE.LineBasicMaterial({
    color: 0x5f93d5,
    transparent: true,
    opacity: 0.24,
  });
  const makeWireFigure = (scale, opacity) => {
    const figure = new THREE.Group();
    const material = wireMaterial.clone();
    material.opacity = opacity;
    const head = new THREE.Mesh(new THREE.IcosahedronGeometry(0.72, 3), material);
    head.position.y = 2.2;
    const torso = new THREE.Mesh(new THREE.CylinderGeometry(0.62, 0.9, 1.9, 14, 5, true), material);
    torso.position.y = 0.72;
    const hips = new THREE.Mesh(new THREE.CylinderGeometry(0.72, 0.53, 0.52, 12, 2, true), material);
    hips.position.y = -0.52;
    figure.add(head, torso, hips);
    for (const [x, angle] of [[-0.88, -0.35], [0.88, 0.35]]) {
      const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.1, 1.95, 8, 2, true), material);
      arm.position.set(x, 0.74, 0);
      arm.rotation.z = angle;
      figure.add(arm);
    }
    for (const x of [-0.34, 0.34]) {
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.13, 2.2, 8, 3, true), material);
      leg.position.set(x, -1.78, 0);
      leg.rotation.z = x * 0.14;
      figure.add(leg);
    }
    figure.scale.setScalar(scale);
    return figure;
  };
  const leftFigure = makeWireFigure(1.08, 0.2);
  leftFigure.position.set(-7.2, -1.4, -5.5);
  leftFigure.rotation.set(0.12, -0.5, -0.1);
  const farFigure = makeWireFigure(0.78, 0.13);
  farFigure.position.set(6.8, 0.5, -7.6);
  farFigure.rotation.set(0, 0.65, 0.06);
  bioObjects.add(leftFigure, farFigure);

  const orbitalRings = new THREE.Group();
  for (let index = 0; index < 4; index += 1) {
    const curve = new THREE.EllipseCurve(0, 0, 4.1 + index * 0.46, 1.55 + index * 0.17, 0, Math.PI * 2, false, 0);
    const points = curve.getPoints(90).map((point) => new THREE.Vector3(point.x, point.y, 0));
    const ring = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), ringMaterial);
    ring.rotation.set(index * 0.45, index * 0.65, index * 0.8);
    orbitalRings.add(ring);
  }
  orbitalRings.position.set(-5.6, 0.1, -4.7);
  bioObjects.add(orbitalRings);

  const molecularField = new THREE.Group();
  const moleculeMaterial = new THREE.MeshStandardMaterial({
    color: 0x4d8fd9,
    emissive: 0x173e71,
    emissiveIntensity: 1.1,
    roughness: 0.3,
    metalness: 0.45,
    transparent: true,
    opacity: 0.52,
  });
  const moleculeWire = new THREE.MeshBasicMaterial({ color: 0x9abce8, wireframe: true, transparent: true, opacity: 0.22 });
  const sphereGeometry = new THREE.SphereGeometry(0.17, 12, 10);
  for (let clusterIndex = 0; clusterIndex < 15; clusterIndex += 1) {
    const cluster = new THREE.Group();
    for (let atomIndex = 0; atomIndex < 5; atomIndex += 1) {
      const atom = new THREE.Mesh(sphereGeometry, atomIndex % 2 ? moleculeMaterial : moleculeWire);
      const angle = (atomIndex / 5) * Math.PI * 2;
      atom.position.set(Math.cos(angle) * 0.48, Math.sin(angle) * 0.36, (atomIndex % 2) * 0.24 - 0.12);
      cluster.add(atom);
    }
    cluster.position.set((Math.random() - 0.5) * 20, (Math.random() - 0.5) * 15, -3 - Math.random() * 7);
    cluster.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
    cluster.userData.speed = 0.08 + Math.random() * 0.12;
    molecularField.add(cluster);
  }
  bioObjects.add(molecularField);

  const grid = new THREE.GridHelper(26, 24, 0x28558c, 0x183150);
  grid.position.set(0, -7.2, -5);
  grid.material.transparent = true;
  grid.material.opacity = 0.18;
  grid.rotation.x = Math.PI * 0.04;
  scene.add(grid);
  const turns = 5.25;
  const strandLength = 14.8;
  const radius = 2.18;
  const segmentCount = 430;
  const pointA = [];
  const pointB = [];
  const basePairs = [];
  for (let index = 0; index <= segmentCount; index += 1) {
    const t = index / segmentCount;
    const angle = t * Math.PI * 2 * turns;
    const y = (t - 0.5) * strandLength;
    const phase = t * Math.PI * 2 * turns;
    pointA.push(new THREE.Vector3(radius * Math.cos(phase), y, radius * Math.sin(phase)));
    pointB.push(new THREE.Vector3(radius * Math.cos(phase + Math.PI), y, radius * Math.sin(phase + Math.PI)));
    if (index % 12 === 0 && index < segmentCount) {
      basePairs.push({ first: pointA[index], second: pointB[index], phase: angle });
    }
  }

  const strandMaterialA = new THREE.MeshStandardMaterial({
    color: 0x72a7ee,
    emissive: 0x123c75,
    emissiveIntensity: 1.05,
    roughness: 0.28,
    metalness: 0.28,
    transparent: true,
    opacity: 0.86,
  });
  const strandMaterialB = new THREE.MeshStandardMaterial({
    color: 0x53647d,
    emissive: 0x16233b,
    emissiveIntensity: 1.1,
    roughness: 0.31,
    metalness: 0.34,
    transparent: true,
    opacity: 0.8,
  });
  const curveA = new THREE.CatmullRomCurve3(pointA);
  const curveB = new THREE.CatmullRomCurve3(pointB);
  helix.add(new THREE.Mesh(new THREE.TubeGeometry(curveA, 520, 0.052, 8, false), strandMaterialA));
  helix.add(new THREE.Mesh(new THREE.TubeGeometry(curveB, 520, 0.052, 8, false), strandMaterialB));

  const rungMaterial = new THREE.MeshStandardMaterial({
    color: 0xaeb7c5,
    emissive: 0x303a4b,
    emissiveIntensity: 0.5,
    metalness: 0.52,
    roughness: 0.3,
    transparent: true,
    opacity: 0.63,
  });
  const rungGeometry = new THREE.CylinderGeometry(0.018, 0.018, 1, 6, 1);
  const rungMeshes = [];
  for (const base of basePairs) {
    const rung = new THREE.Mesh(rungGeometry, rungMaterial);
    rung.position.copy(base.first).add(base.second).multiplyScalar(0.5);
    rung.scale.y = base.first.distanceTo(base.second);
    rung.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      base.second.clone().sub(base.first).normalize(),
    );
    helix.add(rung);
    rungMeshes.push(rung);
  }

  const nodeGeometry = new THREE.SphereGeometry(0.095, 12, 10);
  const nodeA = new THREE.MeshStandardMaterial({
    color: 0xc7dbfc,
    emissive: 0x2f6dbf,
    emissiveIntensity: 1.7,
    roughness: 0.19,
    metalness: 0.1,
  });
  const nodeB = new THREE.MeshStandardMaterial({
    color: 0x9ca8ba,
    emissive: 0x40506b,
    emissiveIntensity: 1.2,
    roughness: 0.25,
    metalness: 0.16,
  });
  for (let index = 0; index < pointA.length; index += 16) {
    const a = new THREE.Mesh(nodeGeometry, index % 2 === 0 ? nodeA : strandMaterialA);
    const b = new THREE.Mesh(nodeGeometry, index % 2 === 0 ? nodeB : strandMaterialB);
    a.position.copy(pointA[index]);
    b.position.copy(pointB[index]);
    helix.add(a, b);
  }

  const dustCount = 250;
  const dustPositions = new Float32Array(dustCount * 3);
  for (let index = 0; index < dustCount; index += 1) {
    dustPositions[index * 3] = (Math.random() - 0.5) * 23;
    dustPositions[index * 3 + 1] = (Math.random() - 0.5) * 20;
    dustPositions[index * 3 + 2] = (Math.random() - 0.5) * 8 - 2;
  }
  const dustGeometry = new THREE.BufferGeometry();
  dustGeometry.setAttribute("position", new THREE.BufferAttribute(dustPositions, 3));
  const dust = new THREE.Points(
    dustGeometry,
    new THREE.PointsMaterial({
      color: 0x6d87aa,
      size: 0.025,
      transparent: true,
      opacity: 0.54,
      sizeAttenuation: true,
    }),
  );
  scene.add(dust);

  let pointerX = 0;
  let pointerY = 0;
  let targetX = 0;
  let targetY = 0;
  let frame = 0;
  const clock = new THREE.Clock();
  const resize = () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.position.z = width < 640 ? 24 : width < 1000 ? 21 : 19;
    camera.updateProjectionMatrix();
    helix.scale.setScalar(width < 640 ? 0.77 : width < 900 ? 0.9 : 1);
    helix.position.x = width < 640 ? 2.5 : width < 900 ? 2.4 : 3.1;
    helix.position.y = -0.3;
  };
  const onPointerMove = (event) => {
    targetX = (event.clientX / window.innerWidth - 0.5) * 0.72;
    targetY = (event.clientY / window.innerHeight - 0.5) * 0.38;
  };
  const animate = () => {
    frame = window.requestAnimationFrame(animate);
    const elapsed = clock.getElapsedTime();
    const follow = reducedMotion ? 0.025 : 0.035;
    pointerX += (targetX - pointerX) * follow;
    pointerY += (targetY - pointerY) * follow;
    helix.rotation.y = elapsed * (reducedMotion ? 0.025 : 0.075) + pointerX;
    helix.rotation.x = 0.08 + pointerY;
    helix.rotation.z = Math.sin(elapsed * 0.19) * 0.025;
    dust.rotation.y = elapsed * 0.009;
    bioObjects.rotation.y = elapsed * (reducedMotion ? 0.008 : 0.022);
    leftFigure.rotation.y = -0.5 + Math.sin(elapsed * 0.21) * 0.18;
    farFigure.rotation.y = 0.65 - Math.sin(elapsed * 0.16) * 0.22;
    orbitalRings.rotation.z = elapsed * (reducedMotion ? 0.01 : 0.045);
    molecularField.children.forEach((cluster, index) => {
      cluster.rotation.y += reducedMotion ? 0.001 : cluster.userData.speed * 0.008;
      cluster.position.y += Math.sin(elapsed * 0.35 + index) * 0.0008;
    });
    grid.position.z = -5 + Math.sin(elapsed * 0.12) * 0.2;
    renderer.render(scene, camera);
  };

  resize();
  window.addEventListener("resize", resize, { passive: true });
  window.addEventListener("pointermove", onPointerMove, { passive: true });
  animate();

  window.addEventListener("pagehide", () => {
    window.cancelAnimationFrame(frame);
    window.removeEventListener("resize", resize);
    window.removeEventListener("pointermove", onPointerMove);
    renderer.dispose();
    dustGeometry.dispose();
    rungGeometry.dispose();
    strandMaterialA.dispose();
    strandMaterialB.dispose();
    rungMaterial.dispose();
    nodeGeometry.dispose();
    nodeA.dispose();
    nodeB.dispose();
    wireMaterial.dispose();
    ringMaterial.dispose();
    moleculeMaterial.dispose();
    moleculeWire.dispose();
    sphereGeometry.dispose();
  }, { once: true });
}
