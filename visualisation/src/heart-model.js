import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';

/**
 * Load and display heart model with animated colors.
 * @param {HTMLElement} div - container div
 * @param {Array} strainData - array of objects with Class, File, Color (string or array)
 */
export function loadHeartModel(div, strainData) {
  const w = div.clientWidth;
  const h = div.clientHeight;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  const camera = new THREE.PerspectiveCamera(75, w / h, 0.1, 1000);
  camera.position.z = -6;

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(w, h);
  div.appendChild(renderer.domElement);

  // LIGHTS
  const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444);
  scene.add(hemiLight);

  const keyLight = new THREE.DirectionalLight(0xffffff, 1);
  keyLight.position.set(-100, 0, 100);

  const fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
  fillLight.position.set(100, 0, 100);

  const backLight = new THREE.DirectionalLight(0xffffff, 0.5);
  backLight.position.set(100, 0, -100).normalize();

  scene.add(keyLight, fillLight, backLight);

  // CONTROLS
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.enableZoom = true;

  // ORIGIN MARKER
  // const pivotMarker = new THREE.Mesh(
  //   new THREE.SphereGeometry(0.1, 16, 16),
  //   new THREE.MeshBasicMaterial({ color: 0xffffff })
  // );
  // pivotMarker.position.set(0, 0, 0);
  // scene.add(pivotMarker);

  // PIVOT & GROUP
  const pivot = new THREE.Object3D();
  scene.add(pivot);

  const group = new THREE.Group();
  scene.add(group);

  const loader = new OBJLoader();
  const animatedMeshes = []; // store meshes for color animation
  let loadCount = 0;

  // LOAD OBJ FILES
  strainData.forEach((item) => {
    loader.load(`/assets/${item.Class}/${item.File}`, (obj) => {
      obj.scale.setScalar(1 / 15);

      obj.traverse((child) => {
        if (child.isMesh) {
          // Prepare color array
          const colors = Array.isArray(item.ColorArray)
            ? item.ColorArray.map(c => new THREE.Color(c))
            : [new THREE.Color(item.Color)];

          // Initialize material with first color (can use MeshToonMaterial or MeshBasicMaterial)
          child.material = new THREE.MeshBasicMaterial({ color: colors[0] });

          // Save mesh for animation
          animatedMeshes.push({ mesh: child, colors, colorIndex: 0, nextIndex: 1, progress: 0 });
        }
      });

      group.add(obj);
      loadCount++;

      if (loadCount === strainData.length) {
        centerItem(group);
        pivot.add(group);
      }
    });
  });

  // CENTER FUNCTION
  function centerItem(model) {
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    model.position.sub(center);
  }

  // ANIMATION LOOP
  function animate() {
    requestAnimationFrame(animate);

    pivot.rotation.x += 0.001;
    pivot.rotation.y += 0.005;
    pivot.rotation.z += 0.0001;

    // Animate colors
    animatedMeshes.forEach((a) => {
      const len = a.colors.length;
      if (len <= 0) return;

      if (len === 1) {
        a.mesh.material.color.copy(a.colors[0]);
        a.mesh.material.needsUpdate = true;
        return;
      }

      // multi-color interpolation
      a.progress += 0.005; // speed
      if (a.progress > 1) {
        a.progress = 0;
        a.colorIndex = a.nextIndex;
        a.nextIndex = (a.nextIndex + 1) % len;
      }

      const c1 = a.colors[a.colorIndex];
      const c2 = a.colors[a.nextIndex];
      if (!c1 || !c2) return;

      a.mesh.material.color.r = c1.r + (c2.r - c1.r) * a.progress;
      a.mesh.material.color.g = c1.g + (c2.g - c1.g) * a.progress;
      a.mesh.material.color.b = c1.b + (c2.b - c1.b) * a.progress;

      a.mesh.material.needsUpdate = true; // required for color change
    });

    controls.update();
    renderer.render(scene, camera);
  }

  animate();

  // RESIZE HANDLER
  window.addEventListener('resize', () => {
    camera.aspect = div.clientWidth / div.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(div.clientWidth, div.clientHeight);
  });
}


