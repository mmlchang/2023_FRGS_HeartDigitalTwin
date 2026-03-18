// https://github.com/simonreisinger/Interactive-3D-Human-Heart-Visualization?tab=readme-ov-file

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader'

const w = window.innerWidth;
const h = window.innerHeight;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff); // green
const camera = new THREE.PerspectiveCamera(75, w / h, 0.1, 1000);
camera.position.z = -6;

const renderer = new THREE.WebGLRenderer();
renderer.setSize(w, h);

document.body.appendChild(renderer.domElement);

// Add Cube
const geometry = new THREE.BoxGeometry();
const material = new THREE.MeshStandardMaterial({ color: 0x00ffbf });
// const cube = new THREE.Mesh(geometry, material);
// scene.add(cube);
// cube.scale.setScalar(2);

const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444);
scene.add(hemiLight);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.enableZoom = true;

//Configure lights
var keyLight = new THREE.DirectionalLight(0xffffff, 1);
keyLight.position.set(-100, 0, 100);

var fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
fillLight.position.set(100, 0, 100);

var backLight = new THREE.DirectionalLight(0xffffff, 0.5);
backLight.position.set(100, 0, -100).normalize();

scene.add(keyLight);
scene.add(fillLight);
scene.add(backLight);

const group = new THREE.Group();
scene.add(group);

//Load model -- CAT
// var objLoader = new OBJLoader();
// let catModel; // Variable to store the loaded model
// objLoader.setPath('assets/');
// objLoader.load('cat.obj', function (object)
// {
//     catModel = object;
//     centerModel(catModel);
//     catModel.scale.setScalar(1/10);
//     group.add(catModel);
//     catModel.position.set(0, 0, 0);

//     catModel.traverse(function (child) {
//         if (child.isMesh) {
//             child.material = new THREE.MeshNormalMaterial();;
//         }
//     });

//     scene.add(catModel);
//     const box = new THREE.Box3().setFromObject(catModel);
//     const helper = new THREE.Box3Helper(box, 0xff0000);
//     scene.add(helper);
// });

// var objLoader2 = new OBJLoader();
// let heartModel; // Variable to store the loaded model
// objLoader2.setPath('assets/');
// objLoader2.load('LateralLV/MM615_BP52010_FMA9563_Lateral.obj', function (object)
// {
//     heartModel = object;
//     centerModel(heartModel);
//     heartModel.scale.setScalar(1);
//     group.add(heartModel);
//     // heartModel.position.set(0, 0, 0);

//     heartModel.traverse(function (child) {
//         if (child.isMesh) {
//             child.material =  material;
//         }
//     });

//     scene.add(heartModel);
//     const box = new THREE.Box3().setFromObject(heartModel);
//     const helper = new THREE.Box3Helper(box, 0xff0000);
//     scene.add(helper);

// });

const loader = new OBJLoader();
fetch('/assets/models.json')
  .then(res => res.json())
  .then(files => {
    files.forEach((file, index) => {
      loader.load(`/assets/${file}`, (obj) => {
        centerModel(obj);
        obj.position.set(index);
        // obj.position.set(0, 0, 0);
        scene.add(obj);
      });
    });
  });

// const box = new THREE.Box3().setFromObject(group);
// const center = box.getCenter(new THREE.Vector3());
// group.position.sub(center); // center group at origin

// function centerModel(model) {
//     const box = new THREE.Box3().setFromObject(model);
//     const center = box.getCenter(new THREE.Vector3());
//     model.position.sub(center); // move geometry center to origin
// }

function centerModel(model) {
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    model.position.sub(center); // centers geometry

    const size = box.getSize(new THREE.Vector3());
    const maxAxis = Math.max(size.x, size.y, size.z);
    if (maxAxis > 0) model.scale.multiplyScalar(1 / maxAxis); // optional scale
}


function animate() {
  requestAnimationFrame(animate);
//   cube.rotation.x += 0.01;
//   cube.rotation.y += 0.01;


    // if (catModel) {
    //     // catModel.rotation.x += 0.01;
    //     // catModel.rotation.y += 0.01;
    //     // catModel.rotation.z += 0.01; // slowly rotate the model
    // }

    // if (heartModel) {
    //     // heartModel.rotation.x += 0.01;
    //     // heartModel.rotation.y += 0.01;
    //     // heartModel.rotation.z += 0.01; // slowly rotate the model
    // }
  controls.update();
  renderer.render(scene, camera);
}

animate();


