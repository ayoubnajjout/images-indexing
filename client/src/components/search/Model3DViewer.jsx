/**
 * 3D Model Viewer Component using Three.js
 * Displays .obj models with interactive controls (rotate, zoom, pan)
 */

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const Model3DViewer = ({ 
  modelUrl, 
  width = 300, 
  height = 300,
  backgroundColor = '#f5f5f5',
  autoRotate = true,
  className = ''
}) => {
  const containerRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Three.js refs
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const modelRef = useRef(null);
  const animationIdRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !modelUrl) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(backgroundColor);
    sceneRef.current = scene;

    // Camera setup - close and centered
    const camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 1000);
    camera.position.set(0, 0, 4);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    containerRef.current.innerHTML = '';
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Orbit Controls for mouse interaction - rotate around center
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0); // Rotation center at origin
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = autoRotate;
    controls.autoRotateSpeed = 1.5;
    controls.enableZoom = true;
    controls.enablePan = false; // Disable pan to keep model centered
    controls.minDistance = 2;
    controls.maxDistance = 10;
    controls.update();
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight1.position.set(5, 5, 5);
    directionalLight1.castShadow = true;
    scene.add(directionalLight1);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight2.position.set(-5, -5, -5);
    scene.add(directionalLight2);

    // Add subtle hemisphere light for better ambient
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.3);
    hemiLight.position.set(0, 20, 0);
    scene.add(hemiLight);

    // Create a pivot group for centered rotation
    const pivotGroup = new THREE.Group();
    scene.add(pivotGroup);

    // Load OBJ model
    const loader = new OBJLoader();
    
    loader.load(
      modelUrl,
      (obj) => {
        // Remove previous model if exists
        if (modelRef.current) {
          pivotGroup.remove(modelRef.current);
        }

        // Compute vertex normals if not present (many OBJ files lack normals)
        obj.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            if (child.geometry) {
              child.geometry.computeVertexNormals();
            }
          }
        });

        // Get bounding box BEFORE any transforms
        const box = new THREE.Box3().setFromObject(obj);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        
        // Scale to fit - make model fill most of the view
        const maxDim = Math.max(size.x, size.y, size.z);
        const targetSize = 2.5;
        if (maxDim > 0) {
          const scale = targetSize / maxDim;
          obj.scale.set(scale, scale, scale);
        }
        
        // Recalculate center after scaling
        const scaledBox = new THREE.Box3().setFromObject(obj);
        const scaledCenter = scaledBox.getCenter(new THREE.Vector3());
        
        // Move model so its center is at origin (pivot point)
        obj.position.sub(scaledCenter);

        // Apply material to all meshes with better defaults
        obj.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            child.material = new THREE.MeshStandardMaterial({
              color: 0xccaa88,
              metalness: 0.1,
              roughness: 0.6,
              side: THREE.DoubleSide,
              flatShading: false
            });
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });

        pivotGroup.add(obj);
        modelRef.current = obj;
        setLoading(false);
      },
      (progress) => {
        // Loading progress
        if (progress.lengthComputable) {
          const percent = (progress.loaded / progress.total) * 100;
          console.log(`Loading: ${percent.toFixed(0)}%`);
        }
      },
      (err) => {
        console.error('Error loading model:', err);
        setError('Failed to load 3D model');
        setLoading(false);
      }
    );

    // Animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Cleanup
    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      if (controlsRef.current) {
        controlsRef.current.dispose();
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
      }
    };
  }, [modelUrl, width, height, backgroundColor, autoRotate]);

  // Handle resize
  useEffect(() => {
    if (cameraRef.current && rendererRef.current) {
      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(width, height);
    }
  }, [width, height]);

  if (error) {
    return (
      <div 
        className={`flex items-center justify-center bg-gray-100 rounded ${className}`}
        style={{ width, height }}
      >
        <p className="text-red-500 text-sm text-center px-2">{error}</p>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`} style={{ width, height }}>
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 rounded z-10">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
            <p className="text-xs text-gray-500">Loading 3D model...</p>
          </div>
        </div>
      )}
      <div 
        ref={containerRef} 
        className="rounded overflow-hidden"
        style={{ width, height }}
      />
      {!loading && !error && (
        <div className="absolute bottom-1 left-1 text-xs text-gray-400 bg-white/70 px-1 rounded">
          Drag to rotate • Scroll to zoom
        </div>
      )}
    </div>
  );
};

export default Model3DViewer;