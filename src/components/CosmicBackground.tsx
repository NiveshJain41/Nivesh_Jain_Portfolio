
import React, { useRef, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Points, PointMaterial, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useSpring, animated } from '@react-spring/three';

// Helper function to generate random points in a sphere
function randomPointsInSphere(count: number, radius: number): Float32Array {
  const points = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const r = radius * Math.cbrt(Math.random());
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const idx = i * 3;
    points[idx] = r * Math.sin(phi) * Math.cos(theta);
    points[idx + 1] = r * Math.sin(phi) * Math.sin(theta);
    points[idx + 2] = r * Math.cos(phi);
  }
  return points;
}

// Torch light effect that follows cursor
function TorchLight({ mouse }: { mouse: { x: number, y: number } }) {
  const light = useRef<THREE.PointLight>(null);
  
  useFrame(() => {
    if (light.current) {
      // Position light slightly in front of the camera to follow cursor
      light.current.position.x = mouse.x * 15;
      light.current.position.y = mouse.y * 10;
      light.current.position.z = 10;
    }
  });
  
  return (
    <pointLight
      ref={light}
      color="#ffffff"
      intensity={2}
      distance={25}
      decay={2}
    />
  );
}

// Central Sacred Geometry (Icosahedron)
function SacredGeometry({ mouse, scrollY }: { mouse: { x: number, y: number }, scrollY: number }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const violetLight = useRef<THREE.PointLight>(null);
  const amberLight = useRef<THREE.PointLight>(null);

  // Animate the geometry with mouse position and scroll
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x += 0.003;
      meshRef.current.rotation.y += 0.005;
      
      // Respond to mouse movement
      meshRef.current.rotation.x += (mouse.y * 0.01 - meshRef.current.rotation.x) * 0.1;
      meshRef.current.rotation.y += (mouse.x * 0.01 - meshRef.current.rotation.y) * 0.1;
      
      // Move up based on scroll position
      meshRef.current.position.y = -scrollY * 0.01;
    }
    
    // Move the lights in circular patterns
    if (violetLight.current) {
      const t = state.clock.getElapsedTime();
      violetLight.current.position.x = Math.sin(t * 0.3) * 3;
      violetLight.current.position.z = Math.cos(t * 0.3) * 3;
      violetLight.current.position.y = -scrollY * 0.01;
    }

    if (amberLight.current) {
      const t = state.clock.getElapsedTime();
      amberLight.current.position.x = Math.sin(t * 0.2 + Math.PI) * 4;
      amberLight.current.position.z = Math.cos(t * 0.2 + Math.PI) * 4;
      amberLight.current.position.y = -scrollY * 0.01;
    }
  });

  return (
    <group>
      {/* Violet/Purple Point Light */}
      <pointLight 
        ref={violetLight}
        position={[2, 2, 2]} 
        intensity={20} 
        color="#9b87f5"
        distance={15}
        decay={2}
      />
      
      {/* Amber/Orange Point Light */}
      <pointLight 
        ref={amberLight}
        position={[-2, -2, -2]} 
        intensity={15} 
        color="#FF7F50"
        distance={12}
        decay={2}
      />
      
      {/* Sacred Geometry - Icosahedron */}
      <mesh ref={meshRef}>
        <icosahedronGeometry args={[2, 1]} />
        <meshStandardMaterial 
          color="#ffffff"
          wireframe={true} 
          emissive="#9b87f5"
          emissiveIntensity={0.4}
          roughness={0.5}
          metalness={1}
        />
      </mesh>
    </group>
  );
}

// Galaxy/Cosmic Dust Particle System
function ParticleField({ mouse, scrollY }: { mouse: { x: number, y: number }, scrollY: number }) {
  const particlesRef = useRef<THREE.Points>(null);
  const [positions] = useState(() => randomPointsInSphere(5000, 15));

  useFrame((state) => {
    if (particlesRef.current) {
      // Rotate particles around center
      particlesRef.current.rotation.y += 0.0005;
      
      // Subtle response to mouse movement
      particlesRef.current.rotation.x += (mouse.y * 0.001 - particlesRef.current.rotation.x) * 0.1;
      particlesRef.current.rotation.z += (mouse.x * 0.001 - particlesRef.current.rotation.z) * 0.1;
      
      // Move up based on scroll position
      particlesRef.current.position.y = -scrollY * 0.01;
    }
  });

  return (
    <group>
      <Points ref={particlesRef} positions={positions} stride={3} frustumCulled={false}>
        <PointMaterial
          transparent
          color="#9b87f5"
          size={0.05}
          sizeAttenuation={true}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </Points>
    </group>
  );
}

// Scene setup and mouse tracking
function Scene() {
  const [mouse, setMouse] = useState({ x: 0, y: 0 });
  const [scrollY, setScrollY] = useState(0);
  const { size } = useThree();

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      // Normalize mouse position
      setMouse({
        x: (event.clientX / size.width) * 2 - 1,
        y: -(event.clientY / size.height) * 2 + 1,
      });
    };
    
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('scroll', handleScroll);
    
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('scroll', handleScroll);
    };
  }, [size]);

  return (
    <>
      <ambientLight intensity={0.1} />
      <SacredGeometry mouse={mouse} scrollY={scrollY} />
      <ParticleField mouse={mouse} scrollY={scrollY} />
      <TorchLight mouse={mouse} />
    </>
  );
}

const CosmicBackground = () => {
  return (
    <div className="fixed top-0 left-0 w-full h-full -z-30">
      <Canvas
        dpr={[1, 2]}
        camera={{ position: [0, 0, 15], fov: 60 }}
        gl={{ antialias: true, alpha: true }}
      >
        <Scene />
        <OrbitControls 
          enableZoom={false}
          enablePan={false}
          autoRotate
          autoRotateSpeed={0.5}
          enableDamping
          dampingFactor={0.05}
        />
      </Canvas>
    </div>
  );
};

export default CosmicBackground;
