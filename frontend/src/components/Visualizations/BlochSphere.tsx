import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, Text, Sphere } from '@react-three/drei';
import * as THREE from 'three';
import { useCircuitStore } from '../../stores/circuitStore';

interface BlochStateProps {
  x: number;
  y: number;
  z: number;
  color?: string;
}

function BlochState({ x, y, z, color = '#0c8ee6' }: BlochStateProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const lineRef = useRef<THREE.Line>(null);

  // Animate to new position
  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.position.lerp(new THREE.Vector3(x, z, y), 0.1);
    }
  });

  const arrowPoints = useMemo(() => {
    return [new THREE.Vector3(0, 0, 0), new THREE.Vector3(x, z, y)];
  }, [x, y, z]);

  return (
    <group>
      {/* State arrow */}
      <Line
        points={arrowPoints}
        color={color}
        lineWidth={3}
      />
      {/* State point */}
      <mesh ref={meshRef} position={[x, z, y]}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} />
      </mesh>
    </group>
  );
}

function BlochSphereGeometry() {
  const { blochVectors, selectedQubit, nQubits } = useCircuitStore();

  // Get the vector to display (selected qubit or first qubit)
  const displayQubit = selectedQubit ?? 0;
  const vector = blochVectors[displayQubit] || { x: 0, y: 0, z: 1 };

  // Create sphere wireframe
  const spherePoints = useMemo(() => {
    const points: THREE.Vector3[] = [];
    const segments = 32;

    // Latitude circles
    for (let lat = 0; lat <= 4; lat++) {
      const theta = (lat / 4) * Math.PI;
      for (let lon = 0; lon <= segments; lon++) {
        const phi = (lon / segments) * 2 * Math.PI;
        const x = Math.sin(theta) * Math.cos(phi);
        const y = Math.sin(theta) * Math.sin(phi);
        const z = Math.cos(theta);
        points.push(new THREE.Vector3(x, z, y));
      }
    }

    return points;
  }, []);

  return (
    <group>
      {/* Transparent sphere */}
      <Sphere args={[1, 32, 32]}>
        <meshStandardMaterial
          color="#1e293b"
          transparent
          opacity={0.1}
          side={THREE.DoubleSide}
        />
      </Sphere>

      {/* Wireframe circles */}
      <group>
        {/* Equator */}
        <Line
          points={Array.from({ length: 65 }, (_, i) => {
            const theta = (i / 64) * 2 * Math.PI;
            return new THREE.Vector3(Math.cos(theta), 0, Math.sin(theta));
          })}
          color="#475569"
          lineWidth={1}
        />

        {/* Vertical circle XZ */}
        <Line
          points={Array.from({ length: 65 }, (_, i) => {
            const theta = (i / 64) * 2 * Math.PI;
            return new THREE.Vector3(Math.cos(theta), Math.sin(theta), 0);
          })}
          color="#475569"
          lineWidth={1}
        />

        {/* Vertical circle YZ */}
        <Line
          points={Array.from({ length: 65 }, (_, i) => {
            const theta = (i / 64) * 2 * Math.PI;
            return new THREE.Vector3(0, Math.sin(theta), Math.cos(theta));
          })}
          color="#475569"
          lineWidth={1}
        />
      </group>

      {/* Axes */}
      <group>
        {/* X axis */}
        <Line
          points={[new THREE.Vector3(-1.3, 0, 0), new THREE.Vector3(1.3, 0, 0)]}
          color="#ef4444"
          lineWidth={2}
        />
        <Text
          position={[1.5, 0, 0]}
          fontSize={0.15}
          color="#ef4444"
        >
          X
        </Text>

        {/* Y axis */}
        <Line
          points={[new THREE.Vector3(0, 0, -1.3), new THREE.Vector3(0, 0, 1.3)]}
          color="#22c55e"
          lineWidth={2}
        />
        <Text
          position={[0, 0, 1.5]}
          fontSize={0.15}
          color="#22c55e"
        >
          Y
        </Text>

        {/* Z axis */}
        <Line
          points={[new THREE.Vector3(0, -1.3, 0), new THREE.Vector3(0, 1.3, 0)]}
          color="#3b82f6"
          lineWidth={2}
        />
        <Text
          position={[0, 1.5, 0]}
          fontSize={0.15}
          color="#3b82f6"
        >
          |0⟩
        </Text>
        <Text
          position={[0, -1.5, 0]}
          fontSize={0.15}
          color="#3b82f6"
        >
          |1⟩
        </Text>
      </group>

      {/* State vectors */}
      <BlochState x={vector.x} y={vector.y} z={vector.z} />

      {/* Reference states */}
      <group>
        {/* |0⟩ - North pole */}
        <mesh position={[0, 1, 0]}>
          <sphereGeometry args={[0.04, 8, 8]} />
          <meshStandardMaterial color="#64748b" />
        </mesh>

        {/* |1⟩ - South pole */}
        <mesh position={[0, -1, 0]}>
          <sphereGeometry args={[0.04, 8, 8]} />
          <meshStandardMaterial color="#64748b" />
        </mesh>

        {/* |+⟩ - X+ */}
        <mesh position={[1, 0, 0]}>
          <sphereGeometry args={[0.04, 8, 8]} />
          <meshStandardMaterial color="#64748b" />
        </mesh>

        {/* |-⟩ - X- */}
        <mesh position={[-1, 0, 0]}>
          <sphereGeometry args={[0.04, 8, 8]} />
          <meshStandardMaterial color="#64748b" />
        </mesh>
      </group>
    </group>
  );
}

export function BlochSphere() {
  const { nQubits, selectedQubit, setSelectedQubit, blochVectors } = useCircuitStore();
  const displayQubit = selectedQubit ?? 0;
  const vector = blochVectors[displayQubit] || { x: 0, y: 0, z: 1 };

  return (
    <div className="space-y-4">
      {/* Qubit selector */}
      {nQubits > 1 && (
        <div className="flex gap-2">
          {Array.from({ length: nQubits }, (_, i) => (
            <button
              key={i}
              onClick={() => setSelectedQubit(i)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                displayQubit === i
                  ? 'bg-quantum-600 text-white'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              q{i}
            </button>
          ))}
        </div>
      )}

      {/* 3D Bloch sphere */}
      <div className="aspect-square bg-slate-900 rounded-lg overflow-hidden">
        <Canvas camera={{ position: [2.5, 2, 2.5], fov: 45 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          <BlochSphereGeometry />
          <OrbitControls
            enableZoom={true}
            enablePan={false}
            minDistance={2}
            maxDistance={6}
          />
        </Canvas>
      </div>

      {/* Coordinates */}
      <div className="grid grid-cols-3 gap-2 text-sm">
        <div className="bg-slate-700/50 rounded p-2 text-center">
          <div className="text-red-400">X</div>
          <div className="font-mono">{vector.x.toFixed(3)}</div>
        </div>
        <div className="bg-slate-700/50 rounded p-2 text-center">
          <div className="text-green-400">Y</div>
          <div className="font-mono">{vector.y.toFixed(3)}</div>
        </div>
        <div className="bg-slate-700/50 rounded p-2 text-center">
          <div className="text-blue-400">Z</div>
          <div className="font-mono">{vector.z.toFixed(3)}</div>
        </div>
      </div>
    </div>
  );
}
