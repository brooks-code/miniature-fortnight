/**
 * @file ThreeDScene.jsx
 * @description
 * React component that renders a 3D scene using React Three Fiber and Drei utilities.
 * It displays either a T-shirt or a shoulder bag model with dynamic environment lighting,
 * shadows, and decals (textures) that update based on application state managed by Valtio.
 *
 * Features:
 *  - Canvas setup with shadows, custom camera position & FOV
 *  - HDR environment via Poly Haven
 *  - Accumulative shadows with randomized lights
 *  - Camera rig with smooth easing-based motion
 *  - Dynamic model switching (T-shirt vs. shoulder bag)
 *  - Color and decal (texture) updates using maath easing
 *  - State management with Valtio (`state` snapshot)
 *  - Model preloading for performance
 *
 * @requires react
 * @requires react-three/fiber
 * @requires react-three/drei
 * @requires maath
 * @requires valtio
 *
 * @author 0x00000050
 * @date 2025-07-07
 * @license MIT
 * @acknowledgments Poimandres collective, Anderson Mancini and Paul Henschel.
 */

import React, { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import {
  useGLTF,
  useTexture,
  AccumulativeShadows,
  RandomizedLight,
  Decal,
  Environment,
  Center
} from '@react-three/drei'
import { easing } from 'maath'
import { useSnapshot } from 'valtio'
import { state } from '../store'

import shirtModel from '../assets/models/shirt.glb?url'
import shoulderBagModel from '../assets/models/sacoche.glb?url'

export const App = ({ position = [0, 0, 2.5], fov = 25 }) => {
  const snap = useSnapshot(state)

  return (
    <Canvas
      shadows
      camera={{ position, fov }}
      gl={{ preserveDrawingBuffer: true }}
      eventSource={document.getElementById('root')}
      eventPrefix="client"
    >
      <ambientLight intensity={0.4} />
      <Environment files="https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/golden_bay_1k.hdr" />
      {/* <Environment files="./assets/models/env/golden_bay_1k.hdr" /> */}
      <CameraRig>
        <Backdrop />
        <Center>
          {snap.product === 'shirt' ? <Shirt /> : <ShoulderBag />}
        </Center>
      </CameraRig>
    </Canvas>
  )
}

function Backdrop() {
  const shadows = useRef()
  useFrame((state, delta) => {
    if (shadows.current) {
      easing.dampC(
        shadows.current.getMesh().material.color,
        state.color,
        0.25,
        delta
      )
    }
  })
  return (
    <AccumulativeShadows
      ref={shadows}
      temporal
      frames={60}
      alphaTest={0.85}
      scale={10}
      rotation={[Math.PI / 2, 0, 0]}
      position={[0, 0, -0.14]}
    >
      <RandomizedLight
        amount={4}
        radius={9}
        intensity={0.55}
        ambient={0.25}
        position={[5, 5, -10]}
      />
      <RandomizedLight
        amount={4}
        radius={5}
        intensity={0.25}
        ambient={0.55}
        position={[-5, 5, -9]}
      />
    </AccumulativeShadows>
  )
}

function CameraRig({ children }) {
  const group = useRef()
  const snap = useSnapshot(state)
  useFrame((state, delta) => {
    easing.damp3(
      state.camera.position,
      [snap.intro ? -state.viewport.width / 4 : 0, 0, 2],
      0.25,
      delta
    )
    if (group.current) {
      easing.dampE(
        group.current.rotation,
        [state.pointer.y / 10, -state.pointer.x / 5, 0],
        0.25,
        delta
      )
    }
  })
  return <group ref={group}>{children}</group>
}

function Shirt(props) {
  const snap = useSnapshot(state)
  const texture = useTexture(snap.decal)
  const { nodes, materials } = useGLTF(shirtModel)
  useFrame((state, delta) =>
    easing.dampC(materials.lambert1.color, snap.color, 0.25, delta)
  )
  return (
    <mesh
      castShadow
      geometry={nodes.T_Shirt_male.geometry}
      material={materials.lambert1}
      material-roughness={1}
      {...props}
      dispose={null}
      scale={1.4}
    >
      <Decal
        position={[0, 0.04, 0.15]}
        rotation={[0, 0, 0]}
        scale={0.2}
        map={texture}
        material-transparent={true}
        material-opacity={0.7}
      />
    </mesh>
  )
}

function ShoulderBag(props) {
  const snap = useSnapshot(state)
  const texture = useTexture(snap.decal)
  const { nodes, materials } = useGLTF(shoulderBagModel)
  useFrame((state, delta) =>
    easing.dampC(materials.defaultMat.color, snap.color, 0.25, delta)
  )

  const toteNode = nodes.Object_2
  return (
    <Center>
      <mesh
        castShadow
        material={materials.defaultMat}
        material-roughness={1}
        geometry={toteNode.geometry}
        scale={[2, 2, 2]}
        //rotation={[2.95, -3.2, -3]}
        rotation={[470, 270, 270]}
        {...props}
        dispose={null}
      >
        <Decal
          position={[0, 0, 0.18]}
          rotation={[1, 0, 0]}
          scale={0.2}
          map={texture}
          transparent={true}
          opacity={0.2}
        />
      </mesh>
    </Center>
  )
}

// Preload models
useGLTF.preload(shirtModel)
useGLTF.preload(shoulderBagModel)

export default App
