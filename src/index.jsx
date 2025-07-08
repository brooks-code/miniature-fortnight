// index.jsx

/**
 * App Component
 * 
 * This component initializes the application by setting a random gradient 
 * background and rendering the main components: PreloadDecals, Canvas, 
 * and Overlay. It uses React and ReactDOM for rendering and manages 
 * styles through an external CSS file.
 * 
 * Author: 0x00000050
 * Date: July 7, 2025
 * License: MIT
 * Acknowledgments: Poimandres collective, Anderson Mancini and Paul Henschel.
 * 
 * Dependencies:
 * - React
 * - ReactDOM
 * - Custom scripts and styles
 * 
 * Usage:
 * Ensure that the necessary assets and components are available 
 * in the specified paths before running the application.
 */

import React from 'react';
import { createRoot } from 'react-dom/client';
import './assets/styles/style.css';
import { getRandomGradient } from './assets/scripts/randomGradient';
import { PreloadDecals } from './PreloadDecals';
import Canvas from './components/ThreeDScene';
import Overlay from './components/Overlay';


// Wait until the DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const bgElement = document.querySelector('.bg')
  if (bgElement) {
    // Set the intial background style to a random gradient.
    bgElement.style.background = getRandomGradient()
  }
})

createRoot(document.getElementById('root')).render(
  <>
    <PreloadDecals />
    <Canvas />
    <Overlay />
  </>
)
