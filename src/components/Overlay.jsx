// Overlay.jsx

/**
 * Overlay Component
 * 
 * This component serves as the main overlay for the application, 
 * managing the display of the introduction section and the customizer. 
 * It utilizes Framer Motion for animations and Valtio for state management. 
 * The overlay includes options for personalization and decal selection, 
 * as well as functionality for saving the canvas as an image.
 * 
 * Author: 0x00000050
 * Date: July 7, 2025
 * License: MIT
 * Acknowledgments: Poimandres collective, Anderson Mancini and Paul Henschel.
 * 
 * Dependencies:
 * - React
 * - Framer Motion
 * - Valtio
 * - React Icons
 * 
 * Usage:
 * Ensure that the necessary assets and components are available 
 * in the specified paths before using this component.
 */

import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  AiFillCamera,
  AiOutlineArrowLeft,
  AiOutlineHighlight,
  AiOutlineShopping
} from 'react-icons/ai';
import { useSnapshot } from 'valtio';
import { state } from '../store';
import DecalModal from '../components/DecalModal';
import '../assets/styles/style.css';

export default function Overlay() {

  const snap = useSnapshot(state);
  const transition = { type: 'spring', duration: 0.8 };
  const motionConfig = {
    initial: { x: -100, opacity: 0, transition: { ...transition, delay: 0.5 } },
    animate: { x: 0, opacity: 1, transition: { ...transition } },
    exit: { x: -100, opacity: 0, transition: { ...transition } }
  };

  return (
    <div className="overlay-container">
      <motion.header
        initial={{ opacity: 0, y: -100 }}
        animate={{ opacity: 1, y: 0 }}
        transition={transition}
        className="header"
      />
      <AnimatePresence mode="wait">
        {snap.intro ? (
          <motion.section key="main" {...motionConfig} className="intro-section">
            <div className="section-container">
              <motion.div
                key="title"
                initial={{ x: 100, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{
                  type: 'spring',
                  damping: 5,
                  stiffness: 40,
                  restDelta: 0.001,
                  duration: 0.3
                }}
              >
                <h1>
                  TOULOUSE&nbsp;
                  <span className="typewriter-container">
                    <span className="typewriter"></span>
                  </span>
                  &nbsp;T'IL?
                </h1>
              </motion.div>
              <div className="support-content">
                <motion.div
                  key="p"
                  initial={{ y: 100, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{
                    type: 'spring',
                    damping: 7,
                    stiffness: 30,
                    restDelta: 0.001,
                    duration: 0.6,
                    delay: 0.2,
                    delayChildren: 0.2
                  }}
                >
                  <p>
                    Une dataviz pour les gouverner toutes!{' '}
                    <strong>Libère ta créativité</strong> et choisis la saveur de <i>joyplot</i> qui convient le mieux à ton envie du moment.
                  </p>
                  <div className="btn-group">
                    <button
                      className="custom-btn"
                      onClick={() => (state.intro = false)}
                    >
                      PERSONNALISATION <AiOutlineHighlight size="1.3em" />
                    </button>
                    <button
                      className="custom-btn"
                      onClick={() => (state.product = snap.product === 'bag' ? 'shirt' : 'bag')}
                    >
                      {snap.product === 'bag'
                        ? 'JE PRÉFÈRE LE T-SHIRT :)'
                        : 'JE VEUX LA SACOCHE!'}{' '}
                      <AiOutlineShopping size="1.3em" />
                    </button>
                  </div>
                </motion.div>
              </div>
            </div>
          </motion.section>
        ) : (
          <motion.section key="custom" {...motionConfig} className="custom-section">
            <Customizer />
          </motion.section>
        )}
      </AnimatePresence>

      <AnimatePresence mode="wait">
        {snap.selectedDecal && (
          <DecalModal
            decal={snap.selectedDecal}
            onClose={() => (state.selectedDecal = null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

function Customizer() {
  const snap = useSnapshot(state);
  const predefinedColors = ['MidnightBlue', 'crimson', 'BurlyWood'];

  useEffect(() => {
    if (snap.color) {
      document.documentElement.style.setProperty('--primary-color', snap.color);
    }
  }, [snap.color]);

  return (
    <div className="customizer">
      <div className="decals">
        <div className="decals-container">
          {snap.decals.map((decal, index) => (
            <div
              key={decal.full}
              className="decal"
              onClick={() => {
                state.decal = decal.full;
                state.selectedDecal = decal;
                state.color = predefinedColors[index % predefinedColors.length];
              }}
            >
              <img src={decal.thumb} alt="decal preview" />
            </div>
          ))}
        </div>
      </div>
      <div className="btn-group" style={{ marginTop: '20px' }}>
        <button
          className="share custom-btn"
          onClick={() => {
            const canvas = document.querySelector('canvas');
            if (canvas) {
              const link = document.createElement('a');
              link.setAttribute('download', 'canvas.png');
              link.setAttribute(
                'href',
                canvas
                  .toDataURL('image/png')
                  .replace('image/png', 'image/octet-stream')
              );
              link.click();
            }
          }}
        >
          SOUVENIR <AiFillCamera size="1.3em" />
        </button>
        <button
          className="exit custom-btn"
          onClick={() => {
            state.intro = true;
            state.selectedDecal = null;
          }}
        >
          RETOUR <AiOutlineArrowLeft size="1.3em" />
        </button>
      </div>
    </div>
  );

}
