/**
 * State Management with Valtio
 * 
 * This module defines the application state using Valtio's proxy. 
 * It includes properties for managing the introduction screen, 
 * available decals, selected decal, product type, and color settings. 
 * The decals array contains information about each decal, including 
 * its full image, thumbnail, legend, and associated slides.
 * 
 * Author: 0x00000050
 * Date: July 7, 2025
 * Acknowledgments: Poimandres collective, Anderson Mancini and Paul Henschel.
 * 
 * Dependencies:
 * - Valtio
 * 
 * Usage:
 * Import the `state` object to access and manipulate the application state 
 * throughout the application. Ensure that the asset paths are correct 
 * for the images used in the decals.
 */

import { proxy } from 'valtio'

import classic from './assets/img/joyplot-classic.png'
import classicThumb from './assets/img/thumbs/classic_thumb.png'
import unknownPleasures from './assets/img/original_joyplot.jpg'
import classicDistribution from './assets/img/joyplot-classic_distribution.png'
import occitania from './assets/img/occitania.png'
import occitaniaThumb from './assets/img/thumbs/occitania_thumb.png'
import multiColor from './assets/img/excess.png'
import multiColorThumb from './assets/img/thumbs/gamut.png'
import multiColorCalm from './assets/img/calm.png'
import extreme from './assets/img/extreme_years.png'

const base = import.meta.env.BASE_URL
const state = proxy({
  intro: true,
  decals: [
    {
      full: classic,
      thumb: classicThumb,
      legend: "Vous avez dit joyplot ? Plutôt connu sous le nom de ridgelines, il s'agit de tracés linéaires combinés par empilement vertical pour permettre de visualiser facilement des changements dans l'espace ou le temps. Les tracés sont légèrement superposés afin de mettre en contraste les changements. Le joyplot le plus célèbre a été conçu par Harold Craft pour visualiser les ondes radios émises par un pulsar, il illustre l'album Unknown Pleasures du groupe Joy Division. Le concept a ensuite été popularisé par Claus Wilke. Le premier graphique représente les niveaux d'eau quotidiens maximaux de la Garonne pour la période 1857-2024. Chaque tracé représente une année, les plus récentes se situant en haut. Le troisième graphique est une variation qui ressemble plus à la couverture de l'album. Il représente la distribution des hauteurs quotidiennes maximales.",
      slides: [classic, unknownPleasures, classicDistribution]
    },
    {
      full: occitania,
      thumb: occitaniaThumb,
      legend: "À première vue, les visuels conçus peuvent sembler brouillons. Mais en fait, ils sont pleins d'informations intéressantes, il suffit de les mettre en avant. De 1857 à 2024, il y a eu 9 crues à Toulouse, certaines ont été hors-normes et dévastatrices, comme celle de 1875 qui a ravagé Saint-Cyprien, ou celle de 2000 qui a causé des dégâts importants. Cette variation du joyplot précédent met en avant les 9 crues historiques."
    },
    {
      full: multiColor,
      thumb: multiColorThumb,
      legend: "Les neuf crues que la Garonne a connu ne sont pas égales. Le premier graphique permet de les envisager en attribuant une couleur en fonction du niveau d'eau maximal mesuré. Plus la couleur est foncée, plus la crue a été importante. Le même travail a été effectué avec le graphique suivant, pour les années les plus calmes, qui seront elles aussi plus foncées (mais en bleu). Quant au dernier graphique, il permet de confronter les deux années les plus à l’opposé du dataset fourni.",
      buttonLink: `${base}cinema/kino.html`,
      slides: [multiColor, multiColorCalm, extreme]
    }
  ],
  color: '#80C670',
  decal: classic,
  selectedDecal: null,
  product: 'shirt',
  //cinemaMode: false // New property to toggle cinema mode (when implemented).
})

export { state }