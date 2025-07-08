# DataSwag, the HackaViz swag configurator

**Customizable T-Shirt designs (and more..) with three.js**

![Banner Image](</processing/img/tshirt_mysterious.png> "A custom tshirt with a mysterious teaser message")
<br>*Teste la version interactive sur... https://brooks-code.github.io/miniature-fortnight*

This data-driven apparel (t‐shirts or bags) is inspired by the flooding dataset of the [hackaviz 2025](https://toulouse-dataviz.fr/hackaviz/2025-contest/) competition. You can pick decal flavors, and download your custom design as a snapshot.

- And if you're curious about the **journey from data processing to visualization**: follow the white rabbit **[into the processing section ♘](/processing/README.md)!**

## Table of contents

<details>
<summary>Contents - click to expand</summary>

- [DataSwag, the HackaViz swag configurator](#dataswag-the-hackaviz-swag-configurator)
  - [Table of contents](#table-of-contents)
  - [Demo](#demo)
  - [Features](#features)
  - [Tech stack](#tech-stack)
  - [Project structure (simplified)](#project-structure-simplified)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [State management (Valtio)](#state-management-valtio)
  - [Deployment guide](#deployment-guide)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Checking the `homepage` field](#2-checking-the-homepage-field)
    - [3. Install the gh-pages package](#3-install-the-gh-pages-package)
    - [4. Update package.json scripts](#4-update-packagejson-scripts)
    - [5. Commit \& push](#5-commit--push)
    - [6. Run the deployment](#6-run-the-deployment)
    - [7. Automate with GitHub actions *(optional)*](#7-automate-with-github-actions-optional)
  - [Data telltales: crafting striking visuals and a storytelling from raw data](#data-telltales-crafting-striking-visuals-and-a-storytelling-from-raw-data)
  - [Goodie: the kino mode screen effect](#goodie-the-kino-mode-screen-effect)
  - [Contributing](#contributing)
  - [License](#license)
    - [Acknowledgements](#acknowledgements)

</details>

---

## Demo

A complete demo is [available here](https://brooks-code.github.io/miniature-fortnight).

## Features

- **3D model rendering**  
Real‐time *customizable* 3D t-shirt or shoulder bag powered by [three.js](https://threejs.org/) and [@react-three/fiber](https://github.com/pmndrs/react-three-fiber).
- **Interactive overlay**  
Modern UI & animations with [Framer Motion](https://www.framer.com/motion/).
- **State management**  
Global reactive state via [Valtio](https://github.com/pmndrs/valtio).
- **Decal preloading**  
Smooth UX with preloaded textures.
- **Gradient background**  
Randomized gradients backgrounds on the landing page at each app relaunch.
- **Download snapshot**  
Capture and download canvas as PNG.
- **Fullscreen & slideshows**  
Modal slideshows for decals, fullscreen preview support.
- **Responsive & accessible**  
Mobile‐friendly layout & proper `alt` text on images & buttons.

## Tech stack

- [React 18](https://reactjs.org/)
- [Vite 5](https://vitejs.dev/)
- [three.js](https://threejs.org/)
- [@react-three/drei](https://github.com/pmndrs/drei)
- [@react-three/fiber](https://github.com/pmndrs/react-three-fiber)
- [valtio](https://github.com/pmndrs/valtio) (state management)
- [Framer Motion](https://www.framer.com/motion/) (animations)
- [maath](https://github.com/gre/gl-matrix)
- Plain CSS

## Project structure (simplified)

```text
.
├── processing 
│   └── ... # The data processing notebook. More about it below.
├── public 
│   └── ...
├── src
│   ├── assets
│   │   ├── models                 # .glb files
│   │   |   └── env
│   │   ├── img                    # visualizations
|   │   │   └── thumbs
│   │   ├── scripts
│   │   │   └── randomGradient.js
│   │   └── styles
│   │       └── style.css
│   ├── cinema 
│   │   └── ...                    # Kino mode
│   ├── components
│   │   ├── ThreeDScene.jsx        # Three.js scene wrapper
│   │   ├── Overlay.jsx            # Main UI overlay & Customizer
│   │   └── DecalModal.jsx         # Decal preview modal
│   ├── PreloadDecals.jsx          # Texture preloader
│   ├── store.js                   # Valtio global state
│   └── index.jsx                  # Entry point
├── package.json
├── index.html
└── README.md
```

## Getting Started

### Prerequisites

- Node.js *(v. 18+)*
- npm (*v. 8+)*

### Installation

Clone the repo:

```bash
git clone https://github.com/brooks-code/miniature-fortnight.git
cd miniature-fortnight
```

Install dependencies:

```bash
npm install
```

Running locally:

```bash
npm run dev
```

- Open `http://localhost:5173` in your browser. The page will reload on changes.

Building for production:

```bash
npm run build
```

- The production‐ready files will be in the  `dist/` directory (using Vite).

### Usage

| Section       | Description                                                                                     |
|---------------|-------------------------------------------------------------------------------------------------|
| **Intro screen** | - Choose between a t-shirt or a shoulder bag. <br> - Hit `PERSONNALISATION` to start customizing. |
| **Customizer**  | - Picking a decal thumbnail (left corner) updates the 3D model’s texture & UI color. <br> - Use the `SOUVENIR` camera button to download your design. <br> - `RETOUR` returns you to the intro page. |
| **Decal modal (side panel)** | - View slideshows of decal images. <br> - Click dots to switch between slides. <br> - Click the image for fullscreen view. <br> - Close the panel with the `FERMER` button or background click. |

### State management (Valtio)

The application’s state lives in `src/store.js`:

```javascript
import { proxy } from 'valtio'

export const state = proxy({
  intro: true,
  product: 'bag',
  color: '#ffffff',
  decal: '',
  selectedDecal: null,
  // Predefined decal list. Each object can have: { full, thumb, slides, legend, buttonLink }
  decals: [
    // Example entry
    //     { full: multiColor, thumb: multiColorThumb, legend: "Lorem ipsum...", buttonLink: 'link', slides: [multiColor, multiColorCalm, extreme] }
  ]
})
```

| Property        | Description                                           |
|-----------------|-------------------------------------------------------|
| `intro`         | Toggles intro vs. customizer state                   |
| `product`       | "shirt" or "bag"                                 |
| `color`         | CSS primary color, harmonizes buttons and apparel colors.              |
| `decal`         | Default decal (classic falvor)           |
| `selectedDecal` | Active decal object for the modal                       |
| `decals`        | Array of available decals                             |

## Deployment guide

This project can be effortlessly deployed to **GitHub pages** by building the production bundle (`npm run build`) and pushing the contents of the `dist/` folder to a branch named `gh-pages`. You can automate this step with a tool like `gh-pages` in the `package.json` script or configure a GitHub action to run the build and publish steps on every push to `main`.

> [!IMPORTANT]
> Keep in mind that GitHub pages is a static host, so server‐side features (like dynamic data fetching or authentication) are not supported out of the box. Additionally, large 3D assets or extensive decal libraries can increase load times and may hit the *100 MB repo* size limit. Consider using a CDN or external asset hosting to offload heavy files and ensure optimal performance for end users.

Follow the steps below to publish DataSwag on GitHub pages.

### 1. Prerequisites

- A GitHub repository for your project, e.g.: `github.com/your-username/repo-name`
- Node.js (v18+) and npm (or Yarn)
- A project set up with a build script that outputs a static folder (`build/` or in our case: `dist/`)

### 2. Checking the `homepage` field

In the `package.json`, add a `homepage` property pointing to the repo’s pages URL in the form of github pages:

```json
{
  "name": "a-name-here",
  "version": "1.0.0",
  "homepage": "https://your-username.github.io/repo-name",
  // …
}
```

### 3. Install the gh-pages package

```bash
npm install gh-pages --save-dev
```

### 4. Update package.json scripts

Add scripts to handle building and deploying:

```json
{
  "scripts": {
    "dev": "vite --host",
    "dev3": "e2e-dev $npm_package_name",
    "build": "tsc && vite build",
    "build2": "tsc && e2e-build $npm_package_name",
    "preview": "vite preview",
    "deploy": "gh-pages -d dist --depth=1"
  },
}
```

> [!NOTE]
> The `-d build` flag tells gh-pages where to find your built assets. For Vite you should need `-d dist`.For Create React App, use `-d build`.

### 5. Commit & push

```bash
git add package.json
git commit -m "Configure gh-pages deployment"
git push origin main
```

### 6. Run the deployment

```bash
npm run deploy
```

This performs:

`npm run build` → creates your production-ready `dist/` (or build/) directory.
`gh-pages -d dist` → pushes the contents of `dist/` to a branch named `gh-pages`.

Voilà! After a minute or two (maybe a bit more ;), your site will be live at:

`https://your-username.github.io/repo-name`

### 7. Automate with GitHub actions *(optional)*

If you'd love to add CI automation, (more [details](https://vite.dev/guide/static-deploy)) to your repo, create a workflow file at `.github/workflows/deploy.yaml`:

<details>
<summary>Sample deploy.yaml - click to expand</summary>

```yaml
# Simple workflow for deploying static content to GitHub Pages
name: Deploy static content to Pages

on:
  # Runs on pushes targeting the default branch
  push:
    branches: ['main']

  # Allows you to run this workflow manually from the Actions tab
  workflow_dispatch:

# Sets the GITHUB_TOKEN permissions to allow deployment to GitHub Pages
permissions:
  contents: read
  pages: write
  id-token: write

# Allow one concurrent deployment
concurrency:
  group: 'pages'
  cancel-in-progress: true

jobs:
  # Single deploy job since we're just deploying
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: lts/*
          cache: 'npm'
      - name: Install dependencies
        run: npm ci
      - name: Build
        run: npm run build
      - name: Setup Pages
        uses: actions/configure-pages@v5
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          # Upload dist folder
          path: './dist'
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

Commit this file and push—GitHub actions will handle your build & deployment on every push to main.
</details>

## Data telltales: crafting striking visuals and a storytelling from raw data

![Footer Image](</processing/img/data_vinyl.png> "A data vinyl") ♫ unknown pleasures (1979)

Got some data nerd vibes? Are you curious about how I achieved these results and generated these visuals? Perhaps you even want to try by yourself and go further? All the information and codebase you need is available in the [processing section](/processing/README.md)! See you there.

## Goodie: the kino mode screen effect

About the vintage cinema experience: this is a JavaScript utility designed to enhance web applications with various visual effects allowing to easily add and manage effects such as snow, VCR noise, and more on a specified parent element.

- **Dynamic Effects**: Add effects like snow, VCR noise, roll, wobble, scanlines, vignette.
- **Responsive Design**: Automatically adjusts effect sizes based on the dimensions of the parent element.
- **Animation Loop**: Utilizes `requestAnimationFrame` for smooth animations and updates.
- **Random Number Generation**: Uses the Web Crypto API to generate cryptographically secure random integers.

## Contributing

Contributions are welcome.

- Fork this repository.
- Create your feature branch (git checkout -b feature/yourFeature).
- Commit your changes (git commit -m 'Add some feature').
- Push to the branch (git push origin feature/yourFeature).
- Open a pull request.

## License

This project is licensed under the [MIT License](/LICENSE).

### Acknowledgements

- This project is a [fork](https://github.com/pmndrs/examples/blob/main/demos/t-shirt-configurator) of a project provided by the [Poimandres collective](https://pmnd.rs/blog/introducing-the-poimandres-blog) based on a design by Anderson Mancini and Paul Henschel. If you're super curious about the whereabouts of the original configurator app. There is a (paid) tutorial explaining a lot available on [Udemy](https://www.udemy.com/course/react-three-fiber-configurator).

- The Kino mode is a fork that heavily relies on a script created by [Karl Saunders](https://github.com/Mobius1/)

- The type writing effect on the landing page is inspired by a snippet created by [Brandon McConnell](https://codepen.io/brandonmcconnell/pen/bZqGdw).

- And.. thanks to the The **HackaViz** community as well!
