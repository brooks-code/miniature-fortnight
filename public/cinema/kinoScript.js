/**
 * ScreenEffect Class (Kino mode)
 * 
 * This is a fork and simplified version of a script provided by K. Saunders.
 * This class manages various visual effects on a specified parent element, 
 * including snow, VCR noise, and other customizable effects. It utilizes 
 * the Web Crypto API for generating random integers and employs the 
 * requestAnimationFrame method for smooth animations. The class supports 
 * adding, removing, and resizing effects dynamically based on the parent 
 * element's dimensions.
 * 
 * Features:
 * - Snow effect generation on a canvas.
 * - VCR noise rendering with customizable parameters.
 * - Support for multiple effect types including roll, wobble, scanlines, 
 *   vignette, image, and video.
 * - Responsive resizing of effects based on the parent element.
 * 
 * Author: 0x00000050
 * Date: July 7, 2025
 * License: MIT
 * Acknowledgments: Karl Saunders, https://github.com/Mobius1/
 * 
 * Dependencies:
 * - Web Crypto API
 * 
 * Usage:
 * Instantiate the ScreenEffect class with a parent element and an optional 
 * configuration array to enable specific effects.
 */

const getRandomIntCrypto = (min, max) => {
  const range = max - min + 1;
  // Create a Uint32Array with one element
  const randBuffer = new Uint32Array(1);
  window.crypto.getRandomValues(randBuffer);
  // Scale the random number to the range [min, max]
  // Since randBuffer[0] is in [0, 2^32-1], we use modulo division.
  return min + (randBuffer[0] % range);
};

class ScreenEffect {
  constructor(parent, effectsConfig = []) {
    // Accept a parent element or selector string
    this.parent = typeof parent === "string" ? document.querySelector(parent) : parent;
    // Effects configuration as an array of effect objects
    // Each effect object: { type: "vcr", options: { ... } }
    this.effectsConfig = effectsConfig;
    // Active effects container
    this.activeEffects = {};
    // Hold per-frame update callbacks for animated effects (e.g., snow, vcr)
    this.animatedEffects = [];
    // Bind resize callback.
    window.addEventListener("resize", this.onResize.bind(this), false);

    // Render the container structure
    this.render();

    // Start the shared animation loop
    this.lastFrame = performance.now();
    this.running = true;
    requestAnimationFrame(this.animationLoop.bind(this));

    // Add any pre-configured effects
    this.effectsConfig.forEach(effect => {
      if (effect.enabled) {
        this.add(effect.type, effect.options);
      }
    });
  }

  render() {
    // Create container & nested wrappers
    const container = document.createElement("div");
    container.classList.add("screen-container");

    const createWrapper = () => {
      const wrapper = document.createElement("div");
      wrapper.classList.add("screen-wrapper");
      return wrapper;
    };

    const wrapper1 = createWrapper();
    const wrapper2 = createWrapper();
    const wrapper3 = createWrapper();

    wrapper1.appendChild(wrapper2);
    wrapper2.appendChild(wrapper3);
    container.appendChild(wrapper1);

    // Insert a container before the parent, then move parent into deepest wrapper.
    this.parent.parentNode.insertBefore(container, this.parent);
    wrapper3.appendChild(this.parent);

    this.nodes = { container, wrapper1, wrapper2, wrapper3 };

    // Update sizes based on parent dimensions.
    this.onResize();
  }

  // Resize handler: update cached rectangle and adjust any effect sizes.
  onResize() {
    this.rect = this.parent.getBoundingClientRect();
    // Adjust VCR canvas if active
    if (this.activeEffects.vcr) {
      const canvas = this.activeEffects.vcr.node;
      canvas.width = this.rect.width;
      canvas.height = this.rect.height;
    }
    // Adjust snow canvas if active
    if (this.activeEffects.snow) {
      const canvas = this.activeEffects.snow.node;
      canvas.width = this.rect.width / 2;
      canvas.height = this.rect.height / 2;
    }
  }

  // The shared animation loop using requestAnimationFrame.
  animationLoop(now) {
    const delta = now - this.lastFrame;
    this.lastFrame = now;

    // Iterate over animated effects and update each.
    for (const updateFn of this.animatedEffects) {
      updateFn(delta);
    }

    if (this.running) {
      requestAnimationFrame(this.animationLoop.bind(this));
    }
  }

  // Add an effect by type & options.
  // Supported types: "snow", "vcr", "roll", "wobblex", "wobbley", "scanlines", "vignette", "image", "video"
  add(type, options = {}) {
    // Default options per effect can be merged in:
    const config = Object.assign({ fps: 30, blur: 1 }, options);

    // If this is already active, do nothing.
    if (this.activeEffects[type]?.enabled) return this;

    switch (type) {
      case "snow": {
        const canvas = document.createElement("canvas");
        canvas.classList.add(type);
        canvas.width = this.rect.width / 2;
        canvas.height = this.rect.height / 2;
        const ctx = canvas.getContext("2d");
        this.nodes.wrapper2.appendChild(canvas);

        const updateSnow = () => {
          this.generateSnow(ctx);
        };

        // Register update callback in a shared loop.
        this.animatedEffects.push(updateSnow);
        this.activeEffects.snow = { node: canvas, config, enabled: true, updateFn: updateSnow };
        break;
      }

      case "vcr": {
        const canvas = document.createElement("canvas");
        canvas.classList.add(type);
        this.nodes.wrapper2.appendChild(canvas);
        canvas.width = this.rect.width;
        canvas.height = this.rect.height;
        const ctx = canvas.getContext("2d");
        // VCR noise update callback.
        const updateVCR = () => {
          this.renderTrackingNoise(ctx, canvas, config);
        };
        this.animatedEffects.push(updateVCR);
        this.activeEffects.vcr = { node: canvas, ctx, config, enabled: true, updateFn: updateVCR };
        break;
      }

      case "roll": {
        this.enableRoll();
        this.activeEffects.roll = { enabled: true };
        break;
      }

      case "wobbley": {
        this.nodes.wrapper2.classList.add(type);
        this.activeEffects[type] = { enabled: true, config };
        break;
      }

      case "scanlines": {
        const node = document.createElement("div");
        node.classList.add(type);
        this.nodes.wrapper2.appendChild(node);
        this.activeEffects.scanlines = { node, enabled: true, config };
        break;
      }

      case "vignette": {
        const node = document.createElement("div");
        node.classList.add(type);
        this.nodes.container.appendChild(node);
        this.activeEffects.vignette = { node, enabled: true, config };
        break;
      }

      case "image": {
        const node = document.createElement("img");
        node.classList.add(type);
        node.src = config.src;
        if (config.blur) {
          node.style.filter = `blur(${config.blur}px)`;
        }
        this.parent.appendChild(node);
        this.activeEffects.image = { node, enabled: true, config };
        break;
      }

      case "video": {
        const node = document.createElement("video");
        node.classList.add(type);
        node.src = config.src;
        node.crossOrigin = "anonymous";
        node.autoplay = true;
        node.muted = true;
        node.loop = true;
        this.parent.appendChild(node);
        this.activeEffects.video = { node, enabled: true, config };
        break;
      }

      default: {
        console.warn("Unrecognized effect type:", type);
      }
    }

    return this;
  }

  remove(type) {
    const effect = this.activeEffects[type];
    if (effect && effect.enabled) {
      effect.enabled = false;
      // Remove animated effects’ update callback from the shared list
      if (effect.updateFn) {
        this.animatedEffects = this.animatedEffects.filter(fn => fn !== effect.updateFn);
      }
      // For roll effect revert the DOM changes if applicable.
      if (type === "roll" && effect.original) {
        this.parent.appendChild(effect.original);
      }
      // Remove canvas or div nodes from the DOM.
      if (effect.node) {
        effect.node.parentNode.removeChild(effect.node);
      } else if (this.nodes.wrapper2) {
        this.nodes.wrapper2.classList.remove(type);
      }
      delete this.activeEffects[type];
    }
    return this;
  }

  // Roll effect rewraps the first element in parent inside a rolling container.
  enableRoll() {
    const originalEl = this.parent.firstElementChild;
    if (originalEl) {
      const roller = document.createElement("div");
      roller.classList.add("roller");
      this.parent.appendChild(roller);
      roller.appendChild(originalEl);
      roller.appendChild(originalEl.cloneNode(true));
      this.activeEffects.roll = { enabled: true, node: roller, original: originalEl };
    }
  }

  // Generates snow effect on a canvas context.
  generateSnow(ctx) {
    const w = ctx.canvas.width,
      h = ctx.canvas.height,
      imageData = ctx.createImageData(w, h),
      data = new Uint32Array(imageData.data.buffer),
      len = data.length;

    for (let i = 0; i < len; i++) {
      data[i] = (getRandomIntCrypto(0, 255) << 24);
    }
    ctx.putImageData(imageData, 0, 0);
  }

  // Renders VCR noise on the provided canvas context.
  renderTrackingNoise(ctx, canvas, config) {
    const radius = 2;
    const posY1Init = config.miny || 0;
    const posY2 = config.maxy || canvas.height;
    let posY1 = posY1Init;
    let posY3 = config.miny2 || 0;
    const num = config.num || 20;

    // Apply blur filter.
    canvas.style.filter = `blur(${config.blur || 1}px)`;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#fff";
    ctx.beginPath();

    for (let i = 0; i <= num; i++) {
      const x = Math.random() * canvas.width;
      const y1 = getRandomIntCrypto(posY1 += 3, posY2);
      const y2 = getRandomIntCrypto(0, posY3 -= 3);
      ctx.fillRect(x, y1, radius, radius);
      ctx.fillRect(x, y2, radius, radius);
      this.renderTail(ctx, x, y1, radius);
      this.renderTail(ctx, x, y2, radius);
    }
    ctx.closePath();
  }

  // Draw a tail of noise starting at (x,y)
  renderTail(ctx, x, y, radius) {
    const n = getRandomIntCrypto(1, 50);
    const dir = Math.random() < 0.5 ? 1 : -1;
    let currentRadius = radius;
    for (let i = 0; i < n; i++) {
      currentRadius = Math.max(currentRadius - 0.01, 0);
      const r = getRandomIntCrypto(Math.floor(currentRadius), Math.floor(radius));
      const dx = getRandomIntCrypto(1, 4) * dir;
      x += dx;
      ctx.fillRect(x, y, r, r);
    }
  }
}

// Example usage:

// Define a simplified configuration as an array of effect objects.
const effectsArray = [
  { type: "roll", enabled: false, options: { speed: 1000 } },
  { type: "image", enabled: true, options: { src: "yearly_animation.gif", blur: 1.2 } },
  { type: "vignette", enabled: true },
  { type: "scanlines", enabled: true },
  { type: "vcr", enabled: true, options: { opacity: 1, miny: 220, miny2: 220, num: 70, fps: 60, blur: 1 } },
  { type: "wobbley", enabled: false },
  { type: "snow", enabled: true, options: { opacity: 0.2 } }
];

// Initialize the ScreenEffect instance with the simplified effects configuration.
const screen = new ScreenEffect("#screen", effectsArray);

/*
  GUI Code: If you are using dat.GUI, you can iterate over the effectsArray
  to generate controls and then call screen.add or screen.remove.
  Below is a simplified example.
*/

/* const gui = new dat.GUI();
const effectsFolder = gui.addFolder("Effects");

effectsArray.forEach(effect => {
  effectsFolder.add(effect, "enabled").name(effect.type).onChange(enabled => {
    if (enabled) {
      screen.add(effect.type, effect.options);
    } else {
      screen.remove(effect.type);
    }
  });
}); */

//effectsFolder.open();
