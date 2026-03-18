// create-color-bar.js
export function createColorBar(containerId, colors, height = 20) {
  const colorBar = document.getElementById(containerId);
  if (!colorBar) {
    console.warn(`Color bar container "${containerId}" not found.`);
    return;
  }

  colorBar.innerHTML = '';
  colorBar.style.display = 'flex';
  colorBar.style.height = `${height}px`; // set visible height

  colors.forEach(color => {
    const block = document.createElement('div');
    block.style.flex = '1';       // equal width per color
    block.style.height = '100%';  // fill container height
    block.style.backgroundColor = color;
    colorBar.appendChild(block);
  });
}