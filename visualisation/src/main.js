import { createAHAPlot } from './AHA-plot.js';
import { loadHeartModel } from './heart-model.js';
import { fetchUpdateColor } from './assign-color.js';
import { createColorBar } from './create-color-bar.js';

window.addEventListener('DOMContentLoaded', async () => {
  const heartCanvas = document.getElementById('heartCanvas');
  const chartCanvas = document.getElementById('AHAPlot');

  const { data: updatedStrainData, colorBarArray: colorbar } = await fetchUpdateColor(-10, 5, 45);

  console.log('Updated Strain Data with Colors:', updatedStrainData);
  console.log('Color Bar Array:', colorbar);

  
  createAHAPlot(chartCanvas, updatedStrainData);
  loadHeartModel(heartCanvas, updatedStrainData);
  createColorBar('colorBar', colorbar);
});
