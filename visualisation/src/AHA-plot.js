import Chart from 'chart.js/auto';
import ChartDataLabels from 'chartjs-plugin-datalabels';

/**
 * Create an AHA segmented doughnut chart with animated colors.
 * @param {HTMLCanvasElement} canvas - Chart canvas
 * @param {Array} strainData - Array of objects with StrainValue, ColorArray, Color, AHAIndex
 */
export function createAHAPlot(canvas, strainData) {
  // Copy & sort descending by AHAIndex
  const sorted = strainData.slice().sort((a, b) => a.AHAIndex - b.AHAIndex).reverse();

  // Split into AHA layers
  const layerOne   = sorted.filter(s => s.AHAIndex >= 1  && s.AHAIndex <= 6);
  const layerTwo   = sorted.filter(s => s.AHAIndex >= 7  && s.AHAIndex <= 12);
  const layerThree = sorted.filter(s => s.AHAIndex >= 13 && s.AHAIndex <= 16);
  const layerFour  = sorted.filter(s => s.AHAIndex === 17);

  // Helper: get first color or fallback
  const getInitialColor = (segment) => {
    if (Array.isArray(segment.ColorArray) && segment.ColorArray.length > 0) return segment.ColorArray[0];
    return segment.Color || '#FFFFFF';
  };

  // Initialize datasets
  const datasets = [
    {
      label: 'BASAL',
      data: Array(layerOne.length).fill(100 / layerOne.length),
      backgroundColor: layerOne.map(getInitialColor),
      borderColor: '#000000',
      labels: layerOne.map(s => s.AHAIndex),
      weight: 1
    },
    {
      label: 'MID-CAVITY',
      data: Array(layerTwo.length).fill(100 / layerTwo.length),
      backgroundColor: layerTwo.map(getInitialColor),
      borderColor: '#000000',
      labels: layerTwo.map(s => s.AHAIndex),
      weight: 1
    },
    {
      label: 'APICAL',
      data: Array(layerThree.length).fill(25),
      backgroundColor: layerThree.map(getInitialColor),
      borderColor: '#000000',
      labels: layerThree.map(s => s.AHAIndex),
      rotation: 45,
      weight: 1
    },
    {
      label: 'Layer 4',
      data: [100],
      backgroundColor: layerFour.map(getInitialColor),
      borderColor: 'transparent',
      labels: layerFour.map(s => s.AHAIndex),
      weight: 1
    }
  ];

  // Create Chart.js doughnut
  const chart = new Chart(canvas, {
    type: 'doughnut',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      aspectRatio: 0.5,
      rotation: 30,
      cutout: '0%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            generateLabels: (chart) => {
              let labels = [];
              chart.data.datasets.forEach((ds, iDs) => {
                const reversedLabels = ds.labels.slice().reverse();
                labels = labels.concat(
                  reversedLabels.map((l) => {
                    const originalIndex = ds.labels.indexOf(l);
                    return {
                      datasetIndex: iDs,
                      labelIndex: originalIndex,
                      text: l,
                      fillStyle: ds.backgroundColor[originalIndex],
                      hidden: chart.getDatasetMeta(iDs).data[originalIndex].hidden,
                      strokeStyle: '#000',
                      fontColor: '#fff',
                    };
                  })
                );
              });
              return labels;
            }
          }
        },
        tooltip: {
          callbacks: {
            label: (tooltipItem) => {
              const dataset = tooltipItem.dataset;
              const label = dataset.labels[tooltipItem.dataIndex];
              const segment = sorted.find(s => s.AHAIndex === label);

              // Get current value based on animated step
              const values = Array.isArray(segment.StrainValue) ? segment.StrainValue : [segment.StrainValue];
              const value = values[chart.currentStep % values.length];

              const segmentClass = segment.AHASegmentName || `Segment ${label}`;
              return `${label}, ${segmentClass}: [${value}]%`;
            }
          }
        },
        datalabels: {
          color: '#000000',
          formatter: (value, context) => context.dataset.labels[context.dataIndex]
        }
      }
    },
    plugins: [ChartDataLabels]
  });

  // -----------------------------
  // Animate colors using ColorArray
  // -----------------------------
  let step = 0;
  const maxSteps = Math.max(...sorted.map(s => Array.isArray(s.ColorArray) ? s.ColorArray.length : 1));
  chart.currentStep = step;

  setInterval(() => {
    datasets.forEach((ds) => {
      ds.backgroundColor = ds.labels.map(label => {
        const segment = sorted.find(s => s.AHAIndex === label);
        const colors = Array.isArray(segment.ColorArray) && segment.ColorArray.length > 0
          ? segment.ColorArray
          : [segment.Color || '#FFFFFF'];
        return colors[step % colors.length];
      });
    });
    chart.currentStep = step;  // update step for tooltip
    chart.update('none');       // instant update, no animation
    step++;
  }, 750); // update every 750ms
}

