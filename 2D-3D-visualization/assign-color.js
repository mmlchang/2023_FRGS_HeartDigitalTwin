export async function fetchUpdateColor(negativeValue, middleValue, positiveValue) {
  const response = await fetch('/assets/models.json');
    if (!response.ok) throw new Error(`Failed to fetch JSON: ${response.status}`);
  const data = await response.json();

  const allValues = data.flatMap(d => Array.isArray(d.StrainValue) ? d.StrainValue : [d.StrainValue]);
  // const minStrain = Math.min(...allValues);
  // const maxStrain = Math.max(...allValues);

  const minStrain = negativeValue;
  const maxStrain = positiveValue;
  const centerStrain = middleValue; // this will map to 0.5 in normalized scale

  function normalize(value) {
    if (value <= centerStrain) {
      // below center: map minStrain → 0, centerStrain → 0.5
      return 0.5 * (value - minStrain) / (centerStrain - minStrain);
    } else {
      // above center: map centerStrain → 0.5, maxStrain → 1
      return 0.5 + 0.5 * (value - centerStrain) / (maxStrain - centerStrain);
    }
  }

  function divergingColor(normalized) {
    let r, g, b;
    if (normalized < 0.5) {
      const t = normalized / 0.5;
      r = Math.round(255 - 255 * t);
      g = Math.round(0 + 255 * t);
      b = 0;
    } else {
      const t = (normalized - 0.5) / 0.5;
      r = Math.round(0 + 255 * t);
      g = Math.round(255 - 255 * t);
      b = 0;
    }
    return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase();
  }

  data.forEach(d => {
    // store array of colors for animation
    d.ColorArray = Array.isArray(d.StrainValue)
      ? d.StrainValue.map(v => divergingColor(normalize(v)))
      : [divergingColor(normalize(d.StrainValue))];

    // optional: set initial color
    d.Color = d.ColorArray[0];
  });

  // Generate colorBar gradient from min → max
  const steps = 50; // smooth gradient
  const colorBarArray = [];
  for (let i = 0; i <= steps; i++) {
    const norm = i / steps;  // normalized 0 → 1
    colorBarArray.push(divergingColor(norm));
  }

  return { data, colorBarArray }; // now each object has StrainValue, ColorArray, and Color (initial)
}

// ############################################################################# //


// async function fetchAndUpdateStrainData() {
//   const response = await fetch('/assets/models.json'); // fetch from public folder
//   const data = await response.json();

//   const minStrain = Math.min(...data.map(d => d.StrainValue));
//   const maxStrain = Math.max(...data.map(d => d.StrainValue));

//   function normalize(value) {
//     return (value - minStrain) / (maxStrain - minStrain);
//   }

//   function heatmapColor(normalized) {
//     let r, g, b;
//     if (normalized < 0.5) {
//       const t = normalized / 0.5;
//       r = Math.round(0 + 255 * t);
//       g = 255;
//       b = 0;
//     } else {
//       const t = (normalized - 0.5) / 0.5;
//       r = 255;
//       g = Math.round(255 - 255 * t);
//       b = 0;
//     }
//     return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase();
//   }

//   data.forEach(d => {
//     d.Color = heatmapColor(normalize(d.StrainValue));
//   });

//   return data; // return updated data in memory
// }



// export async function fetchUpdateColor() {
//   const response = await fetch('/assets/models.json');
//   if (!response.ok) throw new Error(`Failed to fetch JSON: ${response.status}`);
//   const data = await response.json();

//   // const minStrain = Math.min(...data.map(d => d.StrainValue)); 
//   // const maxStrain = Math.max(...data.map(d => d.StrainValue)); 

//   const minStrain = -40; // fixed range for better color consistency
//   const maxStrain = 40;

//   function normalize(value) {
//     // Map -max → 0, 0 → 0.5, +max → 1
//     return (value - minStrain) / (maxStrain - minStrain); // basic 0..1
//   }

//   function divergingColor(normalized) {
//     // normalized: 0 -> minStrain (-30), 0.5 -> zero, 1 -> maxStrain (+30)
//     let r, g, b;

//     if (normalized < 0.5) {
//       // negative values → red to green
//       const t = normalized / 0.5; // 0 -> 0, 0.5 -> 1
//       r = Math.round(255 - 255 * t); // 255 -> 0
//       g = Math.round(0 + 255 * t);   // 0 -> 255
//       b = 0;
//     } else {
//       // positive values → green to red
//       const t = (normalized - 0.5) / 0.5; // 0.5 -> 0, 1 -> 1
//       r = Math.round(0 + 255 * t);   // 0 -> 255
//       g = Math.round(255 - 255 * t); // 255 -> 0
//       b = 0;
//     }

//     return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase();
//   }

//   data.forEach(d => {
//     d.Color = divergingColor(normalize(d.StrainValue));
//   });

//   return data; // 
// }


