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

