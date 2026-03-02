import factoryModule from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';

// Handle CJS/ESM interop: factory may be { default: fn } or fn directly
const createPlotlyComponent =
  typeof factoryModule === 'function'
    ? factoryModule
    : (factoryModule as { default: typeof factoryModule }).default;

const Plot = createPlotlyComponent(Plotly as object);
export default Plot;
