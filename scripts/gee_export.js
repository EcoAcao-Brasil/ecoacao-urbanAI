/**
 * UrbanAI Data Ingestion Script
 * * Pipeline for generating cloud-free Landsat composites for urban heat analysis.
 * Handles the retrieval, radiometric scaling, cloud masking, and export of 
 * satellite imagery for specific temporal windows.
 * * Target: Landsat Collection 2 (Level 2)
 */

// 1. REGION & SENSOR SETUP

// Region of Interest: Defaulting to Palmas, Tocantins (GAUL Level 2).
// To analyze a different city, replace this with your own feature collection or geometry.
var aoi = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
  .filter(ee.Filter.eq('ADM2_NAME', 'Palmas'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Tocantins'));

Map.centerObject(aoi, 10);
Map.addLayer(aoi, {}, 'Study Area Boundary');

// Configuration constants
// Switch SENSOR_NAME to 'L5' or 'L7' for historical analysis (pre-2013).
var SENSOR_NAME = 'L8'; 
var COLLECTION_ID = 'LANDSAT/LC08/C02/T1_L2'; 

var EXPORT_CONFIG = {
  scale: 30,             // Native Landsat resolution (meters)
  folder: 'Landsat_UrbanAI',
  crs: 'EPSG:4326'       // WGS84 for broad compatibility
};

// 2. PREPROCESSING UTILITIES

/**
 * Normalizes raw Digital Numbers (DN) to physical units.
 * * Landsat Collection 2 uses scaling factors to pack data. This restores:
 * - Optical bands to Surface Reflectance (0.0 - 1.0)
 * - Thermal bands to Surface Temperature (Kelvin)
 */
function applyScaleFactors(image) {
  var opticalBands = image.select("SR_B.*").multiply(0.0000275).add(-0.2);
  var thermalBands = image.select("ST_B.*").multiply(0.00341802).add(149.0);
  
  // "true" argument overwrites the original bands to save memory
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBands, null, true);
}

/**
 * Masking strategy for OLI/TIRS (Landsat 8/9).
 * Uses the QA_PIXEL bitmask to exclude clouds (bit 3), shadows (bit 4), and cirrus (bit 2).
 */
function cloudMaskL89(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0)
    .and(qa.bitwiseAnd(1 << 4).eq(0))
    .and(qa.bitwiseAnd(1 << 2).eq(0));
  
  return image.updateMask(mask);
}

/**
 * Masking strategy for TM/ETM+ (Landsat 5/7).
 * Slightly different bit mapping; checks clouds (bit 3) and shadows (bit 4).
 */
function cloudMaskL57(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0)
    .and(qa.bitwiseAnd(1 << 4).eq(0));
  
  return image.updateMask(mask);
}

// Dynamically select the correct masking function based on the sensor ID
var cloudMask = (SENSOR_NAME === 'L5' || SENSOR_NAME === 'L7') 
  ? cloudMaskL57 
  : cloudMaskL89;

/**
 * Wrapper for the standard Export task.
 * Ensures consistent CRS and resolution across all exports.
 */
function createExportTask(image, id) {
  Export.image.toDrive({
    image: image,
    description: id,
    folder: EXPORT_CONFIG.folder,
    fileNamePrefix: id,
    scale: EXPORT_CONFIG.scale,
    region: aoi,
    maxPixels: 1e13,
    crs: EXPORT_CONFIG.crs
  });
}

// 3. TEMPORAL DEFINITIONS

// We use 6-month dry/wet season windows to ensure we have enough pixels 
// to form a clean median composite.
var intervals = [
  ['1985-07-01', '1985-12-31'],
  ['1987-07-01', '1987-12-31'],
  ['1989-07-01', '1989-12-31'],
  ['1991-07-01', '1991-12-31'],
  ['1993-07-01', '1993-12-31'],
  ['1995-07-01', '1995-12-31'],
  ['1997-07-01', '1997-12-31'],
  ['1999-07-01', '1999-12-31'],
  ['2001-07-01', '2001-12-31'],
  ['2003-07-01', '2003-12-31'],
  ['2005-07-01', '2005-12-31'],
  ['2007-07-01', '2007-12-31'],
  ['2009-07-01', '2009-12-31'],
  ['2011-07-01', '2011-12-31'],
  ['2013-07-01', '2013-12-31'],
  ['2015-07-01', '2015-12-31'],
  ['2017-07-01', '2017-12-31'],
  ['2019-07-01', '2019-12-31'],
  ['2021-07-01', '2021-12-31'],
  ['2023-07-01', '2023-12-31'],
  ['2025-07-01', '2025-12-31']
];

// 4. MAIN PROCESSING LOOP

print('Initializing batch export for ' + intervals.length + ' periods...');

intervals.forEach(function(interval) {
  var startDate = interval[0];
  var endDate = interval[1];
  
  // Pipeline: Filter -> Mask -> Scale -> Composite
  // We use median compositing to robustly remove outliers (clouds/shadows) that escaped the mask.
  var collection = ee.ImageCollection(COLLECTION_ID)
    .filterBounds(aoi)
    .filterDate(startDate, endDate)
    .map(cloudMask)
    .map(applyScaleFactors);
  
  // Debug check: ensure we actually have data for this sensor/period
  // (Note: calling .size() can be slow on large collections, but fine for debug here)
  print('Processing window: ' + startDate + ' | Image count: ', collection.size());
  
  var composite = collection.median()
    .clip(aoi)
    .toFloat(); // Ensure consistent data type for export
  
  var description = SENSOR_NAME + '_GeoTIFF_' + startDate + '_' + endDate + '_cropped';
  
  createExportTask(composite, description);
});

print('Batch initialization complete. Please check the "Tasks" tab.');

// 5. VISUALIZATION (PREVIEW)

var lastInterval = intervals[intervals.length - 1];
var lastComposite = ee.ImageCollection(COLLECTION_ID)
  .filterBounds(aoi)
  .filterDate(lastInterval[0], lastInterval[1])
  .map(cloudMask)
  .map(applyScaleFactors)
  .median()
  .clip(aoi);

// True color visualization parameters
var visParams = {
  bands: ['SR_B4', 'SR_B3', 'SR_B2'],
  min: 0,
  max: 0.3,
  gamma: 1.4
};

// Thermal visualization parameters
var thermalParams = {
  bands: ['ST_B10'],
  min: 273,
  max: 323,
  palette: ['blue', 'cyan', 'yellow', 'orange', 'red']
};

Map.addLayer(lastComposite, visParams, 'Latest Composite (True Color)');
Map.addLayer(lastComposite, thermalParams, 'Latest Composite (Thermal)');
