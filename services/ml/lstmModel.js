// src/services/lstmModel.js
// Complete implementation with TensorFlow.js

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

let backendInitialized = false;

const ensureTensorflowBackend = async () => {
  if (backendInitialized) {
    return;
  }

  await tf.setBackend('tensorflow');
  await tf.ready();
  backendInitialized = true;

  console.log('[bottomPeak:lstm] TensorFlow backend:', tf.getBackend());
};

/**
 * Normalizes features for LSTM model
 * @param {Array} features - Array of feature objects
 * @returns {Object} - Object containing normalized features and normalization parameters
 */
export const normalizeFeatures = (features) => {
  // Extract relevant columns for normalization
  const columns = [
    'sma20Ratio', 'sma50Ratio', 'sma200Ratio', 
    'macd', 'macdSignal', 'macdHistogram',
    'rsi', 'volatility', 'volumeRatio', 'pricePosition'
  ];
  
  // Calculate min and max for each column
  const normParams = {};
  
  columns.forEach(col => {
    const values = features.map(f => f[col]);
    normParams[col] = {
      min: Math.min(...values),
      max: Math.max(...values)
    };
  });
  
  // Normalize features
  const normalizedFeatures = features.map(feature => {
    const normalized = { date: feature.date, close: feature.close };
    
    columns.forEach(col => {
      const params = normParams[col];
      // Min-max normalization to range [0, 1]
      normalized[col] = (feature[col] - params.min) / (params.max - params.min);
      
      // Handle edge cases where min equals max
      if (isNaN(normalized[col])) {
        normalized[col] = 0.5; // Set to middle of range
      }
    });
    
    // Labels don't need normalization (already 0 or 1)
    normalized.bottomSignal = feature.bottomSignal;
    normalized.peakSignal = feature.peakSignal;
    
    return normalized;
  });
  
  return {
    normalizedFeatures,
    normParams
  };
};

/**
 * Prepares training data for LSTM
 * @param {Array} normalizedFeatures - Array of normalized feature objects
 * @param {Number} sequenceLength - Length of sequences for LSTM
 * @returns {Object} - Object containing input sequences and labels
 */
export const prepareSequences = (normalizedFeatures, sequenceLength = 20) => {
  const inputColumns = [
    'sma20Ratio', 'sma50Ratio', 'sma200Ratio', 
    'macd', 'macdSignal', 'macdHistogram',
    'rsi', 'volatility', 'volumeRatio', 'pricePosition'
  ];
  
  const sequences = [];
  const bottomLabels = [];
  const peakLabels = [];
  const dates = [];
  const prices = [];
  
  // Create sequences of specified length
  for (let i = sequenceLength; i < normalizedFeatures.length; i++) {
    const sequence = [];
    
    // Get sequence of previous data points
    for (let j = i - sequenceLength; j < i; j++) {
      const featureRow = [];
      
      // Add each input column to the feature row
      inputColumns.forEach(col => {
        featureRow.push(normalizedFeatures[j][col]);
      });
      
      sequence.push(featureRow);
    }
    
    sequences.push(sequence);
    bottomLabels.push(normalizedFeatures[i].bottomSignal);
    peakLabels.push(normalizedFeatures[i].peakSignal);
    dates.push(normalizedFeatures[i].date);
    prices.push(normalizedFeatures[i].close);
  }
  
  return {
    sequences,
    bottomLabels,
    peakLabels,
    dates,
    prices
  };
};

/**
 * Creates one-hot encoded labels for multi-class classification
 * @param {Array} bottomLabels - Array of binary bottom signals
 * @param {Array} peakLabels - Array of binary peak signals
 * @returns {Array} - Array of one-hot encoded labels
 */
export const createOneHotLabels = (bottomLabels, peakLabels) => {
  return bottomLabels.map((bottom, i) => {
    if (bottom === 1) return [0, 1, 0]; // Bottom
    if (peakLabels[i] === 1) return [0, 0, 1]; // Peak
    return [1, 0, 0]; // Neither
  });
};

/**
 * Builds and trains LSTM model for bottom/peak detection
 * @param {Array} features - Extracted technical features
 * @param {Number} epochs - Number of training epochs
 * @param {Number} batchSize - Batch size for training
 * @param {Object} customCallbacks - Custom callbacks for training
 * @returns {Object} - Object containing trained model and training history
 */
export const buildAndTrainLSTMModel = async (features, epochs = 50, batchSize = 32, customCallbacks = {}) => {
  await ensureTensorflowBackend();

  // Normalize features
  const { normalizedFeatures, normParams } = normalizeFeatures(features);
  
  // Prepare sequences
  const sequenceLength = 20;
  const { sequences, bottomLabels, peakLabels, dates, prices } = prepareSequences(normalizedFeatures, sequenceLength);
  
  // Create one-hot encoded labels for multi-class classification
  const oneHotLabels = createOneHotLabels(bottomLabels, peakLabels);

  const labelCounts = oneHotLabels.reduce(
    (acc, label) => {
      const classIdx = label[1] === 1 ? 1 : label[2] === 1 ? 2 : 0;
      acc[classIdx] += 1;
      return acc;
    },
    [0, 0, 0]
  );
  const totalLabels = oneHotLabels.length;
  const classWeights = labelCounts.map((count) => {
    if (!count) return 1;
    return totalLabels / (labelCounts.length * count);
  });

  // Convert data to tensors
  const inputDim = sequences[0][0].length; // Number of features per time step
  
  // Create the model
  const model = tf.sequential();
  
  // Add LSTM layer
  model.add(tf.layers.lstm({
    units: 64,
    returnSequences: true,
    inputShape: [sequenceLength, inputDim]
  }));
  
  // Add another LSTM layer
  model.add(tf.layers.lstm({
    units: 32,
    returnSequences: false
  }));
  
  // Add dropout to prevent overfitting
  model.add(tf.layers.dropout({ rate: 0.2 }));
  
  // Add dense layer
  model.add(tf.layers.dense({
    units: 16,
    activation: 'relu'
  }));
  
  // Add output layer
  model.add(tf.layers.dense({
    units: 3, // 3 classes: neither, bottom, peak
    activation: 'softmax'
  }));
  
  // Compile the model
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  // Prepare validation data (20% of the data)
  const splitIdx = Math.floor(sequences.length * 0.8);

  const trainSequences = sequences.slice(0, splitIdx);
  const trainLabels = oneHotLabels.slice(0, splitIdx);
  const valSequences = sequences.slice(splitIdx);
  const valLabels = oneHotLabels.slice(splitIdx);

  const classBuckets = [[], [], []];
  trainLabels.forEach((label, idx) => {
    const classIdx = label[1] === 1 ? 1 : label[2] === 1 ? 2 : 0;
    classBuckets[classIdx].push(idx);
  });

  const maxClassCount = Math.max(...classBuckets.map((bucket) => bucket.length));
  const balancedIndices = [];
  classBuckets.forEach((bucket) => {
    if (!bucket.length) return;
    for (let i = 0; i < maxClassCount; i += 1) {
      balancedIndices.push(bucket[i % bucket.length]);
    }
  });

  for (let i = balancedIndices.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [balancedIndices[i], balancedIndices[j]] = [balancedIndices[j], balancedIndices[i]];
  }

  const balancedTrainSequences = balancedIndices.map((idx) => trainSequences[idx]);
  const balancedTrainLabels = balancedIndices.map((idx) => trainLabels[idx]);

  const xTrain = tf.tensor3d(balancedTrainSequences);
  const yTrain = tf.tensor2d(balancedTrainLabels, [balancedTrainLabels.length, 3], 'float32');
  const xVal = tf.tensor3d(valSequences);
  const yVal = tf.tensor2d(valLabels, [valLabels.length, 3], 'float32');
  
  // Log model summary
  model.summary();
  
  // Create an object to store training history
  const trainHistory = {
    loss: [],
    acc: [],
    val_loss: [],
    val_acc: []
  };

  console.log('[bottomPeak:lstm] Label distribution:', {
    neither: labelCounts[0],
    bottom: labelCounts[1],
    peak: labelCounts[2],
    total: totalLabels,
    classWeights,
  });

  console.log('[bottomPeak:lstm] Training balance strategy:', {
    strategy: 'oversample-minority-classes',
    originalTrainSize: trainSequences.length,
    balancedTrainSize: balancedTrainSequences.length,
    trainClassCounts: {
      neither: classBuckets[0].length,
      bottom: classBuckets[1].length,
      peak: classBuckets[2].length,
    },
    targetPerClass: maxClassCount,
  });
  
  // Define default callbacks
  const defaultCallbacks = {
    onEpochEnd: (epoch, logs) => {
      const loss = logs.loss ?? null;
      const acc = logs.acc ?? logs.accuracy ?? null;
      const valLoss = logs.val_loss ?? logs.valLoss ?? null;
      const valAcc = logs.val_acc ?? logs.val_accuracy ?? logs.valAccuracy ?? null;

      console.log('[bottomPeak:lstm] Epoch metrics:', {
        epoch: epoch + 1,
        epochs,
        loss,
        accuracy: acc,
        valLoss,
        valAccuracy: valAcc,
      });
      
      // Store training metrics
      trainHistory.loss.push(loss);
      trainHistory.acc.push(acc);
      trainHistory.val_loss.push(valLoss);
      trainHistory.val_acc.push(valAcc);
    }
  };
  
  // Merge default and custom callbacks
  const callbacks = {
    ...defaultCallbacks,
    ...customCallbacks
  };
  
  // Train the model
  const history = await model.fit(xTrain, yTrain, {
    epochs: epochs,
    batchSize: batchSize,
    validationData: [xVal, yVal],
    callbacks: callbacks
  });
  
  // Clean up tensors
  xTrain.dispose();
  yTrain.dispose();
  xVal.dispose();
  yVal.dispose();
  
  return {
    model,
    history: trainHistory,
    normParams,
    sequenceLength,
    inputDim,
    labelStats: {
      counts: {
        neither: labelCounts[0],
        bottom: labelCounts[1],
        peak: labelCounts[2],
      },
      classWeights,
      total: totalLabels,
    },
  };
};

/**
 * Applies normalization to new feature data using existing normalization parameters
 * @param {Array} features - New features to normalize
 * @param {Object} normParams - Normalization parameters from training
 * @returns {Array} - Normalized features
 */
export const normalizeNewFeatures = (features, normParams) => {
  return features.map(feature => {
    const normalized = { date: feature.date, close: feature.close };
    
    Object.keys(normParams).forEach(col => {
      if (feature[col] !== undefined) {
        const params = normParams[col];
        // Min-max normalization to range [0, 1]
        normalized[col] = (feature[col] - params.min) / (params.max - params.min);
        
        // Clamp values to [0, 1] range for out-of-range values
        normalized[col] = Math.max(0, Math.min(1, normalized[col]));
        
        // Handle edge cases
        if (isNaN(normalized[col])) {
          normalized[col] = 0.5; // Set to middle of range
        }
      }
    });
    
    return normalized;
  });
};

/**
 * Predicts bottoms and peaks on new data
 * @param {Object} model - Trained LSTM model
 * @param {Array} features - New features to predict on
 * @param {Object} normParams - Normalization parameters
 * @param {Number} sequenceLength - Length of sequences used in model
 * @param {Number} threshold - Probability threshold for signals
 * @returns {Array} - Array of prediction results
 */
export const predictBottomsPeaks = async (model, features, normParams, sequenceLength, threshold = 0.7) => {
  await ensureTensorflowBackend();

  const modelInstance =
    model && typeof model.predict === 'function'
      ? model
      : model?.model && typeof model.model.predict === 'function'
        ? model.model
        : null;

  if (!modelInstance || !features || features.length < sequenceLength) {
    throw new Error(`Insufficient data for prediction. Need at least ${sequenceLength} data points.`);
  }
  
  // Normalize features
  const normalizedFeatures = normalizeNewFeatures(features, normParams);
  
  // Prepare sequences
  const { sequences, dates, prices } = prepareSequences(normalizedFeatures, sequenceLength);
  
  if (sequences.length === 0) {
    return [];
  }
  
  // Convert to tensor
  const inputTensor = tf.tensor3d(sequences);
  
  // Make predictions
  const predictions = modelInstance.predict(inputTensor);
  const predictionValues = await predictions.array();
  
  // Clean up tensors
  inputTensor.dispose();
  predictions.dispose();
  
  // Process predictions to determine bottoms and peaks
  const results = predictionValues.map((pred, i) => {
    return {
      date: dates[i],
      price: prices[i],
      isBottom: pred[1] > threshold,
      isPeak: pred[2] > threshold,
      bottomProb: pred[1],
      peakProb: pred[2],
      neitherProb: pred[0]
    };
  });
  
  return results;
};

/**
 * Saves the trained model and its metadata to IndexedDB
 * @param {Object} model - Trained TensorFlow.js model
 * @param {Object} normParams - Normalization parameters
 * @param {Number} sequenceLength - Sequence length used in training
 * @param {String} modelName - Name to save the model under
 * @returns {Promise} - Promise that resolves when the model is saved
 */
export const saveModel = async (model, normParams, sequenceLength, modelName = 'lstm-bottom-peak-model') => {
  try {
    // Save the TensorFlow.js model
    const saveResult = await model.save(`indexeddb://${modelName}`);
    
    // Save the metadata separately
    const metadata = {
      normParams,
      sequenceLength,
      savedAt: new Date().toISOString()
    };
    
    // Use localStorage to save metadata since IndexedDB is more complex for simple objects
    localStorage.setItem(`${modelName}-metadata`, JSON.stringify(metadata));
    
    console.log('Model and metadata saved successfully:', saveResult);
    return saveResult;
  } catch (error) {
    console.error('Error saving model:', error);
    throw error;
  }
};

/**
 * Loads a previously saved model and its metadata from IndexedDB
 * @param {String} modelName - Name of the saved model
 * @returns {Promise<Object>} - Promise that resolves with the loaded model and metadata
 */
export const loadModel = async (modelName = 'lstm-bottom-peak-model') => {
  try {
    // Load the TensorFlow.js model
    const model = await tf.loadLayersModel(`indexeddb://${modelName}`);
    
    // Load the metadata
    const metadataStr = localStorage.getItem(`${modelName}-metadata`);
    if (!metadataStr) {
      throw new Error('Model metadata not found. Please retrain the model.');
    }
    
    const metadata = JSON.parse(metadataStr);
    
    console.log('Model and metadata loaded successfully');
    return {
      model,
      normParams: metadata.normParams,
      sequenceLength: metadata.sequenceLength,
      savedAt: metadata.savedAt
    };
  } catch (error) {
    console.error('Error loading model:', error);
    throw error;
  }
};

/**
 * Checks if a model with the given name and its metadata exist
 * @param {String} modelName - Name of the model to check
 * @returns {Promise<boolean>} - Promise that resolves with a boolean indicating if the model and metadata exist
 */
export const modelExists = async (modelName = 'lstm-bottom-peak-model') => {
  try {
    // Check if the TensorFlow.js model exists
    const models = await tf.io.listModels();
    const modelExists = Boolean(models[`indexeddb://${modelName}`]);
    
    // Check if the metadata exists
    const metadataExists = Boolean(localStorage.getItem(`${modelName}-metadata`));
    
    // Both must exist for the model to be considered complete
    return modelExists && metadataExists;
  } catch (error) {
    console.error('Error checking model existence:', error);
    return false;
  }
};

/**
 * Deletes a saved model and its metadata
 * @param {String} modelName - Name of the model to delete
 * @returns {Promise<boolean>} - Promise that resolves with success status
 */
export const deleteModel = async (modelName = 'lstm-bottom-peak-model') => {
  try {
    // Delete the TensorFlow.js model from IndexedDB
    await tf.io.removeModel(`indexeddb://${modelName}`);
    
    // Delete the metadata from localStorage
    localStorage.removeItem(`${modelName}-metadata`);
    
    console.log('Model and metadata deleted successfully');
    return true;
  } catch (error) {
    console.error('Error deleting model:', error);
    return false;
  }
};
/** 
 * @param {Object} model - Trained LSTM model
 * @param {Array} testFeatures - Test features
 * @param {Array} testBottomLabels - True bottom labels for test data
 * @param {Array} testPeakLabels - True peak labels for test data
 * @returns {Object} - Performance metrics
 */
export const evaluateModel = async (model, testFeatures, testBottomLabels, testPeakLabels) => {
  const oneHotLabels = createOneHotLabels(testBottomLabels, testPeakLabels);
  
  // Convert to tensors
  const xTest = tf.tensor3d(testFeatures);
  const yTest = tf.tensor2d(oneHotLabels, [oneHotLabels.length, 3], 'float32');
  
  // Evaluate model
  const evaluation = await model.evaluate(xTest, yTest);
  
  // Get predictions for metrics calculation
  const predictions = model.predict(xTest);
  const predictionValues = await predictions.argMax(1).array();
  
  // Convert one-hot labels back to class indices
  const trueLabels = oneHotLabels.map(label => {
    return label.indexOf(Math.max(...label));
  });
  
  // Calculate metrics
  const confusionMatrix = [
    [0, 0, 0], // Actual: neither, Predicted: [neither, bottom, peak]
    [0, 0, 0], // Actual: bottom, Predicted: [neither, bottom, peak]
    [0, 0, 0]  // Actual: peak, Predicted: [neither, bottom, peak]
  ];
  
  for (let i = 0; i < predictionValues.length; i++) {
    const actual = trueLabels[i];
    const predicted = predictionValues[i];
    confusionMatrix[actual][predicted]++;
  }
  
  // Calculate precision, recall and F1 for each class
  const metrics = {
    loss: (await evaluation[0].data())[0],
    accuracy: (await evaluation[1].data())[0],
    confusionMatrix
  };
  
  // Calculate per-class metrics
  const classNames = ['neither', 'bottom', 'peak'];
  const classMetrics = {};
  
  classNames.forEach((className, i) => {
    const truePositives = confusionMatrix[i][i];
    const falsePositives = confusionMatrix.reduce((sum, row, idx) => sum + (idx !== i ? row[i] : 0), 0);
    const falseNegatives = confusionMatrix[i].reduce((sum, val, idx) => sum + (idx !== i ? val : 0), 0);
    
    const precision = truePositives / (truePositives + falsePositives) || 0;
    const recall = truePositives / (truePositives + falseNegatives) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    classMetrics[className] = { precision, recall, f1 };
  });
  
  metrics.classMetrics = classMetrics;
  
  // Clean up tensors
  xTest.dispose();
  yTest.dispose();
  evaluation.forEach(t => t.dispose());
  predictions.dispose();
  
  return metrics;
};