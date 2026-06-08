// Enhanced Long-Term Random Forest Implementation with Feature Selection
// src/services/longTermRandomForestService.js
// Includes feature importance filtering, redundancy removal, and noise filtering

import { extractTechnicalFeatures } from './technicalIndicators';

// Feature Selection and Filtering Utilities
class FeatureSelector {
  constructor() {
    this.selectedFeatureIndices = null;
    this.featureNames = null;
    this.correlationMatrix = null;
    this.importanceThreshold = 0.001; // Minimum importance to keep feature
    this.correlationThreshold = 0.85; // Maximum correlation to keep both features
    this.varianceThreshold = 0.0001; // Minimum variance to keep feature
  }

  // Calculate correlation matrix
  calculateCorrelationMatrix(features) {
    const numFeatures = features[0].length;
    const correlationMatrix = Array(numFeatures).fill().map(() => Array(numFeatures).fill(0));
    
    for (let i = 0; i < numFeatures; i++) {
      for (let j = i; j < numFeatures; j++) {
        const correlation = this.calculateCorrelation(features, i, j);
        correlationMatrix[i][j] = correlation;
        correlationMatrix[j][i] = correlation;
      }
    }
    
    return correlationMatrix;
  }

  // Calculate Pearson correlation between two features
  calculateCorrelation(features, featureI, featureJ) {
    const valuesI = features.map(sample => sample[featureI]);
    const valuesJ = features.map(sample => sample[featureJ]);
    
    const meanI = valuesI.reduce((sum, val) => sum + val, 0) / valuesI.length;
    const meanJ = valuesJ.reduce((sum, val) => sum + val, 0) / valuesJ.length;
    
    let numerator = 0;
    let denominatorI = 0;
    let denominatorJ = 0;
    
    for (let k = 0; k < valuesI.length; k++) {
      const diffI = valuesI[k] - meanI;
      const diffJ = valuesJ[k] - meanJ;
      
      numerator += diffI * diffJ;
      denominatorI += diffI * diffI;
      denominatorJ += diffJ * diffJ;
    }
    
    const denominator = Math.sqrt(denominatorI * denominatorJ);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  // Calculate feature variance
  calculateVariance(features, featureIndex) {
    const values = features.map(sample => sample[featureIndex]);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + (val - mean) ** 2, 0) / values.length;
    return variance;
  }

  // Remove low-variance features (noise filtering)
  filterLowVarianceFeatures(features, featureNames) {
    const numFeatures = features[0].length;
    const keepIndices = [];
    const keepNames = [];
    
    for (let i = 0; i < numFeatures; i++) {
      const variance = this.calculateVariance(features, i);
      if (variance > this.varianceThreshold) {
        keepIndices.push(i);
        keepNames.push(featureNames[i]);
      }
    }
    
    console.log(`Variance filtering: kept ${keepIndices.length}/${numFeatures} features`);
    return { keepIndices, keepNames };
  }

  // Remove highly correlated features (redundancy removal)
  filterCorrelatedFeatures(features, featureNames, keepIndices, featureImportances = null) {
    if (keepIndices.length <= 1) return { keepIndices, keepNames: featureNames };
    
    // Calculate correlation matrix for remaining features
    const filteredFeatures = features.map(sample => 
      keepIndices.map(idx => sample[idx])
    );
    
    const correlationMatrix = this.calculateCorrelationMatrix(filteredFeatures);
    const finalKeepIndices = [];
    const finalKeepNames = [];
    const processedPairs = new Set();
    
    for (let i = 0; i < keepIndices.length; i++) {
      let shouldKeep = true;
      
      for (let j = 0; j < i; j++) {
        const pairKey = `${Math.min(i, j)}-${Math.max(i, j)}`;
        if (processedPairs.has(pairKey)) continue;
        
        const correlation = Math.abs(correlationMatrix[i][j]);
        
        if (correlation > this.correlationThreshold) {
          processedPairs.add(pairKey);
          
          // If we have importance scores, keep the more important feature
          if (featureImportances) {
            const importanceI = featureImportances[keepIndices[i]] || 0;
            const importanceJ = featureImportances[keepIndices[j]] || 0;
            
            if (importanceI < importanceJ) {
              shouldKeep = false;
              break;
            }
          } else {
            // Otherwise keep the first one (arbitrary but consistent)
            shouldKeep = false;
            break;
          }
        }
      }
      
      if (shouldKeep) {
        finalKeepIndices.push(keepIndices[i]);
        finalKeepNames.push(featureNames[i]);
      }
    }
    
    console.log(`Correlation filtering: kept ${finalKeepIndices.length}/${keepIndices.length} features`);
    return { keepIndices: finalKeepIndices, keepNames: finalKeepNames };
  }

  // Filter features by importance (feature importance filtering)
  filterByImportance(features, featureNames, featureImportances, topN = 50) {
    if (!featureImportances) {
      console.log('No feature importances available, skipping importance filtering');
      return { keepIndices: Array.from({length: features[0].length}, (_, i) => i), keepNames: featureNames };
    }
    
    // Create array of {index, importance, name} and sort by importance
    const importanceData = featureImportances.map((importance, index) => ({
      index,
      importance: importance || 0,
      name: featureNames[index]
    })).sort((a, b) => b.importance - a.importance);
    
    // Keep top N features or features above threshold
    const keepData = importanceData.filter((item, index) => 
      index < topN && item.importance > this.importanceThreshold
    );
    
    const keepIndices = keepData.map(item => item.index);
    const keepNames = keepData.map(item => item.name);
    
    console.log(`Importance filtering: kept ${keepIndices.length}/${featureImportances.length} features`);
    console.log(`Top 10 features by importance:`, keepData.slice(0, 10).map(item => 
      `${item.name}: ${item.importance.toFixed(4)}`
    ));
    
    return { keepIndices, keepNames };
  }

  // Complete feature selection pipeline
  selectFeatures(features, featureNames, featureImportances = null, maxFeatures = 50) {
    console.log(`Starting feature selection with ${features[0].length} features`);
    
    // Step 1: Remove low variance features (noise filtering)
    let { keepIndices, keepNames } = this.filterLowVarianceFeatures(features, featureNames);
    
    // Step 2: Feature importance filtering (if available)
    if (featureImportances) {
      const importanceResult = this.filterByImportance(features, featureNames, featureImportances, maxFeatures * 2);
      
      // Intersect with variance-filtered features
      const importanceSet = new Set(importanceResult.keepIndices);
      keepIndices = keepIndices.filter(idx => importanceSet.has(idx));
      keepNames = keepIndices.map(idx => featureNames[idx]);
      
      console.log(`After importance filtering: ${keepIndices.length} features`);
    }
    
    // Step 3: Remove correlated features (redundancy removal)
    const finalResult = this.filterCorrelatedFeatures(features, featureNames, keepIndices, featureImportances);
    
    // If still too many features, take top ones by importance
    if (finalResult.keepIndices.length > maxFeatures && featureImportances) {
      const importanceMap = {};
      featureImportances.forEach((imp, idx) => {
        importanceMap[idx] = imp || 0;
      });
      
      const sortedIndices = finalResult.keepIndices.sort((a, b) => 
        importanceMap[b] - importanceMap[a]
      );
      
      finalResult.keepIndices = sortedIndices.slice(0, maxFeatures);
      finalResult.keepNames = finalResult.keepIndices.map(idx => featureNames[idx]);
      
      console.log(`Final truncation: kept top ${finalResult.keepIndices.length} features`);
    }
    
    this.selectedFeatureIndices = finalResult.keepIndices;
    this.featureNames = finalResult.keepNames;
    
    console.log(`Feature selection complete: ${finalResult.keepIndices.length} features selected`);
    return finalResult;
  }

  // Apply feature selection to new data
  applySelection(features) {
    if (!this.selectedFeatureIndices) {
      throw new Error('Feature selection not performed yet. Call selectFeatures first.');
    }
    
    return features.map(sample => 
      this.selectedFeatureIndices.map(idx => sample[idx])
    );
  }

  // Apply to single sample
  applySingleSample(sample) {
    if (!this.selectedFeatureIndices) {
      throw new Error('Feature selection not performed yet. Call selectFeatures first.');
    }
    
    return this.selectedFeatureIndices.map(idx => sample[idx]);
  }
}

// Decision Tree Node for Classification
class ClassificationTreeNode {
  constructor() {
    this.feature = null;
    this.threshold = null;
    this.left = null;
    this.right = null;
    this.value = null; // For leaf nodes - most common class
    this.samples = 0;
    this.impurity = 0; // Gini impurity for this node
  }
  
  isLeaf() {
    return this.value !== null;
  }
}

// Decision Tree Implementation for 5-Class Classification
class LongTermDecisionTree {
  constructor(maxDepth = 15, minSamplesSplit = 5, minSamplesLeaf = 3) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.minSamplesLeaf = minSamplesLeaf;
    this.root = null;
    this.featureImportances = null;
  }
  
  // Calculate Gini impurity for 5 classes
  calculateGini(labels) {
    if (labels.length === 0) return 0;
    
    const counts = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
    labels.forEach(label => {
      counts[label] = (counts[label] || 0) + 1;
    });
    
    let gini = 1;
    const total = labels.length;
    
    Object.values(counts).forEach(count => {
      const probability = count / total;
      gini -= probability * probability;
    });
    
    return gini;
  }
  
  // Find best split for a feature
  findBestSplit(features, labels, featureIndex) {
    const values = features.map(sample => sample[featureIndex]);
    const uniqueValues = [...new Set(values)].sort((a, b) => a - b);
    
    let bestThreshold = null;
    let bestGain = -1;
    
    for (let i = 0; i < uniqueValues.length - 1; i++) {
      const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
      
      const leftIndices = [];
      const rightIndices = [];
      
      for (let j = 0; j < features.length; j++) {
        if (features[j][featureIndex] <= threshold) {
          leftIndices.push(j);
        } else {
          rightIndices.push(j);
        }
      }
      
      if (leftIndices.length === 0 || rightIndices.length === 0) continue;
      
      const leftLabels = leftIndices.map(idx => labels[idx]);
      const rightLabels = rightIndices.map(idx => labels[idx]);
      
      const parentGini = this.calculateGini(labels);
      const leftGini = this.calculateGini(leftLabels);
      const rightGini = this.calculateGini(rightLabels);
      
      const weightedGini = (leftLabels.length / labels.length) * leftGini + 
                          (rightLabels.length / labels.length) * rightGini;
      
      const gain = parentGini - weightedGini;
      
      if (gain > bestGain) {
        bestGain = gain;
        bestThreshold = threshold;
      }
    }
    
    return { threshold: bestThreshold, gain: bestGain };
  }
  
  // Find best split across all features
  findBestFeatureSplit(features, labels, availableFeatures) {
    let bestFeature = null;
    let bestThreshold = null;
    let bestGain = -1;
    
    availableFeatures.forEach(featureIndex => {
      const split = this.findBestSplit(features, labels, featureIndex);
      
      if (split.gain > bestGain) {
        bestGain = split.gain;
        bestFeature = featureIndex;
        bestThreshold = split.threshold;
      }
    });
    
    return { feature: bestFeature, threshold: bestThreshold, gain: bestGain };
  }
  
  // Build tree recursively
  buildTree(features, labels, depth = 0, availableFeatures = null) {
    const node = new ClassificationTreeNode();
    node.samples = features.length;
    node.impurity = this.calculateGini(labels);
    
    // Initialize available features on first call
    if (availableFeatures === null) {
      availableFeatures = Array.from({length: features[0].length}, (_, i) => i);
    }
    
    // Check stopping criteria
    if (depth >= this.maxDepth || 
        features.length < this.minSamplesSplit ||
        node.impurity === 0 ||
        availableFeatures.length === 0) {
      
      // Create leaf node - find most common class
      const counts = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
      labels.forEach(label => {
        counts[label] = (counts[label] || 0) + 1;
      });
      
      node.value = parseInt(Object.keys(counts).reduce((a, b) => 
        counts[a] > counts[b] ? a : b
      ));
      
      return node;
    }
    
    // Find best split
    const bestSplit = this.findBestFeatureSplit(features, labels, availableFeatures);
    
    if (bestSplit.feature === null || bestSplit.gain <= 0) {
      // No good split found, create leaf
      const counts = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
      labels.forEach(label => {
        counts[label] = (counts[label] || 0) + 1;
      });
      
      node.value = parseInt(Object.keys(counts).reduce((a, b) => 
        counts[a] > counts[b] ? a : b
      ));
      
      return node;
    }
    
    node.feature = bestSplit.feature;
    node.threshold = bestSplit.threshold;
    
    // Split data
    const leftFeatures = [];
    const leftLabels = [];
    const rightFeatures = [];
    const rightLabels = [];
    
    for (let i = 0; i < features.length; i++) {
      if (features[i][node.feature] <= node.threshold) {
        leftFeatures.push(features[i]);
        leftLabels.push(labels[i]);
      } else {
        rightFeatures.push(features[i]);
        rightLabels.push(labels[i]);
      }
    }
    
    // Check minimum samples per leaf
    if (leftFeatures.length < this.minSamplesLeaf || rightFeatures.length < this.minSamplesLeaf) {
      const counts = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
      labels.forEach(label => {
        counts[label] = (counts[label] || 0) + 1;
      });
      
      node.value = parseInt(Object.keys(counts).reduce((a, b) => 
        counts[a] > counts[b] ? a : b
      ));
      
      return node;
    }
    
    // Recursively build subtrees
    node.left = this.buildTree(leftFeatures, leftLabels, depth + 1, availableFeatures);
    node.right = this.buildTree(rightFeatures, rightLabels, depth + 1, availableFeatures);
    
    return node;
  }
  
  // Train the tree
  train(features, labels) {
    this.root = this.buildTree(features, labels);
    this.calculateFeatureImportances(features, labels);
    return this;
  }
  
  // Calculate feature importances
  calculateFeatureImportances(features, labels) {
    const numFeatures = features[0].length;
    this.featureImportances = new Array(numFeatures).fill(0);
    
    const calculateNodeImportance = (node, totalSamples) => {
      if (node.isLeaf()) return;
      
      const importance = (node.samples / totalSamples) * node.impurity;
      this.featureImportances[node.feature] += importance;
      
      if (node.left) calculateNodeImportance(node.left, totalSamples);
      if (node.right) calculateNodeImportance(node.right, totalSamples);
    };
    
    if (this.root) {
      calculateNodeImportance(this.root, this.root.samples);
      
      // Normalize importances
      const sum = this.featureImportances.reduce((a, b) => a + b, 0);
      if (sum > 0) {
        this.featureImportances = this.featureImportances.map(imp => imp / sum);
      }
    }
  }
  
  // Predict single sample
  predictSample(sample) {
    let node = this.root;
    
    while (!node.isLeaf()) {
      if (sample[node.feature] <= node.threshold) {
        node = node.left;
      } else {
        node = node.right;
      }
    }
    
    return node.value;
  }
  
  // Predict multiple samples
  predict(features) {
    return features.map(sample => this.predictSample(sample));
  }
}

// Enhanced 5-Class Random Forest Classifier with Feature Selection
export class LongTermRandomForestClassifier {
  constructor(nTrees = 100, maxDepth = 20, minSamplesSplit = 2, minSamplesLeaf = 1, maxFeatures = 'sqrt', maxSelectedFeatures = 50) {
    this.nTrees = nTrees;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.minSamplesLeaf = minSamplesLeaf;
    this.maxFeatures = maxFeatures;
    this.maxSelectedFeatures = maxSelectedFeatures;
    this.trees = [];
    this.featureImportances = null;
    this.featureSelector = new FeatureSelector();
    this.trained = false;
    this.selectedFeatureNames = null;
  }
  
  // Bootstrap sampling
  bootstrapSample(features, labels) {
    const n = features.length;
    const indices = [];
    
    for (let i = 0; i < n; i++) {
      indices.push(Math.floor(Math.random() * n));
    }
    
    const sampledFeatures = indices.map(idx => features[idx]);
    const sampledLabels = indices.map(idx => labels[idx]);
    
    return { features: sampledFeatures, labels: sampledLabels };
  }
  
  // Select random features for each tree
  getRandomFeatures(numFeatures) {
    let maxFeaturesCount;
    
    if (this.maxFeatures === 'sqrt') {
      maxFeaturesCount = Math.floor(Math.sqrt(numFeatures));
    } else if (this.maxFeatures === 'log2') {
      maxFeaturesCount = Math.floor(Math.log2(numFeatures));
    } else if (typeof this.maxFeatures === 'number') {
      maxFeaturesCount = Math.min(this.maxFeatures, numFeatures);
    } else {
      maxFeaturesCount = numFeatures;
    }
    
    const allFeatures = Array.from({length: numFeatures}, (_, i) => i);
    const selectedFeatures = [];
    
    for (let i = 0; i < maxFeaturesCount; i++) {
      const randomIndex = Math.floor(Math.random() * allFeatures.length);
      selectedFeatures.push(allFeatures.splice(randomIndex, 1)[0]);
    }
    
    return selectedFeatures;
  }
  
  // Train the forest with feature selection
  train(features, labels, featureNames = null) {
    console.log(`Training Enhanced Long-Term Random Forest with ${this.nTrees} trees for 5-class classification...`);
    
    // Step 1: Initial model training to get feature importances
    console.log('Step 1: Training initial model for feature importance calculation...');
    this.trees = [];
    const numFeatures = features[0].length;
    const allFeatureImportances = [];
    
    // Train a smaller forest first to get importance estimates
    const initialTrees = Math.min(20, this.nTrees);
    for (let i = 0; i < initialTrees; i++) {
      const { features: sampledFeatures, labels: sampledLabels } = this.bootstrapSample(features, labels);
      const selectedFeatures = this.getRandomFeatures(numFeatures);
      
      const tree = new LongTermDecisionTree(this.maxDepth, this.minSamplesSplit, this.minSamplesLeaf);
      tree.selectedFeatures = selectedFeatures;
      
      const filteredFeatures = sampledFeatures.map(sample => 
        selectedFeatures.map(featureIdx => sample[featureIdx])
      );
      
      tree.train(filteredFeatures, sampledLabels);
      
      if (tree.featureImportances) {
        const fullImportances = new Array(numFeatures).fill(0);
        selectedFeatures.forEach((featureIdx, treeFeatureIdx) => {
          fullImportances[featureIdx] = tree.featureImportances[treeFeatureIdx];
        });
        allFeatureImportances.push(fullImportances);
      }
    }
    
    // Calculate average feature importances
    const avgImportances = new Array(numFeatures).fill(0);
    allFeatureImportances.forEach(importances => {
      importances.forEach((importance, idx) => {
        avgImportances[idx] += importance;
      });
    });
    this.featureImportances = avgImportances.map(imp => imp / initialTrees);
    
    // Step 2: Feature selection
    console.log('Step 2: Performing feature selection...');
    
    // Generate default feature names if not provided
    if (!featureNames) {
      featureNames = Array.from({length: numFeatures}, (_, i) => `Feature_${i}`);
    }
    
    const selectionResult = this.featureSelector.selectFeatures(
      features, 
      featureNames, 
      this.featureImportances, 
      this.maxSelectedFeatures
    );
    
    this.selectedFeatureNames = selectionResult.keepNames;
    const selectedFeatures = this.featureSelector.applySelection(features);
    
    console.log(`Selected ${selectedFeatures[0].length} features for final training`);
    
    // Step 3: Train final model with selected features
    console.log('Step 3: Training final model with selected features...');
    this.trees = [];
    const selectedFeatureImportances = [];
    
    for (let i = 0; i < this.nTrees; i++) {
      if (i % 20 === 0) {
        console.log(`Training tree ${i + 1}/${this.nTrees}`);
      }
      
      const { features: sampledFeatures, labels: sampledLabels } = this.bootstrapSample(selectedFeatures, labels);
      const randomFeatures = this.getRandomFeatures(selectedFeatures[0].length);
      
      const tree = new LongTermDecisionTree(this.maxDepth, this.minSamplesSplit, this.minSamplesLeaf);
      tree.selectedFeatures = randomFeatures;
      
      const filteredFeatures = sampledFeatures.map(sample => 
        randomFeatures.map(featureIdx => sample[featureIdx])
      );
      
      tree.train(filteredFeatures, sampledLabels);
      this.trees.push(tree);
      
      if (tree.featureImportances) {
        const fullImportances = new Array(selectedFeatures[0].length).fill(0);
        randomFeatures.forEach((featureIdx, treeFeatureIdx) => {
          fullImportances[featureIdx] = tree.featureImportances[treeFeatureIdx];
        });
        selectedFeatureImportances.push(fullImportances);
      }
    }
    
    // Update feature importances for selected features
    const finalImportances = new Array(selectedFeatures[0].length).fill(0);
    selectedFeatureImportances.forEach(importances => {
      importances.forEach((importance, idx) => {
        finalImportances[idx] += importance;
      });
    });
    
    this.featureImportances = finalImportances.map(imp => imp / this.nTrees);
    
    this.trained = true;
    console.log('Enhanced Long-Term Random Forest training completed');
    console.log(`Final model uses ${selectedFeatures[0].length} selected features`);
    
    return this;
  }
  
  // Predict probabilities for 5 classes
  predictProba(features) {
    if (!this.trained) {
      throw new Error('Model not trained yet!');
    }
    
    // Apply feature selection
    const selectedFeatures = this.featureSelector.applySelection(features);
    
    const predictions = selectedFeatures.map(sample => {
      const votes = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
      
      this.trees.forEach(tree => {
        const filteredSample = tree.selectedFeatures.map(featureIdx => sample[featureIdx]);
        const prediction = tree.predictSample(filteredSample);
        votes[prediction]++;
      });
      
      const total = this.nTrees;
      return [
        votes[0] / total, // Big Down
        votes[1] / total, // Small Down  
        votes[2] / total, // Flat
        votes[3] / total, // Small Up
        votes[4] / total  // Big Up
      ];
    });
    
    return predictions;
  }
  
  // Predict classes (0-4)
  predict(features) {
    const probabilities = this.predictProba(features);
    return probabilities.map(probs => {
      let maxProb = -1;
      let maxClass = 0;
      probs.forEach((prob, classIndex) => {
        if (prob > maxProb) {
          maxProb = prob;
          maxClass = classIndex;
        }
      });
      return maxClass;
    });
  }
  
  // Predict with confidence (for ensemble compatibility)
  predictWithConfidence(features) {
    const probabilities = this.predictProba(features);
    const predictions = this.predict(features);
    
    return features.map((_, index) => {
      const probs = probabilities[index];
      const prediction = predictions[index];
      const confidence = Math.max(...probs);
      
      return {
        prediction: prediction,
        probabilities: probs,
        confidence: confidence,
        agreementScore: confidence
      };
    });
  }
  
  // Evaluate model performance for 5-class classification
  evaluate(features, trueLabels) {
    const predictions = this.predict(features);
    const probabilities = this.predictProba(features);
    
    // Overall accuracy
    let correct = 0;
    for (let i = 0; i < predictions.length; i++) {
      if (predictions[i] === trueLabels[i]) {
        correct++;
      }
    }
    const accuracy = correct / predictions.length;
    
    // Per-class metrics
    const classes = [0, 1, 2, 3, 4];
    const classMetrics = {};
    let avgPrecision = 0;
    let avgRecall = 0;
    let avgF1 = 0;
    
    classes.forEach(cls => {
      let tp = 0, fp = 0, fn = 0;
      
      for (let i = 0; i < predictions.length; i++) {
        if (predictions[i] === cls && trueLabels[i] === cls) tp++;
        if (predictions[i] === cls && trueLabels[i] !== cls) fp++;
        if (predictions[i] !== cls && trueLabels[i] === cls) fn++;
      }
      
      const precision = tp / (tp + fp) || 0;
      const recall = tp / (tp + fn) || 0;
      const f1 = 2 * (precision * recall) / (precision + recall) || 0;
      
      classMetrics[cls] = { precision, recall, f1 };
      avgPrecision += precision;
      avgRecall += recall;
      avgF1 += f1;
    });
    
    avgPrecision /= classes.length;
    avgRecall /= classes.length;
    avgF1 /= classes.length;
    
    // Confusion matrix
    const confusionMatrix = Array(5).fill().map(() => Array(5).fill(0));
    for (let i = 0; i < predictions.length; i++) {
      confusionMatrix[trueLabels[i]][predictions[i]]++;
    }
    
    return {
      accuracy,
      precision: avgPrecision,
      recall: avgRecall,
      f1: avgF1,
      classMetrics,
      confusionMatrix,
      sampleCount: predictions.length
    };
  }
  
  // Get feature selection summary
  getFeatureSelectionSummary() {
    return {
      originalFeatureCount: this.featureSelector.selectedFeatureIndices ? 
        this.featureSelector.selectedFeatureIndices.length : 0,
      selectedFeatureCount: this.selectedFeatureNames ? this.selectedFeatureNames.length : 0,
      selectedFeatureNames: this.selectedFeatureNames,
      featureImportances: this.featureImportances,
      correlationThreshold: this.featureSelector.correlationThreshold,
      varianceThreshold: this.featureSelector.varianceThreshold,
      importanceThreshold: this.featureSelector.importanceThreshold
    };
  }
}

// Enhanced feature preparation with better feature engineering
export const prepareLongTermFeatures = (stockData, lookbackDays = 30, targetDaysAhead = 22) => {
  if (!stockData || stockData.length < Math.max(lookbackDays + targetDaysAhead, 252 + targetDaysAhead)) {
    throw new Error(`Insufficient data. Need at least ${252 + targetDaysAhead} days for long-term prediction.`);
  }

  const formattedData = {
    close: stockData.map(d => d.close),
    high: stockData.map(d => d.high),
    low: stockData.map(d => d.low),
    volume: stockData.map(d => d.volume),
    dates: stockData.map(d => d.date)
  };
  
  const technicalFeatures = extractTechnicalFeatures(formattedData);
  
  if (technicalFeatures.length < lookbackDays + targetDaysAhead) {
    throw new Error(`Insufficient technical feature data for long-term prediction.`);
  }

  const features = [];
  const targets = [];
  const featureNames = [];

  // Generate feature names first (for feature selection)
  for (let day = 0; day < lookbackDays; day++) {
    featureNames.push(`Day${day}_SMA20Ratio`);
    featureNames.push(`Day${day}_SMA50Ratio`);
    featureNames.push(`Day${day}_SMA200Ratio`);
    featureNames.push(`Day${day}_MACD`);
    featureNames.push(`Day${day}_MACDSignal`);
    featureNames.push(`Day${day}_MACDHistogram`);
    featureNames.push(`Day${day}_RSI`);
    featureNames.push(`Day${day}_Volatility`);
    featureNames.push(`Day${day}_VolumeRatio`);
    featureNames.push(`Day${day}_PricePosition`);
    featureNames.push(`Day${day}_BottomSignal`);
    featureNames.push(`Day${day}_PeakSignal`);
  }
  
  // Aggregated feature names
  featureNames.push('AvgRSI');
  featureNames.push('AvgVolatility');
  featureNames.push('TrendStrength');
  featureNames.push('PriceMomentum');
  featureNames.push('VolumeTrend');
  
  for (let k = 1; k <= 5; k++) {
    featureNames.push(`RecentMomentum_${k}d`);
  }
  featureNames.push('VolatilityBreakout');
  featureNames.push('PriceAcceleration');

  for (let i = lookbackDays; i < technicalFeatures.length - targetDaysAhead; i++) {
    const currentFeatures = [];
    
    // Daily features over lookback period
    for (let j = i - lookbackDays; j < i; j++) {
      const feature = technicalFeatures[j];
      
      currentFeatures.push(feature.sma20Ratio);
      currentFeatures.push(feature.sma50Ratio);
      currentFeatures.push(feature.sma200Ratio);
      currentFeatures.push(feature.macd);
      currentFeatures.push(feature.macdSignal);
      currentFeatures.push(feature.macdHistogram);
      currentFeatures.push(feature.rsi / 100);
      currentFeatures.push(feature.volatility);
      currentFeatures.push(feature.volumeRatio);
      currentFeatures.push(feature.pricePosition);
      currentFeatures.push(feature.bottomSignal);
      currentFeatures.push(feature.peakSignal);
    }
    
    // Aggregated features
    const lookbackWindow = technicalFeatures.slice(i - lookbackDays, i);
    
    const avgRSI = lookbackWindow.reduce((sum, f) => sum + f.rsi, 0) / lookbackDays;
    currentFeatures.push(avgRSI / 100);
    
    const avgVol = lookbackWindow.reduce((sum, f) => sum + f.volatility, 0) / lookbackDays;
    currentFeatures.push(avgVol);
    
    const trendStrength = lookbackWindow.reduce((sum, f) => {
      let score = 0;
      if (f.close > f.close * f.sma20Ratio) score += 1;
      if (f.close > f.close * f.sma50Ratio) score += 1;
      if (f.close > f.close * f.sma200Ratio) score += 1;
      return sum + score;
    }, 0) / (lookbackDays * 3);
    currentFeatures.push(trendStrength);
    
    const startPrice = lookbackWindow[0].close;
    const endPrice = lookbackWindow[lookbackWindow.length - 1].close;
    const momentum = (endPrice - startPrice) / startPrice;
    currentFeatures.push(momentum);
    
    const avgVolumeFirst = lookbackWindow.slice(0, lookbackDays/2).reduce((sum, f) => sum + f.volumeRatio, 0) / (lookbackDays/2);
    const avgVolumeSecond = lookbackWindow.slice(lookbackDays/2).reduce((sum, f) => sum + f.volumeRatio, 0) / (lookbackDays/2);
    const volumeTrend = avgVolumeSecond - avgVolumeFirst;
    currentFeatures.push(volumeTrend);

    // Recent momentum
    for (let k = 1; k <= 5; k++) {
      const momentum = (technicalFeatures[i].close - technicalFeatures[i-k].close) / technicalFeatures[i-k].close;
      currentFeatures.push(momentum);
    }

    // Volatility breakout
    const recentVolatility = lookbackWindow.slice(-10).reduce((sum, f) => sum + f.volatility, 0) / 10;
    const avgVolatility = lookbackWindow.reduce((sum, f) => sum + f.volatility, 0) / lookbackDays;
    const volatilityBreakout = avgVolatility > 0 ? recentVolatility / avgVolatility : 1;
    currentFeatures.push(volatilityBreakout);

    // Price acceleration
    if (i >= 2) {
      const currentMomentum = (technicalFeatures[i].close - technicalFeatures[i-1].close) / technicalFeatures[i-1].close;
      const prevMomentum = (technicalFeatures[i-1].close - technicalFeatures[i-2].close) / technicalFeatures[i-2].close;
      const acceleration = currentMomentum - prevMomentum;
      currentFeatures.push(acceleration);
    } else {
      currentFeatures.push(0);
    }

    // Target classification
    const currentPrice = technicalFeatures[i].close;
    const futurePrice = technicalFeatures[i + targetDaysAhead].close;
    const percentReturn = (futurePrice - currentPrice) / currentPrice;

    let targetClass;
    if (percentReturn > 0.02) targetClass = 4;        // Big Up (>2%)
    else if (percentReturn > 0.005) targetClass = 3;  // Small Up (0.5-2%)
    else if (percentReturn < -0.02) targetClass = 0;  // Big Down (<-2%)
    else if (percentReturn < -0.005) targetClass = 1; // Small Down (-2% to -0.5%)
    else targetClass = 2;                              // Flat (-0.5% to +0.5%)

    features.push(currentFeatures);
    targets.push(targetClass);
  }

  console.log(`Prepared ${features.length} samples with ${features[0].length} long-term features each`);
  console.log(`Target distribution:`, {
    'Big Down (0)': targets.filter(t => t === 0).length,
    'Small Down (1)': targets.filter(t => t === 1).length,
    'Flat (2)': targets.filter(t => t === 2).length,
    'Small Up (3)': targets.filter(t => t === 3).length,
    'Big Up (4)': targets.filter(t => t === 4).length
  });

  return { features, targets, featureNames };
};

// Enhanced latest features preparation
export const prepareLatestLongTermFeatures = (stockData, lookbackDays = 30, featureSelector = null) => {
  if (!stockData || stockData.length < Math.max(lookbackDays, 252)) {
    throw new Error(`Insufficient data for prediction. Need at least 252 days for technical indicators.`);
  }
  
  const formattedData = {
    close: stockData.map(d => d.close),
    high: stockData.map(d => d.high),
    low: stockData.map(d => d.low),
    volume: stockData.map(d => d.volume),
    dates: stockData.map(d => d.date)
  };
  
  const technicalFeatures = extractTechnicalFeatures(formattedData);
  
  if (technicalFeatures.length < lookbackDays) {
    throw new Error(`Insufficient technical feature data for prediction.`);
  }
  
  const latestFeatures = technicalFeatures.slice(technicalFeatures.length - lookbackDays);
  const features = [];
  
  // Build features the same way as training
  for (let j = 0; j < lookbackDays; j++) {
    const feature = latestFeatures[j];
    
    features.push(feature.sma20Ratio);
    features.push(feature.sma50Ratio);
    features.push(feature.sma200Ratio);
    features.push(feature.macd);
    features.push(feature.macdSignal);
    features.push(feature.macdHistogram);
    features.push(feature.rsi / 100);
    features.push(feature.volatility);
    features.push(feature.volumeRatio);
    features.push(feature.pricePosition);
    features.push(feature.bottomSignal);
    features.push(feature.peakSignal);
  }
  
  // Aggregated features
  const avgRSI = latestFeatures.reduce((sum, f) => sum + f.rsi, 0) / lookbackDays;
  features.push(avgRSI / 100);
  
  const avgVol = latestFeatures.reduce((sum, f) => sum + f.volatility, 0) / lookbackDays;
  features.push(avgVol);
  
  const trendStrength = latestFeatures.reduce((sum, f) => {
    let score = 0;
    if (f.close > f.close * f.sma20Ratio) score += 1;
    if (f.close > f.close * f.sma50Ratio) score += 1;
    if (f.close > f.close * f.sma200Ratio) score += 1;
    return sum + score;
  }, 0) / (lookbackDays * 3);
  features.push(trendStrength);
  
  const startPrice = latestFeatures[0].close;
  const endPrice = latestFeatures[latestFeatures.length - 1].close;
  const momentum = (endPrice - startPrice) / startPrice;
  features.push(momentum);
  
  const avgVolumeFirst = latestFeatures.slice(0, lookbackDays/2).reduce((sum, f) => sum + f.volumeRatio, 0) / (lookbackDays/2);
  const avgVolumeSecond = latestFeatures.slice(lookbackDays/2).reduce((sum, f) => sum + f.volumeRatio, 0) / (lookbackDays/2);
  const volumeTrend = avgVolumeSecond - avgVolumeFirst;
  features.push(volumeTrend);

  // Recent momentum
  const currentIndex = technicalFeatures.length - 1;
  for (let k = 1; k <= 5; k++) {
    if (currentIndex - k >= 0) {
      const momentum = (technicalFeatures[currentIndex].close - technicalFeatures[currentIndex-k].close) / technicalFeatures[currentIndex-k].close;
      features.push(momentum);
    } else {
      features.push(0);
    }
  }

  // Volatility breakout
  const recentVolatility = latestFeatures.slice(-10).reduce((sum, f) => sum + f.volatility, 0) / 10;
  const avgVolatility = latestFeatures.reduce((sum, f) => sum + f.volatility, 0) / lookbackDays;
  const volatilityBreakout = avgVolatility > 0 ? recentVolatility / avgVolatility : 1;
  features.push(volatilityBreakout);

  // Price acceleration
  if (currentIndex >= 2) {
    const currentMomentum = (technicalFeatures[currentIndex].close - technicalFeatures[currentIndex-1].close) / technicalFeatures[currentIndex-1].close;
    const prevMomentum = (technicalFeatures[currentIndex-1].close - technicalFeatures[currentIndex-2].close) / technicalFeatures[currentIndex-2].close;
    const acceleration = currentMomentum - prevMomentum;
    features.push(acceleration);
  } else {
    features.push(0);
  }
  
  // Apply feature selection if provided
  if (featureSelector) {
    return featureSelector.applySingleSample(features);
  }
  
  return features;
};

// Enhanced training function
export const trainLongTermRandomForest = (stockData, lookbackDays = 30, targetDaysAhead = 22, nTrees = 100, testSplit = 0.2, maxSelectedFeatures = 50) => {
  const { features, targets, featureNames } = prepareLongTermFeatures(stockData, lookbackDays, targetDaysAhead);
  
  console.log('=== ENHANCED LONG-TERM RF DEBUG ===');
  console.log(`Features shape: ${features.length} samples x ${features[0].length} features`);
  console.log(`Target distribution:`, {
    'Big Down (0)': targets.filter(t => t === 0).length,
    'Small Down (1)': targets.filter(t => t === 1).length,
    'Flat (2)': targets.filter(t => t === 2).length,
    'Small Up (3)': targets.filter(t => t === 3).length,
    'Big Up (4)': targets.filter(t => t === 4).length
  });
  
  const hasNaN = features.some(row => row.some(val => !isFinite(val)));
  const targetsHaveNaN = targets.some(val => !isFinite(val));
  console.log(`Features have NaN/Inf: ${hasNaN}`);
  console.log(`Targets have NaN/Inf: ${targetsHaveNaN}`);

  const totalSamples = features.length;
  const testSize = Math.floor(totalSamples * testSplit);
  const trainSize = totalSamples - testSize;
  
  const trainFeatures = features.slice(0, trainSize);
  const trainTargets = targets.slice(0, trainSize);
  const testFeatures = features.slice(trainSize);
  const testTargets = targets.slice(trainSize);
  
  console.log(`Training set: ${trainSize} samples, Test set: ${testSize} samples`);
  
  // Use enhanced classifier with feature selection
  const model = new LongTermRandomForestClassifier(nTrees, 20, 2, 1, 'sqrt', maxSelectedFeatures);
  model.train(trainFeatures, trainTargets, featureNames);
  
  const metrics = model.evaluate(testFeatures, testTargets);
  const featureSelectionSummary = model.getFeatureSelectionSummary();
  
  console.log(`Enhanced Long-Term RF Performance: Accuracy=${(metrics.accuracy * 100).toFixed(1)}%, F1=${(metrics.f1 * 100).toFixed(1)}%`);
  console.log(`Feature selection: ${featureSelectionSummary.originalFeatureCount} → ${featureSelectionSummary.selectedFeatureCount} features`);
  
  // Enhanced feature importance analysis
  const featureImportanceDetails = model.featureImportances.map((importance, index) => ({
    feature: featureSelectionSummary.selectedFeatureNames[index] || `Feature_${index}`,
    importance: importance,
    rank: index + 1
  })).sort((a, b) => b.importance - a.importance);
  
  return {
    model,
    metrics,
    featureImportances: model.featureImportances,
    featureImportanceDetails,
    featureSelectionSummary,
    numTrees: nTrees,
    numFeatures: featureSelectionSummary.selectedFeatureCount,
    originalNumFeatures: features[0].length,
    lookbackDays,
    targetDaysAhead,
    trainingSamples: trainSize,
    testSamples: testSize,
    technicalIndicatorsUsed: [
      'Selected Long-term SMA Analysis', 'Selected MACD Patterns', 'Selected RSI Trends', 
      'Selected Volatility Patterns', 'Selected Volume Trends', 'Selected Price Position Analysis',
      'Selected Trend Strength', 'Selected Momentum Analysis', 'Selected Recent Momentum', 
      'Selected Volatility Breakouts'
    ]
  };
};

export const predictWithLongTermRandomForest = (
  model,
  stockData,
  lookbackDays = 30,
  targetDaysAhead = 22,
  featureSelector = null
) => {
  if (!model || typeof model.predictWithConfidence !== 'function') {
    throw new Error('Long-term Random Forest model is not available for prediction');
  }

  const latestFeatures = prepareLatestLongTermFeatures(stockData, lookbackDays, featureSelector);
  const prediction = model.predictWithConfidence([latestFeatures])?.[0];

  if (!prediction) {
    throw new Error('Long-term Random Forest did not return a prediction result');
  }

  const currentPrice = Number(stockData?.[stockData.length - 1]?.close ?? 0);
  if (!Number.isFinite(currentPrice) || currentPrice <= 0) {
    throw new Error('Unable to resolve current price for long-term Random Forest prediction');
  }

  const categoryToReturn = {
    0: -0.03,
    1: -0.0125,
    2: 0,
    3: 0.0125,
    4: 0.03,
  };

  const category = Number(prediction.prediction ?? 2);
  const expectedReturn = categoryToReturn[category] ?? 0;
  const predictedPrice = currentPrice * (1 + expectedReturn);

  return {
    currentPrice,
    predictedPrice,
    predictedReturn: expectedReturn,
    percentChange: expectedReturn * 100,
    priceChange: predictedPrice - currentPrice,
    direction: expectedReturn > 0 ? 'UP' : (expectedReturn < 0 ? 'DOWN' : 'FLAT'),
    confidence: Number(prediction.confidence ?? prediction.agreementScore ?? 0),
    category,
    probabilities: prediction.probabilities || null,
    horizonDays: Math.max(1, Number(targetDaysAhead) || 22),
    categoryLegend: {
      0: 'Big Down',
      1: 'Small Down',
      2: 'Flat',
      3: 'Small Up',
      4: 'Big Up',
    },
  };
};