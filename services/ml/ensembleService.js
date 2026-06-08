// Ensemble Model Service - Combines Naive Bayes, LSTM, and Random Forest
// src/services/ensembleService.js

import { trainAndEvaluate, prepareLatestForPrediction, discretizePredictionSample } from './mlService';
import { trainRandomForest, prepareLatestRandomForestFeatures } from './randomForestService';
import { buildAndTrainLSTMModel, predictBottomsPeaks, normalizeNewFeatures } from './lstmModel';
import { extractTechnicalFeatures } from './technicalIndicators';

// Ensemble Model Class
export class EnsemblePredictor {
  constructor(ensembleConfig = {}) {
    this.config = {
      // Model weights (how much to trust each model)
      naiveBayesWeight: ensembleConfig.naiveBayesWeight || 0.3,
      randomForestWeight: ensembleConfig.randomForestWeight || 0.4,
      lstmWeight: ensembleConfig.lstmWeight || 0.3,
      
      // Confidence thresholds
      minConfidenceThreshold: ensembleConfig.minConfidenceThreshold || 0.6,
      consensusThreshold: ensembleConfig.consensusThreshold || 0.7,
      
      // Model parameters
      modelParams: {
        naiveBayes: ensembleConfig.naiveBayes || { lookbackDays: 5, bins: 5, testSplit: 0.2 },
        randomForest: ensembleConfig.randomForest || { 
          lookbackDays: 5, 
          nTrees: 50, 
          testSplit: 0.2 
        },
        lstm: ensembleConfig.lstm || { 
          epochs: 50, 
          batchSize: 32,
          sequenceLength: 20
        }
      }
    };
    
    // Model instances and results
    this.models = {
      naiveBayes: null,
      randomForest: null,
      lstm: null
    };
    
    // Training results
    this.trainingResults = {
      naiveBayes: null,
      randomForest: null,
      lstm: null
    };
    
    this.trained = false;
    this.lastTrainingData = null;
  }
  
  // Train all models
  async trainEnsemble(stockData, progressCallback = null) {
    console.log('🎯 Starting Ensemble Training...');
    
    const totalSteps = 6; // 3 models + 3 evaluations
    let currentStep = 0;
    
    const updateProgress = (message) => {
      currentStep++;
      const progress = Math.floor((currentStep / totalSteps) * 100);
      console.log(`[${progress}%] ${message}`);
      if (progressCallback) progressCallback(progress, message);
    };
    
    try {
      this.lastTrainingData = stockData;
      
      // Train Naive Bayes
      updateProgress('Training Naive Bayes model...');
      try {
        const nbResult = trainAndEvaluate(
          stockData, 
          this.config.modelParams.naiveBayes.lookbackDays,
          this.config.modelParams.naiveBayes.bins,
          this.config.modelParams.naiveBayes.testSplit
        );
        
        this.trainingResults.naiveBayes = {
          model: nbResult.model,
          metadata: nbResult.metadata,
          accuracy: nbResult.metrics.accuracy,
          metrics: nbResult.metrics
        };
        this.models.naiveBayes = nbResult.model;
        
        console.log('✅ Naive Bayes trained - Accuracy:', 
          (nbResult.metrics.accuracy * 100).toFixed(1) + '%');
      } catch (error) {
        console.error('❌ Naive Bayes training failed:', error);
        this.trainingResults.naiveBayes = { error: error.message, accuracy: 0 };
      }
      
      // Train Random Forest
      updateProgress('Training Random Forest model...');
      try {
        const rfResult = trainRandomForest(
          stockData,
          this.config.modelParams.randomForest.lookbackDays,
          this.config.modelParams.randomForest.nTrees,
          this.config.modelParams.randomForest.testSplit
        );
        
        this.trainingResults.randomForest = rfResult;
        this.models.randomForest = rfResult.model;
        
        console.log('✅ Random Forest trained - Accuracy:', 
          (rfResult.metrics.accuracy * 100).toFixed(1) + '%');
      } catch (error) {
        console.error('❌ Random Forest training failed:', error);
        this.trainingResults.randomForest = { error: error.message, metrics: { accuracy: 0 } };
      }
      
      // Train LSTM
      updateProgress('Training LSTM model...');
      try {
        // Prepare data for LSTM
        const formattedData = {
          close: stockData.map(d => d.close),
          high: stockData.map(d => d.high),
          low: stockData.map(d => d.low),
          volume: stockData.map(d => d.volume),
          dates: stockData.map(d => d.date)
        };
        
        const technicalFeatures = extractTechnicalFeatures(formattedData);
        
        const lstmResult = await buildAndTrainLSTMModel(
          technicalFeatures,
          this.config.modelParams.lstm.epochs,
          this.config.modelParams.lstm.batchSize
        );
        
        this.trainingResults.lstm = {
          model: lstmResult.model,
          normParams: lstmResult.normParams,
          sequenceLength: lstmResult.sequenceLength,
          accuracy: lstmResult.history.val_acc[lstmResult.history.val_acc.length - 1] || 0,
          history: lstmResult.history
        };
        this.models.lstm = lstmResult.model;
        
        console.log('✅ LSTM trained - Final Val Accuracy:', 
          (this.trainingResults.lstm.accuracy * 100).toFixed(1) + '%');
      } catch (error) {
        console.error('❌ LSTM training failed:', error);
        this.trainingResults.lstm = { error: error.message, accuracy: 0 };
      }
      
      // Calculate ensemble weights based on individual model performance
      updateProgress('Calculating optimal model weights...');
      this.calculateDynamicWeights();
      
      updateProgress('Validating ensemble performance...');
      const ensembleMetrics = await this.validateEnsemble(stockData);
      
      updateProgress('Ensemble training completed!');
      
      this.trained = true;
      
      return {
        success: true,
        individualResults: this.trainingResults,
        ensembleMetrics: ensembleMetrics,
        optimalWeights: {
          naiveBayes: this.config.naiveBayesWeight,
          randomForest: this.config.randomForestWeight,
          lstm: this.config.lstmWeight
        },
        totalModels: this.getTrainedModelCount()
      };
      
    } catch (error) {
      console.error('❌ Ensemble training failed:', error);
      throw new Error(`Ensemble training failed: ${error.message}`);
    }
  }
  
  // Calculate dynamic weights based on model performance
  calculateDynamicWeights() {
    const accuracies = {
      naiveBayes: this.trainingResults.naiveBayes?.accuracy || 0,
      randomForest: this.trainingResults.randomForest?.metrics?.accuracy || 0,
      lstm: this.trainingResults.lstm?.accuracy || 0
    };
    
    console.log('📊 Individual Model Accuracies:', accuracies);
    
    // Performance-based weighting
    const totalAccuracy = Object.values(accuracies).reduce((sum, acc) => sum + acc, 0);
    
    if (totalAccuracy > 0) {
      // Weight models based on their relative performance
      this.config.naiveBayesWeight = accuracies.naiveBayes / totalAccuracy;
      this.config.randomForestWeight = accuracies.randomForest / totalAccuracy;
      this.config.lstmWeight = accuracies.lstm / totalAccuracy;
      
      // Normalize weights to sum to 1
      const totalWeight = this.config.naiveBayesWeight + 
                         this.config.randomForestWeight + 
                         this.config.lstmWeight;
      
      if (totalWeight > 0) {
        this.config.naiveBayesWeight /= totalWeight;
        this.config.randomForestWeight /= totalWeight;
        this.config.lstmWeight /= totalWeight;
      }
    }
    
    console.log('🎯 Calculated Optimal Weights:', {
      naiveBayes: (this.config.naiveBayesWeight * 100).toFixed(1) + '%',
      randomForest: (this.config.randomForestWeight * 100).toFixed(1) + '%',
      lstm: (this.config.lstmWeight * 100).toFixed(1) + '%'
    });
  }
  
  // Make ensemble prediction
  async predictEnsemble(stockData) {
    if (!this.trained) {
      throw new Error('Ensemble not trained yet!');
    }
    
    console.log('🔮 Making Ensemble Prediction...');
    
    const predictions = {
      naiveBayes: null,
      randomForest: null,
      lstm: null
    };
    
    const confidences = {
      naiveBayes: 0,
      randomForest: 0,
      lstm: 0
    };
    
    // Get predictions from each model
    try {
      // Naive Bayes prediction
      if (this.models.naiveBayes && this.trainingResults.naiveBayes?.metadata) {
        const features = prepareLatestForPrediction(
          stockData, 
          this.config.modelParams.naiveBayes.lookbackDays
        );
        const discretizedFeatures = discretizePredictionSample(
          features, 
          this.trainingResults.naiveBayes.metadata
        );
        const probabilities = this.models.naiveBayes.predictProba([discretizedFeatures]);
        
        predictions.naiveBayes = probabilities[0][1] > probabilities[0][0] ? 1 : 0;
        confidences.naiveBayes = Math.max(probabilities[0][0], probabilities[0][1]);
      }
    } catch (error) {
      console.warn('⚠️ Naive Bayes prediction failed:', error);
    }
    
    try {
      // Random Forest prediction
      if (this.models.randomForest) {
        const features = prepareLatestRandomForestFeatures(
          stockData, 
          this.config.modelParams.randomForest.lookbackDays
        );
        const rfProbs = this.models.randomForest.predictProba([features]);
        predictions.randomForest = rfProbs[0][1] > rfProbs[0][0] ? 1 : 0;
        confidences.randomForest = Math.max(rfProbs[0][0], rfProbs[0][1]);
      }
    } catch (error) {
      console.warn('⚠️ Random Forest prediction failed:', error);
    }
    
    try {
      // LSTM prediction
      if (this.models.lstm && this.trainingResults.lstm?.normParams) {
        // Prepare data for LSTM
        const formattedData = {
          close: stockData.map(d => d.close),
          high: stockData.map(d => d.high),
          low: stockData.map(d => d.low),
          volume: stockData.map(d => d.volume),
          dates: stockData.map(d => d.date)
        };
        
        const technicalFeatures = extractTechnicalFeatures(formattedData);
        
        // Get recent features for prediction
        const recentFeatures = technicalFeatures.slice(-this.trainingResults.lstm.sequenceLength - 1);
        
        const lstmResults = await predictBottomsPeaks(
          this.models.lstm,
          recentFeatures,
          this.trainingResults.lstm.normParams,
          this.trainingResults.lstm.sequenceLength,
          0.5 // Lower threshold for more predictions
        );
        
        if (lstmResults.length > 0) {
          const lastResult = lstmResults[lstmResults.length - 1];
          // Convert LSTM bottom/peak predictions to UP/DOWN
          if (lastResult.isBottom) {
            predictions.lstm = 1; // Bottom suggests UP movement
            confidences.lstm = lastResult.bottomProb;
          } else if (lastResult.isPeak) {
            predictions.lstm = 0; // Peak suggests DOWN movement  
            confidences.lstm = lastResult.peakProb;
          } else {
            // Use the stronger signal
            if (lastResult.bottomProb > lastResult.peakProb) {
              predictions.lstm = 1;
              confidences.lstm = lastResult.bottomProb;
            } else {
              predictions.lstm = 0;
              confidences.lstm = lastResult.peakProb;
            }
          }
        }
      }
    } catch (error) {
      console.warn('⚠️ LSTM prediction failed:', error);
    }
    
    console.log('📊 Individual Predictions:', predictions);
    console.log('📊 Individual Confidences:', confidences);
    
    // Calculate weighted ensemble prediction
    const ensemblePrediction = this.calculateWeightedPrediction(predictions, confidences);
    
    return ensemblePrediction;
  }
  
  // Calculate weighted ensemble prediction
  calculateWeightedPrediction(predictions, confidences) {
    let weightedSum = 0;
    let totalWeight = 0;
    let validPredictions = 0;
    
    const modelNames = ['naiveBayes', 'randomForest', 'lstm'];
    const modelWeights = [
      this.config.naiveBayesWeight,
      this.config.randomForestWeight, 
      this.config.lstmWeight
    ];
    
    // Confidence-adjusted weighted voting
    modelNames.forEach((modelName, index) => {
      if (predictions[modelName] !== null) {
        const confidence = confidences[modelName] || 0.5;
        const baseWeight = modelWeights[index];
        
        // Boost weight for high-confidence predictions
        const adjustedWeight = baseWeight * (0.5 + confidence);
        
        weightedSum += predictions[modelName] * adjustedWeight;
        totalWeight += adjustedWeight;
        validPredictions++;
      }
    });
    
    // Calculate final prediction
    const finalPrediction = totalWeight > 0 ? weightedSum / totalWeight : 0.5;
    const predictionDirection = finalPrediction > 0.5 ? 'UP' : 'DOWN';
    const ensembleConfidence = Math.abs(finalPrediction - 0.5) * 2; // Convert to 0-1 scale
    
    // Consensus analysis
    const upVotes = Object.values(predictions).filter(p => p === 1).length;
    const downVotes = Object.values(predictions).filter(p => p === 0).length;
    const consensus = validPredictions > 0 ? Math.max(upVotes, downVotes) / validPredictions : 0;
    
    // Agreement level
    let agreementLevel = 'Low';
    if (consensus >= 0.8) agreementLevel = 'High';
    else if (consensus >= 0.6) agreementLevel = 'Medium';
    
    const currentPrice = this.lastTrainingData[this.lastTrainingData.length - 1]?.close || 0;
    
    return {
      prediction: predictionDirection,
      confidence: ensembleConfidence,
      weightedScore: finalPrediction,
      consensus: consensus,
      agreementLevel: agreementLevel,
      validModels: validPredictions,
      totalModels: 3,
      individualPredictions: predictions,
      individualConfidences: confidences,
      modelWeights: {
        naiveBayes: this.config.naiveBayesWeight,
        randomForest: this.config.randomForestWeight,
        lstm: this.config.lstmWeight
      },
      currentPrice: currentPrice,
      timestamp: new Date(),
      
      // Detailed breakdown
      votingBreakdown: {
        upVotes: upVotes,
        downVotes: downVotes,
        abstains: 3 - validPredictions
      }
    };
  }
  
  // Validate ensemble performance
  async validateEnsemble(stockData) {
    // This would typically involve backtesting
    // For now, return summary of individual model performance
    
    const trainedModels = this.getTrainedModelCount();
    const avgAccuracy = this.getAverageAccuracy();
    
    return {
      trainedModels: trainedModels,
      totalModels: 3,
      averageAccuracy: avgAccuracy,
      ensembleReady: trainedModels >= 2, // Need at least 2 models
      recommendedConfidenceThreshold: this.getRecommendedThreshold()
    };
  }
  
  // Helper methods
  getTrainedModelCount() {
    let count = 0;
    if (this.models.naiveBayes) count++;
    if (this.models.randomForest) count++;
    if (this.models.lstm) count++;
    return count;
  }
  
  getAverageAccuracy() {
    const accuracies = [];
    
    if (this.trainingResults.naiveBayes?.accuracy) {
      accuracies.push(this.trainingResults.naiveBayes.accuracy);
    }
    if (this.trainingResults.randomForest?.metrics?.accuracy) {
      accuracies.push(this.trainingResults.randomForest.metrics.accuracy);
    }
    if (this.trainingResults.lstm?.accuracy) {
      accuracies.push(this.trainingResults.lstm.accuracy);
    }
    
    return accuracies.length > 0 ? 
      accuracies.reduce((sum, acc) => sum + acc, 0) / accuracies.length : 0;
  }
  
  getRecommendedThreshold() {
    const avgAccuracy = this.getAverageAccuracy();
    
    // Higher threshold for lower accuracy models
    if (avgAccuracy > 0.7) return 0.6;
    if (avgAccuracy > 0.6) return 0.7;
    return 0.8;
  }
  
  // Get ensemble status
  getEnsembleStatus() {
    return {
      trained: this.trained,
      trainedModels: this.getTrainedModelCount(),
      totalModels: 3,
      averageAccuracy: this.getAverageAccuracy(),
      weights: {
        naiveBayes: this.config.naiveBayesWeight,
        randomForest: this.config.randomForestWeight,
        lstm: this.config.lstmWeight
      },
      lastTrainingDate: this.lastTrainingData ? new Date() : null
    };
  }
}

// Factory function to create ensemble predictor
export const createEnsemblePredictor = (config = {}) => {
  return new EnsemblePredictor(config);
};

// Main training function
export const trainEnsembleModel = async (stockData, config = {}, progressCallback = null) => {
  const ensemble = createEnsemblePredictor(config);
  const result = await ensemble.trainEnsemble(stockData, progressCallback);
  
  return {
    ensemble: ensemble,
    trainingResult: result
  };
};

// Main prediction function
export const predictWithEnsemble = async (ensemble, stockData) => {
  if (!ensemble || !ensemble.trained) {
    throw new Error('Ensemble model not trained');
  }
  
  return await ensemble.predictEnsemble(stockData);
};