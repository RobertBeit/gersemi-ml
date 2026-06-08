import { extractTechnicalFeatures } from './technicalIndicators';
import { buildAndTrainLSTMModel, predictBottomsPeaks } from './lstmModel';

export const runBottomPeakDetector = async (stockData, options = {}) => {
  if (!Array.isArray(stockData) || stockData.length < 80) {
    throw new Error('Insufficient stock data for Bottom/Peak LSTM. Need at least 80 rows.');
  }

  const {
    epochs = 20,
    batchSize = 32,
    threshold = 0.7,
    maxPredictions = 400,
  } = options;

  const formattedData = {
    dates: stockData.map((row) => row.date),
    open: stockData.map((row) => row.open),
    high: stockData.map((row) => row.high),
    low: stockData.map((row) => row.low),
    close: stockData.map((row) => row.close),
    volume: stockData.map((row) => row.volume),
  };

  const features = extractTechnicalFeatures(formattedData);
  if (!features || features.length < 50) {
    throw new Error(`Insufficient technical features for Bottom/Peak LSTM. Generated ${features?.length || 0}. Need at least 50.`);
  }

  const modelResult = await buildAndTrainLSTMModel(features, epochs, batchSize, {});
  const predictions = await predictBottomsPeaks(
    modelResult.model,
    features,
    modelResult.normParams,
    modelResult.sequenceLength,
    threshold
  );

  const limitedPredictions = predictions.slice(-Math.max(1, maxPredictions));
  const bottomCount = limitedPredictions.filter((item) => item.isBottom).length;
  const peakCount = limitedPredictions.filter((item) => item.isPeak).length;

  const probabilitySummary = limitedPredictions.reduce(
    (acc, item) => {
      const bottom = Number(item.bottomProb) || 0;
      const peak = Number(item.peakProb) || 0;

      acc.bottom.min = Math.min(acc.bottom.min, bottom);
      acc.bottom.max = Math.max(acc.bottom.max, bottom);
      acc.bottom.sum += bottom;

      acc.peak.min = Math.min(acc.peak.min, peak);
      acc.peak.max = Math.max(acc.peak.max, peak);
      acc.peak.sum += peak;

      return acc;
    },
    {
      bottom: { min: Number.POSITIVE_INFINITY, max: Number.NEGATIVE_INFINITY, sum: 0 },
      peak: { min: Number.POSITIVE_INFINITY, max: Number.NEGATIVE_INFINITY, sum: 0 },
    }
  );

  const predictionCount = Math.max(limitedPredictions.length, 1);
  const bottomStats = {
    min: Number.isFinite(probabilitySummary.bottom.min) ? probabilitySummary.bottom.min : 0,
    max: Number.isFinite(probabilitySummary.bottom.max) ? probabilitySummary.bottom.max : 0,
    avg: probabilitySummary.bottom.sum / predictionCount,
  };
  const peakStats = {
    min: Number.isFinite(probabilitySummary.peak.min) ? probabilitySummary.peak.min : 0,
    max: Number.isFinite(probabilitySummary.peak.max) ? probabilitySummary.peak.max : 0,
    avg: probabilitySummary.peak.sum / predictionCount,
  };

  if (modelResult.model && typeof modelResult.model.dispose === 'function') {
    modelResult.model.dispose();
  }

  return {
    algorithm: 'bottomPeakLSTM',
    summary: {
      totalRows: stockData.length,
      featureRows: features.length,
      predictionRows: limitedPredictions.length,
      threshold,
      bottomsDetected: bottomCount,
      peaksDetected: peakCount,
    },
    training: {
      epochs,
      batchSize,
      sequenceLength: modelResult.sequenceLength,
      inputDim: modelResult.inputDim,
      finalLoss: modelResult.history?.loss?.[modelResult.history.loss.length - 1] ?? null,
      finalAccuracy: modelResult.history?.acc?.[modelResult.history.acc.length - 1] ?? null,
      finalValidationLoss: modelResult.history?.val_loss?.[modelResult.history.val_loss.length - 1] ?? null,
      finalValidationAccuracy: modelResult.history?.val_acc?.[modelResult.history.val_acc.length - 1] ?? null,
    },
    debug: {
      optionsApplied: {
        epochs,
        batchSize,
        threshold,
        maxPredictions,
      },
      datasetShape: {
        stockRows: stockData.length,
        featureRows: features.length,
        sequenceLength: modelResult.sequenceLength,
        inputDim: modelResult.inputDim,
      },
      labelStats: modelResult.labelStats || null,
      probabilityStats: {
        bottom: bottomStats,
        peak: peakStats,
      },
      signalStats: {
        bottomsDetected: bottomCount,
        peaksDetected: peakCount,
      },
    },
    predictions: limitedPredictions,
  };
};
