const { randomUUID } = require("crypto");

const { executeMlMethod } = require("./mlExecutionService");
const { ML_BUILD, buildBanner } = require("./buildInfo");

const jobs = [];
let isProcessing = false;

const toFiniteNumber = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const deriveDirection = (predictedPrice, currentPrice, predictedReturn) => {
  const safePredictedPrice = toFiniteNumber(predictedPrice);
  const safeCurrentPrice = toFiniteNumber(currentPrice);
  const safePredictedReturn = toFiniteNumber(predictedReturn);
  const delta = safePredictedPrice !== null && safeCurrentPrice !== null
    ? Number(predictedPrice) - Number(currentPrice)
    : safePredictedReturn;

  if (toFiniteNumber(delta) === null) return null;
  if (delta > 0) return "UP";
  if (delta < 0) return "DOWN";
  return "FLAT";
};

const addDaysToIsoDate = (dateLike, daysAhead = 1) => {
  const candidate = dateLike ? new Date(dateLike) : new Date();
  if (Number.isNaN(candidate.getTime())) {
    return null;
  }
  const safeDays = Number.isFinite(Number(daysAhead)) ? Number(daysAhead) : 1;
  candidate.setDate(candidate.getDate() + safeDays);
  return candidate.toISOString().split("T")[0];
};

const averageAbsoluteReturn = (stockData = [], windowSize = 30) => {
  const recent = stockData.slice(-windowSize);
  const returns = recent
    .map((row, index, arr) => {
      if (index === 0) return null;
      const prev = Number(arr[index - 1]?.close);
      const curr = Number(row?.close);
      if (!Number.isFinite(prev) || !Number.isFinite(curr) || prev <= 0) return null;
      return Math.abs((curr - prev) / prev);
    })
    .filter((value) => Number.isFinite(value));

  if (!returns.length) {
    return 0.01;
  }

  return returns.reduce((sum, value) => sum + value, 0) / returns.length;
};

const derivePriceContext = (metadata = {}) => {
  const intradayMerge = metadata?.intradayMerge;
  if (intradayMerge?.didMerge) {
    return {
      isLivePrice: true,
      currentPriceLabel: "Current Price",
      priceSource: intradayMerge?.reason || "intraday-merged",
    };
  }

  return {
    isLivePrice: false,
    currentPriceLabel: "Last Close",
    priceSource: intradayMerge?.reason || "historical-close",
  };
};

const normalizePredictionSummary = ({ algorithm, horizonDays, stockData, rawForecast, fallbackSource, metadata }) => {
  const currentPrice = toFiniteNumber(rawForecast?.currentPrice) ?? toFiniteNumber(stockData?.[stockData.length - 1]?.close);
  const predictedPrice = toFiniteNumber(rawForecast?.predictedPrice);
  const predictedReturn = toFiniteNumber(rawForecast?.predictedReturn);
  const expectedReturn = toFiniteNumber(rawForecast?.expectedReturn);
  const percentChange = toFiniteNumber(rawForecast?.percentChange)
    ?? (predictedReturn !== null ? predictedReturn * 100 : null)
    ?? expectedReturn;

  const normalizedHorizon = Math.max(1, Number(rawForecast?.horizonDays ?? rawForecast?.daysAhead ?? horizonDays ?? 1) || 1);
  const targetDate = rawForecast?.targetDate
    ? addDaysToIsoDate(rawForecast.targetDate, 0)
    : addDaysToIsoDate(stockData?.[stockData.length - 1]?.date || new Date().toISOString(), normalizedHorizon);
  const priceContext = derivePriceContext(metadata);

  return {
    algorithm,
    source: fallbackSource,
    horizonDays: normalizedHorizon,
    asOfDate: stockData?.[stockData.length - 1]?.date || null,
    targetDate,
    currentPrice,
    predictedPrice,
    priceChange: toFiniteNumber(rawForecast?.priceChange)
      ?? (predictedPrice !== null && currentPrice !== null ? predictedPrice - currentPrice : null),
    percentChange,
    confidence: toFiniteNumber(rawForecast?.confidence),
    direction: rawForecast?.direction || rawForecast?.prediction || deriveDirection(predictedPrice, currentPrice, predictedReturn),
    isLivePrice: priceContext.isLivePrice,
    currentPriceLabel: priceContext.currentPriceLabel,
    priceSource: priceContext.priceSource,
    details: rawForecast,
  };
};

const buildBottomPeakSummary = (stockData = [], resultPayload = {}, metadata = {}) => {
  const predictions = Array.isArray(resultPayload?.predictions) ? resultPayload.predictions : [];
  if (!predictions.length) return null;

  const lastPrediction = predictions[predictions.length - 1] || {};
  const currentPrice = toFiniteNumber(lastPrediction?.price) ?? toFiniteNumber(lastPrediction?.close) ?? toFiniteNumber(stockData?.[stockData.length - 1]?.close);
  const bottomProb = toFiniteNumber(lastPrediction?.bottomProb);
  const peakProb = toFiniteNumber(lastPrediction?.peakProb);
  const score = (bottomProb ?? 0) - (peakProb ?? 0);
  const projectedMove = averageAbsoluteReturn(stockData, 30) * score;
  const predictedPrice = currentPrice !== null ? currentPrice * (1 + projectedMove) : null;
  const priceContext = derivePriceContext(metadata);

  return {
    algorithm: "bottomPeakLSTM",
    source: "derivedFromBottomPeakSignals",
    horizonDays: 1,
    asOfDate: lastPrediction?.date || stockData?.[stockData.length - 1]?.date || null,
    targetDate: addDaysToIsoDate(lastPrediction?.date || stockData?.[stockData.length - 1]?.date || new Date().toISOString(), 1),
    currentPrice,
    predictedPrice,
    priceChange: predictedPrice !== null && currentPrice !== null ? predictedPrice - currentPrice : null,
    percentChange: projectedMove * 100,
    confidence: Math.max(bottomProb ?? 0, peakProb ?? 0),
    direction: score > 0 ? "UP" : (score < 0 ? "DOWN" : "FLAT"),
    isLivePrice: priceContext.isLivePrice,
    currentPriceLabel: priceContext.currentPriceLabel,
    priceSource: priceContext.priceSource,
    details: {
      bottomProbability: bottomProb,
      peakProbability: peakProb,
      heuristic: "Signal probability delta scaled by recent realized volatility",
      predictionRows: predictions.length,
    },
  };
};

const buildPredictionSummaryForJob = async (job, resultPayload) => {
  const algorithm = job?.metadata?.algorithm;
  const stockData = Array.isArray(job?.args?.[0]) ? job.args[0] : [];
  const metadata = job?.metadata || {};
  if (!algorithm || !stockData.length || !resultPayload || typeof resultPayload !== "object") {
    return null;
  }

  const invoke = (service, method, args = []) =>
    executeMlMethod({ service, method, args }).then((execution) => execution?.result);

  try {
    if (algorithm === "linearRegression") {
      const forecast = await invoke("linearRegressionService", "predictWithLinearRegression", [
        resultPayload.model,
        stockData,
        resultPayload.lookbackDays,
      ]);
      return normalizePredictionSummary({
        algorithm,
        horizonDays: resultPayload.targetDaysAhead || 1,
        stockData,
        rawForecast: forecast,
        fallbackSource: "linearRegressionService.predictWithLinearRegression",
        metadata,
      });
    }

    if (algorithm === "institutionalLinearRegression") {
      const forecast = await invoke("institutionalLinearRegressionService", "predictWithInstitutionalModel", [
        resultPayload.model,
        resultPayload.factorManager,
        stockData,
        5,
      ]);
      return normalizePredictionSummary({
        algorithm,
        horizonDays: 1,
        stockData,
        rawForecast: forecast,
        fallbackSource: "institutionalLinearRegressionService.predictWithInstitutionalModel",
        metadata,
      });
    }

    if (algorithm === "randomForest") {
      const forecast = await invoke("randomForestService", "predictWithRandomForest", [
        resultPayload.model,
        stockData,
        resultPayload.lookbackDays,
        1,
      ]);
      return normalizePredictionSummary({
        algorithm,
        horizonDays: 1,
        stockData,
        rawForecast: forecast,
        fallbackSource: "randomForestService.predictWithRandomForest",
        metadata,
      });
    }

    if (algorithm === "longTermRandomForest") {
      const forecast = await invoke("longTermRandomForestService", "predictWithLongTermRandomForest", [
        resultPayload.model,
        stockData,
        resultPayload.lookbackDays,
        resultPayload.targetDaysAhead,
      ]);
      return normalizePredictionSummary({
        algorithm,
        horizonDays: resultPayload.targetDaysAhead || 22,
        stockData,
        rawForecast: forecast,
        fallbackSource: "longTermRandomForestService.predictWithLongTermRandomForest",
        metadata,
      });
    }

    if (algorithm === "longTermNaiveBayes") {
      const forecast = await invoke("longTermNaiveBayesService", "predictWithLongTermNaiveBayes", [
        resultPayload.model,
        resultPayload.metadata,
        stockData,
        resultPayload.lookbackDays,
        resultPayload.targetDaysAhead,
      ]);
      return normalizePredictionSummary({
        algorithm,
        horizonDays: resultPayload.targetDaysAhead || 1,
        stockData,
        rawForecast: forecast,
        fallbackSource: "longTermNaiveBayesService.predictWithLongTermNaiveBayes",
        metadata,
      });
    }

    if (algorithm === "ensemble") {
      const forecast = await invoke("ensembleService", "predictWithEnsemble", [
        resultPayload.ensemble,
        stockData,
      ]);

      const currentPrice = toFiniteNumber(forecast?.currentPrice) ?? toFiniteNumber(stockData[stockData.length - 1]?.close);
      const score = toFiniteNumber(forecast?.weightedScore);
      const signedReturn = score !== null ? (score - 0.5) * 2 * averageAbsoluteReturn(stockData, 30) : null;
      const heuristicPrice = currentPrice !== null && signedReturn !== null ? currentPrice * (1 + signedReturn) : null;

      return normalizePredictionSummary({
        algorithm,
        horizonDays: 1,
        stockData,
        rawForecast: {
          ...forecast,
          predictedPrice: heuristicPrice,
          predictedReturn: signedReturn,
          percentChange: signedReturn !== null ? signedReturn * 100 : null,
        },
        fallbackSource: "ensembleService.predictWithEnsemble",
        metadata,
      });
    }

    if (algorithm === "longTermEnsemble") {
      const forecast = await invoke("longTermEnsembleService", "predictWithLongTermEnsemble", [
        resultPayload.ensemble,
        stockData,
      ]);
      return normalizePredictionSummary({
        algorithm,
        horizonDays: toFiniteNumber(forecast?.daysAhead) || 5,
        stockData,
        rawForecast: {
          ...forecast,
          percentChange: toFiniteNumber(forecast?.expectedReturn),
        },
        fallbackSource: "longTermEnsembleService.predictWithLongTermEnsemble",
        metadata,
      });
    }

    if (algorithm === "longTermLSTM") {
      const forecast = await invoke("longTermLSTMService", "predictWithTrainedLongTermLSTM", [
        resultPayload.model,
        resultPayload.normParams,
        stockData,
        resultPayload.sequenceLength,
        resultPayload.targetDaysAhead,
        20,
      ]);
      return normalizePredictionSummary({
        algorithm,
        horizonDays: resultPayload.targetDaysAhead || 5,
        stockData,
        rawForecast: forecast,
        fallbackSource: "longTermLSTMService.predictWithTrainedLongTermLSTM",
        metadata,
      });
    }

    if (algorithm === "xgboost") {
      const options = job?.args?.[2] || {};
      const forecast = await invoke("xgBoostStockService", "predictWithXGBoost", [
        resultPayload.predictor,
        stockData,
        null,
        options,
      ]);
      const horizonDays = options?.targetColumn === "target_return_5d" ? 5 : 1;
      return normalizePredictionSummary({
        algorithm,
        horizonDays,
        stockData,
        rawForecast: forecast,
        fallbackSource: "xgBoostStockService.predictWithXGBoost",
        metadata,
      });
    }

    if (algorithm === "bottomPeakLSTM") {
      return buildBottomPeakSummary(stockData, resultPayload, metadata);
    }

    return null;
  } catch (error) {
    return {
      algorithm,
      source: "prediction-summary-error",
      error: error?.message || "Unable to build prediction summary",
    };
  }
};

const toPublicJob = (job) => ({
  id: job.id,
  service: job.service,
  method: job.method,
  metadata: job.metadata,
  status: job.status,
  createdAt: job.createdAt,
  startedAt: job.startedAt,
  finishedAt: job.finishedAt,
  error: job.error,
});

const processQueue = async () => {
  if (isProcessing) {
    return;
  }

  isProcessing = true;

  try {
    while (true) {
      const nextJob = jobs.find((job) => job.status === "queued");
      if (!nextJob) {
        break;
      }

      nextJob.status = "running";
      nextJob.startedAt = new Date().toISOString();

      try {
        const execution = await executeMlMethod({
          service: nextJob.service,
          method: nextJob.method,
          args: nextJob.args,
        });

        const resultPayload = execution?.result;
        const predictionSummary = await buildPredictionSummaryForJob(nextJob, resultPayload);
        const tracePayload = execution?.trace || null;

        if (resultPayload && typeof resultPayload === "object" && !Array.isArray(resultPayload)) {
          nextJob.result = {
            ...resultPayload,
            predictionSummary,
            _build: ML_BUILD,
            _buildBanner: buildBanner,
            _debugTrace: tracePayload,
          };
        } else {
          nextJob.result = {
            value: resultPayload,
            predictionSummary,
            _build: ML_BUILD,
            _buildBanner: buildBanner,
            _debugTrace: tracePayload,
          };
        }
        nextJob.status = "completed";
      } catch (error) {
        nextJob.status = "failed";
        nextJob.error = error.message || "Unknown ML execution error";
        nextJob.result = {
          _build: ML_BUILD,
          _buildBanner: buildBanner,
          _debugTrace: error?.executionTrace || null,
        };
      } finally {
        nextJob.finishedAt = new Date().toISOString();
      }
    }
  } finally {
    isProcessing = false;
  }
};

const addJob = async ({ service, method, args = [], metadata = {} }) => {
  const job = {
    id: randomUUID(),
    service,
    method,
    args,
    metadata,
    status: "queued",
    result: null,
    error: null,
    createdAt: new Date().toISOString(),
    startedAt: null,
    finishedAt: null,
  };

  jobs.push(job);
  processQueue().catch((error) => {
    console.error("ML queue processing error:", error);
  });

  return toPublicJob(job);
};

const getAllJobs = () => jobs.map(toPublicJob).sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));

const getJobById = (id) => {
  const job = jobs.find((item) => item.id === id);
  return job ? toPublicJob(job) : null;
};

const getJobResultById = (id) => {
  const job = jobs.find((item) => item.id === id);
  if (!job) {
    return null;
  }

  return {
    job: toPublicJob(job),
    result: job.result,
    error: job.error,
  };
};

const cancelQueuedJob = (id) => {
  const job = jobs.find((item) => item.id === id);
  if (!job || job.status !== "queued") {
    return null;
  }

  job.status = "cancelled";
  job.finishedAt = new Date().toISOString();
  return toPublicJob(job);
};

module.exports = {
  addJob,
  getAllJobs,
  getJobById,
  getJobResultById,
  cancelQueuedJob,
};