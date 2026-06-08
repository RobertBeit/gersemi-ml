const path = require("path");
const jiti = require("jiti")(__filename, { interopDefault: true });
const { ML_BUILD, buildBanner } = require("./buildInfo");

const {
  createRemoteRef,
  getRemoteObject,
} = require("./mlRemoteObjectStore");

const SERVICE_MODULES = {
  mlService: path.resolve(__dirname, "./ml/mlService.js"),
  randomForestService: path.resolve(__dirname, "./ml/randomForestService.js"),
  longTermRandomForestService: path.resolve(__dirname, "./ml/longTermRandomForestService.js"),
  longTermNaiveBayesService: path.resolve(__dirname, "./ml/longTermNaiveBayesService.js"),
  ensembleService: path.resolve(__dirname, "./ml/ensembleService.js"),
  longTermEnsembleService: path.resolve(__dirname, "./ml/longTermEnsembleService.js"),
  longTermLSTMService: path.resolve(__dirname, "./ml/longTermLSTMService.js"),
  linearRegressionService: path.resolve(__dirname, "./ml/linearRegressionService.js"),
  institutionalLinearRegressionService: path.resolve(__dirname, "./ml/institutionalLinearRegressionService.js"),
  xgBoostStockService: path.resolve(__dirname, "./ml/xgBoostStockService.js"),
  bottomPeakDetectorService: path.resolve(__dirname, "./ml/bottomPeakDetectorService.js"),
  lstmModel: path.resolve(__dirname, "./ml/lstmModel.js"),
  technicalIndicators: path.resolve(__dirname, "./ml/technicalIndicators.js"),
};

const moduleCache = new Map();

const nowIso = () => new Date().toISOString();

const summarizeValue = (value, depth = 0) => {
  if (value === null || value === undefined) return value;
  if (typeof value === "string") return value.length > 120 ? `${value.slice(0, 117)}...` : value;
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (value instanceof Date) return value.toISOString();

  if (Array.isArray(value)) {
    if (depth >= 2) return { type: "array", length: value.length };
    return {
      type: "array",
      length: value.length,
      sample: value.slice(0, 3).map((item) => summarizeValue(item, depth + 1)),
    };
  }

  if (typeof value === "object") {
    if (value.__remoteRef) {
      return {
        type: "remoteRef",
        ref: value.__remoteRef,
        className: value.className,
      };
    }

    if (depth >= 2) {
      return {
        type: "object",
        keys: Object.keys(value).slice(0, 20),
      };
    }

    const keys = Object.keys(value);
    const summary = { type: "object", keys: keys.slice(0, 20), sample: {} };
    keys.slice(0, 8).forEach((key) => {
      summary.sample[key] = summarizeValue(value[key], depth + 1);
    });
    return summary;
  }

  return { type: typeof value };
};

const pushTraceStep = (trace, stage, detail = {}) => {
  trace.steps.push({ at: nowIso(), stage, detail });
};

const isPrimitive = (value) =>
  value === null ||
  value === undefined ||
  typeof value === "string" ||
  typeof value === "number" ||
  typeof value === "boolean";

const isPlainObject = (value) => {
  if (!value || typeof value !== "object") {
    return false;
  }

  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
};

const deserializeValue = (value) => {
  if (isPrimitive(value)) {
    return value;
  }

  if (Array.isArray(value)) {
    return value.map(deserializeValue);
  }

  if (value && typeof value === "object" && value.__remoteRef) {
    const referencedObject = getRemoteObject(value.__remoteRef);
    if (!referencedObject) {
      throw new Error(`Remote reference not found: ${value.__remoteRef}`);
    }
    return referencedObject;
  }

  if (isPlainObject(value)) {
    const output = {};
    Object.entries(value).forEach(([key, item]) => {
      output[key] = deserializeValue(item);
    });
    return output;
  }

  return value;
};

const buildSnapshot = (value) => {
  if (!value || typeof value !== "object") {
    return {};
  }

  const snapshot = {};
  Object.entries(value).forEach(([key, item]) => {
    if (isPrimitive(item)) {
      snapshot[key] = item;
    }
  });
  return snapshot;
};

const serializeValue = (value) => {
  if (isPrimitive(value)) {
    return value;
  }

  if (value instanceof Date) {
    return value.toISOString();
  }

  if (Array.isArray(value)) {
    return value.map(serializeValue);
  }

  if (isPlainObject(value)) {
    const output = {};
    Object.entries(value).forEach(([key, item]) => {
      if (typeof item === "function") {
        return;
      }
      output[key] = serializeValue(item);
    });
    return output;
  }

  return createRemoteRef(value, value?.constructor?.name, buildSnapshot(value));
};

const getModule = (service) => {
  if (!SERVICE_MODULES[service]) {
    throw new Error(`Unsupported ML service: ${service}`);
  }

  if (!moduleCache.has(service)) {
    moduleCache.set(service, jiti(SERVICE_MODULES[service]));
  }

  return moduleCache.get(service);
};

const invokeRemoteMethod = async (targetRef, methodName, methodArgs = []) => {
  const target = getRemoteObject(targetRef?.__remoteRef || targetRef);
  if (!target) {
    throw new Error("Remote object not found for method invocation");
  }

  if (typeof target[methodName] !== "function") {
    throw new Error(`Method ${methodName} does not exist on remote object`);
  }

  const deserializedArgs = deserializeValue(methodArgs);
  const output = await target[methodName](...deserializedArgs);
  return serializeValue(output);
};

const executeMlMethod = async ({ service, method, args = [] }) => {
  const trace = {
    service,
    method,
    build: ML_BUILD,
    startedAt: nowIso(),
    steps: [],
  };

  console.log(`${buildBanner} ▶️ execute ${service}.${method}`);

  pushTraceStep(trace, "request.received", {
    args: summarizeValue(args),
  });
  pushTraceStep(trace, "build.info", {
    build: ML_BUILD,
    banner: buildBanner,
  });

  if (service === "__remote__" && method === "invoke") {
    const [targetRef, methodName, methodArgs = []] = args;
    pushTraceStep(trace, "remote.invoke.request", {
      targetRef: summarizeValue(targetRef),
      methodName,
      methodArgs: summarizeValue(methodArgs),
    });

    try {
      const result = await invokeRemoteMethod(targetRef, methodName, methodArgs);
      pushTraceStep(trace, "remote.invoke.success", {
        result: summarizeValue(result),
      });
      trace.finishedAt = nowIso();
      console.log(`${buildBanner} ✅ execute ${service}.${method}`);
      return { result, trace };
    } catch (error) {
      pushTraceStep(trace, "remote.invoke.error", {
        message: error?.message || "Unknown remote invocation error",
      });
      trace.finishedAt = nowIso();
      console.error(`${buildBanner} ❌ execute ${service}.${method}:`, error?.message || "Unknown error");
      error.executionTrace = trace;
      throw error;
    }
  }

  try {
    const serviceModule = getModule(service);
    pushTraceStep(trace, "module.loaded", {
      moduleKeys: Object.keys(serviceModule || {}).slice(0, 25),
    });

    const targetMethod = serviceModule[method];

    if (typeof targetMethod !== "function") {
      throw new Error(`Method ${method} is not available in ${service}`);
    }

    pushTraceStep(trace, "method.resolved", { method });

    const deserializedArgs = deserializeValue(args);
    pushTraceStep(trace, "args.deserialized", {
      args: summarizeValue(deserializedArgs),
    });

    const started = Date.now();
    const rawResult = await targetMethod(...deserializedArgs);
    pushTraceStep(trace, "method.executed", {
      durationMs: Date.now() - started,
      rawResult: summarizeValue(rawResult),
    });

    const serializedResult = serializeValue(rawResult);
    pushTraceStep(trace, "result.serialized", {
      serializedResult: summarizeValue(serializedResult),
    });

    trace.finishedAt = nowIso();
    console.log(`${buildBanner} ✅ execute ${service}.${method}`);
    return { result: serializedResult, trace };
  } catch (error) {
    pushTraceStep(trace, "execution.error", {
      message: error?.message || "Unknown ML execution error",
      stack: error?.stack || null,
    });
    trace.finishedAt = nowIso();
    console.error(`${buildBanner} ❌ execute ${service}.${method}:`, error?.message || "Unknown error");
    error.executionTrace = trace;
    throw error;
  }
};

module.exports = {
  executeMlMethod,
};