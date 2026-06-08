const {
  addJob,
  getAllJobs,
  getJobById,
  getJobResultById,
  cancelQueuedJob,
} = require("../services/mlJobQueueService");

const enqueueJob = async (request, response) => {
  const { service, method, args = [], metadata = {} } = request.body || {};

  if (!service || !method) {
    return response.status(400).json({
      error: "service and method are required",
    });
  }

  if (!Array.isArray(args)) {
    return response.status(400).json({
      error: "args must be an array",
    });
  }

  try {
    const job = await addJob({ service, method, args, metadata });
    return response.status(202).json(job);
  } catch (error) {
    return response.status(500).json({
      error: error.message || "Failed to enqueue ML job",
    });
  }
};

const listJobs = (_request, response) => {
  return response.status(200).json({ jobs: getAllJobs() });
};

const getJob = (request, response) => {
  const job = getJobById(request.params.jobId);
  if (!job) {
    return response.status(404).json({ error: "Job not found" });
  }

  return response.status(200).json(job);
};

const getJobResult = (request, response) => {
  const result = getJobResultById(request.params.jobId);
  if (!result) {
    return response.status(404).json({ error: "Job result not found" });
  }

  return response.status(200).json(result);
};

const cancelJob = (request, response) => {
  const cancelled = cancelQueuedJob(request.params.jobId);
  if (!cancelled) {
    return response.status(409).json({ error: "Only queued jobs can be cancelled" });
  }

  return response.status(200).json(cancelled);
};

module.exports = {
  enqueueJob,
  listJobs,
  getJob,
  getJobResult,
  cancelJob,
};