const express = require("express");
const {
  enqueueJob,
  listJobs,
  getJob,
  getJobResult,
  cancelJob,
} = require("../controllers/mlJobController");

const router = express.Router();

router.post("/", enqueueJob);
router.get("/", listJobs);
router.get("/:jobId", getJob);
router.get("/:jobId/result", getJobResult);
router.post("/:jobId/cancel", cancelJob);

module.exports = router;