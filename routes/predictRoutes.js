const express = require("express");
const { enqueuePredict } = require("../controllers/predictController");

const router = express.Router();

router.post("/", enqueuePredict);

module.exports = router;
