const { execSync } = require("child_process");

const getGitShortSha = () => {
  try {
    return execSync("git rev-parse --short HEAD", {
      stdio: ["ignore", "pipe", "ignore"],
    })
      .toString()
      .trim();
  } catch (error) {
    return "no-git";
  }
};

const gitShortSha = process.env.ML_BUILD_GIT_SHA || getGitShortSha();

const ML_BUILD = {
  version: "ML-2026.05.25-r1",
  label: "Bottom/Peak + Trace Instrumentation",
  emoji: "🧠🚦✨",
  gitShortSha,
  stamp: `ML-2026.05.25-r1+${gitShortSha}`,
};

const buildBanner = `${ML_BUILD.emoji} [${ML_BUILD.version}] ${ML_BUILD.label} 🔖 ${ML_BUILD.gitShortSha}`;

module.exports = {
  ML_BUILD,
  buildBanner,
};
